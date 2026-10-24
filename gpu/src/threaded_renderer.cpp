/*
 * A GsRenderer on a thread of its own (MakeThreadedGsRenderer): the calls
 * that only hand work over - draws, uploads, decodes, swizzles - are copied
 * into a queue and made on the renderer's thread, in order; a call that
 * answers or reads (MemoryRead, TargetToMemory, DisplayTarget...) waits for
 * the queue to empty and is made on the caller's thread, the renderer's own
 * standing still meanwhile. The caller (the host's GS thread) goes on with
 * the next primitives while Vulkan records the last ones.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ps2gpu/gs_renderer.h"

namespace ps2gpu {

extern void (*g_renderer_thread_start)();     // gpu.cpp (SetRendererThreadStart)

namespace {

/* A chunk of the queue: bytes appended, never zeroed first (a vector's
   resize filled each primitive's bytes before they were copied in - its
   memset and the copy a tenth of the GS thread in a race's crash). */
class Chunk {
public:
    const std::uint8_t* data() const { return bytes_.get(); }
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    void clear() { size_ = 0; }
    void reserve(std::size_t capacity) {
        if (capacity <= capacity_) return;
        std::unique_ptr<std::uint8_t[]> bigger(new std::uint8_t[capacity]);
        if (size_) std::memcpy(bigger.get(), bytes_.get(), size_);
        bytes_ = std::move(bigger);
        capacity_ = capacity;
    }
    std::size_t capacity() const { return capacity_; }
    std::uint8_t* at(std::size_t offset) { return bytes_.get() + offset; }
    void append(const void* data, std::size_t size) {
        if (size_ + size > capacity_) reserve(std::max(capacity_ * 2, size_ + size));
        std::memcpy(bytes_.get() + size_, data, size);
        size_ += size;
    }

private:
    std::unique_ptr<std::uint8_t[]> bytes_;
    std::size_t size_ = 0, capacity_ = 0;
};

class ThreadedGsRenderer final : public GsRenderer {
public:
    explicit ThreadedGsRenderer(std::unique_ptr<GsRenderer> inner) : inner_(std::move(inner)) {
        scale_ = inner_->Scale();
        open_.reserve(kChunk + 4096);
        thread_ = std::thread([this] { Run(); });
    }
    ~ThreadedGsRenderer() override {
        Drain();
        {
            std::lock_guard<std::mutex> hold(lock_);
            stop_ = true;
        }
        wake_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    // ---- handed over

    void DrawTriangles(const GsDrawState& state, const GsVertex* vertices, std::size_t count) override {
        const std::uint32_t n = std::uint32_t(count);
        // The images its batch makes (VulkanGsRenderer::Flush): its target,
        // and its depth buffer as large as the target was made.
        // (The last ones kept: a draw's target and depth buffer are nearly
        // always the last draw's, and the sets' hashing for each was 5% of
        // the GS thread in splitscreen.)
        if (n >= 3) {
            const std::uint64_t target = TargetKey(state.fbp, state.fbw, state.psm);
            if (target != last_target_) {
                last_made_ = Made(state.fbp, state.fbw, state.psm, state.width, state.height);
                last_target_ = target;
            }
            if (state.depth) {
                const std::uint64_t depth = DepthKey(state.zbp, last_made_.w, std::min(last_made_.h, kDepthRows));
                if (depth != last_depth_) {
                    depths_.insert(depth);
                    last_depth_ = depth;
                }
            }
        }
        if (has_last_ && std::memcmp(&state, &last_, sizeof state) == 0) {
            Same(vertices, n);
            return;
        } else {
            std::memcpy(&last_, &state, sizeof state);
            has_last_ = true;
            Op(kDraw, sizeof state + sizeof n + n * sizeof(GsVertex));
            Put(&state, sizeof state);
            Put(&n, sizeof n);
        }
        Put(vertices, n * sizeof(GsVertex));
        End();
    }
    /* The last state again (the bridge's fast path): its 200 bytes not
       compared - 4.5% of the GS thread in splitscreen - where `gouraud`, the
       one field that path changes, is the last one's too. */
    void DrawTrianglesAgain(const GsDrawState& state, const GsVertex* vertices, std::size_t count) override {
        if (!has_last_ || state.gouraud != last_.gouraud) return DrawTriangles(state, vertices, count);
        Same(vertices, std::uint32_t(count));
    }
    void Flush() override { Op(kFlush, 0); End(); }
    void Kick() override {
        Op(kKick, 0);
        End();
        Submit();               // the frame's end: the renderer's thread takes what is left now
        static const bool log = std::getenv("PS2HOST_GPU_THREAD_LOG") != nullptr;
        if (log && ++kicks_ % 60 == 0) {
            std::fprintf(stderr, "[gpu thread] last 60 frames, waits for the queue:");
            for (int i = 0; i < kWhys; ++i)
                if (drains_[i]) std::fprintf(stderr, " %s %u", kWhyNames[i], drains_[i]);
            const unsigned wrong = wrong_.exchange(0);
            if (wrong) std::fprintf(stderr, "; %u write-backs answered wrong", wrong);
            std::fprintf(stderr, "\n");
            std::memset(drains_, 0, sizeof drains_);
        }
    }
    // Answered from what the queue has made by then (the images known),
    // made in turn; the renderer's own answer counted when it differs.
    bool TargetToMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t cols,
                        std::uint32_t rows, std::uint32_t first_row, std::uint32_t first_col) override {
        const std::uint32_t known = targets_.count(TargetKey(fbp, fbw, psm)) ? 1 : 0;
        const std::uint32_t a[8] = {fbp, fbw, psm, cols, rows, first_row, known, first_col};
        Op(kTargetToMemory, sizeof a);
        Put(a, sizeof a);
        End();
        return known != 0;
    }
    bool DepthToMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                       std::uint32_t zpsm, std::uint32_t cols, std::uint32_t rows, std::uint32_t first_row,
                       std::uint32_t first_col) override {
        const std::uint32_t known = depths_.count(DepthKey(zbp, width, height)) ? 1 : 0;
        const std::uint32_t a[10] = {zbp, width, height, fbw, zpsm, cols, rows, first_row, known, first_col};
        Op(kDepthToMemory, sizeof a);
        Put(a, sizeof a);
        End();
        return known != 0;
    }
    void UploadTexture(std::uint64_t key, const std::uint32_t* pixels, std::uint32_t width,
                       std::uint32_t height) override {
        const std::size_t bytes = std::size_t(width) * height * 4;
        Op(kUploadTexture, sizeof key + 8 + bytes);
        Put(&key, sizeof key);
        Put(&width, 4);
        Put(&height, 4);
        Put(pixels, bytes);
        End();
    }
    void UseReplacement(std::uint64_t key, std::uint64_t id) override {
        Op(kUseReplacement, 16);
        Put(&key, 8);
        Put(&id, 8);
        End();
    }
    void UploadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, const std::uint32_t* z) override {
        depths_.insert(DepthKey(zbp, width, height));
        const std::size_t bytes = std::size_t(width) * height * 4;
        Op(kUploadDepth, 12 + bytes);
        Put(&zbp, 4);
        Put(&width, 4);
        Put(&height, 4);
        Put(z, bytes);
        End();
    }
    void UploadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, const std::uint32_t* pixels,
                      std::uint32_t width, std::uint32_t height) override {
        Made(fbp, fbw, psm, width, height);
        const std::size_t bytes = std::size_t(width) * height * 4;
        const std::uint32_t a[5] = {fbp, fbw, psm, width, height};
        Op(kUploadTarget, sizeof a + bytes);
        Put(a, sizeof a);
        Put(pixels, bytes);
        End();
    }
    void ReleaseTexture(std::uint64_t key) override {
        Op(kReleaseTexture, 8);
        Put(&key, 8);
        End();
    }
    void MemoryWrite(std::uint32_t begin, std::uint32_t end, const std::uint8_t* bytes) override {
        const std::size_t size = end > begin ? std::size_t(end - begin) * 256 : 0;
        Op(kMemoryWrite, 8 + size);
        Put(&begin, 4);
        Put(&end, 4);
        Put(bytes, size);
        End();
    }
    void MemoryPatch(const std::uint32_t* words, std::size_t count) override {
        const std::uint64_t n = count;
        Op(kMemoryPatch, 8 + count * 12);
        Put(&n, 8);
        Put(words, count * 12);
        End();
    }
    void TargetFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t width,
                          std::uint32_t height) override {
        Made(fbp, fbw, psm, width, height);
        const std::uint32_t a[5] = {fbp, fbw, psm, width, height};
        Op(kTargetFromMemory, sizeof a);
        Put(a, sizeof a);
        End();
    }
    void DepthFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                         std::uint32_t zpsm) override {
        depths_.insert(DepthKey(zbp, width, height));
        const std::uint32_t a[5] = {zbp, width, height, fbw, zpsm};
        Op(kDepthFromMemory, sizeof a);
        Put(a, sizeof a);
        End();
    }
    void TextureFromMemory(std::uint64_t key, const GsTextureDecode& decode) override {
        // A CLUT of the caller's goes along (the decode reads it later).
        const std::uint32_t with_clut = decode.clut ? 1 : 0;
        Op(kTextureFromMemory, 8 + sizeof decode + 4 + (with_clut ? 1024 : 0));
        Put(&key, 8);
        Put(&decode, sizeof decode);
        Put(&with_clut, 4);
        if (with_clut) Put(decode.clut, 1024);
        End();
    }
    void ClutFromMemory(const GsClutLoad& load) override {
        Op(kClutFromMemory, sizeof load);
        Put(&load, sizeof load);
        End();
    }
    bool CanShuffle() const override { return inner_->CanShuffle(); }
    void ChannelShuffle(const GsShuffle& shuffle) override {
        Op(kChannelShuffle, sizeof shuffle);
        Put(&shuffle, sizeof shuffle);
        End();
    }
    bool CanRunEffects() const override { return inner_->CanRunEffects(); }
    void SetScaleRule(std::uint32_t fbp_begin, std::uint32_t fbp_end, std::uint32_t max_fbw) override {
        if (fbp_begin == rule_[0] && fbp_end == rule_[1] && max_fbw == rule_[2]) return;
        rule_[0] = fbp_begin;
        rule_[1] = fbp_end;
        rule_[2] = max_fbw;
        Drain(0);
        inner_->SetScaleRule(fbp_begin, fbp_end, max_fbw);
        targets_.clear();                           // (every image dropped)
        depths_.clear();
        last_target_ = last_depth_ = ~std::uint64_t(0);
    }
    void RunEffect(const GsEffect& effect) override {
        Op(kRunEffect, sizeof effect);
        Put(&effect, sizeof effect);
        End();
    }
    void ClutRows(const std::uint32_t* clut, std::uint32_t rows) override {
        Op(kClutRows, 4 + 1024);
        Put(&rows, 4);
        Put(clut, 1024);
        End();
    }
    void ReloadKeepsColour(bool keep) override {
        const std::uint32_t k = keep;
        Op(kReloadKeepsColour, 4);
        Put(&k, 4);
        End();
    }
    void TargetRowsFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t width,
                              std::uint32_t height, std::uint32_t first_row, std::uint32_t rows,
                              std::uint32_t first_col, std::uint32_t end_col) override {
        Made(fbp, fbw, psm, width, height);         // (made whole if it was not)
        const std::uint32_t a[9] = {fbp, fbw, psm, width, height, first_row, rows, first_col, end_col};
        Op(kTargetRowsFromMemory, sizeof a);
        Put(a, sizeof a);
        End();
    }
    void DepthRowsFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                             std::uint32_t zpsm, std::uint32_t first_row, std::uint32_t rows) override {
        depths_.insert(DepthKey(zbp, width, height));
        const std::uint32_t a[7] = {zbp, width, height, fbw, zpsm, first_row, rows};
        Op(kDepthRowsFromMemory, sizeof a);
        Put(a, sizeof a);
        End();
    }

    // ---- answered: the queue first

    std::uint32_t Scale() const override { return scale_; }
    void SetScale(std::uint32_t scale) override {
        Drain(0);
        inner_->SetScale(scale);
        if (inner_->Scale() != scale_) {            // every image dropped
            targets_.clear();
            depths_.clear();
            last_target_ = last_depth_ = ~std::uint64_t(0);
        }
        scale_ = inner_->Scale();
    }
    /* Queued with its data, in its place among the draws - not the queue
       drained first and the GS thread waiting for the renderer at every new
       replacement (a view's new textures, one after another). Refused here
       as the renderer would: a format it cannot take. */
    bool UploadReplacement(std::uint64_t id, std::uint32_t format, std::uint32_t width, std::uint32_t height,
                           std::uint32_t levels, const std::uint8_t* data, std::size_t bytes) override {
        if (format > 4 || width == 0 || height == 0 || (format != 0 && !BlockCompression())) return false;
        const std::uint32_t a[4] = {format, width, height, levels};
        Op(kUploadReplacement, 8 + sizeof a + bytes);
        Put(&id, 8);
        Put(a, sizeof a);
        Put(data, bytes);
        End();
        queued_replacements_[id] = std::chrono::steady_clock::now();
        return true;
    }
    // Asked each draw of a replaced texture: the renderer's own answer, safe
    // from here, not after the queue drains (that was half the GS thread
    // with a texture pack) - or one queued and not yet made (for a few
    // seconds at most: one the renderer could not make, or made and evicted
    // unseen, is then asked again). One evicted meanwhile is drawn once as
    // the game has it - as the renderer does for any it no longer holds.
    bool HasReplacement(std::uint64_t id) const override {
        if (inner_->HasReplacement(id)) {
            if (!queued_replacements_.empty()) queued_replacements_.erase(id);
            return true;
        }
        const auto queued = queued_replacements_.find(id);
        if (queued == queued_replacements_.end()) return false;
        if (std::chrono::steady_clock::now() - queued->second < std::chrono::seconds(3)) return true;
        queued_replacements_.erase(queued);
        return false;
    }
    bool BlockCompression() const override {
        if (block_compression_ < 0) {
            Drain(3);
            block_compression_ = inner_->BlockCompression() ? 1 : 0;
        }
        return block_compression_ != 0;
    }
    bool HasTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm) const override {
        Drain(4);
        return inner_->HasTarget(fbp, fbw, psm);
    }
    void MemoryRead(std::uint32_t begin, std::uint32_t end, std::uint8_t* bytes) override {
        Drain(5);
        inner_->MemoryRead(begin, end, bytes);
    }
    bool ReadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height,
                   std::vector<std::uint32_t>* z) override {
        Drain(8);
        return inner_->ReadDepth(zbp, width, height, z);
    }
    bool ReadTexture(std::uint64_t key, std::vector<std::uint32_t>* pixels, std::uint32_t* width,
                     std::uint32_t* height) override {
        Drain(9);
        return inner_->ReadTexture(key, pixels, width, height);
    }
    bool DisplayTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint64_t* image, int* layout,
                       std::uint32_t* width, std::uint32_t* height) override {
        Drain(10);
        return inner_->DisplayTarget(fbp, fbw, psm, image, layout, width, height);
    }
    bool ReadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::vector<std::uint32_t>* pixels,
                    std::uint32_t* width, std::uint32_t* height) override {
        Drain(11);
        return inner_->ReadTarget(fbp, fbw, psm, pixels, width, height);
    }
    bool ReadTargetFull(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::vector<std::uint32_t>* pixels,
                        std::uint32_t* width, std::uint32_t* height) override {
        Drain(12);
        return inner_->ReadTargetFull(fbp, fbw, psm, pixels, width, height);
    }

private:
    /* A primitive of the last state: added to the run of them the open
       chunk ends with when it has as many vertices (3, or a sprite's 6) -
       one record for the run, and on the renderer's thread one
       DrawTriangleRun (most of a crash's ~120,000 primitives a picture come
       in runs of one state: a record and a draw call each were a tenth of
       the GS thread and two thirds of the renderer's). */
    void Same(const GsVertex* vertices, std::uint32_t n) {
        const std::size_t bytes = n * sizeof(GsVertex);
        if (run_chunk_ == chunks_ && run_end_ == open_.size() && run_vertices_ == n) {
            std::uint8_t* head = open_.at(run_at_);
            std::uint32_t size, count;
            std::memcpy(&size, head + 4, 4);
            std::memcpy(&count, head + 12, 4);
            size += std::uint32_t(bytes);
            ++count;
            std::memcpy(head + 4, &size, 4);
            std::memcpy(head + 12, &count, 4);
            Put(vertices, bytes);
        } else {
            run_at_ = open_.size();
            run_chunk_ = chunks_;
            run_vertices_ = n;
            const std::uint32_t head[4] = {kDrawRun, std::uint32_t(8 + bytes), n, 1};
            Put(head, sizeof head);
            Put(vertices, bytes);
        }
        run_end_ = open_.size();
        End();
    }
    // The run the open chunk ends with: where its record starts, the chunk
    // it is in (Submit counts them), where it ends, its primitives' size.
    std::size_t run_at_ = 0, run_end_ = 0;
    std::uint64_t run_chunk_ = ~0ull;
    std::uint32_t run_vertices_ = 0;
    mutable std::uint64_t chunks_ = 0;

    enum : std::uint32_t {
        kDrawRun, kDraw, kDrawSame, kFlush, kKick, kUploadTexture, kUseReplacement, kUploadDepth, kUploadTarget,
        kReleaseTexture, kMemoryWrite, kMemoryPatch, kTargetFromMemory, kDepthFromMemory, kTextureFromMemory,
        kClutFromMemory, kClutRows, kReloadKeepsColour, kTargetRowsFromMemory, kDepthRowsFromMemory,
        kTargetToMemory, kDepthToMemory, kChannelShuffle, kRunEffect, kUploadReplacement,
    };
    // UploadReplacement's queued and not yet seen made, by id (the GS
    // thread's alone); the device's block compression, asked once.
    mutable std::unordered_map<std::uint64_t, std::chrono::steady_clock::time_point> queued_replacements_;
    mutable int block_compression_ = -1;
    // The images the queue has made by its end, as VulkanGsRenderer makes
    // them: a target by (FBP, FBW, PSM) at the size first asked for, a depth
    // buffer by (ZBP, width, height); all dropped by a new scale.
    static constexpr std::uint32_t kDepthRows = 512;    // (gs_renderer.cpp's)
    struct Size { std::uint32_t w, h; };
    static std::uint64_t TargetKey(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm) {
        return std::uint64_t(fbp) << 32 | std::uint64_t(fbw) << 16 | psm;
    }
    static std::uint64_t DepthKey(std::uint32_t zbp, std::uint32_t width, std::uint32_t height) {
        return std::uint64_t(zbp) << 32 | std::uint64_t(width) << 16 | height;
    }
    Size Made(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t width, std::uint32_t height) {
        return targets_.emplace(TargetKey(fbp, fbw, psm), Size{width, height}).first->second;
    }
    // What a chunk holds before it goes to the renderer's thread: a few
    // hundred primitives, so that each hand-over (a lock, a wake) is shared.
    static constexpr std::size_t kChunk = 64 * 1024;
    static constexpr std::size_t kAhead = 32;           // chunks queued at most

    // A record: its operation and its payload's size, then the payload.
    void Op(std::uint32_t op, std::size_t size) {
        const std::uint32_t head[2] = {op, std::uint32_t(size)};
        Put(head, sizeof head);
    }
    void Put(const void* data, std::size_t size) {
        if (size) open_.append(data, size);
    }
    void End() {
        if (open_.size() >= kChunk) Submit();
    }
    // The open chunk to the renderer's thread.
    void Submit() const {
        if (open_.empty()) return;
        {
            std::unique_lock<std::mutex> hold(lock_);
            // No further ahead than this: the caller waits for the renderer
            // rather than queue a frame after another without end.
            if (full_.size() >= kAhead) {
                ++drains_[kWhys - 1];
                drained_.wait(hold, [this] { return full_.size() < kAhead; });
            }
            full_.push_back(std::move(open_));
            if (!spare_.empty()) {
                open_ = std::move(spare_.back());
                spare_.pop_back();
            } else {
                open_ = Chunk();
            }
        }
        open_.clear();
        ++chunks_;
        if (open_.capacity() < kChunk) open_.reserve(kChunk + 4096);
        wake_.notify_one();
    }
    // Everything handed over, made: the renderer's thread idle after it.
    void Drain(int why = kWhys - 1) const {
        ++drains_[why];
        Submit();
        std::unique_lock<std::mutex> hold(lock_);
        drained_.wait(hold, [this] { return full_.empty() && !busy_; });
    }

    void Run() {
        if (g_renderer_thread_start) g_renderer_thread_start();
        std::unique_lock<std::mutex> hold(lock_);
        for (;;) {
            wake_.wait(hold, [this] { return stop_ || !full_.empty(); });
            if (full_.empty()) return;
            Chunk chunk = std::move(full_.front());
            full_.pop_front();
            busy_ = true;
            hold.unlock();
            Play(chunk);
            chunk.clear();
            hold.lock();
            spare_.push_back(std::move(chunk));
            if (full_.empty()) busy_ = false;
            drained_.notify_all();      // (Drain, and a caller held back by kAhead)
        }
    }

    template <typename T>
    static T Take(const std::uint8_t*& at) {
        T value;
        std::memcpy(&value, at, sizeof value);
        at += sizeof value;
        return value;
    }

    void Play(const Chunk& chunk) {
        const std::uint8_t* at = chunk.data();
        const std::uint8_t* const end = at + chunk.size();
        GsRenderer& r = *inner_;
        while (at < end) {
            const std::uint32_t op = Take<std::uint32_t>(at);
            const std::uint32_t size = Take<std::uint32_t>(at);
            const std::uint8_t* const next = at + size;
            switch (op) {
                case kDrawRun: {
                    const std::uint32_t n = Take<std::uint32_t>(at), count = Take<std::uint32_t>(at);
                    vertices_.resize(std::size_t(n) * count);
                    std::memcpy(vertices_.data(), at, vertices_.size() * sizeof(GsVertex));
                    r.DrawTriangleRun(played_, vertices_.data(), n, count);
                    break;
                }
                case kDraw:
                case kDrawSame: {
                    if (op == kDraw) std::memcpy(&played_, at, sizeof played_), at += sizeof played_;
                    const std::uint32_t n = Take<std::uint32_t>(at);
                    vertices_.resize(n);
                    std::memcpy(vertices_.data(), at, n * sizeof(GsVertex));
                    r.DrawTriangles(played_, vertices_.data(), n);
                    break;
                }
                case kFlush: r.Flush(); break;
                case kKick: r.Kick(); break;
                case kUploadTexture: {
                    const auto key = Take<std::uint64_t>(at);
                    const auto w = Take<std::uint32_t>(at), h = Take<std::uint32_t>(at);
                    words_.resize(std::size_t(w) * h);
                    std::memcpy(words_.data(), at, words_.size() * 4);
                    r.UploadTexture(key, words_.data(), w, h);
                    break;
                }
                case kUseReplacement: {
                    const auto key = Take<std::uint64_t>(at);
                    r.UseReplacement(key, Take<std::uint64_t>(at));
                    break;
                }
                case kUploadDepth: {
                    const auto zbp = Take<std::uint32_t>(at), w = Take<std::uint32_t>(at), h = Take<std::uint32_t>(at);
                    words_.resize(std::size_t(w) * h);
                    std::memcpy(words_.data(), at, words_.size() * 4);
                    r.UploadDepth(zbp, w, h, words_.data());
                    break;
                }
                case kUploadTarget: {
                    std::uint32_t a[5];
                    std::memcpy(a, at, sizeof a);
                    at += sizeof a;
                    words_.resize(std::size_t(a[3]) * a[4]);
                    std::memcpy(words_.data(), at, words_.size() * 4);
                    r.UploadTarget(a[0], a[1], a[2], words_.data(), a[3], a[4]);
                    break;
                }
                case kReleaseTexture: r.ReleaseTexture(Take<std::uint64_t>(at)); break;
                case kMemoryWrite: {
                    const auto b = Take<std::uint32_t>(at), e = Take<std::uint32_t>(at);
                    r.MemoryWrite(b, e, at);
                    break;
                }
                case kMemoryPatch: {
                    const auto n = Take<std::uint64_t>(at);
                    words_.resize(std::size_t(n) * 3);
                    std::memcpy(words_.data(), at, words_.size() * 4);
                    r.MemoryPatch(words_.data(), std::size_t(n));
                    break;
                }
                case kTargetFromMemory: {
                    std::uint32_t a[5];
                    std::memcpy(a, at, sizeof a);
                    r.TargetFromMemory(a[0], a[1], a[2], a[3], a[4]);
                    break;
                }
                case kDepthFromMemory: {
                    std::uint32_t a[5];
                    std::memcpy(a, at, sizeof a);
                    r.DepthFromMemory(a[0], a[1], a[2], a[3], a[4]);
                    break;
                }
                case kTextureFromMemory: {
                    const auto key = Take<std::uint64_t>(at);
                    GsTextureDecode d = Take<GsTextureDecode>(at);
                    const auto with_clut = Take<std::uint32_t>(at);
                    if (with_clut) {
                        std::memcpy(clut_, at, sizeof clut_);
                        d.clut = clut_;
                    }
                    r.TextureFromMemory(key, d);
                    break;
                }
                case kClutFromMemory: r.ClutFromMemory(Take<GsClutLoad>(at)); break;
                case kChannelShuffle: r.ChannelShuffle(Take<GsShuffle>(at)); break;
                case kRunEffect: r.RunEffect(Take<GsEffect>(at)); break;
                case kUploadReplacement: {
                    const auto id = Take<std::uint64_t>(at);
                    std::uint32_t a[4];
                    std::memcpy(a, at, sizeof a);
                    at += sizeof a;
                    r.UploadReplacement(id, a[0], a[1], a[2], a[3], at, std::size_t(next - at));
                    break;
                }
                case kClutRows: {
                    const auto rows = Take<std::uint32_t>(at);
                    std::memcpy(clut_, at, sizeof clut_);
                    r.ClutRows(clut_, rows);
                    break;
                }
                case kReloadKeepsColour: r.ReloadKeepsColour(Take<std::uint32_t>(at) != 0); break;
                case kTargetRowsFromMemory: {
                    std::uint32_t a[9];
                    std::memcpy(a, at, sizeof a);
                    r.TargetRowsFromMemory(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
                    break;
                }
                case kDepthRowsFromMemory: {
                    std::uint32_t a[7];
                    std::memcpy(a, at, sizeof a);
                    r.DepthRowsFromMemory(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
                    break;
                }
                case kTargetToMemory: {
                    std::uint32_t a[8];
                    std::memcpy(a, at, sizeof a);
                    if (r.TargetToMemory(a[0], a[1], a[2], a[3], a[4], a[5], a[7]) != (a[6] != 0)) ++wrong_;
                    break;
                }
                case kDepthToMemory: {
                    std::uint32_t a[10];
                    std::memcpy(a, at, sizeof a);
                    if (r.DepthToMemory(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[9]) != (a[8] != 0)) ++wrong_;
                    break;
                }
                default: break;
            }
            at = next;
        }
    }

    std::unique_ptr<GsRenderer> inner_;
    std::uint32_t scale_ = 1;
    std::unordered_map<std::uint64_t, Size> targets_;
    std::unordered_set<std::uint64_t> depths_;
    std::uint64_t last_target_ = ~std::uint64_t(0), last_depth_ = ~std::uint64_t(0);
    Size last_made_{};
    std::atomic<unsigned> wrong_{0};
    // PS2HOST_GPU_THREAD_LOG: the waits for the queue, by what waited.
    static constexpr int kWhys = 14;
    static constexpr const char* kWhyNames[kWhys] = {
        "SetScale", "UploadReplacement", "HasReplacement", "BlockCompression", "HasTarget", "MemoryRead",
        "TargetToMemory", "DepthToMemory", "ReadDepth", "ReadTexture", "DisplayTarget", "ReadTarget",
        "ReadTargetFull", "queue full"};
    mutable unsigned drains_[kWhys] = {};
    std::uint32_t rule_[3] = {};      // SetScaleRule's, as last passed on
    unsigned kicks_ = 0;
    // The caller's side.
    mutable Chunk open_;
    GsDrawState last_;
    bool has_last_ = false;
    // Between the two.
    mutable std::mutex lock_;
    mutable std::condition_variable wake_, drained_;
    mutable std::deque<Chunk> full_;
    mutable std::vector<Chunk> spare_;
    mutable bool busy_ = false;
    bool stop_ = false;
    // The renderer's thread.
    GsDrawState played_;
    std::vector<GsVertex> vertices_;
    std::vector<std::uint32_t> words_;
    std::uint32_t clut_[256] = {};
    std::thread thread_;
};

}  // namespace

std::unique_ptr<GsRenderer> MakeThreadedGsRenderer(std::unique_ptr<GsRenderer> inner) {
    if (!inner) return inner;
    return std::make_unique<ThreadedGsRenderer>(std::move(inner));
}

}  // namespace ps2gpu

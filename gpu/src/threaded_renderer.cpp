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

namespace {

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
        if (n >= 3) {
            const Size made = Made(state.fbp, state.fbw, state.psm, state.width, state.height);
            if (state.depth) depths_.insert(DepthKey(state.zbp, made.w, std::min(made.h, kDepthRows)));
        }
        if (has_last_ && std::memcmp(&state, &last_, sizeof state) == 0) {
            Op(kDrawSame, sizeof n + n * sizeof(GsVertex));
            Put(&n, sizeof n);
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
                        std::uint32_t rows, std::uint32_t first_row) override {
        const std::uint32_t known = targets_.count(TargetKey(fbp, fbw, psm)) ? 1 : 0;
        const std::uint32_t a[7] = {fbp, fbw, psm, cols, rows, first_row, known};
        Op(kTargetToMemory, sizeof a);
        Put(a, sizeof a);
        End();
        return known != 0;
    }
    bool DepthToMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                       std::uint32_t zpsm, std::uint32_t cols, std::uint32_t rows, std::uint32_t first_row) override {
        const std::uint32_t known = depths_.count(DepthKey(zbp, width, height)) ? 1 : 0;
        const std::uint32_t a[9] = {zbp, width, height, fbw, zpsm, cols, rows, first_row, known};
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
                              std::uint32_t height, std::uint32_t first_row, std::uint32_t rows) override {
        Made(fbp, fbw, psm, width, height);         // (made whole if it was not)
        const std::uint32_t a[7] = {fbp, fbw, psm, width, height, first_row, rows};
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
        }
        scale_ = inner_->Scale();
    }
    bool UploadReplacement(std::uint64_t id, std::uint32_t format, std::uint32_t width, std::uint32_t height,
                           std::uint32_t levels, const std::uint8_t* data, std::size_t bytes) override {
        Drain(1);
        return inner_->UploadReplacement(id, format, width, height, levels, data, bytes);
    }
    bool HasReplacement(std::uint64_t id) const override {
        Drain(2);
        return inner_->HasReplacement(id);
    }
    bool BlockCompression() const override {
        Drain(3);
        return inner_->BlockCompression();
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
    enum : std::uint32_t {
        kDraw, kDrawSame, kFlush, kKick, kUploadTexture, kUseReplacement, kUploadDepth, kUploadTarget,
        kReleaseTexture, kMemoryWrite, kMemoryPatch, kTargetFromMemory, kDepthFromMemory, kTextureFromMemory,
        kClutFromMemory, kClutRows, kReloadKeepsColour, kTargetRowsFromMemory, kDepthRowsFromMemory,
        kTargetToMemory, kDepthToMemory,
    };
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
        if (!size) return;
        const std::size_t at = open_.size();
        open_.resize(at + size);
        std::memcpy(open_.data() + at, data, size);
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
                open_ = std::vector<std::uint8_t>();
            }
        }
        open_.clear();
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
        std::unique_lock<std::mutex> hold(lock_);
        for (;;) {
            wake_.wait(hold, [this] { return stop_ || !full_.empty(); });
            if (full_.empty()) return;
            std::vector<std::uint8_t> chunk = std::move(full_.front());
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

    void Play(const std::vector<std::uint8_t>& chunk) {
        const std::uint8_t* at = chunk.data();
        const std::uint8_t* const end = at + chunk.size();
        GsRenderer& r = *inner_;
        while (at < end) {
            const std::uint32_t op = Take<std::uint32_t>(at);
            const std::uint32_t size = Take<std::uint32_t>(at);
            const std::uint8_t* const next = at + size;
            switch (op) {
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
                case kClutRows: {
                    const auto rows = Take<std::uint32_t>(at);
                    std::memcpy(clut_, at, sizeof clut_);
                    r.ClutRows(clut_, rows);
                    break;
                }
                case kReloadKeepsColour: r.ReloadKeepsColour(Take<std::uint32_t>(at) != 0); break;
                case kTargetRowsFromMemory: {
                    std::uint32_t a[7];
                    std::memcpy(a, at, sizeof a);
                    r.TargetRowsFromMemory(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
                    break;
                }
                case kDepthRowsFromMemory: {
                    std::uint32_t a[7];
                    std::memcpy(a, at, sizeof a);
                    r.DepthRowsFromMemory(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
                    break;
                }
                case kTargetToMemory: {
                    std::uint32_t a[7];
                    std::memcpy(a, at, sizeof a);
                    if (r.TargetToMemory(a[0], a[1], a[2], a[3], a[4], a[5]) != (a[6] != 0)) ++wrong_;
                    break;
                }
                case kDepthToMemory: {
                    std::uint32_t a[9];
                    std::memcpy(a, at, sizeof a);
                    if (r.DepthToMemory(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]) != (a[8] != 0)) ++wrong_;
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
    std::atomic<unsigned> wrong_{0};
    // PS2HOST_GPU_THREAD_LOG: the waits for the queue, by what waited.
    static constexpr int kWhys = 14;
    static constexpr const char* kWhyNames[kWhys] = {
        "SetScale", "UploadReplacement", "HasReplacement", "BlockCompression", "HasTarget", "MemoryRead",
        "TargetToMemory", "DepthToMemory", "ReadDepth", "ReadTexture", "DisplayTarget", "ReadTarget",
        "ReadTargetFull", "queue full"};
    mutable unsigned drains_[kWhys] = {};
    unsigned kicks_ = 0;
    // The caller's side.
    mutable std::vector<std::uint8_t> open_;
    GsDrawState last_;
    bool has_last_ = false;
    // Between the two.
    mutable std::mutex lock_;
    mutable std::condition_variable wake_, drained_;
    mutable std::deque<std::vector<std::uint8_t>> full_;
    mutable std::vector<std::vector<std::uint8_t>> spare_;
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

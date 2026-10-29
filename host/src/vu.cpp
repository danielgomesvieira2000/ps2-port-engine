/*
 * The vector units in micro mode (vu.h).
 *
 * Instruction fields (both halves): dest x y z w in bits 24-21, ft/it 20-16,
 * fs/is 15-11, fd/id 10-6. The upper's flags: I 31 (the lower word is a float
 * for the I register), E 30 (end), M 29, D 28, T 27. Instruction addresses are
 * in bytes (pairs of 8); data addresses in quadwords.
 */

#include "ps2host/vu.h"

#include <algorithm>
#include <map>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <xmmintrin.h>
#include <unordered_set>
#include <vector>

#include "ps2host/memory.h"
#include "ps2host/profile.h"
#include "ps2host/vu_core.h"
#include "ps2rt/generated.h"
#include "ps2rt/vu_math.h"


namespace ps2host {

namespace {

constexpr Vu::u32 kNone = ~0u;
constexpr int kRegR = 20, kRegI = 21, kRegQ = 22;
constexpr Vu::u32 kCop2 = 0x4A000000u;       // a macro-mode COP2 operation

Vu::u32 Field(Vu::u32 word, int low, int count) { return (word >> low) & ((1u << count) - 1); }
std::int32_t Imm11(Vu::u32 word) { return std::int32_t(word << 21) >> 21; }

// The vector fields a pair reads, for its stalls: (register, lanes as dest bits).
struct Reads {
    int n = 0;
    Vu::u32 reg[6], lanes[6];
    void Add(Vu::u32 r, Vu::u32 l) {
        if (r && l) { reg[n] = r; lanes[n] = l; ++n; }
    }
};

Vu::u32 LaneBit(Vu::u32 lane) { return 8u >> lane; }    // x 0 .. w 3, as dest bits

void UpperReads(Vu::u32 word, Reads& r) {
    const Vu::u32 funct = word & 0x3F, dest = Field(word, 21, 4);
    const Vu::u32 fs = Field(word, 11, 5), ft = Field(word, 16, 5);
    if (funct < 0x1C) { r.Add(fs, dest); r.Add(ft, LaneBit(funct & 3)); return; }   // the broadcasts
    if (funct < 0x28) { r.Add(fs, dest); return; }                                   // Q and I forms
    if (funct < 0x30) {
        const Vu::u32 lanes = funct == 0x2E ? 0xE : dest;                             // OPMSUB: xyz
        r.Add(fs, lanes);
        r.Add(ft, lanes);
        return;
    }
    if (funct < 0x3C) return;
    const Vu::u32 index = (word & 3) | ((word >> 4) & 0x7C);
    if (index == 0x2F) return;                                                       // NOP
    if (index == 0x1F) { r.Add(fs, 0xE); r.Add(ft, 0x1); return; }                   // CLIP
    if (index < 0x10 || (index >= 0x18 && index < 0x1C)) {                           // the broadcasts
        r.Add(fs, dest);
        r.Add(ft, LaneBit(index & 3));
        return;
    }
    if (index < 0x28) { r.Add(fs, dest); return; }      // ITOF FTOI ABS, Q and I forms
    const Vu::u32 lanes = index == 0x2E ? 0xE : dest;   // OPMULA: xyz
    r.Add(fs, lanes);
    r.Add(ft, lanes);
}

void LowerReads(Vu::u32 word, Reads& r) {
    const Vu::u32 op = word >> 25, dest = Field(word, 21, 4);
    const Vu::u32 is = Field(word, 11, 5), it = Field(word, 16, 5);
    const Vu::u32 fsf = LaneBit(Field(word, 21, 2)), ftf = LaneBit(Field(word, 23, 2));
    if (op == 0x01) { r.Add(is, dest); return; }        // SQ
    if (op != 0x40 || (word & 0x3F) < 0x3C) return;
    const Vu::u32 index = (word & 3) | ((word >> 4) & 0x7C);
    switch (index) {
        case 0x30: r.Add(is, dest); return;                                          // MOVE
        case 0x31: r.Add(is, ((dest << 3) | (dest >> 1)) & 0xF); return;             // MR32: x from y ..
        case 0x35: case 0x37: r.Add(is, dest); return;                               // SQI SQD
        case 0x38: case 0x3A: r.Add(is, fsf); r.Add(it, ftf); return;                // DIV RSQRT
        case 0x39: r.Add(it, ftf); return;                                           // SQRT
        case 0x3C: case 0x42: case 0x43: r.Add(is, fsf); return;                     // MTIR RINIT RXOR
        case 0x70: case 0x71: case 0x72: case 0x73: r.Add(is, 0xE); return;         // ESADD .. ERLENG
        case 0x74: r.Add(is, 0xC); return;                                           // EATANxy
        case 0x75: r.Add(is, 0xA); return;                                           // EATANxz
        case 0x76: r.Add(is, 0xF); return;                                           // ESUM
        case 0x78: case 0x79: case 0x7A: case 0x7C: case 0x7D: case 0x7E:
            r.Add(is, fsf);
            return;
        default: return;
    }
}

bool IsEfu(Vu::u32 word) {
    if ((word >> 25) != 0x40 || (word & 0x3F) < 0x3C) return false;
    const Vu::u32 index = (word & 3) | ((word >> 4) & 0x7C);
    return index >= 0x70 && index <= 0x7E && index != 0x77 && index != 0x7B;
}

bool IsDivide(Vu::u32 word) {
    if ((word >> 25) != 0x40 || (word & 0x3F) < 0x3C) return false;
    const Vu::u32 index = (word & 3) | ((word >> 4) & 0x7C);
    return index >= 0x38 && index <= 0x3A;
}

bool IsWait(Vu::u32 word, Vu::u32 which) {             // 0x3B WAITQ, 0x7B WAITP
    return (word >> 25) == 0x40 && (word & 0x3F) >= 0x3C && ((word & 3) | ((word >> 4) & 0x7C)) == which;
}

constexpr int kFmacLatency = 4;

// A pair, decoded for its stalls once: what it reads and whether it waits on
// the divider or the EFU. Kept per address and checked against the words
// there (an MPG may have replaced them).
struct DecodedPair {
    Vu::u32 lower = 0, upper = 0;
    bool valid = false;
    bool immediate = false, q_waits = false, p_waits = false;
    Reads reads;
    bool has_fmac = false;          // the upper is an FMAC operation, decoded
    lifted::FmacOp fmac;
};

DecodedPair& Decoded(std::vector<DecodedPair>& cache, Vu::u32 pc, Vu::u32 lower, Vu::u32 upper) {
    DecodedPair& d = cache[pc / 8];
    if (d.valid && d.lower == lower && d.upper == upper) return d;
    d = DecodedPair{};
    d.lower = lower;
    d.upper = upper;
    d.valid = true;
    d.immediate = (upper & 0x80000000u) != 0;          // I bit: the lower is I's value
    UpperReads(upper, d.reads);
    d.has_fmac = lifted::DecodeFmac(kCop2 | (upper & 0x01FFFFFF), &d.fmac);
    if (!d.immediate) {
        LowerReads(lower, d.reads);
        d.q_waits = IsDivide(lower) || IsWait(lower, 0x3B);
        d.p_waits = IsEfu(lower) || IsWait(lower, 0x7B);
    }
    return d;
}

// ---------------------------------------------------------------- compiled programs

std::unordered_map<std::uint64_t, std::vector<const VuCompiled*>>& Compiled() {
    static std::unordered_map<std::uint64_t, std::vector<const VuCompiled*>> programs;
    return programs;
}

// A program's blocks by pair address (null where none starts), made once.
const std::vector<VuBlock>& BlockTable(const VuCompiled* program, std::size_t pairs, bool fast) {
    // A thread's own (VU1 runs on VIF1's worker, VU0 on the EE's).
    thread_local std::unordered_map<const VuCompiled*, std::vector<VuBlock>> tables[2];
    const VuBlock* blocks = fast && program->fast_blocks ? program->fast_blocks : program->blocks;
    std::vector<VuBlock>& table = tables[blocks == program->blocks ? 0 : 1][program];
    if (table.empty()) {
        table.assign(pairs, nullptr);
        for (std::uint32_t i = 0; i < program->block_count; ++i)
            if (program->starts[i] / 8 < pairs) table[program->starts[i] / 8] = blocks[i];
    }
    return table;
}

// Whether instruction memory still holds the code a program was compiled from.
bool Holds(const Vu::Span& micro, const Vu::u32* code) {
    const Vu::u32 runs = *code++;
    for (Vu::u32 i = 0; i < runs; ++i) {
        const Vu::u32 base = code[0], words = code[1];
        if (base + 4 * words > micro.size() || std::memcmp(micro.data() + base, code + 2, 4 * words) != 0)
            return false;
        code += 2 + words;
    }
    return true;
}

// ---------------------------------------------------------------- recording (--vu-record)

std::FILE* g_record = nullptr;

/* PS2HOST_VU_BY_BLOCK (--vu-bench): the time, entries and pairs of each
   compiled block, by its program's entry and its address. */
struct BlockTime {
    std::uint64_t ticks = 0, entries = 0, pairs = 0;
    std::uint32_t entry = 0, pc = 0;
};
std::unordered_map<const void*, BlockTime>* g_block_times = nullptr;

/* PS2HOST_VU_PC_COUNTS=file (--vu-bench): how many times the interpreter ran
   each pair, a count a pair of the run's program (the loops' iterations). */
std::vector<std::uint64_t>* g_pc_counts = nullptr;

/* PS2HOST_VU_PERTURB (--vu-bench): when the interpreter reaches `pc` the
   nth time, one lane of one VF register is changed (x1.25, or 1 if 0) - does
   a loop's later work depend on it? `vi` is a VI register's value at each
   arrival (the loop's pointer), to place what changed. */
struct PerturbHook {
    std::uint32_t pc = 0;
    std::uint64_t nth = 0, seen = 0;
    int reg = 0, lane = 0, vi_reg = 0;
    std::vector<std::uint32_t> vi;
};
PerturbHook* g_perturb = nullptr;

std::unordered_set<std::size_t>& Recorded() {
    static std::unordered_set<std::size_t> seen;
    return seen;
}

bool IsBranch(Vu::u32 lower, bool* conditional, bool* known) {
    const Vu::u32 op = lower >> 25;
    *conditional = op == 0x28 || op == 0x29 || (op >= 0x2C && op <= 0x2F);
    *known = op == 0x20 || op == 0x21 || *conditional;
    return *known || op == 0x24 || op == 0x25;
}

// The pairs a run started at `entry` can reach: on from each, both ways at a
// branch (its delay slot run first), as far as the pair after an E bit; a
// jump to a register's address is followed no further (the run leaves what
// was compiled there, and the interpreter goes on).
std::vector<Vu::u32> Reachable(const Vu::Span& micro, Vu::u32 entry) {
    const Vu::u32 mask = Vu::u32(micro.size() - 1);
    std::vector<std::uint8_t> seen(micro.size() / 8, 0);     // 1: followed on, 2: included only
    std::vector<std::pair<Vu::u32, bool>> work{{entry, true}};
    auto words = [&](Vu::u32 pc, Vu::u32* lower, Vu::u32* upper) {
        std::memcpy(lower, micro.data() + pc, 4);
        std::memcpy(upper, micro.data() + pc + 4, 4);
    };
    while (!work.empty()) {
        const auto [pc, follow] = work.back();
        work.pop_back();
        std::uint8_t& mark = seen[pc / 8];
        if (mark & 1 || (!follow && mark)) continue;
        mark |= follow ? 1 : 2;
        if (!follow) continue;
        Vu::u32 lower, upper;
        words(pc, &lower, &upper);
        const Vu::u32 next = (pc + 8) & mask;
        bool conditional = false, known = false;
        if (upper & 0x40000000u) {
            work.push_back({next, false});
        } else if (!(upper & 0x80000000u) && IsBranch(lower, &conditional, &known)) {
            work.push_back({next, false});
            if (known) work.push_back({(pc + 8 + Vu::u32(Imm11(lower) * 8)) & mask, true});
            if (conditional) work.push_back({(pc + 16) & mask, true});
        } else {
            work.push_back({next, true});
        }
    }
    std::vector<Vu::u32> pcs;
    for (Vu::u32 i = 0; i < seen.size(); ++i)
        if (seen[i]) pcs.push_back(i * 8);
    return pcs;
}

void RecordProgram(const Vu::Span& micro, int unit, Vu::u32 entry) {
    const std::vector<Vu::u32> pcs = Reachable(micro, entry);
    std::string line = "vu" + std::to_string(unit);
    char text[32];
    std::snprintf(text, sizeof text, " %X ", unsigned(entry));
    line += text;
    for (std::size_t i = 0; i < pcs.size(); ++i) {
        const bool starts = i == 0 || pcs[i - 1] + 8 != pcs[i];
        if (starts) {
            std::snprintf(text, sizeof text, "%s%X:", i ? ";" : "", unsigned(pcs[i]));
            line += text;
        }
        Vu::u32 lower, upper;
        std::memcpy(&lower, micro.data() + pcs[i], 4);
        std::memcpy(&upper, micro.data() + pcs[i] + 4, 4);
        std::snprintf(text, sizeof text, "%s%X,%X", starts ? "" : ",", unsigned(lower), unsigned(upper));
        line += text;
    }
    if (!Recorded().insert(std::hash<std::string>{}(line)).second) return;
    std::fprintf(g_record, "%s\n", line.c_str());
    std::fflush(g_record);
}

}  // namespace

void RegisterVuProgram(const VuCompiled* program) {
    Compiled()[std::uint64_t(program->unit) << 32 | program->entry].push_back(program);
}

void Vu::RecordPrograms(const std::string& path) {
    g_record = std::fopen(path.c_str(), "a");
    if (!g_record) std::fprintf(stderr, "vu: cannot write %s\n", path.c_str());
}

bool Vu::SameState(const Vu& o, bool flags) const {
    // PS2HOST_VU_SAY_DIFF: what differs, on stderr (the host tests' failures).
    static const bool say = std::getenv("PS2HOST_VU_SAY_DIFF") != nullptr;
    if (say) {
        for (int r = 0; r < 32; ++r)
            for (int lane = 0; lane < 4; ++lane) {
                if (regs.vf[r][lane] != o.regs.vf[r][lane]) std::fprintf(stderr, "  vf%d.%d %08X %08X\n", r, lane, regs.vf[r][lane], o.regs.vf[r][lane]);
                if (std::max(vf_ready_[r][lane], cycle_) != std::max(o.vf_ready_[r][lane], o.cycle_))
                    std::fprintf(stderr, "  ready vf%d.%d %llu %llu (cycles %llu %llu)\n", r, lane,
                                 (unsigned long long)vf_ready_[r][lane], (unsigned long long)o.vf_ready_[r][lane],
                                 (unsigned long long)cycle_, (unsigned long long)o.cycle_);
            }
        for (int r = 0; r < 32; ++r)
            if (regs.vc[r] != o.regs.vc[r]) std::fprintf(stderr, "  vc%d %X %X\n", r, regs.vc[r], o.regs.vc[r]);
        if (cycle_ != o.cycle_) std::fprintf(stderr, "  cycle %llu %llu\n", (unsigned long long)cycle_, (unsigned long long)o.cycle_);
        if (mac != o.mac || status != o.status || clip != o.clip) std::fprintf(stderr, "  flags\n");
        if (std::memcmp(data.data(), o.data.data(), data.size())) std::fprintf(stderr, "  data\n");
        if (resume != o.resume) std::fprintf(stderr, "  resume %X %X\n", resume, o.resume);
    }
    auto same = [](const void* a, const void* b, std::size_t n) { return std::memcmp(a, b, n) == 0; };
    if (!same(regs.vf, o.regs.vf, sizeof regs.vf) || !same(regs.vc, o.regs.vc, sizeof regs.vc) ||
        !same(regs.vacc, o.regs.vacc, sizeof regs.vacc))
        return false;
    if ((flags && (mac != o.mac || status != o.status)) || clip != o.clip || p != o.p || resume != o.resume)
        return false;
    // Landing times as far as they can still stall: past ones are all alike
    // (compiled code leaves out the times nothing reads - vurecomp's
    // untimed_writes - and a time left out stays an earlier, past one).
    for (int r = 0; r < 32; ++r)
        for (int lane = 0; lane < 4; ++lane)
            if (std::max(vf_ready_[r][lane], cycle_) != std::max(o.vf_ready_[r][lane], o.cycle_)) return false;
    if (cycle_ != o.cycle_ || q_next_ != o.q_next_ ||
        p_next_ != o.p_next_ || q_ready_ != o.q_ready_ || p_ready_ != o.p_ready_ || q_busy_ != o.q_busy_ ||
        p_busy_ != o.p_busy_)
        return false;
    if (!flags)     // CLIP's in flight still, without the MAC writes between
        return clip_latest_ == o.clip_latest_ && ClipsInFlight() == o.ClipsInFlight() && data.size() == o.data.size() &&
               same(data.data(), o.data.data(), data.size()) && kicks_ == o.kicks_;
    if (mac_latest_ != o.mac_latest_ || status_latest_ != o.status_latest_ || clip_latest_ != o.clip_latest_ ||
        flag_count_ != o.flag_count_)
        return false;
    for (int i = 0; i < flag_count_; ++i) {
        const FlagWrite &a = flag_queue_[(flag_head_ + i) & 7], &b = o.flag_queue_[(o.flag_head_ + i) & 7];
        if (a.ready != b.ready || a.mac != b.mac || a.status != b.status || a.clip != b.clip ||
            a.sets_mac != b.sets_mac || a.sets_clip != b.sets_clip)
            return false;
    }
    return data.size() == o.data.size() && same(data.data(), o.data.data(), data.size()) && kicks_ == o.kicks_;
}

Vu::Vu(int index)
    : own_micro_(index ? 16384 : 4096), own_data_(index ? 16384 : 4096), data_mask_(index ? 0x3FF : 0xFF) {
    micro = {own_micro_.data(), own_micro_.size()};
    data = {own_data_.data(), own_data_.size()};
    regs.vf[0][3] = 0x3F800000u;    // vf0 = (0, 0, 0, 1)
    start_counts_.assign(micro.size() / 8, 0);
    entered_.assign(micro.size() / 8, Entered{});
}

void Vu::Attach(std::uint8_t* micro_memory, std::uint8_t* data_memory) {
    if (micro_memory) {
        micro.p = micro_memory;
        ++micro_generation_;        // other memory: what was found in it, found again
    }
    if (data_memory) data.p = data_memory;
}

const std::map<Vu::u32, Vu::u64>& Vu::starts() const {
    starts_.clear();
    for (std::size_t i = 0; i < start_counts_.size(); ++i)
        if (start_counts_[i]) starts_[u32(i * 8)] = start_counts_[i];
    return starts_;
}

Vu::u32* Vu::Quad(u32 address) {
    return reinterpret_cast<u32*>(data.data() + std::size_t(address & data_mask_) * 16);
}

void Vu::Kick(u32 address) {
    ++kicks_;
    // Data memory's size wraps the address, as the VU does: the runs below
    // are counted from it (an address past the end ran them past the memory).
    address &= data_mask_;
    for (int packets = 0; packets < 4096; ++packets) {
        const u32* tag = Quad(address);
        const u64 low = u64(tag[0]) | u64(tag[1]) << 32, high = u64(tag[2]) | u64(tag[3]) << 32;
        const u32 nloop = u32(low & 0x7FFF), eop = u32(low >> 15) & 1, flg = u32(low >> 58) & 3;
        const u32 nreg = u32(low >> 60) ? u32(low >> 60) : 16;
        const u32 qwords = flg == 0 ? nloop * nreg : flg == 1 ? (nloop * nreg + 1) / 2 : nloop;
        if (accesses)
            for (u32 i = 0; i <= qwords; ++i) accesses->Read((address + i) & data_mask_, 0xF);
        if (on_kick_block) {
            // The tag and its data, in runs up to where data memory wraps.
            for (u32 left = 1 + qwords; left;) {
                const u32 run = std::min(left, data_mask_ + 1 - address);
                on_kick_block(Quad(address), run * 4);
                address = (address + run) & data_mask_;
                left -= run;
            }
            if (eop) return;
            continue;
        }
        if (on_kick) on_kick(low, high);
        address = (address + 1) & data_mask_;
        for (u32 i = 0; i < qwords; ++i) {
            const u32* q = Quad(address);
            if (on_kick) on_kick(u64(q[0]) | u64(q[1]) << 32, u64(q[2]) | u64(q[3]) << 32);
            address = (address + 1) & data_mask_;
        }
        if (eop) return;
    }
}

// The upper instruction, run on the registers; says which vector register and
// fields it wrote (if any), for the lower's writes to give way to.
void Vu::Upper(u32 word, int* written_reg, u32* written_lanes, bool* set_flags, const lifted::FmacOp* fmac) {
    *written_reg = -1;
    *written_lanes = 0;
    *set_flags = false;
    const u32 funct = word & 0x3F, dest = Field(word, 21, 4);
    const u32 ft = Field(word, 16, 5), fd = Field(word, 6, 5);
    bool flags = false;
    int target = -1;
    if (funct < 0x3C) {
        if (funct >= 0x30) return;                          // not upper operations
        target = int(fd);
        // MAX and MINI leave the flags alone.
        flags = !((funct >= 0x10 && funct <= 0x17) || funct == 0x1D || funct == 0x1F ||
                  funct == 0x2B || funct == 0x2F);
    } else {
        const u32 index = (word & 3) | ((word >> 4) & 0x7C);
        if (index == 0x1F) {                                // CLIP: fs.xyz against |ft.w|
            const u32* v = regs.vf[Field(word, 11, 5)];
            const double w = std::fabs(lifted::Value(regs.vf[ft][3]));
            u32 judgement = 0;
            for (int lane = 0; lane < 3; ++lane) {
                const double x = lifted::Value(v[lane]);
                if (x > w) judgement |= 1u << (2 * lane);
                if (x < -w) judgement |= 2u << (2 * lane);
            }
            clip_latest_ = ((clip_latest_ << 6) | judgement) & 0xFFFFFF;
            *set_flags = true;
            return;
        }
        if (index == 0x2F) return;                          // NOP
        if ((index >= 0x10 && index <= 0x17) || index == 0x1D) target = int(ft);   // ITOF, FTOI, ABS
        else flags = true;                                  // the accumulator forms
    }
    if (fmac) lifted::RunFmac(regs, *fmac);
    else lifted::Cop2(regs, kCop2 | (word & 0x01FFFFFF));
    if (target > 0) {
        *written_reg = target;
        *written_lanes = dest;
    }
    if (flags && target == 0) {
        // Into VF0: the flags are the result's, which went nowhere.
        if (fmac) {
            lifted::FmacResult(regs, *fmac, discarded_);
        } else {
            lifted::FmacOp decoded;
            if (lifted::DecodeFmac(kCop2 | (word & 0x01FFFFFF), &decoded)) {
                lifted::FmacResult(regs, decoded, discarded_);
            } else if (funct == 0x2E) {                     // OPMSUB (it wrote nothing: its operands stand)
                const u32 *a = regs.vf[Field(word, 11, 5)], *b = regs.vf[ft], *acc = regs.vacc;
                const u32 out[3] = {lifted::Binary(lifted::kMsub, a[1], b[2], acc[0]),
                                    lifted::Binary(lifted::kMsub, a[2], b[0], acc[1]),
                                    lifted::Binary(lifted::kMsub, a[0], b[1], acc[2])};
                for (int lane = 0; lane < 3; ++lane) discarded_[lane] = (dest & (8u >> lane)) ? out[lane] : 0;
                discarded_[3] = 0;
            }
        }
    }
    if (flags) {
        // MAC: zero (bits 3-0) and sign (7-4) of each written field, x highest.
        const u32* out = target > 0 ? regs.vf[target] : target == 0 ? discarded_ : regs.vacc;
        u32 m = 0;
        for (int lane = 0; lane < 4; ++lane) {
            if (!(dest & (8u >> lane))) continue;
            const u32 bits = lifted::Clean(out[lane]);
            if ((bits & 0x7FFFFFFFu) == 0) m |= 8u >> lane;
            if (bits & 0x80000000u) m |= 0x80u >> lane;
        }
        mac_latest_ = m;
        status_latest_ = (status_latest_ & ~0x3u) | ((m & 0xF) ? 1u : 0u) | ((m & 0xF0) ? 2u : 0u);
        status_latest_ |= (status_latest_ & 3u) << 6;       // the sticky copies
        *set_flags = true;
    }
}

void Vu::Lower(u32 word, u32 pc, Pending& out, u32* branch) {
    const u32 op = word >> 25, dest = Field(word, 21, 4);
    const u32 it = Field(word, 16, 5), is = Field(word, 11, 5);
    u32* vi = regs.vc;
    auto setvi = [&](u32 r, u32 v) { if (r > 0 && r < 16) vi[r] = v & 0xFFFF; };
    // The first field set in dest: what ILW and ILWR read.
    const u32 first_lane = dest & 8 ? 8 : dest & 4 ? 4 : dest & 2 ? 2 : dest & 1 ? 1 : 0;
    auto load = [&](u32 address) {
        if (accesses) accesses->Read(address & data_mask_, dest);
        const u32* q = Quad(address);
        out.vf = int(it);
        out.lanes = dest;
        std::memcpy(out.value, q, 16);
    };
    auto store = [&](u32 address, const u32* v) {
        if (accesses) accesses->Write(address & data_mask_, dest);
        u32* q = Quad(address);
        for (int lane = 0; lane < 4; ++lane)
            if (dest & (8u >> lane)) q[lane] = v[lane];
    };
    auto through_cop2 = [&]() {
        // Run it as macro mode would, then hold its writes back until the upper
        // has read its operands.
        u32 before_vc[32], before_vf[4];
        std::memcpy(before_vc, regs.vc, sizeof before_vc);
        std::memcpy(before_vf, regs.vf[it], sizeof before_vf);
        lifted::Cop2(regs, kCop2 | (word & 0x01FFFFFF));
        if (std::memcmp(before_vf, regs.vf[it], sizeof before_vf) != 0) {
            out.vf = int(it);
            out.lanes = dest;
            std::memcpy(out.value, regs.vf[it], 16);
            std::memcpy(regs.vf[it], before_vf, sizeof before_vf);
        }
        if (std::memcmp(before_vc, regs.vc, sizeof before_vc) != 0) {
            out.vc = true;
            std::memcpy(out.vc_after, regs.vc, sizeof out.vc_after);
            std::memcpy(regs.vc, before_vc, sizeof before_vc);
        }
    };
    auto efu = [&](double value) {
        out.p = true;
        out.p_value = lifted::Result(value);
        static const int kLatency[16] = {11, 18, 18, 24, 54, 54, 12, 0, 12, 18, 12, 0, 29, 54, 44, 0};
        out.p_latency = kLatency[(((word & 3) | ((word >> 4) & 0x7C)) - 0x70) & 15];
    };
    const u32 target = (pc + 8 + u32(Imm11(word) * 8)) & u32(micro.size() - 1);
    const std::int16_t a = std::int16_t(vi[is]), b = std::int16_t(vi[it]);
    switch (op) {
        case 0x40: {
            const u32 funct = word & 0x3F;
            if (funct >= 0x30 && funct <= 0x35) {
                // IADD ISUB IADDI IAND IOR, as the macro-mode unit has them: they
                // write vi1-vi15, which no upper reads, so they need not wait.
                const u32 id = Field(word, 6, 5);
                if (funct == 0x32) {
                    const std::int32_t imm = (id & 0x10) ? std::int32_t(id) - 32 : std::int32_t(id);
                    setvi(it, u32(std::int32_t(vi[is]) + imm));
                    return;
                }
                const u32 x = vi[is], y = vi[it];
                setvi(id, funct == 0x30 ? x + y : funct == 0x31 ? x - y : funct == 0x34 ? (x & y) : (x | y));
                return;
            }
            if (funct < 0x3C) { through_cop2(); return; }
            const u32 index = (word & 3) | ((word >> 4) & 0x7C);
            const u32 fsf = Field(word, 21, 2);
            const u32* v = regs.vf[is];
            switch (index) {
                case 0x34: load(vi[is]); setvi(is, vi[is] + 1); return;                 // LQI
                case 0x35: store(vi[it], regs.vf[is]); setvi(it, vi[it] + 1); return;   // SQI
                case 0x36: setvi(is, vi[is] - 1); load(vi[is]); return;                 // LQD
                case 0x37: setvi(it, vi[it] - 1); store(vi[it], regs.vf[is]); return;   // SQD
                case 0x3E: {                                                            // ILWR
                    if (accesses) accesses->Read(vi[is] & data_mask_, first_lane);
                    const u32* q = Quad(vi[is]);
                    for (int lane = 0; lane < 4; ++lane)
                        if (dest & (8u >> lane)) { setvi(it, q[lane]); break; }
                    return;
                }
                case 0x3F: {                                                            // ISWR
                    if (accesses) accesses->Write(vi[is] & data_mask_, dest);
                    u32* q = Quad(vi[is]);
                    for (int lane = 0; lane < 4; ++lane)
                        if (dest & (8u >> lane)) q[lane] = vi[it] & 0xFFFF;
                    return;
                }
                case 0x64: {                                                            // MFP
                    out.vf = int(it);
                    out.lanes = dest;
                    for (u32& x : out.value) x = p;
                    return;
                }
                case 0x68: setvi(it, top); return;                                      // XTOP
                case 0x69: setvi(it, itop); return;                                     // XITOP
                case 0x6C: Kick(vi[is]); return;                                        // XGKICK
                case 0x70: case 0x71: case 0x72: case 0x73: {   // ESADD ERSADD ELENG ERLENG
                    const double x = lifted::Value(v[0]), y = lifted::Value(v[1]), z = lifted::Value(v[2]);
                    const double sum = x * x + y * y + z * z;
                    efu(index == 0x70 ? sum : index == 0x71 ? 1.0 / sum
                        : index == 0x72 ? std::sqrt(sum) : 1.0 / std::sqrt(sum));
                    return;
                }
                case 0x74: efu(std::atan2(lifted::Value(v[1]), lifted::Value(v[0]))); return;  // EATANxy
                case 0x75: efu(std::atan2(lifted::Value(v[2]), lifted::Value(v[0]))); return;  // EATANxz
                case 0x76: efu(lifted::Value(v[0]) + lifted::Value(v[1]) + lifted::Value(v[2]) +
                               lifted::Value(v[3])); return;                                   // ESUM
                case 0x78: efu(std::sqrt(std::fabs(lifted::Value(v[fsf])))); return;          // ESQRT
                case 0x79: efu(1.0 / std::sqrt(std::fabs(lifted::Value(v[fsf])))); return;    // ERSQRT
                case 0x7A: efu(1.0 / lifted::Value(v[fsf])); return;                          // ERCPR
                case 0x7B: return;                                                            // WAITP: a stall (Run)
                case 0x7C: efu(std::sin(lifted::Value(v[fsf]))); return;                      // ESIN
                case 0x7D: efu(std::atan(lifted::Value(v[fsf]))); return;                     // EATAN
                case 0x7E: efu(std::exp(-lifted::Value(v[fsf]))); return;                     // EEXP
                case 0x38: case 0x39: case 0x3A:                                            // DIV SQRT RSQRT
                    through_cop2();
                    out.q_latency = index == 0x3A ? 13 : 7;
                    return;
                case 0x3B: return;                                                            // WAITQ: a stall (Run)
                default: through_cop2(); return;    // MOVE MR32 MTIR MFIR R*
            }
        }
        case 0x00: load(vi[is] + u32(Imm11(word))); return;                     // LQ
        case 0x01: store(vi[it] + u32(Imm11(word)), regs.vf[is]); return;       // SQ
        case 0x04: {                                                             // ILW
            if (accesses) accesses->Read((vi[is] + u32(Imm11(word))) & data_mask_, first_lane);
            const u32* q = Quad(vi[is] + u32(Imm11(word)));
            for (int lane = 0; lane < 4; ++lane)
                if (dest & (8u >> lane)) { setvi(it, q[lane]); break; }
            return;
        }
        case 0x05: {                                                             // ISW
            if (accesses) accesses->Write((vi[is] + u32(Imm11(word))) & data_mask_, dest);
            u32* q = Quad(vi[is] + u32(Imm11(word)));
            for (int lane = 0; lane < 4; ++lane)
                if (dest & (8u >> lane)) q[lane] = vi[it] & 0xFFFF;
            return;
        }
        case 0x08: setvi(it, vi[is] + ((Field(word, 21, 4) << 11) | Field(word, 0, 11))); return;  // IADDIU
        case 0x09: setvi(it, vi[is] - ((Field(word, 21, 4) << 11) | Field(word, 0, 11))); return;  // ISUBIU
        case 0x10: setvi(1, (clip & 0xFFFFFF) == (word & 0xFFFFFF)); return;                        // FCEQ
        case 0x11: clip = clip_latest_ = word & 0xFFFFFF; return;                                   // FCSET
        case 0x12: setvi(1, (clip & word & 0xFFFFFF) != 0); return;                                 // FCAND
        case 0x13: setvi(1, ((clip | word) & 0xFFFFFF) == 0xFFFFFF); return;                        // FCOR
        case 0x14: case 0x15: case 0x16: case 0x17: {                           // FSEQ FSSET FSAND FSOR
            const u32 imm = (Field(word, 21, 1) << 11) | Field(word, 0, 11);
            if (op == 0x14) setvi(it, (status & 0xFFF) == imm);
            else if (op == 0x15) {
                status = (status & 0x3F) | (imm & 0xFC0);
                status_latest_ = (status_latest_ & 0x3F) | (imm & 0xFC0);
            }
            else if (op == 0x16) setvi(it, status & imm);
            else setvi(it, ((status | imm) & 0xFFF) == 0xFFF);
            return;
        }
        case 0x18: setvi(it, (vi[is] & 0xFFFF) == (mac & 0xFFFF)); return;     // FMEQ
        case 0x1A: setvi(it, vi[is] & mac); return;                              // FMAND
        case 0x1B: setvi(it, vi[is] | mac); return;                              // FMOR
        case 0x1C: setvi(it, clip & 0xFFF); return;                              // FCGET
        case 0x20: *branch = target; return;                                     // B
        case 0x21: setvi(it, (pc + 16) / 8); *branch = target; return;           // BAL
        case 0x24: *branch = (vi[is] * 8) & u32(micro.size() - 1); return;       // JR
        case 0x25: {                                                             // JALR
            const u32 to = (vi[is] * 8) & u32(micro.size() - 1);
            setvi(it, (pc + 16) / 8);
            *branch = to;
            return;
        }
        case 0x28: if (a == b) *branch = target; return;                         // IBEQ
        case 0x29: if (a != b) *branch = target; return;                         // IBNE
        case 0x2C: if (a < 0) *branch = target; return;                          // IBLTZ
        case 0x2D: if (a > 0) *branch = target; return;                          // IBGTZ
        case 0x2E: if (a <= 0) *branch = target; return;                         // IBLEZ
        case 0x2F: if (a >= 0) *branch = target; return;                         // IBGEZ
        default: return;
    }
}

std::vector<std::pair<Vu::u64, Vu::u32>> Vu::ClipsInFlight() const {
    std::vector<std::pair<u64, u32>> out;
    for (int i = 0; i < flag_count_; ++i) {
        const FlagWrite& f = flag_queue_[(flag_head_ + i) & 7];
        if (f.sets_clip) out.emplace_back(f.ready, f.clip);
    }
    return out;
}

void Vu::LandAll() {
    cycle_ += 64;                   // time passes after the program: what was in flight lands
    Land();
}

namespace {

// --vu-snapshot and --vu-bench (Vu::Snapshots, Vu::Bench): a run as it starts.
struct Snapshot {
    u32 pc;
    u32 flags[6];                       // mac, status, clip, p, top, itop
    u32 vf[32][4], vc[32], vacc[4];
};
std::FILE* g_snapshots = nullptr;
std::atomic<u64> g_snapshot_from{~u64(0)};
std::atomic<bool> g_snapshotting{false};
u32 g_snapshots_left = 0;
std::atomic<u32> g_snapshots_written{0};     // runs written so far (the frames' marks)

}  // namespace

void Vu::Snapshots(const std::string& path, u64 from, u32 count) {
    g_snapshots = std::fopen(path.c_str(), "wb");
    if (!g_snapshots) {
        std::fprintf(stderr, "vu: cannot write %s\n", path.c_str());
        return;
    }
    g_snapshot_from = from;
    g_snapshots_left = count;
}

void Vu::SnapshotFrame(u64 frame) {
    if (g_snapshots && frame >= g_snapshot_from.load()) {
        g_snapshotting = true;
        // Where each frame begins among the runs (the EE's frame: the
        // worker's runs may lag it by a little).
        std::fprintf(stderr, "[vu-snapshot] frame %llu at run %u\n", (unsigned long long)frame,
                     g_snapshots_written.load(std::memory_order_relaxed));
    }
}

int Vu::Bench(const std::string& path, int passes, bool fast) {
    if (std::getenv("PS2HOST_VU_DEPS")) return Dependencies(path);
    std::FILE* in = std::fopen(path.c_str(), "rb");
    if (!in) {
        std::fprintf(stderr, "vu: cannot read %s\n", path.c_str());
        return 1;
    }
    struct Run {
        Snapshot s;
        std::vector<std::uint8_t> micro, data;
        u32 image = 0;                  // FNV-1a of the microcode (PS2HOST_VU_BY_PROGRAM)
    };
    std::vector<Run> runs;
    for (;;) {
        Run r;
        r.micro.resize(16384);
        r.data.resize(16384);
        if (std::fread(&r.s, sizeof r.s, 1, in) != 1 || std::fread(r.micro.data(), 1, 16384, in) != 16384 ||
            std::fread(r.data.data(), 1, 16384, in) != 16384)
            break;
        r.image = 2166136261u;
        for (std::uint8_t b : r.micro) r.image = (r.image ^ b) * 16777619u;
        runs.push_back(std::move(r));
    }
    std::fclose(in);
    Vu vu(1);
    vu.fast = fast;
    vu.on_kick = [](u64, u64) {};
    vu.on_kick_block = [](const u32*, u32) {};
    // PS2HOST_VU_FIND=x,y,z,w (hex words): the run that leaves that quadword
    // in data memory, the pair that first stores it (the interpreter, run
    // again with a growing limit) and the registers before and after.
    auto load = [&](const Run& r) {
        std::memcpy(vu.micro.data(), r.micro.data(), 16384);
        vu.MicroWritten();
        std::memcpy(vu.data.data(), r.data.data(), 16384);
        std::memcpy(vu.regs.vf, r.s.vf, sizeof r.s.vf);
        std::memcpy(vu.regs.vc, r.s.vc, sizeof r.s.vc);
        std::memcpy(vu.regs.vacc, r.s.vacc, sizeof r.s.vacc);
        vu.mac = r.s.flags[0];
        vu.status = r.s.flags[1];
        vu.clip = r.s.flags[2];
        vu.p = r.s.flags[3];
        vu.top = r.s.flags[4];
        vu.itop = r.s.flags[5];
    };
    // PS2HOST_VU_PERTURB=entry,pc,nth,reg,lane,vi[,skip] (hex entry and pc):
    // the first run from `entry` (after `skip` of them) that reaches `pc` at
    // least nth+8 times, run twice by the interpreter - as it is, and with
    // vf`reg`'s `lane` (0 = x) changed at the nth arrival - and the data
    // quadwords that differ, with vi`vi` at each arrival.
    if (const char* perturb = std::getenv("PS2HOST_VU_PERTURB")) {
        unsigned entry = 0, pc = 0, skip = 0;
        unsigned long long nth = 0;
        int reg = 0, lane = 0, vi_reg = 0;
        if (std::sscanf(perturb, "%x,%x,%llu,%d,%d,%d,%u", &entry, &pc, &nth, &reg, &lane, &vi_reg, &skip) < 6)
            return 64;
        vu.use_compiled = false;
        PerturbHook hook;
        hook.pc = pc;
        hook.reg = reg;
        hook.lane = lane;
        hook.vi_reg = vi_reg;
        g_perturb = &hook;
        for (std::size_t n = 0; n < runs.size(); ++n) {
            if (runs[n].s.pc != entry) continue;
            load(runs[n]);
            hook.nth = 0;
            hook.seen = 0;
            hook.vi.clear();
            vu.Run(entry);
            if (hook.seen < nth + 8 || skip-- > 0) continue;
            const std::vector<std::uint8_t> before(vu.data.data(), vu.data.data() + 16384);
            const std::vector<std::uint32_t> pointers = hook.vi;
            load(runs[n]);
            hook.nth = nth;
            hook.seen = 0;
            vu.Run(entry);
            g_perturb = nullptr;
            std::printf("run %zu: %llu arrivals at %04X, vf%d lane %d changed at arrival %llu\n", n,
                        (unsigned long long)pointers.size(), pc, reg, lane, nth);
            std::printf("vi%d at arrivals:", vi_reg);
            for (std::uint32_t v : pointers) std::printf(" %X", v);
            std::printf("\nquadwords that differ:");
            int differ = 0;
            for (u32 q = 0; q < 1024; ++q)
                if (std::memcmp(before.data() + q * 16, vu.data.data() + q * 16, 16) != 0) {
                    std::printf(" %X", q);
                    ++differ;
                }
            std::printf("\n%d quadwords differ\n", differ);
            return 0;
        }
        std::printf("no run from %04X reaches %04X %llu times\n", entry, pc, nth + 8);
        return 1;
    }
    if (const char* find = std::getenv("PS2HOST_VU_FIND")) {
        u32 want[4] = {};
        if (std::sscanf(find, "%x,%x,%x,%x", &want[0], &want[1], &want[2], &want[3]) != 4) return 64;
        vu.use_compiled = false;
        auto holds = [&]() -> int {
            for (u32 q = 0; q < 1024; ++q) {
                u32 w[4];
                std::memcpy(w, vu.data.data() + q * 16, 16);
                if ((w[0] & 0xFF) == want[0] && (w[1] & 0xFF) == want[1] && (w[2] & 0xFF) == want[2] && (w[3] & 0xFF) == want[3]) return int(q);    // (PACKED RGBAQ: the low bytes)
                if (want[1] == 0xFFFFFFFFu && (w[0] == want[0] || w[1] == want[0] || w[2] == want[0] || w[3] == want[0])) return int(q);    // x,FFFFFFFF,0,0: one word anywhere
            }
            return -1;
        };
        auto dump = [&](const char* when) {
            std::printf("  %s (pc %04X):\n", when, vu.resume * 1);
            for (int i = 0; i < 32; ++i) {
                float f[4];
                std::memcpy(f, vu.regs.vf[i], 16);
                std::printf("    vf%02d %12g %12g %12g %12g   [%08X %08X %08X %08X]\n", i, f[0], f[1], f[2], f[3],
                            vu.regs.vf[i][0], vu.regs.vf[i][1], vu.regs.vf[i][2], vu.regs.vf[i][3]);
            }
            std::printf("    vi:");
            for (int i = 0; i < 16; ++i) std::printf(" %X", vu.regs.vc[i] & 0xFFFF);
            float q, acc[4];
            std::memcpy(&q, &vu.regs.vc[22], 4);
            std::memcpy(acc, vu.regs.vacc, 16);
            std::printf("\n    Q %g  I %08X  ACC %g %g %g %g\n", q, vu.regs.vc[21], acc[0], acc[1], acc[2], acc[3]);
        };
        // PS2HOST_VU_STEPS=run,first,last: that run's registers after each pair count.
        if (const char* steps = std::getenv("PS2HOST_VU_STEPS")) {
            unsigned run = 0; unsigned long long first = 0, last = 0;
            std::sscanf(steps, "%u,%llu,%llu", &run, &first, &last);
            for (unsigned long long k = first; k <= last && run < runs.size(); ++k) {
                load(runs[run]);
                vu.Run(runs[run].s.pc, k);
                float acc[4];
                std::memcpy(acc, vu.regs.vacc, 16);
                std::printf("%llu pc %04X ACC %g %g %g %g", k, vu.resume * 1, acc[0], acc[1], acc[2], acc[3]);
                for (int r : {18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}) {
                    float f[4];
                    std::memcpy(f, vu.regs.vf[r], 16);
                    std::printf(" | vf%d %g %g %g %g", r, f[0], f[1], f[2], f[3]);
                }
                std::printf("\n");
            }
            return 0;
        }
        for (std::size_t n = 0; n < runs.size(); ++n) {
            load(runs[n]);
            if (holds() >= 0) continue;                 // there before it ran
            const u64 total = vu.Run(runs[n].s.pc);
            const int q = holds();
            if (q < 0) continue;
            std::printf("run %zu (pc %04X, %llu pairs) stores it at quadword %d\n", n, runs[n].s.pc,
                        (unsigned long long)total, q);
            u64 lo = 0, hi = total;                     // after hi pairs it is there; after lo not
            while (hi - lo > 1) {
                const u64 mid = (lo + hi) / 2;
                load(runs[n]);
                vu.Run(runs[n].s.pc, mid);
                (holds() >= 0 ? hi : lo) = mid;
            }
            load(runs[n]);
            vu.Run(runs[n].s.pc, lo);
            std::printf("  the storing pair: after %llu pairs\n", (unsigned long long)lo);
            dump("before the storing pair");
            load(runs[n]);
            vu.Run(runs[n].s.pc, hi);
            dump("after it");
            return 0;
        }
        std::printf("no run stores it\n");
        return 1;
    }
    double best = 0;
    u64 pairs = 0, hash = 0, data_hash = 0;
    // PS2HOST_VU_BY_PROGRAM: the best pass's time and pairs by the
    // microprogram - the microcode loaded (its hash: a game loads several
    // at the same addresses) and the address each run started at - most first.
    static const bool by_program = std::getenv("PS2HOST_VU_BY_PROGRAM") != nullptr;
    std::unordered_map<const void*, BlockTime> block_times;
    if (std::getenv("PS2HOST_VU_BY_BLOCK")) g_block_times = &block_times;
    // PS2HOST_VU_PC_COUNTS: one pass, interpreted, every pair counted by program.
    const char* pc_counts_path = std::getenv("PS2HOST_VU_PC_COUNTS");
    std::map<u64, std::vector<std::uint64_t>> pc_counts;
    if (pc_counts_path) {
        passes = 1;
        vu.use_compiled = false;
    }
    std::map<u64, std::pair<double, u64>> programs, best_programs;
    std::map<u64, u64> program_runs;
    for (int pass = 0; pass < passes; ++pass) {
        double seconds = 0;
        programs.clear();
        u64 pass_pairs = 0, pass_hash = 1469598103934665603ull, pass_data_hash = 1469598103934665603ull;
        std::uint8_t* micro_now = nullptr;
        for (Run& r : runs) {
            if (micro_now != r.micro.data() &&
                (!micro_now || std::memcmp(micro_now, r.micro.data(), 16384) != 0)) {
                std::memcpy(vu.micro.data(), r.micro.data(), 16384);
                vu.MicroWritten();
            }
            micro_now = r.micro.data();
            std::memcpy(vu.data.data(), r.data.data(), 16384);
            std::memcpy(vu.regs.vf, r.s.vf, sizeof r.s.vf);
            std::memcpy(vu.regs.vc, r.s.vc, sizeof r.s.vc);
            std::memcpy(vu.regs.vacc, r.s.vacc, sizeof r.s.vacc);
            vu.mac = r.s.flags[0];
            vu.status = r.s.flags[1];
            vu.clip = r.s.flags[2];
            vu.p = r.s.flags[3];
            vu.top = r.s.flags[4];
            vu.itop = r.s.flags[5];
            if (pc_counts_path) {
                auto& counts = pc_counts[u64(r.image) << 32 | r.s.pc];
                counts.resize(16384 / 8);
                g_pc_counts = &counts;
            }
            const auto began = std::chrono::steady_clock::now();
            const u64 run_pairs = vu.Run(r.s.pc);
            const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
            pass_pairs += run_pairs;
            seconds += took;
            if (by_program) {
                const u64 program = u64(r.image) << 32 | r.s.pc;
                auto& [t, p] = programs[program];
                t += took;
                p += run_pairs;
                if (pass == 0) ++program_runs[program];
            }
            // FNV-1a, a word at a time, over what the run left: registers, flags, data memory.
            // The second (data_hash) all but MAC and STATUS: what fast mode
            // (--vu-fast) keeps exact.
            auto mix = [&pass_hash, &pass_data_hash](const void* p, std::size_t n, bool data = true) {
                const std::uint8_t* b = static_cast<const std::uint8_t*>(p);
                for (std::size_t i = 0; i + 8 <= n; i += 8) {      // (every size here is a multiple of 8)
                    std::uint64_t word;
                    std::memcpy(&word, b + i, 8);
                    pass_hash = (pass_hash ^ word) * 1099511628211ull;
                    if (data) pass_data_hash = (pass_data_hash ^ word) * 1099511628211ull;
                }
            };
            mix(vu.regs.vf, sizeof vu.regs.vf);
            mix(vu.regs.vc, sizeof vu.regs.vc);
            mix(vu.regs.vacc, sizeof vu.regs.vacc);
            const u32 flags[4] = {vu.mac, vu.status, vu.clip, vu.p};
            mix(flags, sizeof flags, false);
            const u32 kept[2] = {vu.clip, vu.p};
            pass_data_hash = (pass_data_hash ^ (u64(kept[0]) | u64(kept[1]) << 32)) * 1099511628211ull;
            mix(vu.data.data(), 16384);
        }
        if (pass == 0 || seconds < best) {
            best = seconds;
            best_programs = programs;
        }
        pairs = pass_pairs;
        hash = pass_hash;
        data_hash = pass_data_hash;
    }
    if (by_program) {
        std::vector<std::pair<double, u64>> order;
        for (const auto& [program, tp] : best_programs) order.emplace_back(tp.first, program);
        std::sort(order.rbegin(), order.rend());
        double so_far = 0;
        for (const auto& [t, program] : order) {
            so_far += t;
            std::printf("  program %08X:%04X: %.2f ms (%.1f%%, %.1f%% so far), %llu runs, %llu pairs, %.2f ns a pair\n",
                        u32(program >> 32), u32(program), t * 1000, best ? 100 * t / best : 0,
                        best ? 100 * so_far / best : 0, (unsigned long long)program_runs[program],
                        (unsigned long long)best_programs[program].second,
                        best_programs[program].second ? t * 1e9 / double(best_programs[program].second) : 0.0);
        }
    }
    if (pc_counts_path) {
        g_pc_counts = nullptr;
        if (std::FILE* out = std::fopen(pc_counts_path, "w")) {
            // A line a pair run: image, entry, pair address (bytes), count.
            for (const auto& [program, counts] : pc_counts)
                for (std::size_t i = 0; i < counts.size(); ++i)
                    if (counts[i])
                        std::fprintf(out, "%08X %04X %04X %llu\n", u32(program >> 32), u32(program), u32(i * 8),
                                     (unsigned long long)counts[i]);
            std::fclose(out);
        }
    }
    if (g_block_times) {
        // Over every pass (the clock around each block slows them all alike).
        std::vector<const BlockTime*> order;
        u64 total = 0;
        for (const auto& [block, b] : block_times) { order.push_back(&b); total += b.ticks; }
        std::sort(order.begin(), order.end(), [](const BlockTime* a, const BlockTime* b) { return a->ticks > b->ticks; });
        double shown = 0;
        for (std::size_t i = 0; i < order.size() && i < 40; ++i) {
            const BlockTime& b = *order[i];
            shown += 100.0 * double(b.ticks) / double(total);
            std::printf("  block %04X:%04X %.1f%% (%.1f%% so far), %llu entries, %.1f pairs an entry, %.2f ns a pair\n",
                        b.entry, b.pc, 100.0 * double(b.ticks) / double(total), shown,
                        (unsigned long long)b.entries, double(b.pairs) / double(b.entries),
                        b.pairs ? double(b.ticks) / double(b.pairs) : 0.0);
        }
        u64 entries = 0, block_pairs = 0;
        for (const auto& [block, b] : block_times) { entries += b.entries; block_pairs += b.pairs; }
        std::printf("  %zu blocks, %llu entries a pass, %.1f pairs an entry\n", block_times.size(),
                    (unsigned long long)(entries / u64(passes)), entries ? double(block_pairs) / double(entries) : 0.0);
        g_block_times = nullptr;
    }
    std::printf("vu-bench: %zu runs, %llu pairs, best pass %.1f ms, %.2f ns a pair; state hash %016llX"
                " (but MAC and STATUS: %016llX)\n",
                runs.size(), (unsigned long long)pairs, best * 1000, pairs ? best * 1e9 / double(pairs) : 0.0,
                (unsigned long long)hash, (unsigned long long)data_hash);
    return 0;
}

namespace {
/* The VUs' arithmetic as the console's: rounded toward zero, denormals flushed
   (inputs and results) - the host's SSE control word, for a run. Burnout 3's
   particles reduce an angle by adding 2^23 and taking it away again: rounded
   to nearest it came out in [0, 1], not [-0.5, 0.5], and the sine polynomial
   after it gave corners twelve times too long - huge smoke puffs lit with
   wrapped colours (the race's white-outs and stretched textures). */
struct VuRounding {
    unsigned saved = lifted::FpuControl();
    VuRounding() { lifted::SetFpuControl(lifted::VuFpuControl(saved)); }
    ~VuRounding() { lifted::SetFpuControl(saved); }
};
}  // namespace

/* PS2HOST_VU_DEPS (vu.h, Dependencies). Each run of the snapshot, in order,
   on the interpreter, from its own snapshot with the pipeline settled:
   - data memory: the fields it reads before writing them itself (Accesses),
     and the fields it writes;
   - registers and flags: run again with one changed to random bits (twice);
     where what it sends (XGKICK), leaves in memory or leaves in the other
     registers differs, it read it. A register is written where it ends
     other than it began (one left as it was needs no writer: its value is
     the one before);
   - between two runs, the fields of data memory that differ from where the
     first left them are the VIF's (or the EE's): fresh, nobody's run.
   A run depends on the runs that last wrote what it reads. Its weight: its
   pairs. Reported: the runs that depend on none, the longest chain against
   the whole, workers side by side (each run starting once what it needs is
   done and a worker is free, in order), and what the edges are. */
int Vu::Dependencies(const std::string& path) {
    std::FILE* in = std::fopen(path.c_str(), "rb");
    if (!in) {
        std::fprintf(stderr, "vu: cannot read %s\n", path.c_str());
        return 1;
    }
    struct Run {
        Snapshot s;
        std::vector<std::uint8_t> micro, data;
    };
    std::vector<Run> runs;
    for (;;) {
        Run r;
        r.micro.resize(16384);
        r.data.resize(16384);
        if (std::fread(&r.s, sizeof r.s, 1, in) != 1 || std::fread(r.micro.data(), 1, 16384, in) != 16384 ||
            std::fread(r.data.data(), 1, 16384, in) != 16384)
            break;
        runs.push_back(std::move(r));
    }
    std::fclose(in);
    const std::size_t n = runs.size();
    if (!n) return 1;

    Vu vu(1);
    vu.use_compiled = false;
    u64 kick_hash = 0;
    vu.on_kick = [](u64, u64) {};
    vu.on_kick_block = [&kick_hash](const u32* words, u32 count) {
        for (u32 i = 0; i < count; ++i) kick_hash = (kick_hash ^ words[i]) * 1099511628211ull;
        kick_hash = (kick_hash ^ 0xFFFFFFFFu) * 1099511628211ull;      // where a block ends
    };

    // The registers and flags looked at, by index: VF1-31, VI1-15, ACC, R, I, Q, MAC, STATUS, CLIP, P.
    enum { kVf = 0, kVi = 32, kAcc = 48, kR = 49, kI = 50, kQ = 51, kMac = 52, kStatus = 53, kClip = 54,
           kP = 55, kRegs = 56 };
    auto reg_name = [](int x) {
        static const char* named[] = {"ACC", "R", "I", "Q", "MAC", "STATUS", "CLIP", "P"};
        char text[16];
        if (x < kVi) std::snprintf(text, sizeof text, "VF%02d", x);
        else if (x < kAcc) std::snprintf(text, sizeof text, "VI%02d", x - kVi);
        else return std::string(named[x - kAcc]);
        return std::string(text);
    };
    auto valid_reg = [](int x) { return (x > kVf && x < kVi) || (x > kVi && x < kAcc) || (x >= kAcc && x < kRegs); };
    // A register's value, as words (VF and ACC four, the rest one).
    auto get = [](const Vu& v, int x, u32* out) -> int {
        if (x < kVi) { std::memcpy(out, v.regs.vf[x], 16); return 4; }
        if (x < kAcc) { out[0] = v.regs.vc[x - kVi]; return 1; }
        switch (x) {
            case kAcc: std::memcpy(out, v.regs.vacc, 16); return 4;
            case kR: out[0] = v.regs.vc[20]; return 1;
            case kI: out[0] = v.regs.vc[21]; return 1;
            case kQ: out[0] = v.regs.vc[22]; return 1;
            case kMac: out[0] = v.mac; return 1;
            case kStatus: out[0] = v.status; return 1;
            case kClip: out[0] = v.clip; return 1;
            default: out[0] = v.p; return 1;
        }
    };
    auto set = [](Vu& v, int x, const u32* in) {
        if (x < kVi) std::memcpy(v.regs.vf[x], in, 16);
        else if (x < kAcc) v.regs.vc[x - kVi] = in[0] & 0xFFFF;
        else switch (x) {
            case kAcc: std::memcpy(v.regs.vacc, in, 16); break;
            case kR: v.regs.vc[20] = in[0]; break;
            case kI: v.regs.vc[21] = in[0]; break;
            case kQ: v.regs.vc[22] = in[0]; break;
            case kMac: v.mac = v.mac_latest_ = in[0] & 0xFFFF; break;
            case kStatus: v.status = v.status_latest_ = in[0] & 0xFFF; break;
            case kClip: v.clip = v.clip_latest_ = in[0] & 0xFFFFFF; break;
            default: v.p = in[0]; break;
        }
    };
    const std::uint8_t* micro_now = nullptr;
    // A run's snapshot loaded, the pipeline settled (nothing in flight, the
    // latest flags the landed ones): every replay of it starts the same.
    auto load = [&](const Run& r) {
        if (micro_now != r.micro.data() && (!micro_now || std::memcmp(micro_now, r.micro.data(), 16384) != 0)) {
            std::memcpy(vu.micro.data(), r.micro.data(), 16384);
            vu.MicroWritten();
        }
        micro_now = r.micro.data();
        std::memcpy(vu.data.data(), r.data.data(), 16384);
        std::memcpy(vu.regs.vf, r.s.vf, sizeof r.s.vf);
        std::memcpy(vu.regs.vc, r.s.vc, sizeof r.s.vc);
        std::memcpy(vu.regs.vacc, r.s.vacc, sizeof r.s.vacc);
        vu.mac = vu.mac_latest_ = r.s.flags[0];
        vu.status = vu.status_latest_ = r.s.flags[1];
        vu.clip = vu.clip_latest_ = r.s.flags[2];
        vu.p = r.s.flags[3];
        vu.top = r.s.flags[4];
        vu.itop = r.s.flags[5];
        vu.cycle_ = 0;
        std::memset(vu.vf_ready_, 0, sizeof vu.vf_ready_);
        vu.q_busy_ = vu.p_busy_ = false;
        vu.q_ready_ = vu.p_ready_ = 0;
        vu.qp_due_ = ~u64(0);
        vu.flag_head_ = vu.flag_count_ = 0;
        std::memset(vu.discarded_, 0, sizeof vu.discarded_);
        kick_hash = 1469598103934665603ull;
    };
    struct Outcome {
        u64 kicks = 0, pairs = 0;
        std::vector<std::uint8_t> data;
        u32 regs[kRegs][4] = {};
    };
    auto outcome = [&](u64 pairs) {
        Outcome o;
        o.kicks = kick_hash;
        o.pairs = pairs;
        o.data.assign(vu.data.data(), vu.data.data() + 16384);
        for (int x = 0; x < kRegs; ++x)
            if (valid_reg(x)) get(vu, x, o.regs[x]);
        return o;
    };

    struct Tracker : Accesses {
        std::uint8_t written[1024] = {}, read[1024] = {};
        std::vector<u32> reads, writes;     // quad*4 + lane, each once
        void Read(u32 q, u32 lanes) override {
            for (u32 lane = 0; lane < 4; ++lane) {
                const u32 bit = 8u >> lane;
                if ((lanes & bit) && !(written[q] & bit) && !(read[q] & bit)) {
                    read[q] |= bit;
                    reads.push_back(q * 4 + lane);
                }
            }
        }
        void Write(u32 q, u32 lanes) override {
            for (u32 lane = 0; lane < 4; ++lane) {
                const u32 bit = 8u >> lane;
                if ((lanes & bit) && !(written[q] & bit)) {
                    written[q] |= bit;
                    writes.push_back(q * 4 + lane);
                }
            }
        }
    };

    u64 rng = 0x9E3779B97F4A7C15ull;
    auto random = [&rng]() {
        u64 z = (rng += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return u32(z ^ (z >> 31));
    };

    std::vector<int> mem_writer(4096, -1), reg_writer(kRegs, -1);
    std::vector<std::vector<int>> deps(n);
    std::vector<u64> weight(n);
    std::map<std::string, u64> edge_kinds;       // what made an edge (a run's each reason once)
    // Predicted: what a run reads from an earlier run, as its program's
    // previous run began with (or, for a register, that value moved on by
    // the step between its two before) - a worker could start with that and
    // check it once the writer is done.
    std::map<std::string, u64> edge_predicted;
    std::map<std::string, u64> anomaly_kinds;
    struct Start { Outcome now, before; int seen = 0; };
    std::map<u32, Start> starts;
    std::vector<std::vector<int>> hard(n);      // the edges not predicted
    std::vector<std::map<int, std::string>> edge_reasons(n);
    u64 reg_anomalies = 0, limited = 0, to_previous = 0;
    const auto began = std::chrono::steady_clock::now();
    const bool trace = std::getenv("PS2HOST_VU_DEPS")[0] == '2';
    // PS2HOST_VU_DEPS_FLAGS=1: MAC and STATUS as a run leaves them compared too.
    const bool keep_flags = std::getenv("PS2HOST_VU_DEPS_FLAGS") != nullptr;
    for (std::size_t k = 0; k < n; ++k) {
        const Run& r = runs[k];
        if (trace) {
            std::fprintf(stderr, "run %zu pc %X\n", k, unsigned(r.s.pc));
            std::fflush(stderr);
        }
        Tracker track;
        load(r);
        vu.accesses = &track;
        const u64 pairs = vu.Run(r.s.pc);
        vu.accesses = nullptr;
        const Outcome normal = outcome(pairs);
        weight[k] = std::max<u64>(pairs, 1);
        // Registers: written where they ended otherwise; read where changing them changes what it does.
        bool reg_written[kRegs] = {}, reg_read[kRegs] = {};
        load(r);
        const Outcome before = outcome(0);
        for (int x = 0; x < kRegs; ++x)
            reg_written[x] = valid_reg(x) && std::memcmp(before.regs[x], normal.regs[x], 16) != 0;
        for (int x = 0; x < kRegs; ++x) {
            if (!valid_reg(x)) continue;
            for (int trial = 0; trial < 2 && !reg_read[x]; ++trial) {
                if (trace) {
                    std::fprintf(stderr, "  %s %d\n", reg_name(x).c_str(), trial);
                    std::fflush(stderr);
                }
                load(r);
                u32 value[4] = {random(), random(), random(), random()};
                set(vu, x, value);
                const u64 got = vu.Run(r.s.pc, std::max<u64>(4 * pairs, 4096));
                if (got >= std::max<u64>(4 * pairs, 4096)) ++limited;
                const Outcome changed = outcome(got);
                bool differs = changed.kicks != normal.kicks || changed.data != normal.data;
                for (int y = 0; y < kRegs && !differs; ++y) {
                    if (!valid_reg(y) || (y == x && !reg_written[x])) continue;
                    if (!keep_flags && (y == kMac || y == kStatus)) continue;
                    differs = std::memcmp(changed.regs[y], normal.regs[y], 16) != 0;
                }
                reg_read[x] = differs;
            }
        }
        // The runs it depends on: the last writers of what it read.
        std::map<int, std::vector<std::string>> why;
        std::map<int, bool> missed;             // an edge's reads not all predicted
        Start& last = starts[r.s.pc];
        auto reg_predicted = [&](int x) {
            if (last.seen < 1) return false;
            if (std::memcmp(before.regs[x], last.now.regs[x], 16) == 0) return true;
            if (last.seen >= 2 && std::memcmp(before.regs[x], last.before.regs[x], 16) == 0) return true;
            if (last.seen < 2 || x < kVi || x >= kAcc) return false;
            const u32 step = last.now.regs[x][0] - last.before.regs[x][0];
            return ((last.now.regs[x][0] + step) & 0xFFFF) == before.regs[x][0];
        };
        std::map<int, std::pair<u64, u64>> mem_counts;    // writer: fields read, of them predicted
        for (u32 field : track.reads) {
            const int w = mem_writer[field];
            if (w < 0) continue;
            why[w].push_back("memory");
            auto& [all, hit] = mem_counts[w];
            ++all;
            if ((last.seen >= 1 && std::memcmp(&before.data[field * 4], &last.now.data[field * 4], 4) == 0) ||
                (last.seen >= 2 && std::memcmp(&before.data[field * 4], &last.before.data[field * 4], 4) == 0))
                ++hit;
        }
        for (const auto& [w, counts] : mem_counts)
            if (counts.second != counts.first) missed[w] = true;
        for (int x = 0; x < kRegs; ++x)
            if (reg_read[x] && reg_writer[x] >= 0) {
                why[reg_writer[x]].push_back(reg_name(x));
                if (reg_predicted(x)) ++edge_predicted[reg_name(x)];
                else missed[reg_writer[x]] = true;
            }
        for (auto& [writer, reasons] : why) {
            deps[k].push_back(writer);
            if (missed[writer]) hard[k].push_back(writer);
            if (writer == int(k) - 1) ++to_previous;
            std::sort(reasons.begin(), reasons.end());
            reasons.erase(std::unique(reasons.begin(), reasons.end()), reasons.end());
            for (const std::string& reason : reasons) {
                ++edge_kinds[reason];
                std::string& joined = edge_reasons[k][writer];
                joined += (joined.empty() ? "" : "+") + reason;
            }
        }
        for (const auto& [w, counts] : mem_counts)
            if (counts.second == counts.first) ++edge_predicted["memory"];
        last.before = std::move(last.now);
        last.now = before;
        ++last.seen;
        // What it leaves for the runs after.
        for (u32 field : track.writes) mem_writer[field] = int(k);
        for (int x = 0; x < kRegs; ++x)
            if (reg_written[x]) reg_writer[x] = int(k);
        // Between it and the next run: the VIF's (or the EE's) writes are nobody's run's.
        if (k + 1 < n) {
            const Run& next = runs[k + 1];
            for (u32 q = 0; q < 1024; ++q)
                for (u32 lane = 0; lane < 4; ++lane) {
                    const std::size_t at = q * 16 + lane * 4;
                    if (std::memcmp(&normal.data[at], &next.data[at], 4) != 0) mem_writer[q * 4 + lane] = -1;
                }
            load(next);
            const Outcome then = outcome(0);
            for (int x = 0; x < kRegs; ++x)
                if (valid_reg(x) && std::memcmp(then.regs[x], normal.regs[x], 16) != 0) {
                    ++reg_anomalies;
                    ++anomaly_kinds[reg_name(x)];
                    reg_writer[x] = -1;
                }
        }
        if (k % 2000 == 1999)
            std::fprintf(stderr, "vu-deps: %zu of %zu runs (%.0f s)\n", k + 1, n,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count());
    }

    // PS2HOST_VU_DEPS_OUT=file: the graph, a line a run - index, program, pairs;
    // then each edge "writer:reasons" (reasons '+'-joined, '!' before a
    // writer whose edge was not predicted).
    if (const char* out_path = std::getenv("PS2HOST_VU_DEPS_OUT")) {
        if (std::FILE* out = std::fopen(out_path, "w")) {
            for (std::size_t k = 0; k < n; ++k) {
                std::fprintf(out, "%zu %X %llu", k, unsigned(runs[k].s.pc), (unsigned long long)weight[k]);
                for (int d : deps[k]) {
                    const bool is_hard = std::find(hard[k].begin(), hard[k].end(), d) != hard[k].end();
                    std::fprintf(out, " %s%d:%s", is_hard ? "!" : "", d, edge_reasons[k][d].c_str());
                }
                std::fprintf(out, "\n");
            }
            std::fclose(out);
        }
    }
    std::printf("vu-deps: registers changed between runs, by register:");
    for (const auto& [name, count] : anomaly_kinds) std::printf(" %s %llu", name.c_str(), (unsigned long long)count);
    std::printf("\n");
    std::printf("vu-deps: edges predicted (the program's previous start, or its step):");
    for (const auto& [name, count] : edge_predicted)
        std::printf(" %s %llu/%llu", name.c_str(), (unsigned long long)count, (unsigned long long)edge_kinds[name]);
    std::printf("\n");
    for (int pass = 0; pass < 2; ++pass) {
    if (pass == 1) {
        deps = hard;
        std::printf("vu-deps: --- with the predicted edges gone ---\n");
    }
    // The schedule: the chain, and workers side by side in order.
    u64 total = 0, free_runs = 0, free_pairs = 0;
    std::vector<u64> finish(n);
    u64 longest = 0;
    for (std::size_t k = 0; k < n; ++k) {
        total += weight[k];
        u64 ready = 0;
        for (int d : deps[k]) ready = std::max(ready, finish[d]);
        if (deps[k].empty()) ++free_runs, free_pairs += weight[k];
        finish[k] = ready + weight[k];
        longest = std::max(longest, finish[k]);
    }
    std::printf("vu-deps: %zu runs, %llu pairs; %llu runs (%.1f%% of the pairs) depend on no earlier run\n", n,
                (unsigned long long)total, (unsigned long long)free_runs, 100.0 * double(free_pairs) / double(total));
    std::printf("vu-deps: the longest chain %llu pairs: at most %.2fx side by side\n", (unsigned long long)longest,
                double(total) / double(longest));
    for (int workers : {2, 3, 4, 6, 8}) {
        std::vector<u64> free_at(workers, 0), done(n);
        u64 end = 0;
        for (std::size_t k = 0; k < n; ++k) {
            u64 ready = 0;
            for (int d : deps[k]) ready = std::max(ready, done[d]);
            auto w = std::min_element(free_at.begin(), free_at.end());
            const u64 start = std::max(ready, *w);
            done[k] = start + weight[k];
            *w = done[k];
            end = std::max(end, done[k]);
        }
        std::printf("vu-deps: %d workers: %.2fx\n", workers, double(total) / double(end));
    }
    }
    std::printf("vu-deps: edges to the run just before: %llu; registers changed between runs: %llu; "
                "replays at their limit: %llu\n",
                (unsigned long long)to_previous, (unsigned long long)reg_anomalies, (unsigned long long)limited);
    std::vector<std::pair<u64, std::string>> kinds;
    for (const auto& [kind, count] : edge_kinds) kinds.emplace_back(count, kind);
    std::sort(kinds.rbegin(), kinds.rend());
    std::printf("vu-deps: edges by what was read:");
    for (std::size_t i = 0; i < kinds.size() && i < 16; ++i)
        std::printf(" %s %llu", kinds[i].second.c_str(), (unsigned long long)kinds[i].first);
    std::printf("\n");
    return 0;
}

Vu::u64 Vu::Run(u32 pc, u64 limit) {
    Timed timed(data_mask_ == 0x3FF ? Section::Vu1 : Section::Vu0);
    VuRounding rounding;
    const int unit = data_mask_ == 0x3FF ? 1 : 0;
    pc &= u32(micro.size() - 1) & ~7u;
    ++start_counts_[pc / 8];
    if (g_record) RecordProgram(micro, unit, pc);
    if (unit == 1 && g_snapshotting.load(std::memory_order_relaxed) && g_snapshots_left) {
        Snapshot s{};
        s.pc = pc;
        const u32 flags[6] = {mac, status, clip, p, top, itop};
        std::memcpy(s.flags, flags, sizeof flags);
        std::memcpy(s.vf, regs.vf, sizeof s.vf);
        std::memcpy(s.vc, regs.vc, sizeof s.vc);
        std::memcpy(s.vacc, regs.vacc, sizeof s.vacc);
        std::fwrite(&s, sizeof s, 1, g_snapshots);
        std::fwrite(micro.data(), 1, 16384, g_snapshots);
        std::fwrite(data.data(), 1, 16384, g_snapshots);
        g_snapshots_written.fetch_add(1, std::memory_order_relaxed);
        if (--g_snapshots_left == 0) {
            std::fclose(g_snapshots);
            g_snapshots = nullptr;
            g_snapshotting = false;
        }
    }
    VuRunState state;
    state.pc = pc;
    unclean_ = ~1u;                     // nothing known about the registers yet
    acc_unclean_ = true;
    RunCompiled(state, limit);
    const u64 compiled = state.count;
    if (!state.done) Interpret(state, limit);
    LandAll();
    resume = state.pc;
    pairs_ += state.count;
    // A run far longer than a program's usual (a frame's whole VU1 work is a
    // few million pairs): said, with how much of it the interpreter ran.
    if (state.count >= (u64(1) << 21)) {
        static int said = 0;
        if (said < 20) {
            ++said;
            std::fprintf(stderr, "[vu] VU%d run from 0x%X: %llu pairs (%llu interpreted), ended at 0x%X%s\n", unit,
                         unsigned(pc), (unsigned long long)state.count,
                         (unsigned long long)(state.count - compiled), unsigned(state.pc),
                         state.count >= limit ? " - its limit" : "");
        }
    }
    return state.count;
}

bool Vu::RunCompiled(VuRunState& state, u64 limit) {
    if (!use_compiled || Compiled().empty()) return false;
    const int unit = data_mask_ == 0x3FF ? 1 : 0;
    bool ran = false;
    // From program to program (a jump to a register's address may go to
    // another), as long as one starts here and its code is in memory.
    while (!state.done && !state.interpret && state.count < limit && state.delayed == kNone && !state.stopping) {
        Entered* entered = state.pc < micro.size() ? &entered_[state.pc / 8] : nullptr;
        const VuCompiled* program = nullptr;
        const std::vector<VuBlock>* blocks = nullptr;
        if (entered && entered->generation == micro_generation_ && entered->fast == fast) {
            program = static_cast<const VuCompiled*>(entered->program);
            blocks = static_cast<const std::vector<VuBlock>*>(entered->table);
        } else {
            const auto found = Compiled().find(std::uint64_t(unit) << 32 | state.pc);
            if (found == Compiled().end()) {
                if (g_record && ran) RecordProgram(micro, unit, state.pc);
                break;
            }
            for (const VuCompiled* candidate : found->second) {
                u64& held_at = held_at_[candidate];
                if (held_at == micro_generation_) { program = candidate; break; }
                compiled_checked_ += 4 * (candidate->code[2] + 2);
                if (Holds(micro, candidate->code)) {
                    held_at = micro_generation_;
                    program = candidate;
                    break;
                }
            }
            if (!program) {
                ++compiled_stale_;
                break;
            }
            blocks = &BlockTable(program, micro.size() / 8, fast);
            if (entered) *entered = Entered{program, blocks, micro_generation_, fast};
        }
        ++compiled_entries_;
        const std::vector<VuBlock>& table = *blocks;
        const u64 before = state.count;
        const u32 started = state.pc;
        while (!state.done && !state.interpret && state.count < limit) {
            // A jump to a register's address: into a block that knows fields
            // clean from the ways into it the generator knew, the interpreter
            // goes on (that block's clamps were left out for them).
            if (state.jumped) {
                state.jumped = false;
                if (program->knows) {
                    const u32 pc = state.pc & u32(micro.size() - 1);
                    const u32* end = program->starts + program->block_count;
                    const u32* at = std::lower_bound(program->starts, end, pc);     // (in order of address)
                    if (at != end && *at == pc && program->knows[at - program->starts]) {
                        state.interpret = true;
                        break;
                    }
                }
            }
            const VuBlock block = table[(state.pc & u32(micro.size() - 1)) / 8];
            if (!block) break;
            const u64 at = state.count;
            if (g_block_times) {
                const u32 pc = state.pc;
                const auto began = std::chrono::steady_clock::now();
                block(*this, state, limit);
                BlockTime& b = (*g_block_times)[reinterpret_cast<const void*>(block)];
                b.ticks += u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - began).count());
                ++b.entries;
                b.pairs += state.count - at;
                b.entry = program->entry;
                b.pc = pc;
                if (state.count == at) break;
                continue;
            }
            block(*this, state, limit);
            if (state.count == at) break;   // it would not start here: the interpreter goes on
        }
        compiled_pairs_ += state.count - before;
        ran = true;
        if (state.pc == started && state.count == before) break;     // nothing taken
    }
    return ran;
}

void Vu::Interpret(VuRunState& state, u64 limit) {
    unclean_ = ~1u;                     // the interpreter does not keep track
    acc_unclean_ = true;
    u64 count = state.count;
    u32 delayed = state.delayed;
    bool stopping = state.stopping;
    u32 pc = state.pc;
    bool delayed_by_register = false;   // the branch delayed is JR's or JALR's
    const int unit = data_mask_ == 0x3FF ? 1 : 0;
    // One cache a unit size; an entry stands only while its words are there.
    static std::vector<DecodedPair> caches[2];
    std::vector<DecodedPair>& decoded_ = caches[data_mask_ == 0x3FF ? 1 : 0];
    if (decoded_.size() != micro.size() / 8) decoded_.assign(micro.size() / 8, DecodedPair{});
    while (count < limit) {
        u32 lower, upper;
        std::memcpy(&lower, micro.data() + pc, 4);
        std::memcpy(&upper, micro.data() + pc + 4, 4);
        const DecodedPair& d = Decoded(decoded_, pc, lower, upper);
        const bool immediate = d.immediate;
        if (g_pc_counts) ++(*g_pc_counts)[pc >> 3];
        if (g_perturb && pc == g_perturb->pc) {
            g_perturb->vi.push_back(regs.vc[g_perturb->vi_reg] & 0xFFFF);
            if (++g_perturb->seen == g_perturb->nth) {
                float f;
                std::memcpy(&f, &regs.vf[g_perturb->reg][g_perturb->lane], 4);
                f = f != 0.0f ? f * 1.25f : 1.0f;
                std::memcpy(&regs.vf[g_perturb->reg][g_perturb->lane], &f, 4);
            }
        }

        // Stalls: operands still in the pipeline, a busy divider or EFU, the waits.
        const Reads& reads = d.reads;
        u64 issue = cycle_;
        for (int i = 0; i < reads.n; ++i)
            for (int lane = 0; lane < 4; ++lane)
                if (reads.lanes[i] & (8u >> lane)) issue = std::max(issue, vf_ready_[reads.reg[i]][lane]);
        if (q_busy_ && d.q_waits) issue = std::max(issue, q_ready_);
        if (p_busy_ && d.p_waits) issue = std::max(issue, p_ready_);
        cycle_ = issue;
        Land();

        Pending pending;
        u32 branch = kNone;
        if (!immediate) Lower(lower, pc, pending, &branch);
        int written = -1;
        u32 lanes = 0;
        bool set_flags = false;
        Upper(upper, &written, &lanes, &set_flags, d.has_fmac ? &d.fmac : nullptr);
        // LOI: the upper beside it read the I from before.
        if (immediate) regs.vc[kRegI] = lower;
        if (pending.vf > 0) {
            for (int lane = 0; lane < 4; ++lane) {
                const u32 bit = 8u >> lane;
                if ((pending.lanes & bit) && !(written == pending.vf && (lanes & bit))) {
                    regs.vf[pending.vf][lane] = pending.value[lane];
                    vf_ready_[pending.vf][lane] = cycle_ + kFmacLatency;
                }
            }
        }
        if (written > 0)
            for (int lane = 0; lane < 4; ++lane)
                if (lanes & (8u >> lane)) vf_ready_[written][lane] = cycle_ + kFmacLatency;
        if (set_flags)
            PushFlags({cycle_ + kFmacLatency, mac_latest_, status_latest_, clip_latest_, (upper & 0x7FF) != 0x1FF,
                       (upper & 0x7FF) == 0x1FF});
        if (pending.q_latency) {
            // (A quotient equal to the Q before changes no register: it is that Q.)
            q_next_ = pending.vc ? pending.vc_after[kRegQ] : regs.vc[kRegQ];
            q_ready_ = cycle_ + u64(pending.q_latency);
            q_busy_ = true;
            qp_due_ = std::min(qp_due_, q_ready_);
            if (pending.vc) pending.vc_after[kRegQ] = regs.vc[kRegQ];
        }
        if (pending.vc) std::memcpy(regs.vc, pending.vc_after, sizeof pending.vc_after);
        if (pending.p) {
            p_next_ = pending.p_value;
            p_ready_ = cycle_ + u64(pending.p_latency ? pending.p_latency : 1);
            p_busy_ = true;
            qp_due_ = std::min(qp_due_, p_ready_);
        }
        regs.vc[0] = 0;
        ++cycle_;
        ++count;
        if (stopping) {                      // the pair after an E bit runs, then the program ends
            pc = (pc + 8) & u32(micro.size() - 1);
            state.done = true;
            break;
        }
        if (upper & 0x40000000u) stopping = true;
        const bool jumped = delayed != kNone, by_register = delayed_by_register;
        const u32 next = jumped ? delayed : (pc + 8) & u32(micro.size() - 1);
        delayed = branch;
        delayed_by_register = branch != kNone && ((lower >> 25) == 0x24 || (lower >> 25) == 0x25);
        pc = next;
        if (jumped && delayed == kNone && !stopping) {
            // Where a jump to a register's address goes is a program of its own
            // (one recorded, and compiled, from there); compiled code, if any
            // starts here, goes on.
            if (g_record && by_register) RecordProgram(micro, unit, pc);
            state.pc = pc;
            state.count = count;
            state.delayed = delayed;
            state.stopping = stopping;
            if (RunCompiled(state, limit)) {
                if (state.done) return;
                pc = state.pc;
                count = state.count;
                delayed = state.delayed;
                stopping = state.stopping;
                delayed_by_register = false;
            }
        }
    }
    state.pc = pc;
    state.count = count;
    state.delayed = delayed;
    state.stopping = stopping;
}

Vu& Vu1() {
    static Vu vu(1);
    static const bool attached = [] {
        // The EE sees VU1's memories at 0x11008000 (instructions) and
        // 0x1100C000 (data).
        vu.Attach(GuestMemory().Translate(0x11008000u, 16384), GuestMemory().Translate(0x1100C000u, 16384));
        return true;
    }();
    (void)attached;
    return vu;
}

Vu& Vu0() {
    static Vu vu(0);
    static const bool attached = [] {
        // VU0's at 0x11000000 and 0x11004000.
        vu.Attach(GuestMemory().Translate(0x11000000u, 4096), GuestMemory().Translate(0x11004000u, 4096));
        return true;
    }();
    (void)attached;
    return vu;
}

}  // namespace ps2host

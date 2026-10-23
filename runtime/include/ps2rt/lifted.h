/*
 * Support for lifted functions: machine code translated instruction by instruction.
 *
 * A derived function (generated.h) is an expression over its arguments and reads
 * like source. A lifted function is a mechanical translation over an explicit
 * register context: every instruction becomes one statement that does exactly
 * what the R5900 does, branches become gotos, and calls pass the whole context.
 * It is less readable, but every function the interpreter can run can be lifted,
 * whatever its size or control flow, and each one is verified the same way: run
 * against the original machine code, comparing the entire register file and every
 * byte of memory written.
 *
 * The semantics follow engine/ps2decomp/emu.py exactly - the interpreter is the
 * reference both are tested against.
 */

#ifndef PS2RT_LIFTED_H
#define PS2RT_LIFTED_H

#include "ps2rt/generated.h"
#include "ps2rt/types.h"

namespace lifted {

/* The register state a function runs on, and the only state it has. */
struct Context {
    u64 r[32];      /* general registers, low 64 bits */
    u64 rh[32];     /* their upper 64 bits, for lq/sq and the 128-bit MMI ops */
    u64 hi, lo;     /* multiply/divide results */
    u64 hi1, lo1;   /* the second pipeline's, as mult1/div1 use */
    u32 f[32];      /* FPU registers, as raw bit patterns */
    u32 acc;        /* the FPU accumulator, as a bit pattern */
    u32 fcc;        /* the FPU condition flag */
    s64 budget;     /* loop iterations left before the run is abandoned */
    u32 vf[32][4];  /* VU0 vector registers, x y z w, as bit patterns */
    u32 vc[32];     /* VU0 control registers: VI0-VI15, then R (20), I (21), Q (22) */
    u32 vacc[4];    /* VU0 accumulator */
    u32 sa;         /* the shift amount qfsrv uses, in bits */
    /* Set by another thread to get control at this context's next loop head
       (Tick), whatever its budget: the host's blank timer, in real time. */
    u32 attention;
};

/* Thrown, and caught at the entry point, when a run cannot go on: a trap
   instruction, an exhausted loop budget, or a callee that failed. */
struct Abort {
    u32 address;
    u32 reason;     /* 1 trap, 2 budget, 3 callee failed, 4 memory fault, 5 COP2 */
};

/* Calls `target` with the whole context, as a jal or jalr does; returns when the
   callee returns. Defined by whatever hosts the lifted code. */
void Call(Context& c, u32 target);

/* The same, with the link register set first: a jal written as one call. */
inline void Call(Context& c, u32 target, u32 return_address) {
    c.r[31] = return_address;
    Call(c, target);
}

/* Transfers control to `target` for good, as a jump out of the function does:
   it runs until control comes back to `return_address` with the stack pointer at
   or above `entry_sp`, which is when the lifted function itself has returned. */
void TailCall(Context& c, u32 target, u32 return_address, u32 entry_sp);

/* A syscall: the host decides what the service does. */
void Syscall(Context& c);

/* A word result, sign-extended into its 64-bit register as the EE does. */
inline u64 Sx32(u64 value) { return (u64)(s64)(s32)(u32)value; }

inline u64 Read64(u32 address) { return generated::ReadU64(address); }

/* A loop head reached with the budget spent. Without a hook the run is
   abandoned. A host that sets `loop_head_hook` is called instead: it can zero a
   context's budget to get control at that context's next loop head - to deliver
   an interrupt, say - and throws Abort itself when the run must stop. */
extern void (*loop_head_hook)(Context& c, u32 address);
void BudgetSpent(Context& c, u32 address);

/* At every loop head, so a loop that never ends cannot hang the host - and
   where another thread gets control of it (`attention`). Not by zeroing the
   budget from there: a loop stores its budget every few cycles, and the store
   still in its core's buffer lands after the other thread's zero - Resident
   Evil Outbreak's spin on a flag (0x0018D650) lost nearly every one, and its
   menus' frames waited 20-80 ms for a blank the timer had long delivered.
   The flag is only read here, never written, so it is never lost. */
inline void Tick(Context& c, u32 address) {
    if (--c.budget < 0 || *static_cast<volatile u32*>(&c.attention)) BudgetSpent(c, address);
}

inline void Trap(u32 address) { throw Abort{address, 1}; }

/* 32x32 multiply into HI/LO, as mult/multu/madd do; the caller copies LO to rd. */
void Multiply(u64& hi, u64& lo, u64 a, u64 b, bool is_signed, bool accumulate);
void Divide(u64& hi, u64& lo, u64 a, u64 b, bool is_signed);

/* Single-precision arithmetic is generated.h's (FloatAdd, IntFromFloat, ...),
   shared with the derived code; comparisons use FloatValue. This is the one the
   accumulator needs besides: acc + a * b (or acc - a * b), rounded once. */
u32 FloatMulAdd(u32 acc, u32 a, u32 b, bool subtract);

/* Per-byte 128-bit add and subtract (paddb, psubb). */
void AddBytes(u64& lo, u64& hi, u64 alo, u64 ahi, u64 blo, u64 bhi, bool subtract);

/* Lane-wise 128-bit MMI operations: each lane of rs (a) and rt (b). */
enum LaneOp { kPmaxw = 1, kPminw, kPsubw, kPadduw, kPaddub, kPabsw, kPaddw, kPaddh,
              /* all ones where rs > rt (signed) or rs == rt; signed halfword max/min */
              kPcgtb, kPcgth, kPcgtw, kPceqb, kPceqh, kPceqw, kPmaxh, kPminh };
void Lanes(int op, u64& lo, u64& hi, u64 alo, u64 ahi, u64 blo, u64 bhi);
/* Lanes and Interleave lane by lane: the model their SSE2 forms are checked against. */
void LanesScalar(int op, u64& lo, u64& hi, u64 alo, u64 ahi, u64 blo, u64 bhi);
void InterleaveScalar(u64& lo, u64& hi, u64 a, u64 b, u32 width);
void ShiftHalfwordsRight(u64& lo, u64& hi, u64 blo, u64 bhi, u32 amount);   /* psrlh */

/* More 128-bit MMI: lane shifts (kind 0 sll, 1 srl, 2 sra), the low-doubleword
   interleave (pextlb/pextlh), ppach, and qfsrv's funnel shift by `sa` bits. */
void ShiftLanes(u64& lo, u64& hi, u64 blo, u64 bhi, u32 width, u32 kind, u32 amount);
/* ShiftLanes lane by lane: the model its SSE2 form is checked against. */
void ShiftLanesScalar(u64& lo, u64& hi, u64 blo, u64 bhi, u32 width, u32 kind, u32 amount);
void Interleave(u64& lo, u64& hi, u64 a, u64 b, u32 width);
void PackHalfwords(u64& lo, u64& hi, u64 alo, u64 ahi, u64 blo, u64 bhi);
void PackBytes(u64& lo, u64& hi, u64 alo, u64 ahi, u64 blo, u64 bhi);     /* ppacb */
void FunnelShift(u64& lo, u64& hi, u64 alo, u64 ahi, u64 blo, u64 bhi, u32 sa);

/* rsqrt.s: fs / sqrt(|ft|), the largest magnitude with fs's sign for a zero root. */
u32 FloatRsqrt(u32 fs, u32 ft);

/* Unaligned loads (they merge into the register's old value) and stores. */
u64 LoadWordLeft(u32 address, u64 old);
u64 LoadWordRight(u32 address, u64 old);
u64 LoadDoubleLeft(u32 address, u64 old);
u64 LoadDoubleRight(u32 address, u64 old);
void StoreLeft(u32 address, u64 value, u32 size);
void StoreRight(u32 address, u64 value, u32 size);

/* VU0 in macro mode (runtime/src/vu0.cpp, mirroring engine/ps2decomp/vu0.py):
   an arithmetic COP2 instruction, and the moves between it and the core.
   Cop2 (below, inline) runs the FMAC operations itself; Cop2Slow every one,
   decoded as it runs. */
void Cop2Slow(Context& c, u32 word);
/* PS2RT_EE_ROUND_NEAREST: macro mode left rounding to nearest (vu0.cpp). */
extern const bool kMacroNearest;
/* Macro mode sets the VU's rounding for each instruction - unless the thread
   running the EE's code runs in it throughout (ps2host's RunThreads clears
   this: two control-word writes an instruction were a fifth of its time). */
extern bool g_macro_switch;
/* COP2's FMAC operations (ADD, SUB, MUL, MADD, MSUB, MAX, MINI, with their
   broadcast, Q, I and accumulator forms), decoded once and run as often as
   needed - the VU interpreter keeps them per instruction. Cop2 runs them the
   same way. */
struct FmacOp {
    int op = 0;                     /* the arithmetic (vu0.cpp's Op) */
    int operand = 0;                /* the second operand: 0-3 a lane of ft, 4 Q, 5 I, 6 ft */
    bool to_acc = false;            /* into ACC rather than fd */
    u32 fd = 0, fs = 0, ft = 0, dest = 0;
};
bool DecodeFmac(u32 word, FmacOp* out);
void RunFmac(Context& c, const FmacOp& op);
/* Its result alone, the fields dest names (the rest 0), written nowhere: for
   an operation into VF0, which keeps 0, 0, 0, 1 but whose flags are this
   result's. */
void FmacResult(const Context& c, const FmacOp& op, u32 out[4]);
/* The same, a lane at a time through the arithmetic model (vu_math.h):
   what RunFmac's four lanes at once must equal. */
void RunFmacScalar(Context& c, const FmacOp& op);
void Cop2Move(Context& c, u32 word, u32 rt);
/* vcallms / vcallmsr: VU0 runs the microprogram at `address` (bytes in its
   instruction memory) on the registers macro mode shares with it. Defined by
   whatever hosts the lifted code. */
void Vcallms(Context& c, u32 address);

}  // namespace lifted

#include "ps2rt/vu_math.h"

namespace lifted {

/* An FMAC's form from its word (vu0.cpp's Special1 and AccumulatorForm, the
   same tables): foldable where the word is a constant - the lifted code's
   34,000 COP2 instructions each are - so each compiles to its arithmetic
   alone, not a decode and a call each time it runs (5% of the EE's time in
   a race). */
struct FmacForm {
    int op = 0, operand = 0;
    bool to_acc = false, valid = false;
};

constexpr PS2RT_VU_INLINE FmacForm MakeForm(int op, int operand, bool to_acc) { return FmacForm{op, operand, to_acc, true}; }

constexpr PS2RT_VU_INLINE FmacForm Special1Form(u32 funct) {
    constexpr int kBroadcast[7] = {kAdd, kSub, kMadd, kMsub, kMax_, kMini, kMul};
    if (funct <= 0x1B) return MakeForm(kBroadcast[funct >> 2], int(funct & 3), false);
    switch (funct) {
        case 0x1C: return MakeForm(kMul, 4, false);
        case 0x1D: return MakeForm(kMax_, 5, false);
        case 0x1E: return MakeForm(kMul, 5, false);
        case 0x1F: return MakeForm(kMini, 5, false);
        case 0x20: return MakeForm(kAdd, 4, false);
        case 0x21: return MakeForm(kMadd, 4, false);
        case 0x22: return MakeForm(kAdd, 5, false);
        case 0x23: return MakeForm(kMadd, 5, false);
        case 0x24: return MakeForm(kSub, 4, false);
        case 0x25: return MakeForm(kMsub, 4, false);
        case 0x26: return MakeForm(kSub, 5, false);
        case 0x27: return MakeForm(kMsub, 5, false);
        case 0x28: return MakeForm(kAdd, 6, false);
        case 0x29: return MakeForm(kMadd, 6, false);
        case 0x2A: return MakeForm(kMul, 6, false);
        case 0x2B: return MakeForm(kMax_, 6, false);
        case 0x2C: return MakeForm(kSub, 6, false);
        case 0x2D: return MakeForm(kMsub, 6, false);
        case 0x2F: return MakeForm(kMini, 6, false);
        default: return FmacForm{};
    }
}

constexpr PS2RT_VU_INLINE FmacForm AccumulatorFormOf(u32 index) {
    constexpr int kBroadcast[4] = {kAdd, kSub, kMadd, kMsub};
    if (index <= 0x0F) return MakeForm(kBroadcast[index >> 2], int(index & 3), true);
    if (index >= 0x18 && index <= 0x1B) return MakeForm(kMul, int(index & 3), true);
    switch (index) {
        case 0x1C: return MakeForm(kMul, 4, true);
        case 0x1E: return MakeForm(kMul, 5, true);
        case 0x20: return MakeForm(kAdd, 4, true);
        case 0x21: return MakeForm(kMadd, 4, true);
        case 0x22: return MakeForm(kAdd, 5, true);
        case 0x23: return MakeForm(kMadd, 5, true);
        case 0x24: return MakeForm(kSub, 4, true);
        case 0x25: return MakeForm(kMsub, 4, true);
        case 0x26: return MakeForm(kSub, 5, true);
        case 0x27: return MakeForm(kMsub, 5, true);
        case 0x28: return MakeForm(kAdd, 6, true);
        case 0x29: return MakeForm(kMadd, 6, true);
        case 0x2A: return MakeForm(kMul, 6, true);
        case 0x2C: return MakeForm(kSub, 6, true);
        case 0x2D: return MakeForm(kMsub, 6, true);
        default: return FmacForm{};
    }
}

constexpr PS2RT_VU_INLINE FmacForm FmacFormOf(u32 word) {
    const u32 funct = word & 0x3F;
    return funct < 0x3C ? Special1Form(funct) : AccumulatorFormOf((word & 3) | ((word >> 4) & 0x7C));
}

#if PS2RT_VU_SSE
/* An FMAC as Cop2Slow runs it (RunFmac's Lanes: LanesSse<Op>, every lane
   read before any is written; dest's lanes stored), in the same rounding. */
template <int Op>
inline __m128i MacroLanes(__m128i a, __m128i b, __m128i acc) { return LanesSse<Op>(a, b, acc); }

inline void MacroFmac(Context& c, u32 word, FmacForm f) {
    const u32 fd = (word >> 6) & 0x1F, fs = (word >> 11) & 0x1F, ft = (word >> 16) & 0x1F;
    const u32 dest = (word >> 21) & 0xF;
    const bool switching = g_macro_switch;
    const unsigned saved = switching ? _mm_getcsr() : 0u;
    if (switching) _mm_setcsr((saved & ~0x6000u) | 0x6000u | 0x8000u | 0x0040u);
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c.vf[fs]));
    __m128i b;
    if (f.operand == 6) b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c.vf[ft]));
    else b = _mm_set1_epi32(int(f.operand < 4 ? c.vf[ft][f.operand] : f.operand == 4 ? c.vc[kRegQ] : c.vc[kRegI]));
    const __m128i acc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c.vacc));
    __m128i r;
    switch (f.op) {
        case kAdd: r = MacroLanes<kAdd>(a, b, acc); break;
        case kSub: r = MacroLanes<kSub>(a, b, acc); break;
        case kMul: r = MacroLanes<kMul>(a, b, acc); break;
        case kMadd: r = MacroLanes<kMadd>(a, b, acc); break;
        case kMsub: r = MacroLanes<kMsub>(a, b, acc); break;
        case kMax_: r = MacroLanes<kMax_>(a, b, acc); break;
        default: r = MacroLanes<kMini>(a, b, acc); break;
    }
    u32 out[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), r);
    u32* to = f.to_acc ? c.vacc : fd ? c.vf[fd] : nullptr;
    if (to)
        for (int lane = 0; lane < 4; ++lane)
            if (dest & (8u >> lane)) to[lane] = out[lane];
    if (switching) _mm_setcsr(saved);
}
#endif

/* Always inlined, with the decode above: at a lifted call site the word is a
   constant, and only there does the decode fold away (a profile-guided build
   left Cop2 out of line - the decode run every time, 2-3% of the EE). What it
   calls is the compiler's to place. */
PS2RT_VU_INLINE void Cop2(Context& c, u32 word) {
#if PS2RT_VU_SSE
    const FmacForm f = FmacFormOf(word);
    if (f.valid) {
        MacroFmac(c, word, f);
        return;
    }
#endif
    Cop2Slow(c, word);
}

}  // namespace lifted

#endif  /* PS2RT_LIFTED_H */

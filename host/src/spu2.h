/*
 * The SPU2, the PS2's sound processor, as libsd drives it (iop_libs.cpp).
 *
 * Voices play through SPU RAM in real time, a sample at a time at 48 kHz:
 * their ADPCM is decoded, shaped by the ADSR envelope and mixed by their
 * volumes into the stereo output (an Output sink - the host's audio device -
 * takes it). Games synchronise with the same machinery: loop flags, ENDX, and
 * the SPU2 interrupt raised when a voice's playback (or a transfer) reaches the
 * interrupt address IRQA - the clock sound drivers stream by.
 *
 * Voice data is the SPU's ADPCM: 16-byte blocks of 28 samples. Byte 0 holds
 * the shift (bits 0-3) and the prediction filter (bits 4-6); byte 1 the flags:
 * bit 0 loop end, bit 1 repeat (go back to the loop start rather than stop),
 * bit 2 loop start. A voice plays at pitch/4096 of 48 kHz.
 *
 * Reverb (the effect unit): each core's voices with VMIXEL/VMIXER set feed it;
 * it runs at 24 kHz on its work area at the top of SPU RAM, with the
 * registers libsd's presets set (SetEffectAttr: sceSdSetEffectAttr), and its
 * output joins the core's at EVOL. The algorithm is the PS1 SPU's, which the
 * SPU2 kept (no-cash psx-spx, "SPU Reverb Formula").
 *
 * Sound-data input: a core's ring in SPU RAM - planar 16-bit PCM, 512 left
 * samples at 0x4000 then 512 right at 0x4400 (core 0), 0x4800/0x4C00 (core
 * 1) - read a stereo sample per 48 kHz tick at the core's BVOL, into its dry
 * and wet mixes as MMIX bits 4-7 say. A driver writes it itself (Burnout 3's
 * engine sound), or a block transfer (sceSdBlockTrans - how CRI's ADX streams
 * its music) fills it from IOP memory a sample a tick, as the hardware's
 * AutoDMA does: 0x200 bytes of left samples, then 0x200 of right, looping
 * over the buffer.
 *
 * Capture: the SPU2 writes what it mixes back into SPU RAM, 512-sample rings
 * - voices 1 and 3 of each core, core 0's output, each core's dry and wet
 * mixes - and the writes raise IRQA, so a driver can read the sound back to
 * process it.
 *
 * The cores (PCSX2's mixer): each sums its voices (VMIX), its sound-data
 * input (BVOL) and its external input into a dry and a wet mix, as MMIX lets
 * each in; the wet goes through the reverb (EVOL); the sum, at MVOL, is the
 * core's output. Core 0's is core 1's external input (at core 1's AVOL), and
 * core 1's is what is heard.
 *
 * Not modelled yet: noise, pitch modulation.
 *
 * libsd's entries (ps2sdk libsd-common.h): a voice parameter or address is
 * (param << 8) | core | voice << 1; a switch or core address (n << 8) | core.
 */

#ifndef PS2HOST_SPU2_H
#define PS2HOST_SPU2_H

#include <cstdint>
#include <functional>
#include <vector>

namespace ps2host {
namespace iop {

class Spu2 {
public:
    using u8 = std::uint8_t;
    using u16 = std::uint16_t;
    using s16 = std::int16_t;
    using u32 = std::uint32_t;
    using s32 = std::int32_t;
    using u64 = std::uint64_t;

    Spu2() : ram_(2u << 20) {
        cores_[0].mmix = 0xFF0;
        cores_[1].mmix = 0xFFC;
    }

    // Host gain for a core's PCM input, applied after the game's own volume.
    // A 5 ms ramp makes live menu changes click-free; capture RAM is unchanged.
    void SetInputVolume(u32 core, u32 percent);
    void SetParam(u32 entry, u32 value);
    u32 GetParam(u32 entry) const;
    void SetSwitch(u32 entry, u32 value);
    u32 GetSwitch(u32 entry) const;
    void SetAddr(u32 entry, u32 value);
    u32 GetAddr(u32 entry) const;
    void SetCoreAttr(u32 entry, u32 value);
    u32 GetCoreAttr(u32 entry) const;
    /* sceSdSetEffectAttr: a reverb preset (SD_EFFECT_MODE_*, 0x100 to clear
       the work area), its depth, and for echo and delay their delay and
       feedback - as libsd programs the registers. -1 for a bad mode. */
    int SetEffectAttr(u32 core, u32 mode, s32 depth_left, s32 depth_right, u32 delay, u32 feedback);
    /* What the last SetEffectAttr was given (libsd keeps it). */
    void GetEffectAttr(u32 core, u32* mode, s32* depth_left, s32* depth_right, u32* delay, u32* feedback) const;

    /* A core's sound-data input: `size` bytes of PCM at `data` (it stays
       there, the source's memory), from its start; null stops it. */
    void SetInput(u32 core, const u8* data, u32 size);

    /* Data into (or out of) SPU RAM at a byte address, as a transfer would. */
    void Write(u32 address, const u8* data, u32 size);
    void Read(u32 address, u8* data, u32 size) const;

    /* Advance playback to `now_us`; `irq(core)` for each interrupt raised.
       With `input_half`, a core's sound-data input reaching a half of its
       buffer calls it (the block transfer's interrupt: the half before is
       free to refill) and playback stops there, the rest owed (Behind): the
       IOP's handler and the stream's decoder run before the SPU2 reads on
       (PlayedUs: where it stopped). */
    void Advance(u64 now_us, const std::function<void(u32 core)>& irq,
                 const std::function<void(u32 core)>& input_half = nullptr);
    /* The time played to: the last Advance's, or short of it where it stopped. */
    u64 PlayedUs() const { return start_us_ + clock_ * 1000000 / 48000; }
    /* The half of its buffer a core's sound-data input is playing (0 or 1). */
    u32 InputHalf(u32 core) const;
    bool InputActive(u32 core) const { return cores_[core & 1].input != nullptr; }

    /* Where the mixed output goes: interleaved 16-bit stereo at 48 kHz. */
    using Output = std::function<void(const s16* frames, u32 count)>;
    void SetOutput(Output output) { output_ = std::move(output); }

    /* Exposed for tests: one ADPCM block decoded, with the filter history. */
    static void DecodeBlock(const u8* block, s32 out[28], s32* older, s32* old);

private:
    enum class Phase : u8 { Off, Attack, Decay, Sustain, Release };
    /* A volume as it is now (a voice's left or right, a core's main): set to
       a register's fixed value (bit 15 clear: 15 bits, signed), or moved a
       sample at a time by its sweep (bit 15 set: bit 14 exponential, 13
       decreasing, 12 the phase inverted, 0-6 the rate, as an envelope's)
       from where it was - the game fades a voice it leaves running so. */
    struct Level {
        u32 reg = 0;
        s32 now = 0;                    // -0x7FFF to 0x7FFF
        u32 counter = 0;
        explicit Level(u32 value = 0) { Set(value); }
        void Set(u32 value);
        // A sample of a sweep; a fixed volume (bit 15 clear) stays as it is.
        // Inline: 100 levels step at 48 kHz, nearly all of them fixed.
        void Step() {
            if (reg & 0x8000) Sweep();
        }
        void Sweep();
    };
    struct Voice {
        u32 pitch = 0, ssa = 0, lsax = 0, nax = 0;
        u32 adsr1 = 0, adsr2 = 0, voll = 0, volr = 0;
        Level left, right;              // VOLL, VOLR as they are now
        bool playing = false;           // sounding (a stopped voice still runs on)
        u32 counter = 0;                // 4.12 fixed point: samples into the block
        // The last block's last three samples (the Gaussian interpolation's
        // history), then the current block, decoded: one run, so the four
        // taps are read straight from it.
        s32 window[3 + 28] = {};
        s32 older = 0, old = 0;         // the decoder's history
        Phase phase = Phase::Off;
        s32 level = 0;                  // the envelope, 0-0x7FFF
        u32 envelope_counter = 0;
        // The phase's rate as Envelope works it out from ADSR1/2, kept while
        // the phase and the registers stay (a voice's every sample asked).
        u64 rate_key = ~0ull;
        s32 rate_step = 0;
        u32 rate_increment = 0, rate_shift = 0;
        bool rate_decreasing = false, rate_exponential = false;
    };
    struct Core {
        Voice voices[24];
        u32 endx = 0, irqa = 0, attr = 0, esa = 0, eea = 0, tsa = 0;
        u32 switches[16] = {};          // by switch number - 0x13
        s32 input_gain = 65536, input_gain_target = 65536;   // SetInputVolume's, 16.16
        u32 input_gain_remaining = 0;
        u32 mvoll = 0x3FFF, mvolr = 0x3FFF;
        // The inputs' volumes: AVOL the external input's (core 1's: core 0's
        // output), BVOL the sound-data input's - at their maximum until set.
        u32 avoll = 0x7FFF, avolr = 0x7FFF, bvoll = 0x7FFF, bvolr = 0x7FFF;
        // MMIX, what goes where, wet R, wet L, dry R, dry L in each four bits:
        // 0-3 the external input, 4-7 the sound-data input, 8-11 the core's
        // voices (PCSX2's gates). After sceSdInit 0xFF0 and 0xFFC (Spu2()).
        u32 mmix = 0xFFF;
        const u8* input = nullptr;      // the sound-data input, while a block transfer runs
        u32 input_size = 0, input_sample = 0;
        // PS2HOST_SPU_LOG: halves begun, and those begun as they were the
        // last time round (not refilled: the decoder behind its reader).
        u32 input_halves = 0, input_stale = 0;
        u64 input_hash[2] = {};
        s32 peak_voices = 0, peak_input = 0, peak_reverb = 0, voice_peak[24] = {};   // PS2HOST_SPU_LOG: the second's
        Level main_left{0x3FFF}, main_right{0x3FFF};     // MVOLL, MVOLR as they are now
        bool irq_enabled = false;
        // Reverb: the registers (libsd's order: dAPF1, dAPF2, vIIR, vCOMB1-4,
        // vWALL, vAPF1, vAPF2, then the 20 buffer offsets in 8-byte units,
        // vLIN, vRIN), EVOL, where it is in its work area, what feeds it.
        u16 reverb[32] = {};
        u32 evoll = 0, evolr = 0;
        u32 effect_mode = 0, effect_delay = 0, effect_feedback = 0;
        s32 effect_depth_left = 0, effect_depth_right = 0;
        u32 reverb_position = 0;        // bytes into the work area
        s32 reverb_out_left = 0, reverb_out_right = 0;   // the last step's
        // The reverb's resampling (ReverbResample): the last 64 samples into
        // it at 48 kHz, and out of it - its steps, a zero between each.
        s32 reverb_in[2][64] = {}, reverb_up[2][64] = {};
        u32 reverb_at = 0;
    };
    Core cores_[2];
    std::vector<u8> ram_;
    u64 start_us_ = 0;               // the clock's start: the first Advance past 0
    u64 clock_ = 0;                  // samples since then, played (or passed over)
    u64 logged_second_ = ~0ull;      // PS2HOST_SPU_LOG: the last second logged
    Output output_;
    std::vector<s16> mixed_;

    void InputHalfBegun(Core& c);
    static u32 CoreOf(u32 entry) { return entry & 1; }
    static u32 VoiceOf(u32 entry) { return ((entry >> 1) & 0x1F) % 24; }
    void CheckIrq(u32 core_hint, u32 from, u32 to, const std::function<void(u32)>& irq);
    void Start(Voice& v);
    void Decode(Voice& v);
    void NextBlock(Core& c, u32 index, const std::function<void(u32)>& irq);
    void Envelope(Voice& v);
    void Reverb(Core& c, s32 in_left, s32 in_right);
    u32 EffectEnd(const Core& c) const;
    std::vector<u32> pending_irqs_;
};

}  // namespace iop
}  // namespace ps2host

#endif  /* PS2HOST_SPU2_H */

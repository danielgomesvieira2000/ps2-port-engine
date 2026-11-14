/*
 * The port's sound (--audio): the SPU2's mixed output (spu2.cpp) played through
 * the system's audio engine (WASAPI, shared mode), or its wave device (waveOut)
 * where that cannot open.
 *
 * The SPU2 produces samples as the IOP runs, in bursts (a frame's worth, or a
 * half-buffer's), by the emulated clock; the device takes them at its own,
 * steadily. Between the two is a ring: AudioFrames puts the samples in, an
 * audio thread takes them out as the device asks. The ring is kept near a
 * target fill (kTargetMs: enough to ride out a burst and a late frame) by
 * playing it a little faster or slower than it came - at most half a percent,
 * a pitch no one hears - so neither a clock's drift nor a hitch builds into
 * lag: the old queue only ever grew after each hitch, until every buffer was
 * full and samples were dropped (crackling, and a picture 170 ms ahead of its
 * sound). An empty ring plays silence, faded out over a few samples rather
 * than cut off, and fills to the target again before it plays on.
 *
 * Lag is capped: the rate control's half percent drains a backlog of 100 ms
 * in 20 seconds, and a hitch (a room loading, a menu's stall) left one as
 * long as itself - the device dry through it, then the SPU2's whole due
 * catch-up at once: the sound a fifth of a second behind the picture for a
 * minute. When the ring has held more than kExcessMs over its target for a
 * whole window (kWindowMs: its fill never that low through it - bursts
 * alone are not lag), the excess is skipped, the old and the new sound
 * crossfaded over kCrossFrames. And a ring that ran dry for a hitch (under a
 * quarter of a second; not a pause) raises its target (kTargetStepMs a time,
 * kTargetMostMs over the start's at most: a PC whose frames come late holds
 * more), back down slowly while it plays on.
 *
 * --audio-wav records the SPU2's output itself to a WAV file, instead of (or as
 * well as) playing it: how sound is checked without anyone listening
 * (tools/audiocheck.py).
 *
 * PS2HOST_AUDIO_LOG=1: every two seconds, how the sound came and went.
 * PS2HOST_AUDIO_MUTE=1: played (the device measured) but silent.
 * PS2HOST_AUDIO_LATENCY_MS=n: the ring's target, 20 to 250 ms.
 * PS2HOST_AUDIO_WAVEOUT=1: the wave device even where WASAPI would open.
 *
 * Elsewhere (Android), SDL's audio stream: AAudio's low-latency output, its
 * callback taking the ring's samples as WASAPI's event does.
 */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <mmsystem.h>
#include <mmreg.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#undef small            // rpcndr.h's, through the COM headers; run.h names a parameter so
#else
#include <SDL3/SDL.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "ps2host/music.h"
#include "ps2host/run.h"
#include "ps2host/timeline.h"

#ifdef _WIN32
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "ole32.lib")
#endif

namespace ps2host {

namespace {

constexpr int kRate = 48000;
constexpr unsigned kRingFrames = 1u << 15;      // 683 ms: room for a long hitch's catch-up
constexpr double kMaxStretch = 0.005;           // the rate control's reach: +-0.5%
constexpr unsigned kFadeFrames = 96;            // 2 ms: into and out of silence
constexpr unsigned kExcessMs = 40;              // lag over the target, a whole window: skipped
constexpr unsigned kWindowMs = 300;
constexpr unsigned kCrossFrames = 240;          // 5 ms: the skip's crossfade
constexpr unsigned kTargetStepMs = 10, kTargetMostMs = 60;

// The ring between the SPU2 (one writer) and the audio thread (one reader):
// interleaved stereo floats; positions count frames and only ever grow.
struct Ring {
    std::vector<float> data = std::vector<float>(kRingFrames * 2);
    std::atomic<unsigned long long> written{0}, read{0};
    // A skip's crossfade's old frames, behind `read` (~0: none): not the
    // writer's to fill until the crossfade is done.
    std::atomic<unsigned long long> kept{~0ull};
    unsigned long long Filled() const { return written.load(std::memory_order_acquire) - read.load(std::memory_order_acquire); }
};

struct Stats {
    std::atomic<unsigned> batches{0}, ran_dry{0}, skips{0};
    std::atomic<unsigned long long> dropped{0};
    std::atomic<double> longest_ms{0};
    std::atomic<int> fewest_ms{1 << 30}, most_ms{0};
    std::atomic<double> stretch_min{1}, stretch_max{1};
};

struct Device {
    Ring ring;
    Stats stats;
    unsigned target_frames = kRate * 45 / 1000;    // (grown after running dry: kTargetStepMs)
    bool mute = false;
    bool log = false;
    std::atomic<bool> open{false}, stop{false};
    std::atomic<bool> resync{false};    // AudioResync: the backlog dropped at the next Render
    std::atomic<int> opened{0};         // the audio thread's answer: 1 a device, -1 none
    std::thread thread;
    std::FILE* wav = nullptr;
    unsigned long long wav_frames = 0;
    std::chrono::steady_clock::time_point last_batch{};

    // The reader's state (the audio thread's alone).
    double phase = 0;                   // between the frame `read` and the next, 0-1
    double smoothed_fill = 0;           // the ring's fill, averaged over ~100 ms
    bool buffering = true;              // filling to the target before playing on
    float last[2] = {0, 0};             // the last frame played: faded out from
    unsigned fade = 0;                  // frames left of a fade in
    unsigned base_target = 0;           // the target asked for (target_frames grows after running dry)
    unsigned long long window_low = ~0ull;  // the ring's lowest fill this window
    unsigned window_frames = 0;         // device frames played this window
    unsigned long long cross_from = 0;  // a skip's old place, crossfaded from
    unsigned cross = 0;                 // frames left of its crossfade
    unsigned long long since_dry = 0;   // device frames played since the ring last ran dry
    bool dry = false;                   // buffering after running dry (not the start's)
    unsigned long long dry_frames = 0;  // ...for so many device frames

    ~Device() { Close(); }
    void Close() {
        stop = true;
        if (thread.joinable()) thread.join();
        open = false;
        if (!wav) return;
        WriteHeader(unsigned(wav_frames * 4));
        std::fclose(wav);
        wav = nullptr;
    }
    void WriteHeader(unsigned data) {
        unsigned char h[44] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
                               16, 0, 0, 0, 1, 0, 2, 0, 0x80, 0xBB, 0, 0, 0, 0xEE, 2, 0,
                               4, 0, 16, 0, 'd', 'a', 't', 'a', 0, 0, 0, 0};
        const unsigned riff = 36 + data;
        std::memcpy(h + 4, &riff, 4);
        std::memcpy(h + 40, &data, 4);
        std::fseek(wav, 0, SEEK_SET);
        std::fwrite(h, 1, 44, wav);
        std::fseek(wav, 0, SEEK_END);
    }

    // `frames` stereo frames for the device, taken from the ring at the
    // controlled rate. Called by the audio thread only.
    void Render(float* out, unsigned frames);
};

Device& D() {
    static Device device;
    return device;
}

template <typename T>
void AtomicMax(std::atomic<T>& a, T v) {
    T now = a.load();
    while (v > now && !a.compare_exchange_weak(now, v)) {
    }
}
template <typename T>
void AtomicMin(std::atomic<T>& a, T v) {
    T now = a.load();
    while (v < now && !a.compare_exchange_weak(now, v)) {
    }
}

// Catmull-Rom through four frames, at t (0-1) between the middle two.
inline float Cubic(float a, float b, float c, float d, float t) {
    return b + 0.5f * t * (c - a + t * (2.0f * a - 5.0f * b + 4.0f * c - d + t * (3.0f * (b - c) + d - a)));
}

void Device::Render(float* out, unsigned frames) {
    if (resync.exchange(false) && ring.Filled() > target_frames) {
        ring.read.store(ring.written.load(std::memory_order_acquire) - target_frames, std::memory_order_release);
        smoothed_fill = double(target_frames);
        fade = kFadeFrames;
    }
    if (!base_target) base_target = target_frames;
    unsigned long long available = ring.Filled();
    // Lag capped: the ring's lowest fill through a window above the target
    // by more than the excess - the excess skipped (crossfaded).
    if (!buffering) {
        window_low = std::min(window_low, available);
        window_frames += frames;
        if (window_frames >= kRate * kWindowMs / 1000) {
            const unsigned long long excess = unsigned(kRate * kExcessMs / 1000);
            if (window_low != ~0ull && window_low > target_frames + excess && !cross) {
                const unsigned long long skip = window_low - target_frames;
                cross_from = ring.read.load(std::memory_order_relaxed);
                ring.kept.store(cross_from, std::memory_order_release);
                ring.read.store(cross_from + skip, std::memory_order_release);
                cross = kCrossFrames;
                smoothed_fill = double(available - skip);
                available -= skip;
                ++stats.skips;
                static int told = 0;
                if (told < 20) {
                    ++told;
                    std::fprintf(stderr, "[audio] %u ms behind: skipped to %u ms\n", unsigned(window_low * 1000 / kRate),
                                 target_frames * 1000 / kRate);
                }
            }
            window_low = ~0ull;
            window_frames = 0;
        }
        // Played on without running dry: the target back down, a millisecond each 2 s.
        since_dry += frames;
        if (since_dry >= kRate * 2u && target_frames > base_target) {
            since_dry = 0;
            target_frames = std::max(base_target, target_frames - kRate / 1000);
        }
    }
    // The fill the rate control steers by: the ring's, smoothed - it jumps by a
    // whole batch each time the SPU2 delivers one.
    smoothed_fill += (double(available) - smoothed_fill) * 0.05;
    if (buffering) {
        if (available < target_frames) {
            std::fill(out, out + 2 * frames, 0.0f);
            if (dry) dry_frames += frames;
            return;
        }
        // Back from running dry: a hitch (the game's frames late, under a
        // quarter of a second) - more kept from now on, up to kTargetMostMs
        // more; a pause (the PC settings panel, the window put away: no
        // sound made for longer) is no lateness, the target as it was (else
        // the sound came later after every pause, for minutes).
        if (dry && dry_frames < unsigned(kRate / 4))
            target_frames = std::min(base_target + kRate * kTargetMostMs / 1000, target_frames + kRate * kTargetStepMs / 1000);
        dry = false;
        dry_frames = 0;
        buffering = false;
        fade = kFadeFrames;
        smoothed_fill = double(available);
    }
    // Faster when fuller than the target, slower when emptier: the step, in
    // ring frames per device frame.
    const double error = (smoothed_fill - double(target_frames)) / double(target_frames);
    const double step = 1.0 + std::clamp(error * kMaxStretch * 2.0, -kMaxStretch, kMaxStretch);
    if (log) {
        AtomicMin(stats.stretch_min, step);
        AtomicMax(stats.stretch_max, step);
    }
    const float* data = ring.data.data();
    unsigned long long at = ring.read.load(std::memory_order_relaxed);
    const unsigned long long end = ring.written.load(std::memory_order_acquire);
    auto frame = [&](unsigned long long n, int ch) { return data[2 * (n % kRingFrames) + ch]; };
    unsigned k = 0;
    for (; k < frames; ++k) {
        // Four frames around the position: at-1, at, at+1, at+2.
        if (at + 2 >= end) break;
        const float t = float(phase);
        for (int ch = 0; ch < 2; ++ch) {
            const float before = at ? frame(at - 1, ch) : frame(at, ch);
            float v = Cubic(before, frame(at, ch), frame(at + 1, ch), frame(at + 2, ch), t);
            if (fade) v *= float(kFadeFrames - fade) / float(kFadeFrames);
            if (cross) {
                // (a skip's: the sound it left, fading out under the new)
                const float g = float(cross) / float(kCrossFrames);
                const unsigned long long old = cross_from + (kCrossFrames - cross);
                v = v * (1.0f - g) + frame(old, ch) * g;
            }
            out[2 * k + ch] = v;
        }
        if (fade) --fade;
        if (cross && !--cross) ring.kept.store(~0ull, std::memory_order_release);
        last[0] = out[2 * k];
        last[1] = out[2 * k + 1];
        phase += step;
        const double whole = std::floor(phase);
        at += (unsigned long long)whole;
        phase -= whole;
    }
    ring.read.store(at, std::memory_order_release);
    if (k < frames) {
        // Ran dry: what was playing fades to silence, and the ring fills to
        // the target before it plays again.
        if (log) ++stats.ran_dry;
        // (its target raised as it plays again, if it was a hitch: above)
        dry = true;
        dry_frames = 0;
        since_dry = 0;
        window_low = ~0ull;
        window_frames = 0;
        cross = 0;
        ring.kept.store(~0ull, std::memory_order_release);
        for (unsigned n = 0; k < frames; ++k, ++n) {
            const float g = n < kFadeFrames ? float(kFadeFrames - n) / float(kFadeFrames) : 0.0f;
            out[2 * k] = last[0] * g;
            out[2 * k + 1] = last[1] * g;
        }
        last[0] = last[1] = 0;
        buffering = true;
    }
    if (mute) std::fill(out, out + 2 * frames, 0.0f);
    if (log) {
        const int ms = int(available * 1000 / kRate);
        AtomicMin(stats.fewest_ms, ms);
        AtomicMax(stats.most_ms, ms);
    }
}

// PS2HOST_AUDIO_LOG: every two seconds, from the writer's side.
void LogStats(Device& d) {
    static auto since = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - since < std::chrono::seconds(2)) return;
    since = now;
    Stats& s = d.stats;
    const int fewest = s.fewest_ms.exchange(1 << 30);
    std::fprintf(stderr, "[audio] %u batches, longest wait %.1f ms; ring %d-%d ms (target %u), ran dry %u times, "
                         "%u skipped, %llu frames dropped, rate %.4f-%.4f\n",
                 s.batches.exchange(0), s.longest_ms.exchange(0), fewest == (1 << 30) ? 0 : fewest, s.most_ms.exchange(0), d.target_frames * 1000 / kRate, s.ran_dry.exchange(0), s.skips.exchange(0), s.dropped.exchange(0),
                 s.stretch_min.exchange(1), s.stretch_max.exchange(1));
}

#ifdef _WIN32
// ---------------------------------------------------------------- WASAPI

// KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, without ksuser.lib.
const GUID kFloatFormat = {0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};

template <typename T>
void Release(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

// Plays until stopped; false if the device could not be opened at all (the
// caller falls back to waveOut). A device that goes away (headphones pulled)
// is opened again - the default one, whatever it now is.
bool RunWasapi(Device& d) {
    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) return false;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    bool ever = false;
    while (!d.stop) {
        IMMDeviceEnumerator* enumerator = nullptr;
        IMMDevice* device = nullptr;
        IAudioClient* client = nullptr;
        IAudioRenderClient* render = nullptr;
        ISimpleAudioVolume* volume = nullptr;
        bool ok = SUCCEEDED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                             __uuidof(IMMDeviceEnumerator), reinterpret_cast<void**>(&enumerator))) &&
                  SUCCEEDED(enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device)) &&
                  SUCCEEDED(device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                                             reinterpret_cast<void**>(&client)));
        UINT32 buffer_frames = 0;
        if (ok) {
            // 48 kHz float stereo, the system converting to its mix format
            // (another rate, more channels) where that differs.
            WAVEFORMATEXTENSIBLE f = {};
            f.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
            f.Format.nChannels = 2;
            f.Format.nSamplesPerSec = kRate;
            f.Format.wBitsPerSample = 32;
            f.Format.nBlockAlign = 8;
            f.Format.nAvgBytesPerSec = kRate * 8;
            f.Format.cbSize = sizeof f - sizeof f.Format;
            f.Samples.wValidBitsPerSample = 32;
            f.dwChannelMask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT;
            f.SubFormat = kFloatFormat;
            const DWORD flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
                                AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
            ok = SUCCEEDED(client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 200000 /* 20 ms */, 0,
                                              &f.Format, nullptr)) &&
                 SUCCEEDED(client->SetEventHandle(event)) && SUCCEEDED(client->GetBufferSize(&buffer_frames)) &&
                 SUCCEEDED(client->GetService(__uuidof(IAudioRenderClient), reinterpret_cast<void**>(&render)));
        }
        if (ok) {
            if (SUCCEEDED(client->GetService(__uuidof(ISimpleAudioVolume), reinterpret_cast<void**>(&volume)))) {
                // PS2HOST_AUDIO_RESET_VOLUME=1: this program's volume in the
                // mixer back to full (an earlier build's mute set it to
                // nothing, and the mixer keeps it).
                if (std::getenv("PS2HOST_AUDIO_RESET_VOLUME")) volume->SetMasterVolume(1.0f, nullptr);
                float level = 1;
                volume->GetMasterVolume(&level);
                if (!ever)
                    std::fprintf(stderr, "[audio] playing (WASAPI, %.0f ms buffer, %u ms target); this program's "
                                         "volume in the system's mixer %.0f%%\n",
                                 1000.0 * buffer_frames / kRate, d.target_frames * 1000 / kRate, 100.0 * level);
            }
            ever = true;
            d.opened.store(1);
            BYTE* out = nullptr;
            if (SUCCEEDED(render->GetBuffer(buffer_frames, &out)))
                render->ReleaseBuffer(buffer_frames, AUDCLNT_BUFFERFLAGS_SILENT);
            ok = SUCCEEDED(client->Start());
        }
        while (ok && !d.stop) {
            WaitForSingleObject(event, 100);
            UINT32 padding = 0;
            if (FAILED(client->GetCurrentPadding(&padding))) break;      // the device went away
            const UINT32 room = buffer_frames - padding;
            if (!room) continue;
            BYTE* out = nullptr;
            if (FAILED(render->GetBuffer(room, &out))) break;
            d.Render(reinterpret_cast<float*>(out), room);
            render->ReleaseBuffer(room, 0);
        }
        if (client) client->Stop();
        Release(volume);
        Release(render);
        Release(client);
        Release(device);
        Release(enumerator);
        if (!ever) break;                   // never opened: waveOut instead
        if (!d.stop) std::this_thread::sleep_for(std::chrono::milliseconds(500));   // a new default device
    }
    CloseHandle(event);
    CoUninitialize();
    return ever;
}

// ---------------------------------------------------------------- waveOut

void RunWaveOut(Device& d) {
    constexpr int kBuffers = 4;
    constexpr unsigned kFrames = 480;           // 10 ms each
    WAVEFORMATEX format = {};
    format.wFormatTag = WAVE_FORMAT_PCM;
    format.nChannels = 2;
    format.nSamplesPerSec = kRate;
    format.wBitsPerSample = 16;
    format.nBlockAlign = 4;
    format.nAvgBytesPerSec = kRate * 4;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    HWAVEOUT out = nullptr;
    if (waveOutOpen(&out, WAVE_MAPPER, &format, DWORD_PTR(event), 0, CALLBACK_EVENT) != MMSYSERR_NOERROR) {
        CloseHandle(event);
        d.opened.store(-1);
        return;
    }
    std::fprintf(stderr, "[audio] playing (waveOut, %u ms target)\n", d.target_frames * 1000 / kRate);
    d.opened.store(1);
    WAVEHDR headers[kBuffers] = {};
    std::vector<short> data[kBuffers];
    std::vector<float> scratch(kFrames * 2);
    for (int i = 0; i < kBuffers; ++i) {
        data[i].assign(kFrames * 2, 0);
        headers[i].lpData = reinterpret_cast<LPSTR>(data[i].data());
        headers[i].dwBufferLength = kFrames * 4;
        headers[i].dwFlags = WHDR_DONE;
    }
    while (!d.stop) {
        for (int i = 0; i < kBuffers; ++i) {
            WAVEHDR& h = headers[i];
            if (!(h.dwFlags & WHDR_DONE)) continue;
            if (h.dwFlags & WHDR_PREPARED) waveOutUnprepareHeader(out, &h, sizeof h);
            d.Render(scratch.data(), kFrames);
            for (unsigned k = 0; k < kFrames * 2; ++k)
                data[i][k] = short(std::lrint(std::clamp(scratch[k], -1.0f, 1.0f) * 32767.0f));
            h.dwFlags = 0;
            waveOutPrepareHeader(out, &h, sizeof h);
            waveOutWrite(out, &h, sizeof h);
        }
        WaitForSingleObject(event, 20);
    }
    waveOutReset(out);
    for (WAVEHDR& h : headers)
        if (h.dwFlags & WHDR_PREPARED) waveOutUnprepareHeader(out, &h, sizeof h);
    waveOutClose(out);
    CloseHandle(event);
}
#else
// ---------------------------------------------------------------- SDL

void RunSdl(Device& d) {
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        std::fprintf(stderr, "[audio] no audio: %s\n", SDL_GetError());
        d.opened.store(-1);
        return;
    }
    SDL_AudioSpec spec{SDL_AUDIO_F32, 2, kRate};
    // The device's own period: as small as it takes (AAudio's low-latency path).
    SDL_SetHint(SDL_HINT_AUDIO_DEVICE_SAMPLE_FRAMES, "256");
    SDL_AudioStream* stream = SDL_OpenAudioDeviceStream(
        SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec,
        [](void* user, SDL_AudioStream* stream, int additional, int) {
            Device& d = *static_cast<Device*>(user);
            thread_local std::vector<float> scratch;
            const unsigned frames = unsigned(additional) / 8;
            if (frames == 0) return;
            scratch.resize(frames * 2);
            d.Render(scratch.data(), frames);
            SDL_PutAudioStreamData(stream, scratch.data(), int(frames * 8));
        },
        &d);
    if (!stream) {
        std::fprintf(stderr, "[audio] no audio device: %s\n", SDL_GetError());
        d.opened.store(-1);
        return;
    }
    int frames = 0;
    SDL_AudioSpec device_spec{};
    SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(stream), &device_spec, &frames);
    std::fprintf(stderr, "[audio] playing (SDL %s, %d frames a period at %d Hz, %u ms target)\n",
                 SDL_GetCurrentAudioDriver(), frames, device_spec.freq, d.target_frames * 1000 / kRate);
    SDL_ResumeAudioStreamDevice(stream);
    d.opened.store(1);
    while (!d.stop) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    SDL_DestroyAudioStream(stream);
}
#endif

}  // namespace

void CloseAudio() { D().Close(); }

bool OpenAudio() {
    Device& d = D();
    d.mute = std::getenv("PS2HOST_AUDIO_MUTE") != nullptr;
    d.log = std::getenv("PS2HOST_AUDIO_LOG") != nullptr;
    if (const char* ms = std::getenv("PS2HOST_AUDIO_LATENCY_MS"))
        d.target_frames = unsigned(kRate * std::clamp(std::atoi(ms), 20, 250) / 1000);
    const bool wave_only = std::getenv("PS2HOST_AUDIO_WAVEOUT") != nullptr;
    d.thread = std::thread([&d, wave_only] {
        timeline::NameThread("audio");
#ifdef _WIN32
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        if (wave_only || !RunWasapi(d)) RunWaveOut(d);
#else
        (void)wave_only;
        RunSdl(d);
#endif
    });
    // Until the thread says whether a device opened.
    while (d.opened.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (d.opened.load() < 0) {
        d.stop = true;
        d.thread.join();
        return false;
    }
    d.open = true;
    return true;
}

bool OpenWavDump(const std::string& path) {
    Device& d = D();
    d.wav = std::fopen(path.c_str(), "wb");
    if (!d.wav) return false;
    d.WriteHeader(0);
    return true;
}

void AudioResync() { D().resync = true; }

bool AudioActive() {
    Device& d = D();
    return d.open || d.wav;
}

void AudioFrames(const short* frames, unsigned count) {
    Device& d = D();
    // PS2HOST_MUSIC_TEST=file.mp3: that MP3 played over the game from its
    // first sound (music.h's check).
    static const bool music_test = [] {
        if (const char* path = std::getenv("PS2HOST_MUSIC_TEST")) MusicPlay(path);
        return true;
    }();
    (void)music_test;
    // The host's music mixed in (music.h): into a copy, the SPU2's own as it is.
    static thread_local std::vector<short> mixed;
    if (LoadingScreenShown()) {             // a port's loading screen: the game behind it unheard
        mixed.assign(2 * std::size_t(count), 0);
        frames = mixed.data();
    } else if (MusicMixing()) {
        mixed.assign(frames, frames + 2 * std::size_t(count));
        MusicMix(mixed.data(), count);
        frames = mixed.data();
    }
    if (d.wav) {
        std::fwrite(frames, 4, count, d.wav);
        d.wav_frames += count;
    }
    if (!d.open) return;
    {
        // PS2HOST_AUDIO_TEST_STALL=FRAME:MS (a test's): the game held MS once
        // at FRAME - a hitch, the sound's catch-up and skip back exercised.
        static unsigned long long stall_at = 0;
        static int stall_ms = 0;
        static const bool stall = [] {
            const char* e = std::getenv("PS2HOST_AUDIO_TEST_STALL");
            return e && std::sscanf(e, "%llu:%d", &stall_at, &stall_ms) == 2;
        }();
        if (stall && FrameNumber() >= stall_at && stall_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(stall_ms));
            stall_ms = 0;
        }
    }
    {
        // A long wait for the game's sound (its IOP held: the game busy
        // between two frames) - a gap heard, then the sound skipped back to
        // the picture: said in the log with its frame, a few times, for a
        // report (the first: RANDOM's figure drawn as Character Select opened).
        static std::chrono::steady_clock::time_point before{};
        static int told = 0;
        const auto now = std::chrono::steady_clock::now();
        if (before.time_since_epoch().count() && told < 12) {
            const double ms = std::chrono::duration<double, std::milli>(now - before).count();
            if (ms > 100.0) {
                ++told;
                std::fprintf(stderr, "[audio] no sound from the game for %.0f ms (frame %llu)\n", ms,
                             (unsigned long long)FrameNumber());
            }
        }
        before = now;
    }
    if (d.log) {
        const auto now = std::chrono::steady_clock::now();
        if (d.last_batch.time_since_epoch().count())
            AtomicMax(d.stats.longest_ms, std::chrono::duration<double, std::milli>(now - d.last_batch).count());
        d.last_batch = now;
        ++d.stats.batches;
        LogStats(d);
    }
    Ring& r = d.ring;
    const unsigned long long w = r.written.load(std::memory_order_relaxed);
    const unsigned long long oldest = std::min(r.read.load(std::memory_order_acquire), r.kept.load(std::memory_order_acquire));
    const unsigned long long room = w - oldest + 4 >= kRingFrames ? 0 : kRingFrames - 4 - (w - oldest);
    if (count > room) {
        // A burst the ring cannot hold (the emulation far ahead of the device:
        // virtual time, a catch-up): the newest are dropped.
        if (d.log) d.stats.dropped += count - room;
        count = unsigned(room);
    }
    float* data = r.data.data();
    for (unsigned k = 0; k < count; ++k) {
        const unsigned long long at = (w + k) % kRingFrames;
        data[2 * at] = frames[2 * k] * (1.0f / 32768.0f);
        data[2 * at + 1] = frames[2 * k + 1] * (1.0f / 32768.0f);
    }
    r.written.store(w + count, std::memory_order_release);
}

}  // namespace ps2host

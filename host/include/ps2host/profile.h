/*
 * Where the host's time goes, for the report: a few sections timed with the
 * steady clock, exclusively (a section entered inside another pauses the
 * outer one, so VU1's time does not include the GS drawing its XGKICKs cause).
 * Cheap enough to stay on: a clock read on entry and exit.
 */

#ifndef PS2HOST_PROFILE_H
#define PS2HOST_PROFILE_H

#include <cstdint>
#include <string>

namespace ps2host {

enum class Section { GsDraw, GsUpload, Vu1, Vu0, GpuBridge, GpuRender, GpuSync, Dma, Iop, Spu2, GsWait, Count };

class Timed {
public:
    explicit Timed(Section section);
    ~Timed();
    Timed(const Timed&) = delete;
    Timed& operator=(const Timed&) = delete;

private:
    Section section_;
};

/* "GS drawing 61.2%, ..." of the run's wall time so far. */
std::string ProfileSummary();

/* --sample: a sampling profiler on the calling thread (the EE's, which runs
   the translated code, the host's devices and the GPU bridge): a thousand
   times a second, the function it is in and seven callers. At the
   end, the functions most often found running ("self"), and those most
   often on the stack at all ("total"), by name from the executable's
   symbols. Windows only (elsewhere it reports nothing). */
void StartSampler();
/* --sample-gs: the sampler on the GS thread instead (gs_thread.cpp starts it
   there), where the GS model and the GPU bridge run. */
void SetSampleGsThread(bool on);
bool SampleGsThread();
/* --sample-vif1: on VIF1's worker (real time), where VU1 runs. */
void SetSampleVif1Thread(bool on);
/* The GPU renderer's own thread instead (--sample-renderer; a threaded renderer's). */
void SetSampleRendererThread(bool on);
bool SampleRendererThread();
/* On the renderer's own thread as it starts: named, and its CPU time to be
   read (RendererThreadCpuMs, milliseconds; 0 without one). */
void NoteRendererThread();
double RendererThreadCpuMs();
/* Its cycles (Windows: exact; elsewhere its CPU time in nanoseconds). */
std::uint64_t RendererThreadCycles();
bool SampleVif1Thread();
/* A host crash (an access violation and the like, on any thread): its code,
   address and the stack by name, to stderr, before the process ends. */
void InstallCrashReport();
/* --sample-from N: no samples until frame N (devices.cpp calls this each frame). */
void SetSampleFrom(unsigned long long frame);
void SampleFrame(unsigned long long frame);
/* The report counts only samples taken within the spans given (steady clock
   ns): with PS2HOST_HITCH, the times a vertical blank waited for the EE to
   take it. None given, it counts them all. On the EE's thread the report
   gives the guest's call stacks too. */
void SampleOnly(unsigned long long from_ns, unsigned long long to_ns);
std::string SampleReport(int top = 30);

/* PS2HOST_STALL_WATCH=<ms>: while the GS thread has been on one batch of its
   entries longer than that, a thread of its own takes the GS thread's and
   the GPU renderer's stacks (every 200 ms, six at most a stall) and prints
   them, named - what a stall of seconds that no span of the GS thread's
   covers was doing (the timeline only says "busy"). The GS thread's calls
   around each batch: StallWorkBegan, StallWorkEnded. */
void StartStallWatch();
void StallWorkBegan();
void StallWorkEnded();

}  // namespace ps2host

#endif  /* PS2HOST_PROFILE_H */

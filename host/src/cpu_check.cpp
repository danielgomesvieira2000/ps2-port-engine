// The game executable's entry (ps2_add_port links it as /ENTRY), before the
// C runtime starts: whether this processor can run this build.
//
// A port is built for AVX2 by default (engine/host/CMakeLists.txt,
// PS2HOST_AVX2) - and so is everything in it, the runtime's start and the
// static constructors included: on a processor without AVX2 (Intel's
// Pentium and Celeron of before 2021: Pentium Silver, Celeron N) the first
// of its instructions would stop the program with no word said. So this file
// is built without AVX, and runs first: on such a processor it starts the
// build for it beside this one (<name>-sse4.exe: PS2HOST_AVX2=OFF), with the
// same command line, or says what is needed. The SSE4 build checks for
// SSE4.1 in turn (its intrinsics: the VU's lanes, the IPU).
//
// Windows: only the Windows API here, nothing of the C runtime is up yet.
// Linux: a function in the executable's .preinit_array, which the C library
// runs before every constructor (its own, libstdc++'s, the port's): the SSE4
// build (<name>-sse4) takes the process over (execve), with the same
// arguments and environment.

#if defined(_WIN32) && defined(_MSC_VER)

#if defined(__AVX__)
#error "cpu_check.cpp must be built without AVX (it runs on processors without it)"
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>

extern "C" DWORD mainCRTStartup(LPVOID);   // the runtime's start: main

namespace {

bool HasAvx2() {
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuid(info, 1);
    const bool osxsave = (info[2] >> 27) & 1, avx = (info[2] >> 28) & 1;
    if (!osxsave || !avx) return false;
    if ((_xgetbv(0) & 6) != 6) return false;        // the OS saves the YMM registers
    __cpuidex(info, 7, 0);
    return (info[1] >> 5) & 1;
}

bool HasSse41() {
    int info[4];
    __cpuid(info, 1);
    return (info[2] >> 19) & 1;
}

/* PS2HOST_TEST_NO_AVX2=1: as on a processor without AVX2 (to test the way to
   the SSE4 build on one with it). */
bool TestNoAvx2() {
    wchar_t value[8];
    return GetEnvironmentVariableW(L"PS2HOST_TEST_NO_AVX2", value, 8) > 0;
}

/* The SSE4 build beside this executable: its own path with "-sse4" before
   ".exe", or "" when there is none. */
bool SiblingPath(wchar_t* out, DWORD size) {
    wchar_t self[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, self, MAX_PATH);
    if (n == 0 || n >= MAX_PATH || n < 4) return false;
    DWORD stem = n;
    if (self[n - 4] == L'.') stem = n - 4;         // before ".exe"
    const wchar_t suffix[] = L"-sse4.exe";
    if (stem + (sizeof suffix / sizeof suffix[0]) > size) return false;
    for (DWORD i = 0; i < stem; ++i) out[i] = self[i];
    for (DWORD i = 0; i < sizeof suffix / sizeof suffix[0]; ++i) out[stem + i] = suffix[i];
    return GetFileAttributesW(out) != INVALID_FILE_ATTRIBUTES;
}

/* The other build, with this one's command line; its exit code is this one's. */
DWORD RunSibling(const wchar_t* path) {
    const wchar_t* line = GetCommandLineW();
    DWORD length = 0;
    while (line[length]) ++length;
    HANDLE heap = GetProcessHeap();
    wchar_t* copy = static_cast<wchar_t*>(HeapAlloc(heap, 0, (length + 1) * sizeof(wchar_t)));
    if (!copy) return 71;
    for (DWORD i = 0; i <= length; ++i) copy[i] = line[i];
    // Its output where this one's goes (a terminal, a pipe: a scripted run's).
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(path, copy, nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) return 71;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 71;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return code;
}

}  // namespace

extern "C" DWORD Ps2CheckedStart(LPVOID peb) {
#if defined(PS2HOST_BUILT_FOR_AVX2)
    if (HasAvx2() && !TestNoAvx2()) return mainCRTStartup(peb);
    wchar_t sibling[MAX_PATH];
    if (SiblingPath(sibling, MAX_PATH)) ExitProcess(RunSibling(sibling));
    MessageBoxW(nullptr,
                L"This build needs a processor with AVX2 (Intel Haswell or AMD Excavator, 2013-2015, or later).\n\n"
                L"For a processor without it (Intel Pentium or Celeron before 2021), use the \"-sse4\" "
                L"executable that comes with it, in the same folder.",
                L"Processor not supported", MB_OK | MB_ICONERROR);
    ExitProcess(70);
#else
    if (HasSse41()) return mainCRTStartup(peb);
    MessageBoxW(nullptr, L"This build needs a processor with SSE4.1 (2008 or later).",
                L"Processor not supported", MB_OK | MB_ICONERROR);
    ExitProcess(70);
#endif
}

#elif defined(__linux__) && defined(__x86_64__) && !defined(__ANDROID__)

#if defined(__AVX__)
#error "cpu_check.cpp must be built without AVX (it runs on processors without it)"
#endif

#include <cpuid.h>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace {

[[maybe_unused]] bool HasAvx2() {
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __get_cpuid(1, &a, &b, &c, &d);
    const bool osxsave = (c >> 27) & 1, avx = (c >> 28) & 1;
    if (!osxsave || !avx) return false;
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    if ((lo & 6) != 6) return false;            // the OS saves the YMM registers
    __get_cpuid_count(7, 0, &a, &b, &c, &d);
    return (b >> 5) & 1;
}

[[maybe_unused]] bool HasSse41() {
    unsigned a = 0, b = 0, c = 0, d = 0;
    __get_cpuid(1, &a, &b, &c, &d);
    return (c >> 19) & 1;
}

void Say(const char* text) {
    if (write(2, text, std::strlen(text)) < 0) return;
}

// The environment's variable `name`, from envp: getenv is not up yet here (the
// C library sets environ in its own initializer, after the executable's
// .preinit_array).
[[maybe_unused]] bool HasVariable(char** envp, const char* name) {
    const size_t length = std::strlen(name);
    for (char** e = envp; e && *e; ++e)
        if (std::strncmp(*e, name, length) == 0 && (*e)[length] == '=') return true;
    return false;
}

void CheckProcessor(int argc, char** argv, char** envp) {
    (void)argc;
#if defined(PS2HOST_BUILT_FOR_AVX2)
    // PS2HOST_TEST_NO_AVX2=1: as on a processor without AVX2.
    if (HasAvx2() && !HasVariable(envp, "PS2HOST_TEST_NO_AVX2")) return;
    char self[4096];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 8);
    if (n > 0) {
        std::memcpy(self + n, "-sse4", 6);
        if (access(self, X_OK) == 0) {
            argv[0] = self;
            execve(self, argv, envp);
        }
    }
    Say("This build needs a processor with AVX2 (Intel Haswell or AMD Excavator, 2013-2015, or later).\n"
        "For a processor without it (Intel Pentium or Celeron before 2021), run the \"-sse4\" program "
        "that comes with it, in the same folder.\n");
    _exit(70);
#else
    (void)argv, (void)envp;
    if (HasSse41()) return;
    Say("This build needs a processor with SSE4.1 (2008 or later).\n");
    _exit(70);
#endif
}

}  // namespace

__attribute__((section(".preinit_array"), used)) void (*const ps2_check_processor)(int, char**, char**) =
    CheckProcessor;

#endif

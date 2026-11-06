"""Compile the generated C++ and run it against the original machine code.

The derivations are verified as expression trees, evaluated in Python. The thing
the project actually ships is the C++ those trees are rendered to, and nothing
checked that the two mean the same. They did not: the renderer printed a signed
and an unsigned comparison identically, and C++'s promotion rules quietly
narrowed 64-bit register arithmetic to `int` - 419 shifts in the emitted file were
undefined behaviour, and the file compiled without a single error.

So this closes the loop. The emitted source is compiled into a shared library
together with a dispatch table and memory helpers that call back into Python,
and every function is run on the same cases the derivation was verified on,
against the interpreter. A function only counts if its *compiled* form agrees.

The compiler is MSVC, found through vswhere, on Windows, and clang++ elsewhere
($CXX names another). Where none is installed the check reports itself
unavailable rather than passing.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# The runtime rounds floats as the console does (toward zero, denormals
# flushed: eefpu.cpp, vu0.cpp); the interpreter it is compared against rounds
# to nearest. Verification checks the translation, not the FPU model: the
# compiled code runs in the interpreter's mode. Set before any harness loads.
os.environ["PS2RT_EE_ROUND_NEAREST"] = "1"

MASK64 = (1 << 64) - 1

# The C++ runtime the generated code is compiled against: engine/runtime.
RUNTIME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime")
# The runtime sources derived code is linked with - one list, so no tool that
# compiles derived code can forget one (it once linked without the VU0 helpers).
RUNTIME_SOURCES = [os.path.join(RUNTIME, "src", name)
                   for name in ("eefpu.cpp", "eeint.cpp", "vu0.cpp")]

VSWHERE = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"

# Functions the emitted code calls but which were not derived: declared, not defined.
EXTERN = re.compile(r"^(u64|u32) (func_([0-9A-F]{8}))\(([^)]*)\);  "
                    r"/\* not yet derived(?:; returns a float, as its bits)? \*/$", re.M)

SIGNATURE = re.compile(r"^(s32|u32|void) (func_([0-9A-F]{8}))\(([^)]*)\) \{$", re.M)

READERS = [
    ("ReadS8", 1, 1), ("ReadU8", 1, 0), ("ReadS16", 2, 1), ("ReadU16", 2, 0),
    ("ReadS32", 4, 1), ("ReadU32", 4, 0), ("ReadU64", 8, 0),
]
WRITERS = [("Write8", 1, "u32"), ("Write16", 2, "u32"), ("Write32", 4, "u32"),
           ("Write64", 8, "u64")]

READ_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32,
                                 ctypes.c_uint32)
WRITE_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32)
CALL_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint32,
                                 ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,
                                 ctypes.c_uint32)


_BUILDS = 0

from . import layout


class CompilerUnavailable(Exception):
    pass


class CompileError(Exception):
    pass


# A check library: a DLL on Windows, a shared library elsewhere.
SHARED_SUFFIX = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"

# Outside Windows the check libraries are compiled as the port is: no
# multiply-add fused into one rounding (engine/host/CMakeLists.txt), and -O3
# as /O2 is MSVC's - the optimiser gets the same chance at undefined
# behaviour here as in the port. The harnesses spell their exports
# __declspec(dllexport), which this makes a visible symbol.
CLANG_FLAGS = ["-std=c++17", "-O3", "-fPIC", "-fexceptions", "-ffp-contract=off", "-w", "-ferror-limit=0",
               "-D__declspec(x)=__attribute__((visibility(\"default\")))"]

_CLANG_ERROR = re.compile(r"^(.*?):(\d+):\d+: (?:fatal )?error: (.*)$")


# On Arm the runtime's SSE intrinsics are NEON's, through sse2neon (MIT) behind
# the x86 headers' names: the same file, by hash, as engine/host/CMakeLists.txt.
SSE2NEON_URL = "https://raw.githubusercontent.com/DLTcollab/sse2neon/v1.8.0/sse2neon.h"
SSE2NEON_SHA256 = "07723c9f9457dd4316f1fde3dd4eb6f31dd67d9955f6c21f4e609ac1698be48a"
X86_HEADERS = ("immintrin.h", "emmintrin.h", "xmmintrin.h", "pmmintrin.h", "tmmintrin.h", "smmintrin.h", "nmmintrin.h")


def sse2neon_include() -> str:
    """The folder of sse2neon and the x86 header names that include it, kept
    in the user's cache and fetched once; the empty string where the processor
    is not Arm."""
    if platform.machine().lower() not in ("arm64", "aarch64"):
        return ""
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    folder = os.path.join(cache, "ps2-port-engine", "sse2neon-" + SSE2NEON_SHA256[:12])
    header = os.path.join(folder, "sse2neon.h")
    if not os.path.exists(header):
        os.makedirs(folder, exist_ok=True)
        with urllib.request.urlopen(SSE2NEON_URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SSE2NEON_SHA256:
            raise CompilerUnavailable("sse2neon.h from %s is not the pinned file" % SSE2NEON_URL)
        with open(header, "wb") as handle:
            handle.write(data)
    for name in X86_HEADERS:
        shim = os.path.join(folder, name)
        if not os.path.exists(shim):
            with open(shim, "w") as handle:
                handle.write('#pragma once\n#include "sse2neon.h"\n')
    return folder


def find_compiler() -> str:
    """clang++ (or $CXX) outside Windows; CompilerUnavailable where there is none."""
    found = shutil.which(os.environ.get("CXX") or "clang++")
    if not found:
        raise CompilerUnavailable("no clang++ on the path (or in $CXX)")
    return found


def compile_shared(sources, out: str, include_dirs, work_dir: str) -> None:
    """Outside Windows: `sources` built into the shared library `out`, a
    compiler each side by side. Errors are raised as CompileError in MSVC's
    spelling, `file(line) : error : text`, which the callers read to drop the
    functions that do not compile. A harness (its dispatch switch of thousands
    of cases) is built unoptimised, as its MSVC build is."""
    clang = find_compiler()
    work_dir = os.path.abspath(work_dir)
    objects_dir = os.path.join(work_dir, "objects_" + os.path.splitext(os.path.basename(out))[0])
    os.makedirs(objects_dir, exist_ok=True)
    includes = ["-I" + os.path.abspath(path) for path in include_dirs]
    neon = sse2neon_include()
    if neon:
        includes = ["-isystem", neon] + includes

    def compile_one(item):
        number, source = item
        stem = os.path.basename(source)
        obj = os.path.join(objects_dir, "%03d_%s.o" % (number, stem))
        optimise = ["-O0"] if "harness" in stem else []
        result = subprocess.run([clang, "-c"] + CLANG_FLAGS + optimise + includes + [source, "-o", obj],
                                capture_output=True, text=True)
        return obj, result

    jobs = max(1, min(os.cpu_count() or 4, 8))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        compiled = list(pool.map(compile_one, enumerate(os.path.abspath(path) for path in sources)))
    errors, other = [], []
    for _obj, result in compiled:
        if result.returncode == 0:
            continue
        for line in (result.stderr or "").splitlines():
            match = _CLANG_ERROR.match(line)
            if match:
                errors.append("%s(%s) : error : %s" % match.groups())
            elif "error" in line:
                other.append(line)
    if errors or other:
        raise CompileError("\n".join(errors + other) or "the compiler failed")
    link = subprocess.run([clang, "-shared", "-o", out] + [obj for obj, _ in compiled],
                          capture_output=True, text=True)
    if link.returncode != 0 or not os.path.exists(out):
        raise CompileError("link: error : " + (link.stderr or "the link failed")[-3000:])


def syntax_check(source: str, include_dirs) -> None:
    """Outside Windows: the front end alone over `source` (seconds, where the
    optimised build is minutes), raising CompileError in MSVC's spelling for
    every error. A caller that drops what does not compile, and rebuilds, then
    pays for the optimiser once instead of once per round."""
    includes = ["-I" + os.path.abspath(path) for path in include_dirs]
    neon = sse2neon_include()
    if neon:
        includes = ["-isystem", neon] + includes
    flags = [flag for flag in CLANG_FLAGS if flag not in ("-O3", "-fPIC")]
    result = subprocess.run([find_compiler(), "-fsyntax-only"] + flags + includes + [os.path.abspath(source)],
                            capture_output=True, text=True)
    if result.returncode == 0:
        return
    errors = ["%s(%s) : error : %s" % match.groups()
              for match in map(_CLANG_ERROR.match, (result.stderr or "").splitlines()) if match]
    raise CompileError("\n".join(errors) or (result.stderr or "the compiler failed")[-2000:])


def find_vcvars() -> str:
    if not os.path.exists(VSWHERE):
        raise CompilerUnavailable("vswhere not found; no Visual Studio installed")
    result = subprocess.run(
        [VSWHERE, "-latest", "-products", "*", "-property", "installationPath"],
        capture_output=True, text=True,
    )
    root = result.stdout.strip()
    vcvars = os.path.join(root, "VC", "Auxiliary", "Build", "vcvars64.bat")
    if not root or not os.path.exists(vcvars):
        raise CompilerUnavailable("no MSVC x64 toolset found")
    return vcvars


def functions_in(source: str) -> list:
    """(address, name, return type, parameter indices) for every emitted function."""
    found = []
    for match in SIGNATURE.finditer(source):
        return_type, name, address, params = match.groups()
        found.append((int(address, 16), name, return_type, _indices(params, "v0_in"),
                      params))
    return found


def _indices(params: str, v0_name: str) -> list:
    """The vector entry of each parameter, in order: arg0..arg7 are entries 0..7,
    farg0..farg7 entries 8..15, and the incoming $v0 (named `v0_name`) entry 16."""
    indices = []
    for f, n, v0 in re.findall(r"\b(?:(f?)arg(\d+)|(%s))\b" % v0_name, params):
        indices.append(16 if v0 else int(n) + (8 if f else 0))
    return indices


def externs_in(source: str) -> list:
    """(address, name, parameter indices, returns a float) for every
    not-yet-derived callee."""
    found = []
    for match in EXTERN.finditer(source):
        return_type, name, address, params = match.groups()
        found.append((int(address, 16), name, _indices(params, "v0"),
                      return_type == "u32"))
    return found


def port_memory(read: str, write: str) -> list:
    """generated.h's memory hooks for a harness: no main RAM given to the code
    inline (ram_base null), so every access goes through `read` (of a, size,
    is_signed) and `write` (of a, v, size)."""
    return [
        "u8* ram_base = nullptr;",
        "bool ram_watched = false;",
        "u8* scratchpad_base = nullptr;",
        "u64 PortRead(u32 a, u32 size, bool is_signed) { return %s; }" % read,
        "void PortWrite(u32 a, u64 v, u32 size) { %s; }" % write,
        # Only cases the original completes are run, so no trap is reached.
        "void TrapAt(u32) {}",
    ]


def harness_source(functions, typed: bool, externs=()) -> str:
    """A dispatch table plus memory helpers that forward to Python callbacks."""
    # The dispatch switch has a case per function - over two thousand - and
    # optimising it ran the compiler out of heap. Only the generated code has to be
    # checked at /O2; the harness that calls it does not.
    lines = ['#pragma optimize("", off)', '#include "ps2rt/generated.h"']
    if typed:
        lines.append('#include "types.h"')
    lines += ["", "namespace generated {"]
    for _address, name, return_type, _indices, params in functions:
        # The declaration repeats the definition's own parameter list, typed views
        # included; the u64 arguments below convert to them implicitly.
        lines.append("%s %s(%s);" % (return_type, name, params))
    lines += [
        "}",
        "",
        "typedef u64 (*ReadFn)(u32 address, u32 size, u32 is_signed);",
        "typedef void (*WriteFn)(u32 address, u64 value, u32 size);",
        "typedef u64 (*CallFn)(u32 address, const u64* arguments, u32 stack_top,",
        "                      u32 float_result);",
        "static ReadFn g_read;",
        "static WriteFn g_write;",
        "static CallFn g_call;",
        "static u32 g_sp = 0x%08X;  /* the interpreter's default stack top */"
        % layout.CURRENT.stack_top,
        "",
        "namespace generated {",
        "u64 StackEnter(u32 size) { g_sp -= size; return g_sp; }",
        "u64 CallPointer(u32 address, u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5,",
        "                u64 a6, u64 a7, u64 f0, u64 f1, u64 f2, u64 f3, u64 f4, u64 f5,",
        "                u64 f6, u64 f7) {",
        "    u64 vector[17] = {a0, a1, a2, a3, a4, a5, a6, a7,",
        "                      f0, f1, f2, f3, f4, f5, f6, f7, 0};",
        "    return g_call(address, vector, g_sp, 0);",
        "}",
        "void StackLeave(u32 size) { g_sp += size; }",
    ]
    # A callee that was not derived runs as the original machine code.
    for address, name, indices, returns_float in externs:
        params = ", ".join("u64 p%d" % index for index in indices) or "void"
        lines.append("%s %s(%s) {" % ("u32" if returns_float else "u64", name, params))
        lines.append("    u64 vector[17] = {0};")
        for index in indices:
            lines.append("    vector[%d] = p%d;" % (index, index))
        # Callees run at the caller's $sp, as in the original and the model
        # (paths.Execution): StackEnter has taken the derived function's frame.
        if returns_float:
            lines.append("    return (u32)g_call(0x%08X, vector, g_sp, 1);" % address)
        else:
            lines.append("    return g_call(0x%08X, vector, g_sp, 0);" % address)
        lines.append("}")
    lines += port_memory("g_read(a, size, is_signed)", "g_write(a, v, size)")
    lines += [
        "}",
        "",
        'extern "C" __declspec(dllexport) void b3_set_memory(ReadFn r, WriteFn w, CallFn c) {',
        "    g_read = r;",
        "    g_write = w;",
        "    g_call = c;",
        "}",
        "",
        "/* Returns 1 with *result set, 2 for a void function, 0 if unknown. */",
        'extern "C" __declspec(dllexport) int b3_call(u32 address, const u64* args, s32* result) {',
        # Every case starts at the stack top: a path that traps returns without
        # its StackLeave.
        "    g_sp = 0x%08X;" % layout.CURRENT.stack_top,
        "    switch (address) {",
    ]
    for address, name, return_type, indices, _params in functions:
        call = "generated::%s(%s)" % (name, ", ".join("args[%d]" % i for i in indices))
        if return_type == "void":
            lines.append("    case 0x%08X: %s; return 2;" % (address, call))
        else:
            lines.append("    case 0x%08X: *result = %s; return 1;" % (address, call))
    lines += ["    }", "    return 0;", "}", ""]
    return "\n".join(lines)


def build(generated_path: str, runtime_sources, include_dir: str, work_dir: str) -> tuple:
    """Compile the generated file into a DLL; returns (dll path, functions)."""
    vcvars = find_vcvars() if os.name == "nt" else find_compiler()
    # Absolute: the build script changes into work_dir before compiling, so a
    # relative path (a relative --game) would name a file that is not there.
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    with open(generated_path, encoding="utf-8") as handle:
        functions = functions_in(handle.read())
    harness = os.path.join(work_dir, "harness.cpp")
    generated_dir = os.path.dirname(os.path.abspath(generated_path))
    typed = os.path.exists(os.path.join(generated_dir, "types.h"))
    with open(generated_path, encoding="utf-8") as handle:
        externs = externs_in(handle.read())
    with open(harness, "w", encoding="utf-8") as handle:
        handle.write(harness_source(functions, typed, externs))

    # A fresh name per build: Windows keeps a loaded DLL locked, so rebuilding in
    # the same process under the same name would fail.
    global _BUILDS
    _BUILDS += 1
    dll = os.path.join(work_dir, "generated_check_%d_%d%s" % (os.getpid(), _BUILDS, SHARED_SUFFIX))
    sources = [os.path.abspath(generated_path), harness] + [
        os.path.abspath(path) for path in runtime_sources
    ]
    # /O2 on purpose: an optimiser is what turns undefined behaviour into wrong
    # answers, so checking an unoptimised build would hide exactly the problems
    # this exists to find.
    if os.name != "nt":
        syntax_check(generated_path, [include_dir, generated_dir])
        compile_shared(sources, dll, [include_dir, generated_dir], work_dir)
        return dll, functions
    script = os.path.join(work_dir, "build.bat")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write("@echo off\n")
        handle.write('call "%s" >nul\n' % vcvars)
        handle.write('cd /d "%s"\n' % work_dir)
        handle.write('cl /nologo /std:c++17 /O2 /W3 /bigobj /LD /I"%s" /I"%s" %s /Fe"%s" >build.log 2>&1\n'
                     % (os.path.abspath(include_dir), generated_dir,
                        " ".join('"%s"' % path for path in sources), dll))
        handle.write("exit /b %errorlevel%\n")
    result = subprocess.run(["cmd", "/c", script], capture_output=True, text=True)
    if result.returncode != 0:
        with open(os.path.join(work_dir, "build.log"), encoding="utf-8",
                  errors="replace") as handle:
            log = handle.read()
        errors = [line for line in log.splitlines() if " error " in line]
        # Every error, not a sample: the caller drops each function named here.
        raise CompileError("\n".join(errors) or log[-2000:])
    return dll, functions


class CompiledFunctions:
    """The compiled library, with memory routed to an interpreter's memory."""

    def __init__(self, dll_path: str, machine, budget: int = 100_000) -> None:
        self.machine = machine
        self.budget = budget
        self.memory = machine.memory
        # Keep the callback objects alive for as long as the library can call them.
        self._read = READ_CALLBACK(self._on_read)
        self._write = WRITE_CALLBACK(self._on_write)
        self._call = CALL_CALLBACK(self._on_call)
        self.library = ctypes.CDLL(dll_path)
        self.library.b3_set_memory(self._read, self._write, self._call)
        self.library.b3_call.restype = ctypes.c_int
        self.library.b3_call.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32)
        ]
        self.error = None

    def _on_read(self, address, size, signed):
        try:
            return self.memory.read_int(address, size, bool(signed)) & MASK64
        except Exception as exc:     # an exception cannot cross the C boundary
            self.error = exc
            return 0

    def _on_write(self, address, value, size):
        # Written straight to memory, inside the verifier's journal: a callee run
        # later in the same function must see it, as it would on the console.
        try:
            self.memory.write_int(address, value & ((1 << (8 * size)) - 1), size)
        except Exception as exc:
            self.error = exc

    def _on_call(self, address, arguments, stack_top, float_result):
        try:
            self.machine.call(address, [arguments[i] for i in range(17)],
                              max_instructions=self.budget, stack_top=stack_top)
            if float_result:
                return self.machine.fpr[0] & 0xFFFFFFFF
            return self.machine.get(2) & MASK64
        except Exception as exc:
            self.error = exc
            return 0

    def evaluator(self, address: int):
        """An `evaluate(args)` for derive.run_cases, backed by compiled code."""
        def evaluate(args):
            self.error = None
            values = (ctypes.c_uint64 * 17)(
                *[(a & MASK64) for a in (list(args) + [0] * 17)[:17]])
            result = ctypes.c_int32(0)
            kind = self.library.b3_call(address, values, ctypes.byref(result))
            if self.error is not None:
                raise self.error
            if kind == 0:
                raise KeyError("0x%08X is not in the compiled library" % address)
            return (None if kind == 2 else result.value), []
        return evaluate

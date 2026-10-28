/*
 * Fibers and address waits where Windows' are not (fiber.h): Android, Linux.
 */

#ifndef _WIN32

#include "fiber.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

struct Fiber {
    void* sp = nullptr;             // its stack as the switch away from it left it
    void* stack = nullptr;          // (none: a thread's own, ConvertThreadToFiber)
    std::size_t size = 0;
};

thread_local Fiber* t_current = nullptr;
thread_local Fiber* t_thread = nullptr;     // the thread's own (ConvertThreadToFiber)

}  // namespace

/* fn(arg) with the stack pointer `sp`, back on the caller's stack after. */
extern "C" void ps2_call_on_stack(void* sp, void (*fn)(void*), void* arg);

/* Saves the registers a call keeps on the stack it runs on, its stack
   pointer to *save, and goes on with the stack `load` (a switch's, or a new
   fiber's first frame: fiber_start's). */
extern "C" void ps2_fiber_switch(void** save, void* load);
extern "C" void ps2_fiber_start();

#if defined(__aarch64__)
__asm__(
    ".text\n"
    ".globl ps2_fiber_switch\n"
    ".type ps2_fiber_switch, %function\n"
    ".p2align 4\n"
    "ps2_fiber_switch:\n"
    "    sub sp, sp, #160\n"
    "    stp x19, x20, [sp, #0]\n"
    "    stp x21, x22, [sp, #16]\n"
    "    stp x23, x24, [sp, #32]\n"
    "    stp x25, x26, [sp, #48]\n"
    "    stp x27, x28, [sp, #64]\n"
    "    stp x29, x30, [sp, #80]\n"
    "    stp d8, d9, [sp, #96]\n"
    "    stp d10, d11, [sp, #112]\n"
    "    stp d12, d13, [sp, #128]\n"
    "    stp d14, d15, [sp, #144]\n"
    "    mov x9, sp\n"
    "    str x9, [x0]\n"
    "    mov sp, x1\n"
    "    ldp x19, x20, [sp, #0]\n"
    "    ldp x21, x22, [sp, #16]\n"
    "    ldp x23, x24, [sp, #32]\n"
    "    ldp x25, x26, [sp, #48]\n"
    "    ldp x27, x28, [sp, #64]\n"
    "    ldp x29, x30, [sp, #80]\n"
    "    ldp d8, d9, [sp, #96]\n"
    "    ldp d10, d11, [sp, #112]\n"
    "    ldp d12, d13, [sp, #128]\n"
    "    ldp d14, d15, [sp, #144]\n"
    "    add sp, sp, #160\n"
    "    ret\n"
    ".size ps2_fiber_switch, .-ps2_fiber_switch\n"
    ".globl ps2_fiber_start\n"
    ".type ps2_fiber_start, %function\n"
    ".p2align 4\n"
    "ps2_fiber_start:\n"            // x19 the start routine, x20 its parameter
    "    mov x0, x20\n"
    "    blr x19\n"
    "    brk #0\n"                  // (a fiber's routine never returns)
    ".size ps2_fiber_start, .-ps2_fiber_start\n"
    ".globl ps2_call_on_stack\n"
    ".type ps2_call_on_stack, %function\n"
    ".p2align 4\n"
    "ps2_call_on_stack:\n"          // x0 the stack, x1 the routine, x2 its parameter
    "    stp x29, x30, [sp, #-16]!\n"
    "    mov x29, sp\n"             // (kept by the call)
    "    mov sp, x0\n"
    "    mov x0, x2\n"
    "    blr x1\n"
    "    mov sp, x29\n"
    "    ldp x29, x30, [sp], #16\n"
    "    ret\n"
    ".size ps2_call_on_stack, .-ps2_call_on_stack\n");
constexpr std::size_t kFrame = 160;
#elif defined(__x86_64__)
__asm__(
    ".text\n"
    ".globl ps2_fiber_switch\n"
    ".type ps2_fiber_switch, @function\n"
    ".p2align 4\n"
    "ps2_fiber_switch:\n"
    "    pushq %rbp\n"
    "    pushq %rbx\n"
    "    pushq %r12\n"
    "    pushq %r13\n"
    "    pushq %r14\n"
    "    pushq %r15\n"
    "    movq %rsp, (%rdi)\n"
    "    movq %rsi, %rsp\n"
    "    popq %r15\n"
    "    popq %r14\n"
    "    popq %r13\n"
    "    popq %r12\n"
    "    popq %rbx\n"
    "    popq %rbp\n"
    "    ret\n"
    ".size ps2_fiber_switch, .-ps2_fiber_switch\n"
    ".globl ps2_fiber_start\n"
    ".type ps2_fiber_start, @function\n"
    ".p2align 4\n"
    "ps2_fiber_start:\n"            // r12 the start routine, r13 its parameter
    "    movq %r13, %rdi\n"
    "    callq *%r12\n"
    "    ud2\n"
    ".size ps2_fiber_start, .-ps2_fiber_start\n"
    ".globl ps2_call_on_stack\n"
    ".type ps2_call_on_stack, @function\n"
    ".p2align 4\n"
    "ps2_call_on_stack:\n"          // rdi the stack, rsi the routine, rdx its parameter
    "    pushq %rbp\n"
    "    movq %rsp, %rbp\n"         // (kept by the call)
    "    movq %rdi, %rsp\n"
    "    movq %rdx, %rdi\n"
    "    callq *%rsi\n"
    "    movq %rbp, %rsp\n"
    "    popq %rbp\n"
    "    ret\n"
    ".size ps2_call_on_stack, .-ps2_call_on_stack\n");
#else
#error "fiber.cpp: no context switch for this processor"
#endif

void* ConvertThreadToFiber(void*) {
    t_current = t_thread = new Fiber();
    return t_current;
}

void OnThreadStack(void (*fn)(void*), void* arg) {
    if (!t_thread || t_current == t_thread) {
        fn(arg);
        return;
    }
    // Below where the thread's own stack was left when it switched to the
    // fibers (a red zone's worth spared), 16-aligned.
    auto sp = reinterpret_cast<std::uintptr_t>(t_thread->sp) - 512;
    sp &= ~std::uintptr_t(15);
    ps2_call_on_stack(reinterpret_cast<void*>(sp), fn, arg);
}

void* CreateFiber(std::size_t stack_size, void (*start)(void*), void* parameter) {
    const std::size_t page = std::size_t(sysconf(_SC_PAGESIZE));
    stack_size = (stack_size + page - 1) / page * page;
    // A page below the stack that faults: an overflow stops at once.
    void* memory = mmap(nullptr, stack_size + page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) return nullptr;
    mprotect(memory, page, PROT_NONE);
    Fiber* f = new Fiber();
    f->stack = memory;
    f->size = stack_size + page;
    auto top = reinterpret_cast<std::uintptr_t>(memory) + f->size;
    top &= ~std::uintptr_t(15);
#if defined(__aarch64__)
    auto* frame = reinterpret_cast<std::uint64_t*>(top - kFrame);
    for (std::size_t i = 0; i < kFrame / 8; ++i) frame[i] = 0;
    frame[0] = reinterpret_cast<std::uint64_t>(start);          // x19
    frame[1] = reinterpret_cast<std::uint64_t>(parameter);      // x20
    frame[11] = reinterpret_cast<std::uint64_t>(&ps2_fiber_start);     // x30
    f->sp = frame;
#else
    // r15 r14 r13 r12 rbx rbp, then the return: ps2_fiber_start, entered
    // with the stack 16-aligned (its call leaves the routine's as a call's).
    auto* frame = reinterpret_cast<std::uint64_t*>(top - 72);
    for (int i = 0; i < 9; ++i) frame[i] = 0;
    frame[2] = reinterpret_cast<std::uint64_t>(parameter);      // r13
    frame[3] = reinterpret_cast<std::uint64_t>(start);          // r12
    frame[6] = reinterpret_cast<std::uint64_t>(&ps2_fiber_start);
    f->sp = frame;
#endif
    return f;
}

void SwitchToFiber(void* fiber) {
    Fiber* to = static_cast<Fiber*>(fiber);
    Fiber* from = t_current;
    if (to == from) return;
    t_current = to;
    ps2_fiber_switch(&from->sp, to->sp);
}

void DeleteFiber(void* fiber) {
    Fiber* f = static_cast<Fiber*>(fiber);
    if (!f) return;
    if (f->stack) munmap(f->stack, f->size);
    delete f;
}

bool WaitOnAddress(volatile void* address, void* compare, std::size_t, unsigned ms) {
    timespec timeout{time_t(ms / 1000), long(ms % 1000) * 1000000L};
    syscall(SYS_futex, const_cast<void*>(address), FUTEX_WAIT_PRIVATE, *static_cast<int*>(compare), &timeout,
            nullptr, 0);
    return true;
}

void WakeByAddressAll(void* address) {
    syscall(SYS_futex, address, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
}

#endif

/*
 * The EE threads' fibers and the timer's address waits (threads.cpp) with
 * Win32's names on every system: on Windows windows.h's own, elsewhere
 * fiber.cpp's - a fiber a stack of its own and the registers a call keeps
 * (switched by a few instructions of assembly: AArch64, x86-64), the waits
 * Linux's futexes. As Windows' fibers by default, a switch keeps the
 * floating point's control register (rounding) as it is.
 */

#pragma once

#ifdef _WIN32
#include "ps2host/platform.h"
#include <timeapi.h>
#else

#include <cstddef>

#define CALLBACK

void* ConvertThreadToFiber(void* parameter);
void* CreateFiber(std::size_t stack_size, void (*start)(void*), void* parameter);
void SwitchToFiber(void* fiber);
void DeleteFiber(void* fiber);
/* Asleep while the 4 bytes at `address` equal those at `compare`, for at
   most `ms` milliseconds (or until woken): true. */
bool WaitOnAddress(volatile void* address, void* compare, std::size_t size, unsigned ms);
void WakeByAddressAll(void* address);

#endif

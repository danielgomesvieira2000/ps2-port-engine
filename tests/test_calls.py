"""Tests for derived functions that call functions which were not derived.

The caller is derived; the callee is called, not inlined - in verification it runs
as the original machine code in the interpreter. What has to hold is ordering:
the callee sees every store made before the call, and loads after the call read
memory as the callee left it.
"""

import os
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp.emu import Machine
from ps2decomp.paths import Execution, derive_paths
from ps2decomp.symexec import Load

BASE = 0x00100000
A0, A1, V0, V1, S0, SP, RA = 4, 5, 2, 3, 16, 29, 31


def i_type(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def r_type(rs, rt, rd, funct):
    return (rs << 21) | (rt << 16) | (rd << 11) | funct


def lw(rt, offset, rs): return i_type(0x23, rs, rt, offset)
def sw(rt, offset, rs): return i_type(0x2B, rs, rt, offset)
def ld(rt, offset, rs): return i_type(0x37, rs, rt, offset)
def sd(rt, offset, rs): return i_type(0x3F, rs, rt, offset)
def addiu(rt, rs, imm): return i_type(0x09, rs, rt, imm)
def addu(rd, rs, rt): return r_type(rs, rt, rd, 0x21)
def move(rd, rs): return r_type(rs, 0, rd, 0x2D)
def jal(target): return (3 << 26) | ((target >> 2) & 0x03FFFFFF)


JR_RA = r_type(RA, 0, 0, 0x08)
NOP = 0

# callee(p, n): v0 = *p + n; p[1] = v0; return v0
CALLEE = [lw(V0, 0, A0), addu(V0, V0, A1), sw(V0, 4, A0), JR_RA, NOP]


def caller_words(callee):
    """caller(p): *p = 7; r = callee(p, 5); return r + p[1]"""
    return [
        addiu(SP, SP, -32),
        sd(RA, 16, SP),
        sd(S0, 0, SP),
        move(S0, A0),
        addiu(V1, 0, 7),
        sw(V1, 0, A0),
        jal(callee),
        addiu(A1, 0, 5),                 # delay slot: the second argument
        lw(V1, 4, S0),                   # what the callee wrote
        addu(V0, V0, V1),
        ld(RA, 16, SP),
        ld(S0, 0, SP),
        JR_RA,
        addiu(SP, SP, 32),
    ]


class TestOpaqueCalls(unittest.TestCase):
    def setUp(self):
        callee = BASE
        self.caller = BASE + 4 * len(CALLEE)
        words = CALLEE + caller_words(callee)
        self.image = struct.pack("<%dI" % len(words), *words)
        self.callee = callee
        self.parameters = {callee: (0, 1)}

    def derive(self):
        return derive_paths(self.image, BASE, self.caller,
                            self.caller + 4 * len(caller_words(self.callee)),
                            known={}, parameters=self.parameters)

    def test_call_to_an_underived_function_is_kept_as_a_call(self):
        derivation = self.derive()
        self.assertTrue(derivation.has_calls)
        kinds = [statement[0] for statement in derivation.tree()]
        self.assertEqual(kinds, ["store", "call", "return"])

    def test_an_overlays_external_callee_has_parameters(self):
        """A callee outside the overlay's code (analyze.py's external_functions)
        is analysed with its own functions: the call is to a known function."""
        from ps2decomp import calls
        end = self.caller + 4 * len(caller_words(self.callee))
        entry = lambda start, stop: {"addr": start, "end": stop, "size": stop - start, "label": "f",
                                     "confidence": "called", "calls": []}
        database = {"functions": [entry(self.caller, end)],
                    "external_functions": [entry(self.callee, self.caller)]}
        found = calls.parameters(database, self.image, BASE)
        self.assertIn(self.callee, found)
        self.assertEqual(tuple(found[self.callee]), (0, 1))
        derivation = derive_paths(self.image, BASE, self.caller, end, known={}, parameters=found)
        self.assertTrue(derivation.has_calls)

    def test_f0_as_scratch_and_a_callees_v0_is_a_void_function(self):
        """One path calls the callee (its $v0 left as is) and uses $f0 as scratch;
        the other writes neither. Not a float function - $f0 is not set on every
        path - so a void one, as a function that never writes $v0 is."""
        bnez = lambda rs, words: (5 << 26) | (rs << 21) | (words & 0xFFFF)
        mtc1_zero_f0 = (0x11 << 26) | (4 << 21) | (0 << 16) | (0 << 11)
        words = [
            addiu(SP, SP, -16),
            sd(RA, 0, SP),
            bnez(A1, 4),                     # to the epilogue
            NOP,
            jal(self.callee),
            NOP,
            mtc1_zero_f0,                    # $f0 = 0.0: scratch, nothing returned in it
            ld(RA, 0, SP),
            JR_RA,
            addiu(SP, SP, 16),
        ]
        start = BASE + 4 * len(CALLEE)
        image = struct.pack("<%dI" % (len(CALLEE) + len(words)), *(CALLEE + words))
        derivation = derive_paths(image, BASE, start, start + 4 * len(words), known={},
                                  parameters=self.parameters)
        self.assertTrue(all(path.expression is None for path in derivation.paths))

    def test_a_leftover_from_an_earlier_call_is_no_value_either(self):
        """The last call preserves $v0 (calls.v0_preserving), so the path ends with
        the first callee's $v0; the other path writes none. With no caller known to
        read $v0, the function is void (Resident Evil Outbreak's 0x005AFAD0)."""
        from ps2decomp.calls import Parameters
        bnez = lambda rs, words: (5 << 26) | (rs << 21) | (words & 0xFFFF)
        keeper = BASE + 4 * len(CALLEE)                 # jr $ra; nop: $v0 as it was
        start = keeper + 8
        words = [
            addiu(SP, SP, -16),
            sd(RA, 0, SP),
            bnez(A1, 6),                     # to the epilogue
            NOP,
            jal(self.callee),
            NOP,
            jal(keeper),
            NOP,
            NOP,
            ld(RA, 0, SP),
            JR_RA,
            addiu(SP, SP, 16),
        ]
        image = struct.pack("<%dI" % (len(CALLEE) + 2 + len(words)), *(CALLEE + [JR_RA, NOP] + words))
        parameters = Parameters({self.callee: (0, 1), keeper: ()})
        parameters.preserves_v0 = frozenset({keeper})
        derivation = derive_paths(image, BASE, start, start + 4 * len(words), known={},
                                  parameters=parameters)
        self.assertTrue(all(path.expression is None for path in derivation.paths))
        # A caller that reads $v0 reads a value: the first callee's on one path,
        # its own $v0 back on the other (V0_IN).
        from ps2decomp.symexec import Arg, CallResult, V0_IN
        parameters.returns_int = frozenset({start})
        returned = [path.expression for path in derive_paths(
            image, BASE, start, start + 4 * len(words), known={}, parameters=parameters).paths]
        self.assertIn(Arg(V0_IN), returned)
        self.assertTrue(any(isinstance(value, CallResult) for value in returned))
        # So does one reached through a pointer, whose callers no scan sees.
        parameters.returns_int = frozenset()
        parameters.address_taken = frozenset({start})
        self.assertFalse(derive_paths(image, BASE, start, start + 4 * len(words), known={},
                                      parameters=parameters).is_void)

    def test_without_parameters_the_call_is_still_refused(self):
        """Opaque calls are opt-in: the old behaviour must remain available."""
        from ps2decomp.symexec import Unsupported

        with self.assertRaises(Unsupported):
            derive_paths(self.image, BASE, self.caller,
                         self.caller + 4 * len(caller_words(self.callee)), known={})

    def test_a_load_after_the_call_reads_the_next_epoch(self):
        derivation = self.derive()
        loads = []
        stack = [derivation.paths[0].expression]
        while stack:
            node = stack.pop()
            if isinstance(node, Load):
                loads.append(node)
            from ps2decomp.symexec import children
            stack.extend(children(node))
        self.assertEqual([load.epoch for load in loads], [1])

    def test_execution_agrees_with_the_original(self):
        """The callee must see the store made before the call, and the load after
        the call must see what the callee wrote: 7 + 5 = 12, and 12 + 12."""
        derivation = self.derive()
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        pointer = 0x01800000
        machine.memory.write(pointer, b"\x00" * 16)
        expected = machine.call(self.caller, (pointer,))
        self.assertEqual(expected, 24)

        machine.memory.write(pointer, b"\x00" * 16)
        machine.memory.begin_journal()
        result = derivation.run((pointer,) + (0,) * 15, Execution(machine))
        self.assertEqual(result, 24)
        self.assertEqual(machine.memory.read_int(pointer + 4, 4, False), 12)
        machine.memory.rollback()
        machine.memory.end_journal()

    def test_emitted_statements_call_in_order(self):
        import derive
        from ps2decomp.calls import Parameters

        # The callee sets $v0 on its one path, so the call hands over none.
        self.parameters = Parameters(self.parameters)
        self.parameters.sets_v0 = frozenset({self.callee})
        derive._SIGNATURES.clear()
        derive._EXTERNS.clear()
        lines = derive.render_body(self.derive())
        text = "\n".join(lines)
        self.assertLess(text.index("Write32"), text.index("func_%08X(" % self.callee))
        self.assertIn("func_%08X(arg0, 5)" % self.callee, text)
        self.assertEqual(derive._EXTERNS, {self.callee: (0, 1)})
        # The result is used, so it is named; the load after the call reads live
        # memory there and needs no capture.
        self.assertIn("const u64 r1 = ", text)
        self.assertTrue(lines[-1].startswith("return r1 + "))


    def test_value_loaded_before_a_call_is_captured_before_it(self):
        """x = *p; r = callee(p, 5); return r + x  - the callee may change *p, so x
        must be read into a local before the call, not after."""
        import derive

        words = CALLEE + [
            addiu(SP, SP, -32),
            sd(RA, 16, SP),
            sd(S0, 0, SP),
            lw(S0, 0, A0),
            jal(BASE),
            addiu(A1, 0, 5),
            addu(V0, V0, S0),
            ld(RA, 16, SP),
            ld(S0, 0, SP),
            JR_RA,
            addiu(SP, SP, 32),
        ]
        image = struct.pack("<%dI" % len(words), *words)
        start = BASE + 4 * len(CALLEE)
        derivation = derive_paths(image, BASE, start, BASE + 4 * len(words), known={},
                                  parameters=self.parameters)
        derive._SIGNATURES.clear()
        derive._EXTERNS.clear()
        # The frame the call is made from is entered first (callees run at its $sp).
        lines = [line for line in derive.render_body(derivation)
                 if not line.startswith("const u64 sp = StackEnter")]
        call_line = next(i for i, line in enumerate(lines) if "func_" in line)
        self.assertTrue(lines[0].startswith("const u64 t1 = ReadS32(arg0)"), lines)
        self.assertLess(0, call_line)
        self.assertIn("t1", lines[-1])



class TestSharedCode(unittest.TestCase):
    """A jump into the middle of another function continues there; it is not a call.

    func_A is `j shared; nop`, where `shared` is the tail of func_B: `lw $v0,
    4($a0); jr $ra; nop`. func_A returns the field at 4, exactly as func_B's tail
    does.
    """

    def setUp(self):
        self.words = [
            lw(V0, 0, A0),                       # func_B starts here
            (2 << 26) | ((BASE + 8) >> 2),       # j shared (stands in for B's body)
            lw(V0, 4, A0),                       # shared: B's tail
            JR_RA,
            NOP,
            (2 << 26) | ((BASE + 8) >> 2),       # func_A: j shared
            NOP,
        ]
        self.image = struct.pack("<%dI" % len(self.words), *self.words)
        self.func_a = BASE + 20

    def test_jump_into_shared_code_is_followed(self):
        derivation = derive_paths(self.image, BASE, self.func_a, self.func_a + 8,
                                  known={}, parameters={BASE: (0,), self.func_a: (0,)})
        self.assertFalse(derivation.has_calls)
        self.assertEqual(derivation.paths[0].expression.render(), "*(s32*)(arg0 + 4)")

    def test_jump_to_a_function_start_is_still_a_tail_call(self):
        derivation = derive_paths(self.image, BASE, self.func_a, self.func_a + 8,
                                  known={}, parameters={BASE: (0,), BASE + 8: (0,),
                                                        self.func_a: (0,)})
        self.assertTrue(derivation.has_calls)

    def test_without_function_starts_nothing_is_shared(self):
        from ps2decomp.symexec import Unsupported

        with self.assertRaises(Unsupported):
            derive_paths(self.image, BASE, self.func_a, self.func_a + 8, known={})



class TestBarriers(unittest.TestCase):
    """A load that may or may not overlap an earlier store reads memory after it.

    f(p, q): *p = 7; return *q - with no way to tell from the addresses whether p
    and q are the same. It used to be refused; now a barrier applies the store
    first, so the answer is right whether they alias or not.
    """

    WORDS = [addiu(V1, 0, 7), sw(V1, 0, A0), JR_RA, lw(V0, 0, A1)]

    def derive(self):
        image = struct.pack("<%dI" % len(self.WORDS), *self.WORDS)
        return image, derive_paths(image, BASE, BASE, BASE + 4 * len(self.WORDS), known={})

    def test_unprovable_overlap_becomes_a_barrier(self):
        _image, derivation = self.derive()
        self.assertTrue(derivation.sequential)
        self.assertEqual([s[0] for s in derivation.tree()], ["store", "barrier", "return"])

    def test_barrier_is_exact_whether_or_not_the_pointers_alias(self):
        image, derivation = self.derive()
        machine = Machine(BASE, image, 0x02000000 - BASE)
        for p, q in ((0x01800000, 0x01800000), (0x01800000, 0x01800010)):
            machine.memory.write(0x01800000, bytes(range(32)))
            expected = machine.call(BASE, (p, q))
            machine.memory.write(0x01800000, bytes(range(32)))
            machine.memory.begin_journal()
            actual = derivation.run((p, q) + (0,) * 14, Execution(machine))
            machine.memory.rollback()
            machine.memory.end_journal()
            self.assertEqual(actual, expected, "p=%#x q=%#x" % (p, q))

    def test_barrier_emits_the_store_before_the_load(self):
        import derive

        derive._SIGNATURES.clear()
        derive._EXTERNS.clear()
        lines = derive.render_body(self.derive()[1])
        self.assertEqual(lines, ["Write32(arg0, 7);", "return ReadS32(arg1);"])



class TestOutParameters(unittest.TestCase):
    """f(): int local; fill(&local); return local + 1  - the callee writes the frame.

    fill(p) stores 41 at *p. The caller's frame has to be real memory for that to
    be seen, and the callee has to run clear of it.
    """

    def setUp(self):
        fill = [addiu(V1, 0, 41), JR_RA, sw(V1, 0, A0)]
        caller = [
            addiu(SP, SP, -32),
            sd(RA, 16, SP),
            jal(BASE),
            addiu(A0, SP, 0),                   # &local, in the delay slot
            lw(V0, 0, SP),
            addiu(V0, V0, 1),
            ld(RA, 16, SP),
            JR_RA,
            addiu(SP, SP, 32),
        ]
        self.words = fill + caller
        self.image = struct.pack("<%dI" % len(self.words), *self.words)
        self.caller = BASE + 4 * len(fill)
        self.parameters = {BASE: (0,), self.caller: ()}

    def derive(self):
        return derive_paths(self.image, BASE, self.caller, BASE + 4 * len(self.words),
                            known={}, parameters=self.parameters)

    def test_out_parameter_is_read_back_after_the_call(self):
        derivation = self.derive()
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        self.assertEqual(machine.call(self.caller, ()), 42)
        machine.memory.begin_journal()
        result = derivation.run((0,) * 16, Execution(machine))
        machine.memory.rollback()
        machine.memory.end_journal()
        self.assertEqual(result, 42)

    def test_emitted_code_allocates_a_guest_frame(self):
        import derive

        derive._SIGNATURES.clear()
        derive._EXTERNS.clear()
        lines = derive.render_body(self.derive())
        self.assertTrue(lines[0].startswith("const u64 sp = StackEnter("), lines)
        self.assertTrue(any(line.startswith("func_%08X(" % BASE) and "sp" in line
                            for line in lines), lines)
        self.assertTrue(lines[-2].startswith("StackLeave("), lines)
        # The $ra spill is bookkeeping and must not be written to the guest stack.
        self.assertFalse(any("0xDEAD0000" in line for line in lines), lines)



class TestAddressOfALocalInMemory(unittest.TestCase):
    """f(): int local; G = &local; fill(); return local + 1  - fill writes 41
    through G. The address reaches the callee through memory, not a register,
    and the frame has to be real memory all the same."""

    G = 0x01800000

    def setUp(self):
        lui = lambda rt, imm: i_type(0x0F, 0, rt, imm)
        T0 = 8
        fill = [lui(V1, self.G >> 16), lw(V1, 0, V1), addiu(T0, 0, 41), JR_RA, sw(T0, 0, V1)]
        caller = [
            addiu(SP, SP, -32),
            sd(RA, 16, SP),
            lui(V1, self.G >> 16),
            addiu(A0, SP, 0),
            sw(0, 0, SP),                       # local = 0
            sw(A0, 0, V1),                      # G = &local
            jal(BASE),
            NOP,
            lw(V0, 0, SP),
            addiu(V0, V0, 1),
            ld(RA, 16, SP),
            JR_RA,
            addiu(SP, SP, 32),
        ]
        self.words = fill + caller
        self.image = struct.pack("<%dI" % len(self.words), *self.words)
        self.caller = BASE + 4 * len(fill)
        self.parameters = {BASE: (), self.caller: ()}

    def test_the_local_is_read_back_after_the_call(self):
        derivation = derive_paths(self.image, BASE, self.caller, BASE + 4 * len(self.words),
                                  known={}, parameters=self.parameters)
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        self.assertEqual(machine.call(self.caller, ()), 42)
        machine.memory.begin_journal()
        result = derivation.run((0,) * 16, Execution(machine))
        machine.memory.rollback()
        machine.memory.end_journal()
        self.assertEqual(result, 42)


class TestRegisterKeptAcrossACall(unittest.TestCase):
    """A caller-saved register the callee never touches survives the call.

        g(p): return *p;     f(p): t1 = p; v = g(p); return v + t1;

    CodeWarrior relies on it (Resident Evil Outbreak's 0x00190100 and $t1);
    dropping every caller-saved register at a call refused f.
    """

    T1 = 9

    def setUp(self):
        self.g = BASE
        g = [lw(V0, 0, A0), JR_RA, NOP]
        self.f = BASE + 4 * len(g)
        f = [addiu(SP, SP, -32), sd(RA, 16, SP), move(self.T1, A0), jal(self.g), NOP,
             addu(V0, V0, self.T1), ld(RA, 16, SP), JR_RA, addiu(SP, SP, 32)]
        self.end = self.f + 4 * len(f)
        self.image = struct.pack("<%dI" % (len(g) + len(f)), *(g + f))
        self.database = {"functions": [
            {"addr": self.g, "end": self.f, "size": self.f - self.g},
            {"addr": self.f, "end": self.end, "size": self.end - self.f}]}

    def test_the_callee_preserves_what_it_never_writes(self):
        from ps2decomp.calls import preserved_registers

        kept = preserved_registers(self.database, self.image, BASE)
        self.assertIn(self.T1, kept[self.g])
        self.assertNotIn(self.T1, kept.get(self.f, ()))     # f writes $t1

    def test_a_caller_reading_it_after_the_call_derives_and_agrees(self):
        from ps2decomp.calls import parameters

        derivation = derive_paths(self.image, BASE, self.f, self.end, known={},
                                  parameters=parameters(self.database, self.image, BASE))
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        for value in (0, 5, -3):
            machine.memory.write_int(0x01800000, value & 0xFFFFFFFF, 4)
            expected = machine.call(self.f, (0x01800000,))
            machine.memory.begin_journal()
            self.assertEqual(derivation.run((0x01800000,) + (0,) * 15, Execution(machine)),
                             expected)
            machine.memory.rollback()
            machine.memory.end_journal()

    def test_without_the_fact_it_is_refused(self):
        from ps2decomp.calls import Parameters
        from ps2decomp.symexec import Unsupported

        with self.assertRaises(Unsupported):
            derive_paths(self.image, BASE, self.f, self.end, known={},
                         parameters=Parameters({self.g: (0,), self.f: (0,)}))

    def test_a_callee_that_calls_through_a_register_preserves_nothing(self):
        from ps2decomp.calls import preserved_registers

        jalr_a1 = r_type(5, 0, RA, 0x09)
        g = [jalr_a1, NOP, JR_RA, NOP]
        image = struct.pack("<4I", *g)
        kept = preserved_registers({"functions": [{"addr": BASE, "end": BASE + 16, "size": 16}]},
                                   image, BASE)
        self.assertNotIn(BASE, kept)

    def test_a_callee_writing_it_through_its_own_callee_does_not_preserve_it(self):
        from ps2decomp.calls import preserved_registers

        h = [move(self.T1, A0), JR_RA, NOP]                  # h writes $t1
        g = [addiu(SP, SP, -16), sd(RA, 0, SP), jal(BASE), NOP, ld(RA, 0, SP), JR_RA,
             addiu(SP, SP, 16)]
        image = struct.pack("<%dI" % (len(h) + len(g)), *(h + g))
        g_at, end = BASE + 12, BASE + 12 + 4 * len(g)
        kept = preserved_registers({"functions": [
            {"addr": BASE, "end": g_at, "size": 12},
            {"addr": g_at, "end": end, "size": end - g_at}]}, image, BASE)
        self.assertNotIn(self.T1, kept[g_at])
        self.assertIn(10, kept[g_at])                         # $t2: nobody writes it


class TestIncomingV0(unittest.TestCase):
    """f(p): if (p == 0) return; return 5;  - $v0 left as the caller had it.

    A state handler in a table does this (Resident Evil Outbreak's game.bin:
    395 refused as "inconsistent void and value-returning paths"). Its result
    is the caller's own $v0 on one path, which a caller through the table may
    read, so it is derived as returning a value: V0_IN there.
    """

    def setUp(self):
        def beq(rs, rt, here, target):
            return i_type(0x04, rs, rt, (target - (here + 4)) // 4)

        self.words = [beq(A0, 0, BASE, BASE + 12), NOP, addiu(V0, 0, 5), JR_RA, NOP]
        self.image = struct.pack("<%dI" % len(self.words), *self.words)

    def test_it_returns_the_incoming_v0_on_the_path_leaving_it(self):
        import derive
        from ps2decomp.symexec import V0_IN

        derivation = derive_paths(self.image, BASE, BASE, BASE + 4 * len(self.words),
                                  known={})
        self.assertFalse(derivation.is_void)
        self.assertIn(V0_IN, derive.derivation_args(derivation))
        func = {"addr": BASE, "end": BASE + 20, "size": 20}
        self.assertIn("u64 v0_in", derive._signature(func, derivation, None)[1])
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        for a0, v0 in ((0, 77), (0, 0x01800010), (1, 77)):
            args = (a0,) + (0,) * 15 + (v0,)
            expected = machine.call(BASE, args)
            self.assertEqual(expected, v0 if a0 == 0 else 5)
            machine.memory.begin_journal()
            self.assertEqual(derivation.run(args, Execution(machine)), expected)
            machine.memory.rollback()
            machine.memory.end_journal()

    def test_a_function_that_never_sets_v0_is_still_void(self):
        image = struct.pack("<2I", JR_RA, NOP)
        self.assertTrue(derive_paths(image, BASE, BASE, BASE + 8, known={}).is_void)


class TestCalleeLeavingV0(unittest.TestCase):
    """g(p): if (p != 0) $v0 = 3;   f(p): $v0 = 7; return g(p);

    g returns f's 7 when p is 0. The model ran g with $v0 0 and disagreed;
    every call now hands the callee the caller's $v0, in the model, in the
    compiled check, and in the emitted call where its result is used.
    """

    def setUp(self):
        def beq(rs, rt, here, target):
            return i_type(0x04, rs, rt, (target - (here + 4)) // 4)

        g = [beq(A0, 0, BASE, BASE + 12), NOP, addiu(V0, 0, 3), JR_RA, NOP]
        self.f = BASE + 4 * len(g)
        f = [addiu(SP, SP, -32), sd(RA, 16, SP), addiu(V0, 0, 7), jal(BASE), NOP,
             ld(RA, 16, SP), JR_RA, addiu(SP, SP, 32)]
        self.end = self.f + 4 * len(f)
        self.image = struct.pack("<%dI" % (len(g) + len(f)), *(g + f))
        self.database = {"functions": [
            {"addr": BASE, "end": self.f, "size": self.f - BASE},
            {"addr": self.f, "end": self.end, "size": self.end - self.f}]}

    def derive(self):
        from ps2decomp.calls import parameters

        return derive_paths(self.image, BASE, self.f, self.end, known={},
                            parameters=parameters(self.database, self.image, BASE))

    def test_the_model_hands_the_callee_the_callers_v0(self):
        derivation = self.derive()
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        for a0 in (0, 1):
            expected = machine.call(self.f, (a0,))
            self.assertEqual(expected, 7 if a0 == 0 else 3)
            machine.memory.begin_journal()
            self.assertEqual(derivation.run((a0,) + (0,) * 15, Execution(machine)), expected)
            machine.memory.rollback()
            machine.memory.end_journal()

    def test_the_emitted_call_passes_it(self):
        import derive

        derive._SIGNATURES.clear()
        derive._EXTERNS.clear()
        derive._V0_EXTERNS.clear()
        text = chr(10).join(derive.render_body(self.derive()))
        self.assertIn("func_%08X(arg0, 7)" % BASE, text)
        self.assertIn(BASE, derive._V0_EXTERNS)


class TestV0Setting(unittest.TestCase):
    """Which functions write $v0 on every path, so that a call to one need not
    hand over the caller's (calls.v0_setting)."""

    def setting(self, words, functions=None):
        from ps2decomp.calls import v0_setting

        image = struct.pack("<%dI" % len(words), *words)
        end = BASE + 4 * len(words)
        functions = functions or [(BASE, end)]
        return v0_setting({"functions": [{"addr": a, "end": b, "size": b - a}
                                         for a, b in functions]}, image, BASE)

    def beq(self, rs, rt, here, target):
        return i_type(0x04, rs, rt, (target - (here + 4)) // 4)

    def test_a_write_on_every_path(self):
        self.assertIn(BASE, self.setting(CALLEE))

    def test_a_path_without_one(self):
        words = [self.beq(A0, 0, BASE, BASE + 12), NOP, addiu(V0, 0, 3), JR_RA, NOP]
        self.assertNotIn(BASE, self.setting(words))

    def test_the_return_delay_slot_counts(self):
        self.assertIn(BASE, self.setting([JR_RA, addiu(V0, 0, 1)]))

    def test_a_branch_likely_skips_its_slot_when_not_taken(self):
        beql = lambda rs, rt, here, target: i_type(0x14, rs, rt, (target - (here + 4)) // 4)
        # beql $a0, $zero, ret; addiu $v0, 1 (taken only); ...; ret: jr $ra; nop
        words = [beql(A0, 0, BASE, BASE + 12), addiu(V0, 0, 1), NOP, JR_RA, NOP]
        self.assertNotIn(BASE, self.setting(words))

    def test_a_call_to_a_setter_sets_it(self):
        g = [JR_RA, addiu(V0, 0, 1)]
        f_at = BASE + 8
        f = [addiu(SP, SP, -16), sd(RA, 0, SP), jal(BASE), NOP, ld(RA, 0, SP), JR_RA,
             addiu(SP, SP, 16)]
        end = f_at + 4 * len(f)
        found = self.setting(g + f, [(BASE, f_at), (f_at, end)])
        self.assertEqual(found, {BASE, f_at})

    def test_a_call_to_anything_else_does_not(self):
        g = [JR_RA, NOP]
        f_at = BASE + 8
        f = [addiu(SP, SP, -16), sd(RA, 0, SP), jal(BASE), NOP, ld(RA, 0, SP), JR_RA,
             addiu(SP, SP, 16)]
        end = f_at + 4 * len(f)
        self.assertEqual(self.setting(g + f, [(BASE, f_at), (f_at, end)]), frozenset())

    def test_a_conditional_move_is_not_a_write(self):
        movz = r_type(A0, A1, V0, 0x0A)
        self.assertNotIn(BASE, self.setting([movz, JR_RA, NOP]))


class TestCalleeStackPointer(unittest.TestCase):
    """A callee runs at its caller's $sp: g() { *G = $sp; }, f() { frame; g(); }.

    The model ran callees at a fixed depth, and one that stored a stack address
    (an RPC's request block) disagreed with the original (Resident Evil
    Outbreak's 0x00119B28 and some 20 callers).
    """

    G = 0x01800000

    def test_the_callee_sees_the_original_stack_pointer(self):
        from ps2decomp.calls import parameters

        lui_v0 = i_type(0x0F, 0, V0, self.G >> 16)
        g = [lui_v0, JR_RA, sw(SP, 0, V0)]
        f_at = BASE + 4 * len(g)
        f = [addiu(SP, SP, -48), sd(RA, 32, SP), jal(BASE), NOP, ld(RA, 32, SP), JR_RA,
             addiu(SP, SP, 48)]
        end = f_at + 4 * len(f)
        image = struct.pack("<%dI" % (len(g) + len(f)), *(g + f))
        database = {"functions": [{"addr": BASE, "end": f_at, "size": f_at - BASE},
                                  {"addr": f_at, "end": end, "size": end - f_at}]}
        derivation = derive_paths(image, BASE, f_at, end, known={},
                                  parameters=parameters(database, image, BASE))
        machine = Machine(BASE, image, 0x02000000 - BASE)
        machine.call(f_at, ())
        expected = machine.memory.read_int(self.G, 4, False)
        machine.memory.write_int(self.G, 0, 4)
        machine.memory.begin_journal()
        derivation.run((0,) * 16, Execution(machine))
        self.assertEqual(machine.memory.read_int(self.G, 4, False), expected)
        machine.memory.rollback()
        machine.memory.end_journal()


class TestStackArguments(unittest.TestCase):
    """h() returns 0($sp), its ninth argument; f() { 0($sp) = 7; return h(); }.

    The caller's frame was symbolic, so h read nothing there (Resident Evil
    Outbreak's 0x00135078 and 0x00127FC8).
    """

    def setUp(self):
        h = [lw(V0, 0, SP), JR_RA, NOP]
        self.f = BASE + 4 * len(h)
        f = [addiu(SP, SP, -32), sd(RA, 16, SP), addiu(V1, 0, 7), sw(V1, 0, SP),
             jal(BASE), NOP, ld(RA, 16, SP), JR_RA, addiu(SP, SP, 32)]
        self.end = self.f + 4 * len(f)
        self.image = struct.pack("<%dI" % (len(h) + len(f)), *(h + f))
        self.database = {"functions": [
            {"addr": BASE, "end": self.f, "size": self.f - BASE},
            {"addr": self.f, "end": self.end, "size": self.end - self.f}]}

    def test_a_callee_reading_above_its_frame_is_found(self):
        from ps2decomp.calls import stack_readers

        readers = stack_readers(self.database, self.image, BASE)
        self.assertIn(BASE, readers)
        self.assertNotIn(self.f, readers)

    def test_the_caller_agrees_with_the_original(self):
        from ps2decomp.calls import parameters

        derivation = derive_paths(self.image, BASE, self.f, self.end, known={},
                                  parameters=parameters(self.database, self.image, BASE))
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        self.assertEqual(machine.call(self.f, ()), 7)
        machine.memory.begin_journal()
        self.assertEqual(derivation.run((0,) * 16, Execution(machine)), 7)
        machine.memory.rollback()
        machine.memory.end_journal()


class TestKernelStubs(unittest.TestCase):
    def test_a_syscall_stub_reads_the_integer_arguments(self):
        from ps2decomp.calls import live_parameters

        # ReferThreadStatus(tid, &status): addiu $v1, $zero, 0x30; syscall; jr $ra
        words = [addiu(V1, 0, 0x30), 0x0000000C, JR_RA, NOP]
        image = struct.pack("<%dI" % len(words), *words)
        database = {"functions": [{"addr": BASE, "end": BASE + 16, "size": 16}]}
        self.assertEqual(live_parameters(database, image, BASE)[BASE], tuple(range(8)))

    def test_a_conditional_move_reads_its_destination(self):
        from ps2decomp.calls import live_parameters

        T0 = 8
        # movz $t0, $zero, $a0; jr $ra; daddu $v0, $t0, $zero: $t0 survives a0 != 0.
        words = [r_type(0, A0, T0, 0x0A), JR_RA, r_type(T0, 0, V0, 0x2D)]
        image = struct.pack("<%dI" % len(words), *words)
        database = {"functions": [{"addr": BASE, "end": BASE + 12, "size": 12}]}
        self.assertEqual(live_parameters(database, image, BASE)[BASE], (0, 4))


class TestPointerCalls(unittest.TestCase):
    """f(obj): return obj->method(obj)  - a call through a function pointer.

    method(p) returns p + 1. The pointer is read from obj, and obj itself is passed
    on unchanged in $a0, the shape of a virtual call.
    """

    def setUp(self):
        method = [JR_RA, addiu(V0, A0, 1)]
        jalr_v0 = r_type(V0, 0, RA, 0x09)
        caller = [
            addiu(SP, SP, -32),
            sd(RA, 16, SP),
            lw(V0, 0, A0),
            jalr_v0,
            NOP,
            ld(RA, 16, SP),
            JR_RA,
            addiu(SP, SP, 32),
        ]
        self.words = method + caller
        self.image = struct.pack("<%dI" % len(self.words), *self.words)
        self.caller = BASE + 4 * len(method)
        self.parameters = {BASE: (0,), self.caller: (0,)}

    def derive(self):
        return derive_paths(self.image, BASE, self.caller, BASE + 4 * len(self.words),
                            known={}, parameters=self.parameters)

    def test_call_through_a_pointer_runs_what_it_points_at(self):
        derivation = self.derive()
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        obj = 0x01800000
        machine.memory.write_int(obj, BASE, 4)
        self.assertEqual(machine.call(self.caller, (obj,)), obj + 1)
        machine.memory.begin_journal()
        result = derivation.run((obj,) + (0,) * 15, Execution(machine))
        machine.memory.rollback()
        machine.memory.end_journal()
        self.assertEqual(result, obj + 1)

    def test_pointer_call_passes_the_object_and_nothing_else(self):
        import derive

        derive._SIGNATURES.clear()
        derive._EXTERNS.clear()
        lines = derive.render_body(self.derive())
        self.assertIn("const u64 r1 = CallPointer(ReadS32(arg0), arg0);", lines)



class TestStructCopies(unittest.TestCase):
    """`lq` then `sq` is the EE's 16-byte struct copy, and must not be refused.

    The model is 64 bits wide, so the copy becomes two 8-byte copies.
    """

    LQ_V0_A1 = (0x1E << 26) | (A1 << 21) | (V0 << 16)
    SQ_V0_A0 = (0x1F << 26) | (A0 << 21) | (V0 << 16)

    def test_sixteen_byte_copy_is_two_eight_byte_copies(self):
        words = [self.LQ_V0_A1, JR_RA, self.SQ_V0_A0]
        image = struct.pack("<%dI" % len(words), *words)
        derivation = derive_paths(image, BASE, BASE, BASE + 12, known={})
        stores = derivation.paths[0].stores
        self.assertEqual([store.size for store in stores], [8, 8])

        machine = Machine(BASE, image, 0x02000000 - BASE)
        source, dest = 0x01800000, 0x01800100
        machine.memory.write(source, bytes(range(16)))
        machine.memory.write(dest, b"\x00" * 16)
        read = lambda a, size, signed: machine.memory.read_int(a, size, signed)
        written = derivation.evaluate_stores((dest, source) + (0,) * 14, read)
        for address, value, size in written:
            machine.memory.write_int(address, value, size)
        self.assertEqual(machine.memory.read(dest, 16), bytes(range(16)))



class TestLoops(unittest.TestCase):
    """strlen: a loop whose trip count depends entirely on memory.

        v0 = 0; for (;;) { t0 = *p; if (t0 == 0) break; v0++; p++; } return v0;

    It cannot be unrolled, so it is derived once over variables.
    """

    T0 = 8

    def setUp(self):
        def beq(rs, rt, here, target):
            return i_type(0x04, rs, rt, (target - (here + 4)) // 4)

        self.words = [
            move(V0, 0),                                       # 0
            i_type(0x20, A0, self.T0, 0),                      # 4: lb $t0, 0($a0)
            beq(self.T0, 0, BASE + 8, BASE + 28),              # 8
            NOP,                                               # 12
            addiu(V0, V0, 1),                                  # 16
            beq(0, 0, BASE + 20, BASE + 4),                    # 20
            addiu(A0, A0, 1),                                  # 24 (delay slot)
            JR_RA,                                             # 28
            NOP,                                               # 32
        ]
        self.image = struct.pack("<%dI" % len(self.words), *self.words)

    def derive(self, loops):
        return derive_paths(self.image, BASE, BASE, BASE + 4 * len(self.words),
                            known={}, loops=loops)

    def test_without_loops_the_function_is_refused(self):
        from ps2decomp.symexec import Unsupported

        with self.assertRaises(Unsupported):
            self.derive(False)

    def test_loop_runs_to_the_right_count(self):
        derivation = self.derive(True)
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        for text in (b"", b"a", b"hello", b"x" * 40):
            machine.memory.write(0x01800000, text + bytes(1))
            expected = machine.call(BASE, (0x01800000,))
            machine.memory.begin_journal()
            actual = derivation.run((0x01800000,) + (0,) * 15, Execution(machine))
            machine.memory.rollback()
            machine.memory.end_journal()
            self.assertEqual(actual, expected, text)
            self.assertEqual(actual, len(text))

    def test_loop_is_emitted_as_a_loop(self):
        import derive

        derive._SIGNATURES.clear()
        derive._EXTERNS.clear()
        text = chr(10).join(derive.render_body(self.derive(True)))
        self.assertIn("for (;;) {", text)
        self.assertIn("break;", text)
        self.assertIn("continue;", text)


class TestLoopWithACall(unittest.TestCase):
    """A loop calling a function leaves the argument registers dead at its exit.

        f(p): while (*p != 0) g(p);    g(p): *p -= 1;

    The call makes every argument register a loop variable. Kept as variables
    because an exit still holds them, they gave f every argument as a
    parameter (Resident Evil Outbreak's 0x00102F08 took sixteen), and no caller
    passing what liveness found - only p - could be emitted.
    """

    def setUp(self):
        def beq(rs, rt, here, target):
            return i_type(0x04, rs, rt, (target - (here + 4)) // 4)

        T0 = 8
        self.callee = BASE
        callee = [lw(T0, 0, A0), addiu(T0, T0, -1), sw(T0, 0, A0), JR_RA, NOP]
        f = BASE + 4 * len(callee)
        self.f = f
        body = [
            addiu(SP, SP, -32),                     # 0
            sd(RA, 16, SP),                         # 4
            sd(S0, 0, SP),                          # 8
            move(S0, A0),                           # 12
            lw(V0, 0, S0),                          # 16: the head
            beq(V0, 0, f + 20, f + 44),             # 20
            NOP,                                    # 24
            jal(self.callee),                       # 28
            move(A0, S0),                           # 32
            beq(0, 0, f + 36, f + 16),              # 36
            NOP,                                    # 40
            ld(RA, 16, SP),                         # 44
            ld(S0, 0, SP),                          # 48
            JR_RA,                                  # 52
            addiu(SP, SP, 32),                      # 56
        ]
        self.end = f + 4 * len(body)
        self.words = callee + body
        self.image = struct.pack("<%dI" % len(self.words), *self.words)

    def derive(self):
        from ps2decomp.calls import Parameters

        parameters = Parameters({self.callee: (0,), self.f: (0,)})
        return derive_paths(self.image, BASE, self.f, self.end, known={},
                            parameters=parameters, loops=True)

    def test_the_function_takes_only_the_argument_it_reads(self):
        import derive

        self.assertEqual(derive.derivation_args(self.derive()), {0})

    def test_it_still_agrees_with_the_original(self):
        derivation = self.derive()
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        for count in (0, 1, 5):
            machine.memory.write_int(0x01800000, count, 4)
            machine.memory.begin_journal()
            derivation.run((0x01800000,) + (0,) * 15, Execution(machine))
            self.assertEqual(machine.memory.read_int(0x01800000, 4, False), 0)
            machine.memory.rollback()
            machine.memory.end_journal()



class TestValueReadInALoopUsedAfterIt(unittest.TestCase):
    """f(p): for (;;) { x = *p; *p = x + 1; if (x < 0) return x; p++; }

    The value returned was read before the store over it. Emitted as a read
    after the loop, it returned x + 1 (Resident Evil Outbreak's 0x00107ED0).
    """

    T0, T1 = 8, 9

    def setUp(self):
        def branch(op, rs, rt, here, target):
            return i_type(op, rs, rt, (target - (here + 4)) // 4)

        self.words = [
            lw(self.T0, 0, A0),                                # 0
            addiu(self.T1, self.T0, 1),                        # 4
            sw(self.T1, 0, A0),                                # 8
            branch(0x01, self.T0, 0, BASE + 12, BASE + 28),    # 12: bltz $t0
            NOP,                                               # 16
            branch(0x04, 0, 0, BASE + 20, BASE + 0),           # 20
            addiu(A0, A0, 4),                                  # 24 (delay slot)
            JR_RA,                                             # 28
            move(V0, self.T0),                                 # 32
        ]
        self.image = struct.pack("<%dI" % len(self.words), *self.words)

    def test_the_exit_keeps_the_value_it_read(self):
        import derive

        derivation = derive_paths(self.image, BASE, BASE, BASE + 4 * len(self.words),
                                  known={}, loops=True)
        derive._SIGNATURES.clear()
        derive._EXTERNS.clear()
        text = chr(10).join(derive.render_body(derivation))
        returned = [line for line in text.splitlines() if "return" in line]
        self.assertTrue(returned and all("Read" not in line for line in returned), text)
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        for values in ((-5,), (3, 0, -2), (1, 1, 1, -1)):
            for index, value in enumerate(values):
                machine.memory.write_int(0x01800000 + 4 * index, value & 0xFFFFFFFF, 4)
            machine.memory.begin_journal()
            expected = machine.call(BASE, (0x01800000,))
            machine.memory.rollback()
            actual = derivation.run((0x01800000,) + (0,) * 15, Execution(machine))
            machine.memory.rollback()
            machine.memory.end_journal()
            self.assertEqual(actual, expected)
            self.assertEqual(actual, values[-1])


class TestLoopOverALocal(unittest.TestCase):
    """f(n): int total = 0; while (n--) total += 2; return total  - with total
    kept in the stack frame, as code compiled without optimisation keeps it."""

    def setUp(self):
        def beq(rs, rt, here, target):
            return i_type(0x04, rs, rt, (target - (here + 4)) // 4)

        self.words = [
            addiu(SP, SP, -16),                     # 0
            sw(0, 0, SP),                           # 4: total = 0
            beq(A0, 0, BASE + 8, BASE + 36),        # 8: the head
            NOP,                                    # 12
            lw(V0, 0, SP),                          # 16
            addiu(V0, V0, 2),                       # 20
            sw(V0, 0, SP),                          # 24
            beq(0, 0, BASE + 28, BASE + 8),         # 28
            addiu(A0, A0, -1),                      # 32
            lw(V0, 0, SP),                          # 36
            JR_RA,                                  # 40
            addiu(SP, SP, 16),                      # 44
        ]
        self.image = struct.pack("<%dI" % len(self.words), *self.words)

    def test_the_frame_becomes_memory_and_the_loop_agrees(self):
        derivation = derive_paths(self.image, BASE, BASE, BASE + 4 * len(self.words),
                                  known={}, loops=True)
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        for n in (0, 1, 7):
            expected = machine.call(BASE, (n,))
            machine.memory.begin_journal()
            actual = derivation.run((n,) + (0,) * 15, Execution(machine))
            machine.memory.rollback()
            machine.memory.end_journal()
            self.assertEqual((actual, expected), (2 * n, 2 * n))


class TestBranchMerging(unittest.TestCase):
    """Two independent if/else assignments merge into one path.

        x = (a0 == 0) ? 2 : 1;  y = (a1 == 0) ? 20 : 10;  return x + y;

    Enumerated, each such branch doubles the paths; merged, the walk carries one.
    """

    def setUp(self):
        def beq(rs, rt, here, target):
            return i_type(0x04, rs, rt, (target - (here + 4)) // 4)

        b = lambda here, target: beq(0, 0, here, target)
        self.words = [
            beq(A0, 0, BASE + 0, BASE + 20),        # 0
            NOP,                                    # 4
            addiu(V0, 0, 1),                        # 8
            b(BASE + 12, BASE + 24),                # 12
            NOP,                                    # 16
            addiu(V0, 0, 2),                        # 20
            beq(A1, 0, BASE + 24, BASE + 44),       # 24
            NOP,                                    # 28
            addiu(V1, 0, 10),                       # 32
            b(BASE + 36, BASE + 48),                # 36
            NOP,                                    # 40
            addiu(V1, 0, 20),                       # 44
            JR_RA,                                  # 48
            addu(V0, V0, V1),                       # 52
        ]
        self.image = struct.pack("<%dI" % len(self.words), *self.words)

    def derive(self, merge):
        return derive_paths(self.image, BASE, BASE, BASE + 4 * len(self.words),
                            known={}, merge=merge)

    def test_join_points_are_found(self):
        from ps2decomp.paths import _join_points

        joins = _join_points(self.image, BASE, BASE, BASE + 4 * len(self.words))
        self.assertEqual(joins.get(BASE), BASE + 24)
        self.assertEqual(joins.get(BASE + 24), BASE + 48)

    def test_merging_gives_one_path_with_the_same_values(self):
        self.assertEqual(len(self.derive(False).paths), 4)
        merged = self.derive(True)
        self.assertEqual(len(merged.paths), 1)
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        read = lambda a, size, signed: machine.memory.read_int(a, size, signed)
        for a0, a1 in ((0, 0), (0, 5), (3, 0), (3, 5)):
            self.assertEqual(merged.evaluate((a0, a1) + (0,) * 14, read),
                             machine.call(BASE, (a0, a1)))


class TestFloatResultWithoutEvidence(unittest.TestCase):
    """f(p, n): callee(p, n); return *(float *)(p + 8) - in $f0, while $v0 is
    only what the callee left. No caller says which is the result."""

    def setUp(self):
        callee = BASE
        self.f = BASE + 4 * len(CALLEE)
        lwc1 = i_type(0x31, S0, 0, 8)                     # lwc1 $f0, 8($s0)
        self.words = CALLEE + [
            addiu(SP, SP, -16),
            sd(RA, 0, SP),
            sd(S0, 8, SP),
            jal(callee),
            move(S0, A0),
            lwc1,
            ld(RA, 0, SP),
            ld(S0, 8, SP),
            JR_RA,
            addiu(SP, SP, 16),
        ]
        self.image = struct.pack("<%dI" % len(self.words), *self.words)
        self.callee = callee

    def derive(self, returns_int):
        from ps2decomp.calls import Parameters

        parameters = Parameters({self.callee: (0, 1), self.f: (0, 1)})
        parameters.returns_int = frozenset({self.f} if returns_int else ())
        return derive_paths(self.image, BASE, self.f, BASE + 4 * len(self.words), known={},
                            parameters=parameters)

    def test_without_evidence_it_returns_the_float(self):
        self.assertTrue(self.derive(returns_int=False).returns_float)

    def test_a_caller_reading_v0_keeps_it_refused(self):
        from ps2decomp.symexec import Unsupported

        with self.assertRaises(Unsupported):
            self.derive(returns_int=True)


class TestTrapPath(unittest.TestCase):
    """f(n): if (n == 0) break; return n + 1 - an assertion: the path to the
    `break` ends there, and the function still derives."""

    def setUp(self):
        bne = i_type(0x05, A0, 0, 2)                 # bne $a0, $zero, +2
        self.words = [bne, NOP, 0x0000000D, JR_RA, addiu(V0, A0, 1)]
        self.image = struct.pack("<%dI" % len(self.words), *self.words)

    def derive(self):
        return derive_paths(self.image, BASE, BASE, BASE + 4 * len(self.words), known={},
                            parameters={BASE: (0,)})

    def test_the_trap_ends_its_path(self):
        derivation = self.derive()
        leaves = [path.leaf for path in derivation.paths]
        self.assertIn(("trap", BASE + 8), leaves)
        read = lambda a, size, signed: 0
        self.assertEqual(derivation.evaluate((5,) + (0,) * 15, read), 6)

    def test_reaching_the_trap_is_not_a_result(self):
        from ps2decomp.symexec import Unsupported

        read = lambda a, size, signed: 0
        with self.assertRaises(Unsupported):
            self.derive().evaluate((0,) * 16, read)


class TestDivisionGuard(unittest.TestCase):
    """f(a, b): a / b, with the compiler's guard - beql b, zero, next; break -
    the trap in a branch-likely's delay slot, run only when b is zero."""

    def setUp(self):
        beql = i_type(0x14, A1, 0, 1)                # beql $a1, $zero, +1 (the next pair)
        div = r_type(A0, A1, 0, 0x1A)                # div $a0, $a1
        mflo = r_type(0, 0, V0, 0x12)                # mflo $v0
        self.words = [beql, 0x0000000D, div, mflo, JR_RA, NOP]
        self.image = struct.pack("<%dI" % len(self.words), *self.words)

    def test_the_guard_traps_only_on_zero(self):
        from ps2decomp.symexec import Unsupported

        derivation = derive_paths(self.image, BASE, BASE, BASE + 4 * len(self.words), known={},
                                  parameters={BASE: (0, 1)})
        self.assertIn(("trap", BASE + 4), [path.leaf for path in derivation.paths])
        read = lambda a, size, signed: 0
        self.assertEqual(derivation.evaluate((7, 2) + (0,) * 14, read), 3)
        with self.assertRaises(Unsupported):
            derivation.evaluate((7, 0) + (0,) * 14, read)


if __name__ == "__main__":
    unittest.main()

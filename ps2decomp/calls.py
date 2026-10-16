"""Which arguments each function reads, for calling it without deriving it.

A derived function that calls one that has not been derived passes it the values
in the argument registers. Only the ones the callee actually reads are part of
the call: a register that merely still holds something is not an argument, and
passing it would make every call site's signature different.

Integer arguments come from structs.argument_use. Float arguments ($f12..$f19)
get the same treatment here: a function reads one if an instruction reads the
register before anything writes it, or passes it on to a callee that reads it.
"""

from __future__ import annotations

import collections
import re
import struct

from .r5900 import DISASM
from .structs import argument_use
from .symexec import FLOAT_ARG_BASE, FLOAT_ARG_REGS

_FLOAT_REGISTER = re.compile(r"\$f(\d+)")


def _float_operands(ins):
    """(read, written) float registers of one instruction."""
    m = ins.mnemonic
    names = [int(n) for op in ins.operands for n in _FLOAT_REGISTER.findall(str(op))]
    if not names:
        return set(), set()
    if m in ("swc1", "mfc1") or m.startswith("c."):
        return set(names), set()
    if m in ("lwc1", "mtc1", "ctc1"):
        return set(), set(names)
    if m.startswith(("adda.", "suba.", "mula.", "madda.", "msuba.")):
        return set(names), set()          # these write the accumulator, not an FPR
    # Arithmetic, moves and conversions: the first operand is the destination.
    return set(names[1:]), {names[0]}


def float_argument_use(database, image, base) -> set:
    """(function, vector index) for every float argument a function reads."""
    functions = {f["addr"]: f for f in database["functions"] if f["size"]}
    used = set()
    passes = collections.defaultdict(set)
    for func in functions.values():
        start, end = func["addr"], func["end"]
        original = {reg: FLOAT_ARG_BASE + k for k, reg in enumerate(FLOAT_ARG_REGS)}
        address = start
        while address < end:
            ins = DISASM.decode(struct.unpack_from("<I", image, address - base)[0], address)
            sequence = [ins]
            if ins.has_delay_slot and address + 4 < end:
                sequence.insert(0, DISASM.decode(
                    struct.unpack_from("<I", image, address + 4 - base)[0], address + 4))
            for step in sequence:
                if step.is_call or (step.is_jump and step.mnemonic == "j"):
                    target = step.target
                    if target in functions and not (start <= target < end):
                        for reg, index in original.items():
                            passes[(target, index)].add((start, index))
                    if step.is_call:
                        original = {}          # $f0..$f19 are caller-saved
                    continue
                read, written = _float_operands(step)
                for reg in read:
                    if reg in original:
                        used.add((start, original[reg]))
                for reg in written:
                    original.pop(reg, None)
            address += 8 if len(sequence) == 2 else 4
    work = list(used)
    while work:
        slot = work.pop()
        for caller in passes.get(slot, ()):
            if caller not in used:
                used.add(caller)
                work.append(caller)
    return used


class Parameters(dict):
    """{function: argument-vector indices it reads}, plus `preserves_v0`: the
    functions after a call to which $v0 still holds the caller's value, and
    `returns_float`: the functions whose callers read their result from $f0;
    `returns_int`: those whose callers read $v0 after calling them;
    `address_taken`: those a pointer can reach; `preserved`: {function: the
    caller-saved registers it leaves untouched}."""

    preserves_v0 = frozenset()
    returns_float = frozenset()
    returns_int = frozenset()
    address_taken = frozenset()
    preserved = {}
    stack_readers = frozenset()
    sets_v0 = frozenset()


def parameters(database, image, base) -> dict:
    """{function: argument-vector indices it reads}, for every known function.

    Live-in analysis over each function's control flow, to a fixed point over the
    call graph (see live_parameters).
    """
    # An overlay's callees in the executable and the other overlays resident
    # with it (analyze.py's `external_functions`) are analysed with its own:
    # a call to one is a call to a known function.
    if database.get("external_functions"):
        database = dict(database, functions=database["functions"] + database["external_functions"])
    found = Parameters(live_parameters(database, image, base))
    found.preserves_v0 = v0_preserving(database, image, base)
    found.returns_float = float_returning(database, image, base)
    found.returns_int = int_returning(database, image, base)
    found.address_taken = address_taken(database, image, base)
    found.preserved = preserved_registers(database, image, base)
    found.stack_readers = stack_readers(database, image, base)
    found.sets_v0 = v0_setting(database, image, base)
    return found


def address_taken(database, image, base) -> frozenset:
    """Functions whose address a word of the image holds or code builds (the
    analysis's xrefs): reached through a pointer, by callers no scan of call
    sites sees - so int_returning's silence says nothing about them."""
    starts = {f["addr"] for f in database["functions"]}
    words = struct.unpack_from("<%dI" % (len(image) // 4), image, 0)
    taken = {word for word in words if word in starts}
    taken |= {x["to"] for x in database.get("xrefs", ()) if x["to"] in starts}
    return frozenset(taken)


# How far after a call a read of $f0 still counts as reading its result.
_RESULT_WINDOW = 24


def float_returning(database, image, base) -> frozenset:
    """Functions that return a float: some caller reads $f0 after calling them,
    before anything writes it.

    What a function computes into $f0 says nothing on its own - $f0 is also
    every float computation's scratch register - so the evidence is taken where
    the result is used. The scan after a call is straight-line: it stops at a
    branch, a jump, another call or the window's end, which can only miss
    evidence, never invent it. A function that ends by jumping to another hands
    that one's result to its own callers, so the evidence passes down tail calls.
    """
    functions = {f["addr"]: f for f in database["functions"] if f["size"]}

    def decode(pc):
        return DISASM.decode(struct.unpack_from("<I", image, pc - base)[0], pc)

    found, tails = set(), collections.defaultdict(set)
    for func in functions.values():
        start, end = func["addr"], func["end"]
        for pc in range(start, end, 4):
            ins = decode(pc)
            if ins.is_jump and ins.mnemonic == "j" and ins.target in functions                     and not (start <= ins.target < end):
                tails[start].add(ins.target)
            if not ins.is_call or ins.target not in functions:
                continue
            for after in range(pc + 8, min(end, pc + 8 + 4 * _RESULT_WINDOW), 4):
                step = decode(after)
                if step.is_branch or step.is_jump or step.is_call:
                    break
                read, written = _float_operands(step)
                if 0 in read:
                    found.add(ins.target)
                    break
                if 0 in written:
                    break
    work = list(found)
    while work:
        for target in tails.get(work.pop(), ()):
            if target not in found:
                found.add(target)
                work.append(target)
    return frozenset(found)


def int_returning(database, image, base) -> frozenset:
    """Functions some caller reads $v0 after calling, before anything writes
    it: evidence of a result in $v0, found as float_returning finds a result
    in $f0 (straight-line after the call; it can miss evidence, never invent
    it). Passed down tail calls the same way."""
    from .structs import _read_registers, _written_register

    functions = {f["addr"]: f for f in database["functions"] if f["size"]}

    def decode(pc):
        return DISASM.decode(struct.unpack_from("<I", image, pc - base)[0], pc)

    found, tails = set(), collections.defaultdict(set)
    for func in functions.values():
        start, end = func["addr"], func["end"]
        for pc in range(start, end, 4):
            ins = decode(pc)
            if ins.is_jump and ins.mnemonic == "j" and ins.target in functions \
                    and not (start <= ins.target < end):
                tails[start].add(ins.target)
            if not ins.is_call or ins.target not in functions:
                continue
            for after in range(pc + 8, min(end, pc + 8 + 4 * _RESULT_WINDOW), 4):
                step = decode(after)
                if 2 in _read_registers(step):
                    found.add(ins.target)
                    break
                if step.is_branch or step.is_jump or step.is_call or _written_register(step) == 2:
                    break
    work = list(found)
    while work:
        for target in tails.get(work.pop(), ()):
            if target not in found:
                found.add(target)
                work.append(target)
    return frozenset(found)


def v0_preserving(database, image, base) -> frozenset:
    """Functions that leave $v0 exactly as the caller had it.

    A function that never writes $v0, and calls only functions that never write it
    either, returns with the caller's $v0 still in place - and callers do rely on
    that, returning a value they set before the call. Treating such a call's $v0 as
    the callee's result made the model return whatever the callee happened to start
    with. A call through a register or a kernel call (which returns in $v0) rules
    a function out; the rest is a fixed point over the call graph.
    """
    from .structs import _written_register

    functions = {f["addr"]: f for f in database["functions"] if f["size"]}
    candidates, needs = set(), {}
    for address, func in functions.items():
        writes, callees = False, set()
        for pc in range(func["addr"], func["end"], 4):
            ins = DISASM.decode(struct.unpack_from("<I", image, pc - base)[0], pc)
            if _written_register(ins) == 2 or ins.mnemonic in ("syscall", "jalr"):
                writes = True
                break
            if ins.is_call or (ins.is_jump and not ins.is_return and ins.mnemonic != "jr"):
                if not (func["addr"] <= ins.target < func["end"]):
                    callees.add(ins.target)
            elif ins.mnemonic == "jr" and not ins.is_return:
                writes = True                     # a jump table: unknown targets
                break
        if not writes:
            candidates.add(address)
            needs[address] = callees
    preserving = set(candidates)
    changed = True
    while changed:
        changed = False
        for address in list(preserving):
            if not needs[address] <= preserving:
                preserving.discard(address)
                changed = True
    return frozenset(preserving)


def v0_setting(database, image, base) -> frozenset:
    """Functions that write $v0 on every path to their return.

    Any other may hand back the $v0 it was called with - a handler returning a
    value on one path and nothing on another - so a call to it passes the
    caller's $v0 along (paths.Call.v0). A path is followed from the entry
    while $v0 is unwritten: a write ends it (a conditional move does not
    write for sure), as does a call or tail call to a function in the set or a
    kernel call (which returns in $v0); a return, a call through a register, a
    jump table, a branch or jump out of the function to anything else, or
    running off its end means $v0 may come back as it was. Recursion is
    resolved as a greatest fixed point: a path that only returns through
    itself returns through one of its other paths.
    """
    from .structs import _written_register

    functions = {f["addr"]: f for f in database["functions"] if f["size"]}

    def decode(pc):
        return DISASM.decode(struct.unpack_from("<I", image, pc - base)[0], pc)

    def writes(ins):
        return _written_register(ins) == 2 and ins.mnemonic not in ("movz", "movn")

    def sets(func, setting) -> bool:
        start, end = func["addr"], func["end"]
        seen, work = set(), [start]
        while work:
            pc = work.pop()
            if pc in seen:
                continue
            seen.add(pc)
            if not (start <= pc < end) or pc + 4 > base + len(image):
                return False
            ins = decode(pc)
            if not ins.has_delay_slot:
                if ins.mnemonic == "syscall" or writes(ins):
                    continue
                if ins.is_jump or ins.is_branch:
                    return False
                work.append(pc + 4)
                continue
            if pc + 8 > end:
                return False
            slot = decode(pc + 4)
            slot_writes = writes(slot)
            if ins.is_call:
                if slot_writes:
                    continue
                if ins.mnemonic == "jal" and ins.target in setting:
                    continue
                if ins.mnemonic != "jal":
                    return False                  # through a register
                work.append(pc + 8)
                continue
            if ins.is_return:
                if not slot_writes:
                    return False
                continue
            targets = []
            if ins.is_branch:
                if not slot_writes:
                    targets.append(ins.target)    # taken: the slot runs
                if ins.is_likely or not slot_writes:
                    work.append(pc + 8)           # not taken: a likely one skips it
            elif ins.mnemonic in ("j", "b") or (ins.is_jump and ins.target >= 0):
                if slot_writes:
                    continue
                targets.append(ins.target)
            else:
                return False                      # a jump table
            for target in targets:
                if start <= target < end:
                    work.append(target)
                elif target not in setting:
                    return False                  # a tail call or a chunk
        return True

    setting = set(functions)
    changed = True
    while changed:
        changed = False
        for address in sorted(setting):
            if not sets(functions[address], setting):
                setting.discard(address)
                changed = True
    return frozenset(setting)


# Caller-saved general registers a callee may leave as they were: $v1, $a0-$t7,
# $t8, $t9 ($v0 is v0_preserving's).
PRESERVABLE = (3,) + tuple(range(4, 16)) + (24, 25)

# Instructions that name a general register first and only read it.
_READS_FIRST = frozenset({
    "sb", "sh", "sw", "sd", "sq", "swl", "swr", "sdl", "sdr", "swc1", "sdc1", "sqc2",
    "mtc0", "mtc1", "ctc1", "qmtc2", "ctc2", "mthi", "mtlo", "mthi1", "mtlo1",
    "mtsa", "mtsab", "mtsah", "pmthi", "pmtlo", "pmthl", "jr", "cache", "pref", "sync",
})


def _may_write(ins):
    """The general register an instruction may write, or None. Over-reports
    rather than under: whatever names a register first, other than a store, a
    branch or a move to another unit, is taken to write it."""
    if ins.is_call:
        return 31
    if ins.is_branch or ins.is_jump or ins.mnemonic in _READS_FIRST:
        return None
    from .structs import _NUMBER, _REGISTER
    for operand in ins.operands:
        for name in _REGISTER.findall(str(operand)):
            if name in _NUMBER:
                return _NUMBER[name]
    return None


def preserved_registers(database, image, base) -> dict:
    """{function: the PRESERVABLE registers it leaves as the caller had them}.

    CodeWarrior keeps a value in a caller-saved register across a call to a
    function that never touches it (Resident Evil Outbreak's 0x00190100 holds
    its argument in $t1 across a call to 0x0018F570, and reads it after).
    A register is preserved by a function that writes it nowhere - not in its
    body, its cold chunks, or any function it calls or jumps to. A call through
    a register, a kernel call, a jump table, a branch or a fall-through out of
    the function's known code, or a call into the middle of something, and
    nothing is: a fixed point over the call graph, from the conservative side.
    """
    functions = {f["addr"]: f for f in database["functions"] if f["size"]}
    every = frozenset(PRESERVABLE)
    kept, needs = {}, {}
    for address, func in functions.items():
        ranges = [(func["addr"], func["end"])] + [tuple(c) for c in func.get("chunks") or ()]

        def inside(target):
            return any(low <= target < high for low, high in ranges)

        written, callees, unknown = set(), set(), False
        for low, high in ranges:
            last = DISASM.decode(struct.unpack_from("<I", image, high - 8 - base)[0], high - 8)
            if not last.ends_block:
                unknown = True                       # runs on past what is known
            for pc in range(low, high, 4):
                ins = DISASM.decode(struct.unpack_from("<I", image, pc - base)[0], pc)
                if ins.mnemonic in ("syscall", "jalr") or (ins.mnemonic == "jr" and not ins.is_return):
                    unknown = True
                    break
                register = _may_write(ins)
                if register is not None:
                    written.add(register)
                if (ins.is_call or ins.is_jump or ins.is_branch) and not ins.is_return:
                    if ins.target is None or ins.target < 0:
                        unknown = True
                    elif ins.is_call or not inside(ins.target):
                        if ins.is_branch and not ins.is_call:
                            unknown = True           # a branch out of the function
                        elif ins.target in functions:
                            callees.add(ins.target)
                        else:
                            unknown = True
            if unknown:
                break
        kept[address] = frozenset() if unknown else every - written
        needs[address] = callees
    changed = True
    while changed:
        changed = False
        for address, callees in needs.items():
            value = kept[address]
            for callee in callees:
                value &= kept[callee]
            if value != kept[address]:
                kept[address] = value
                changed = True
    return {address: value for address, value in kept.items() if value}


_LOADS = frozenset({"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld", "lq", "lwl", "lwr", "ldl",
                    "ldr", "lwc1", "ldc1", "lqc2"})


def stack_readers(database, image, base) -> frozenset:
    """Functions that read their caller's frame: arguments past the eighth, which
    EABI passes at 0($sp) and up.

    A load from `$sp` at an offset of at least the function's own frame (its
    `addiu $sp, $sp, -F`), or such an address taken (a va_list), reads above
    it. A function with no frame reads its caller's too through what it calls
    or jumps to. The model keeps a caller's frame symbolic, so a callee reading
    its arguments there read nothing (Resident Evil Outbreak's 0x00135078
    passes a pointer at 0($sp) to 0x00127FC8, which wrote through null).
    """
    functions = {f["addr"]: f for f in database["functions"] if f["size"]}
    readers, frameless_calls = set(), {}
    for address, func in functions.items():
        frame, reads, callees = 0, False, set()
        for pc in range(func["addr"], func["end"], 4):
            ins = DISASM.decode(struct.unpack_from("<I", image, pc - base)[0], pc)
            if ins.mnemonic in ("addiu", "daddiu") and ins.rs == 29:
                if ins.rt == 29:
                    if ins.imm < 0 and not frame:
                        frame = -ins.imm
                    continue
                if ins.imm >= frame:
                    reads = True
            elif ins.mnemonic in _LOADS and ins.rs == 29 and ins.imm >= frame:
                reads = True
            elif ins.mnemonic in ("daddu", "addu", "or") and 29 in (ins.rs, ins.rt) and not frame:
                if ins.rd != 29:
                    reads = True                 # $sp copied with no frame: its caller's
            if (ins.is_call or ins.is_jump) and ins.target is not None and ins.target >= 0:
                if not func["addr"] <= ins.target < func["end"]:
                    callees.add(ins.target)
        if reads:
            readers.add(address)
        elif not frame:
            frameless_calls[address] = callees
    changed = True
    while changed:
        changed = False
        for address, callees in frameless_calls.items():
            if address not in readers and callees & readers:
                readers.add(address)
                changed = True
    return frozenset(readers)


def scanned_parameters(database, image, base) -> dict:
    """{function: sorted argument-vector indices it reads}, for every known function.

    A function that reads no argument maps to (), so that "takes nothing" and "not
    a function anyone found" stay distinguishable - a jump into the middle of
    some other code is not a call that can be made with no arguments.
    """
    found = {f["addr"]: set() for f in database["functions"] if f["size"]}
    for function, index in argument_use(database, image, base):
        found.setdefault(function, set()).add(index)
    for function, index in float_argument_use(database, image, base):
        found.setdefault(function, set()).add(index)
    return {function: tuple(sorted(indices)) for function, indices in found.items()}


# -- liveness ------------------------------------------------------------------
#
# The scans above follow each function in address order, so a register written on
# one branch and read after the paths rejoin looks written-before-read, although
# on the other path it is read first. That missed real arguments - a caller set
# $t1 and $t2 for a callee whose analysis said it took five - and every miss
# surfaced as a disagreement, since the call then passed nothing in those
# registers. Live-in analysis over the control-flow graph has no such blind spot.

ALL_ARGUMENTS = (1 << 16) - 1
_INT_INDEX = {reg: index for index, reg in enumerate((4, 5, 6, 7, 8, 9, 10, 11))}
_FLOAT_INDEX = {reg: FLOAT_ARG_BASE + k for k, reg in enumerate(FLOAT_ARG_REGS)}


def _effects(ins) -> tuple:
    """(uses, defs) of one instruction, as bitmasks over argument-vector indices."""
    from .structs import _read_registers, _written_register

    uses = defs = 0
    if ins.mnemonic == "syscall":
        # The kernel takes its arguments where a function does: a stub of
        # `addiu $v1, N; syscall; jr $ra` reads them though no instruction
        # names them. Read as none, a caller passing a local's address to
        # ReferThreadStatus or SetOsdConfigParam kept its frame symbolic and
        # read back a slot the kernel had filled (Resident Evil Outbreak, 14
        # functions). How many a service takes is not recorded, so all eight.
        uses |= (1 << FLOAT_ARG_BASE) - 1
    for reg in _read_registers(ins):
        if reg in _INT_INDEX:
            uses |= 1 << _INT_INDEX[reg]
    written = _written_register(ins)
    if ins.mnemonic in ("movz", "movn"):
        # A conditional move keeps the old value when its condition fails: it
        # reads its destination, and kills nothing (Resident Evil Outbreak's
        # 0x00197900 passes on $t0 untouched when its `movz` does not move).
        if written in _INT_INDEX:
            uses |= 1 << _INT_INDEX[written]
        written = None
    if written in _INT_INDEX:
        defs |= 1 << _INT_INDEX[written]
    read, wrote = _float_operands(ins)
    for reg in read:
        if reg in _FLOAT_INDEX:
            uses |= 1 << _FLOAT_INDEX[reg]
    for reg in wrote:
        if reg in _FLOAT_INDEX:
            defs |= 1 << _FLOAT_INDEX[reg]
    return uses, defs


def _graph(func, image, base, functions) -> list:
    """The function as nodes (uses, defs, successors, call target) in address order.

    A call is its own node after its delay slot: it uses whatever the callee reads
    and defines every argument register, all of which are caller-saved. A call
    target of None means "reads everything" - a call through a register, a jump
    table, a jump to somewhere that is not a known function.
    """
    start, end = func["addr"], func["end"]
    nodes = {}
    slots = set()

    def decode(address):
        return DISASM.decode(struct.unpack_from("<I", image, address - base)[0], address)

    address = start
    while address < end:
        if address in slots:
            address += 4
            continue
        ins = decode(address)
        uses, defs = _effects(ins)
        if not ins.has_delay_slot or address + 4 >= end:
            successors = [address + 4] if address + 4 < end else []
            if ins.has_delay_slot or (ins.is_jump and not ins.is_call):
                successors = []
            nodes[address] = (uses, defs, successors, False)
            address += 4
            continue

        slot = address + 4
        slots.add(slot)
        slot_uses, slot_defs = _effects(decode(slot))
        after = address + 8
        target = ins.target
        inside = start <= target < end

        if ins.is_call:
            callee = target if ins.mnemonic == "jal" and target in functions else None
            nodes[address] = (uses, defs, [slot], False)
            nodes[slot] = (slot_uses, slot_defs, [("call", address)], False)
            nodes[("call", address)] = (callee, ALL_ARGUMENTS, [after], True)
        elif ins.is_return:
            nodes[address] = (uses, defs, [slot], False)
            nodes[slot] = (slot_uses, slot_defs, [], False)
        elif ins.is_branch:
            branch_targets = [target] if inside else [("tail", address)]
            nodes[address] = (uses, defs, [slot] + ([after] if ins.is_likely else []), False)
            nodes[slot] = (slot_uses, slot_defs,
                           branch_targets + ([] if ins.is_likely else [after]), False)
            if not inside:
                nodes[("tail", address)] = (None, 0, [], True)
        else:                                     # j, b, jr to somewhere else
            nodes[address] = (uses, defs, [slot], False)
            if inside:
                nodes[slot] = (slot_uses, slot_defs, [target], False)
            else:
                callee = target if target in functions else None
                nodes[slot] = (slot_uses, slot_defs, [("tail", address)], False)
                nodes[("tail", address)] = (callee, 0, [], True)
        address += 8
    return nodes


def _live_in(nodes, start, parameters) -> int:
    return live_map(nodes, parameters).get(start, 0)


def live_map(nodes, parameters) -> dict:
    """{node: mask of the argument-vector indices live on entry to it}."""
    live = {key: 0 for key in nodes}

    def uses_of(node):
        first, _defs, _successors, is_call = node
        if not is_call:
            return first
        if first is None:
            return ALL_ARGUMENTS
        mask = 0
        for index in parameters.get(first, ()):
            mask |= 1 << index
        return mask

    order = sorted((key for key in nodes if isinstance(key, int)), reverse=True)
    extra = [key for key in nodes if not isinstance(key, int)]
    changed = True
    while changed:
        changed = False
        for key in extra + order:
            node = nodes[key]
            out = 0
            for successor in node[2]:
                out |= live.get(successor, 0)
            value = uses_of(node) | (out & ~node[1])
            if value != live[key]:
                live[key] = value
                changed = True
    return live


def live_parameters(database, image, base, rounds: int = 40) -> dict:
    """{function: argument-vector indices live on entry}, for every known function.

    Solved to a fixed point over the call graph: what a function reads includes
    what it passes on to callees that read it.
    """
    functions = {f["addr"]: f for f in database["functions"] if f["size"]}
    graphs = {address: _graph(func, image, base, functions)
              for address, func in functions.items()}
    parameters = {address: () for address in functions}
    for _round in range(rounds):
        changed = False
        for address, nodes in graphs.items():
            mask = _live_in(nodes, address, parameters)
            found = tuple(i for i in range(16) if mask >> i & 1)
            if found != parameters[address]:
                parameters[address] = found
                changed = True
        if not changed:
            break
    return parameters

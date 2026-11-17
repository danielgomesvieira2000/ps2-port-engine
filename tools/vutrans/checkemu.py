"""Checks vu1sym against burnout3-decomp's vu1emu (read-only, imported from its directory) on snapshot runs:
the same final data memory, registers, pc, pair count and kicked packets (bits); and that every kicked word's
expression, evaluated from the run's initial state alone, gives the kicked bits.

    python -I engine/tools/vutrans/checkemu.py SNAP IMAGE [IMAGE ...] [--runs N] [--skip N] [--no-emu]
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vu1sym  # noqa: E402

EMU_DIR = "C:/Users/danyg/claude-projects/burnout3-decomp/games/burnout3/build/native"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--no-emu", action="store_true", help="only the DAG evaluation check (and timing)")
    ap.add_argument("--emu-dir", default=EMU_DIR)
    ap.add_argument("--sim", action="store_true", help="compare with spec_7E86891B_sim.py (it has RSQRT, ERSQRT, "
                    "and the engine's MOVE timing) instead of vu1emu")
    a = ap.parse_args()
    vu1emu = None
    if not a.no_emu:
        sys.path.append(a.emu_dir)
        if a.sim:
            import spec_7E86891B_sim as vu1emu  # noqa: F811
        else:
            import vu1emu  # noqa: F401
    total_bad = 0
    for img in a.images:
        image = int(img, 16)
        ipdom = None
        n = ok = words = 0
        t_sym = t_emu = 0.0
        pairs = 0
        for i, head, m, d in vu1sym.image_runs(a.snap, image, a.runs, a.skip):
            if ipdom is None:
                ipdom = vu1sym.build_ipdom(m)
                dec = vu1sym.decode_image(m)
            t0 = time.perf_counter()
            s = vu1sym.SymVu(head, m, d, decoded=dec, ipdom=ipdom, engine_moves=a.sim)
            s.run()
            t_sym += time.perf_counter() - t0
            pairs += s.count
            bad = []
            if vu1emu is not None:
                t0 = time.perf_counter()
                e = vu1emu.Vu(head, m, d)
                e.run()
                t_emu += time.perf_counter() - t0
                if a.sim:
                    if e.mem != s.mem:
                        bad.append("memory")
                    if e.vf != s.vf or e.acc != s.acc or e.vi[1:16] != [x & 0xFFFF for x in s.vc[1:16]]:
                        bad.append("registers")
                    ek = [[tuple(q) for ad, pk in out for q in pk] for c, out in s.kicks]
                    ek = [[tuple(b for b, _ in q) for q in k] for k in ek]
                    if ek != [list(k) for k in e.kicks]:
                        bad.append("kicks")
                elif bytes(e.data) != s.data_bytes():
                    bad.append("memory")
                if a.sim:
                    pass
                elif e.pc != s.pc or e.count != s.count:
                    bad.append("pc/count %X/%d vs %X/%d" % (e.pc, e.count, s.pc, s.count))
                if a.sim:
                    pass
                elif e.vf != s.vf:
                    bad.append("vf")
                if a.sim:
                    pass
                elif [x & 0xFFFF for x in e.vc[:16]] != [x & 0xFFFF for x in s.vc[:16]] or e.vc[21:23] != s.vc[21:23]:
                    bad.append("vi/I/Q")
                if not a.sim and e.acc != s.acc:
                    bad.append("acc")
                ek = [] if a.sim else [(c, [(ad, [list(q) for q in pk]) for ad, pk in out]) for c, out in e.kicks]
                sk = [] if a.sim else [(c, [(ad, [[b for b, _ in q] for q in pk]) for ad, pk in out]) for c, out in s.kicks]
                if ek != sk:
                    bad.append("kicks")
            leafval = s.leafval_fn(head, d)
            memo = {}
            for c, out in s.kicks:
                for ad, pk in out:
                    for q in pk:
                        for b, node in q:
                            words += 1
                            if s.dag.evaluate(node, leafval, memo) != b:
                                bad.append("dag word")
                                break
            n += 1
            if bad:
                total_bad += 1
                print("%08X run %d (snapshot %d): MISMATCH %s" % (image, n - 1, i, ", ".join(sorted(set(bad)))))
            else:
                ok += 1
        print("%08X: %d runs, %d identical to the reference%s; %d kicked words re-evaluated from their DAGs; "
              "%d pairs; vu1sym %.1f runs/s (%.0f pairs/s)%s" %
              (image, n, ok, "" if vu1emu else " (emu skipped)", words, pairs, n / max(t_sym, 1e-9),
               pairs / max(t_sym, 1e-9), (", vu1emu %.1f runs/s" % (n / t_emu)) if t_emu else ""))
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())

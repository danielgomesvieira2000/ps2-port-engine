#!/usr/bin/env python3
"""What a recording of the port's sound (--audio-wav) is like, second by second.

    python engine/tools/audiocheck.py out.wav                 # a line a second
    python engine/tools/audiocheck.py out.wav --from 100 --to 130 --every 0.5

For each span: the level (RMS) and peak; samples at full scale (clipping:
runs of two or more at +-32767/-32768); clicks - a step from one sample to
the next far larger than the signal's own steps around it (a buffer dropped
or repeated, a voice cut off mid-wave); and gaps - runs of exact silence
inside sound (a device or mixer that ran dry). A summary at the end counts
them over the whole recording, to compare two builds.
"""

from __future__ import annotations

import argparse
import struct
import sys

import numpy as np


def read_wav(path: str):
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SystemExit("%s: not a WAV file" % path)
    position, rate, channels, samples = 12, 48000, 2, None
    while position + 8 <= len(data):
        chunk, size = data[position:position + 4], struct.unpack_from("<I", data, position + 4)[0]
        body = data[position + 8:position + 8 + size]
        if chunk == b"fmt ":
            channels, rate = struct.unpack_from("<HI", body, 2)
        elif chunk == b"data":
            # A recording cut short (the run killed) keeps a size of 0: the rest of the file.
            if size == 0:
                body = data[position + 8:]
            samples = np.frombuffer(body[:len(body) // (2 * channels) * 2 * channels], dtype="<i2")
            break
        position += 8 + size + (size & 1)
    if samples is None:
        raise SystemExit("%s: no data chunk" % path)
    return rate, samples.reshape(-1, channels).astype(np.int32)


def clipped_runs(channel: np.ndarray) -> int:
    """Runs of two or more samples at full scale."""
    full = (channel >= 32767) | (channel <= -32768)
    if not full.any():
        return 0
    starts = np.flatnonzero(full[1:] & full[:-1] & ~np.concatenate(([False], full[:-2])))
    return len(starts)


def clicks(channel: np.ndarray, rate: int) -> np.ndarray:
    """Where a step between samples stands out from the steps around it: more
    than 8 times the local typical step (median over ~5 ms) and over 2,000."""
    step = np.abs(np.diff(channel))
    if len(step) < 512:
        return np.array([], dtype=np.int64)
    window = max(64, rate // 200)
    # The local typical step: a running mean of the steps (robust enough
    # against a single outlier, as the window is long).
    kernel = np.ones(window) / window
    local = np.convolve(step, kernel, mode="same")
    found = np.flatnonzero((step > 2000) & (step > 8 * (local + 50)))
    # One click a few samples long counts once.
    if len(found):
        found = found[np.concatenate(([True], np.diff(found) > 32))]
    return found


def gaps(frames: np.ndarray, rate: int) -> list:
    """Runs of exact silence (both channels 0) from 2 ms to 1 s long, with
    sound on both sides: a dropout, not a pause."""
    silent = np.all(frames == 0, axis=1)
    edges = np.flatnonzero(np.diff(silent.astype(np.int8)))
    found = []
    for begin, end in zip(edges[::2], edges[1::2]):
        if silent[begin]:       # a run of sound, not of silence
            continue
        length = end - begin
        if rate // 500 <= length <= rate:
            found.append((begin + 1, length))
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav")
    parser.add_argument("--from", dest="start", type=float, default=0.0, help="seconds")
    parser.add_argument("--to", type=float, default=None, help="seconds")
    parser.add_argument("--every", type=float, default=1.0, help="seconds a line")
    parser.add_argument("--quiet", action="store_true", help="the summary only")
    args = parser.parse_args(argv)
    rate, frames = read_wav(args.wav)
    first = int(args.start * rate)
    last = len(frames) if args.to is None else min(len(frames), int(args.to * rate))
    frames = frames[first:last]
    span = max(1, int(args.every * rate))
    total_clip = total_click = 0
    all_gaps = gaps(frames, rate)
    click_at = np.unique(np.concatenate([clicks(frames[:, ch], rate) for ch in range(frames.shape[1])]))
    if not args.quiet:
        print("    time      RMS   peak  clip  clicks  gaps")
    for begin in range(0, len(frames), span):
        part = frames[begin:begin + span]
        if not len(part):
            break
        rms = float(np.sqrt(np.mean(part.astype(np.float64) ** 2)))
        peak = int(np.max(np.abs(part)))
        clip = sum(clipped_runs(part[:, ch]) for ch in range(part.shape[1]))
        click = int(np.count_nonzero((click_at >= begin) & (click_at < begin + span)))
        gap = sum(1 for at, _ in all_gaps if begin <= at < begin + span)
        total_clip += clip
        total_click += click
        if not args.quiet:
            print("%8.1f %8.0f %6d %5d %7d %5d" % ((first + begin) / rate, rate and rms, peak, clip, click, gap))
    seconds = len(frames) / rate
    sounding = float(np.mean(np.any(frames != 0, axis=1))) if len(frames) else 0.0
    print("summary: %.1f s, sound in %.0f%%, RMS %.0f, peak %d; %d clipped runs, %d clicks, %d gaps"
          % (seconds, 100 * sounding, float(np.sqrt(np.mean(frames.astype(np.float64) ** 2))) if len(frames) else 0,
             int(np.max(np.abs(frames))) if len(frames) else 0, total_clip, total_click, len(all_gaps)))
    for at, length in all_gaps[:20]:
        print("  gap at %.3f s, %.1f ms" % ((first + at) / rate, 1000 * length / rate))
    shown = click_at[:20]
    if len(shown):
        print("  clicks at: " + ", ".join("%.3f" % ((first + c) / rate) for c in shown))
    return 0


if __name__ == "__main__":
    sys.exit(main())

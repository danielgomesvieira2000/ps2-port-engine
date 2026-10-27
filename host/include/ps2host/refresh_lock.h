/*
 * The vertical blank timer locked to the screen's refresh (threads.cpp's
 * VblankTimer, each blank): a blank k of the screen's refreshes long, k the
 * whole number the game's own blank is (60 a second on a 60, 120 or 240 Hz
 * screen: 1, 2, 4; FRAME RATE DISPLAY: 1), and each blank nudged onto one of
 * the screen's. Free-running by the processor's clock, the game's 59.94
 * blanks a second against a 60 Hz screen gained a picture every 17 seconds
 * - one shown twice, or one dropped - and 144.00 against a 143.98 Hz screen
 * one a minute: the stutter no pacing undoes.
 *
 * Within half a percent of the game's own blank only (the sound's ring
 * plays up to half a percent faster or slower, audio.cpp): the game runs
 * that much faster or slower, as on a console whose clock was the screen's.
 * A VRR screen's refresh follows the pictures - the blanks would chase it
 * and it them: a lock still pulling in after ten seconds, or one whose
 * nudges add up (more than a millisecond in ten seconds once in place),
 * gives up until the screen's period changes. The VRR setting turns it off.
 *
 * Each blank a `lead` before the screen's (Step's): the picture a blank
 * makes is done on the GPU some time after it; on the screen's blank, it
 * waited out nearly a whole refresh for the next - the lead (the host's
 * longest such time and a margin, as the screen's refreshes go) has it done
 * just before one, shown at once. The lead taken up at the slew's pace.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ps2host {
namespace refresh_lock {

/* How many of the screen's refreshes a blank of `blank_ns` is: 1 to 4 when
   within half a percent of a whole number, else 0 (120 a second on a 144 Hz
   screen: none). */
inline int Refreshes(double blank_ns, double refresh_ns) {
    if (!(refresh_ns > 0) || !(blank_ns > 0)) return 0;
    const long long k = std::llround(blank_ns / refresh_ns);
    if (k < 1 || k > 4) return 0;
    return std::fabs(blank_ns - double(k) * refresh_ns) <= 0.005 * blank_ns ? int(k) : 0;
}

/* Where `since_anchor` (a time less a screen's blank) falls: after the
   nearest of the screen's blanks by this much, -period/2 to period/2. */
inline double Phase(double since_anchor, double period) {
    double p = std::fmod(since_anchor, period);
    if (p < 0) p += period;
    return p > period / 2 ? p - period : p;
}

/* The first middle of one of the screen's refreshes at or after `since_anchor`,
   as the time to add to it: where a picture handed over is shown at the next
   refresh whichever way a sleep's wake-up errs. */
inline double ToMiddle(double since_anchor, double period) {
    double into = std::fmod(since_anchor, period);
    if (into < 0) into += period;
    return into <= period / 2 ? period / 2 - into : 1.5 * period - into;
}

class Lock {
public:
    enum class State { kOff, kPulling, kLocked, kGaveUp };
    static constexpr double kGain = 0.1;                // of the phase's error, each blank
    static constexpr double kSlewShare = 0.004;         // of a refresh, the most a blank is nudged
    static constexpr double kPullNs = 10e9;             // pulling in for longer: given up
    static constexpr double kWindowNs = 10e9;           // once locked, nudges summed over this...
    static constexpr double kDriftNs = 1e6;             // ...more than this beyond the error closed: given up

    /* The step to the blank after `next` (steady clock ns), the game's own
       being `nominal` ns; the screen's refresh `period` ns with a blank at
       `anchor` (valid false: not known); each blank wanted `lead` ns before
       one of the screen's (0 to a period). */
    double Step(double nominal, std::uint64_t next, bool valid, double period, std::uint64_t anchor,
                double lead = 0) {
        const int k = valid ? Refreshes(nominal, period) : 0;
        if (state_ == State::kGaveUp) {
            if (k && std::fabs(period - gave_up_period_) < 0.01 * period) return nominal;
            state_ = State::kOff;
        }
        if (!k) {
            if (state_ != State::kOff) Change(State::kOff);
            return nominal;
        }
        if (state_ == State::kOff || k != k_) {
            k_ = k;
            Pull();
        }
        const double slew = kSlewShare * period;
        // The lead followed a slew at a time once locked - the phase's error
        // stays a few hundredths of a refresh (never pulling in again), and
        // what it adds is no drift (the nudges' sum less it).
        const double wanted = Phase(lead, period);
        const double moved = state_ == State::kLocked ? std::clamp(Phase(wanted - lead_, period), -slew, slew) : 0.0;
        lead_ += moved;
        window_sum_ -= moved;
        // The blank after this step, unnudged: after the screen's nearest (less
        // the lead) by e.
        const double target = double(next) + double(k) * period;
        const double e = Phase(target + lead_ - double(anchor), period);
        const double nudge = std::clamp(e * kGain, -slew, slew);
        elapsed_ += nominal;
        if (state_ == State::kPulling) {
            if (std::fabs(e) < period / 16) {
                Change(State::kLocked);
                window_from_ = elapsed_;
                window_sum_ = 0;
                window_e_ = e - nudge;
            } else if (elapsed_ > kPullNs) {
                GiveUp(period);
                return nominal;
            }
        } else if (std::fabs(e) > period / 8) {
            Pull();                             // knocked off (a stall): in again
        } else {
            // Drift: nudges beyond the error they closed (and the lead taken
            // up) - a screen that follows the blanks, its error never closing.
            window_sum_ += nudge;
            if (std::fabs(window_sum_ - (window_e_ - (e - nudge))) > kDriftNs) {
                GiveUp(period);
                return nominal;
            }
            if (elapsed_ - window_from_ > kWindowNs) {
                window_from_ = elapsed_;
                window_sum_ = 0;
                window_e_ = e - nudge;
            }
        }
        period_ = period;
        anchor_ = anchor;
        return double(k) * period - nudge;
    }

    /* Pulling in again from here (the timer's own restart after a stall). */
    void Pull() {
        Change(State::kPulling);
        elapsed_ = 0;
    }

    State state() const { return state_; }
    int refreshes() const { return k_; }
    double period() const { return period_; }
    std::uint64_t anchor() const { return anchor_; }
    /* Counted up at each change of state (the timer logs it). */
    unsigned changes() const { return changes_; }

private:
    void Change(State s) {
        if (s != state_) ++changes_;
        state_ = s;
    }
    void GiveUp(double period) {
        gave_up_period_ = period;
        Change(State::kGaveUp);
    }

    State state_ = State::kOff;
    int k_ = 0;
    double elapsed_ = 0, window_from_ = 0, window_sum_ = 0, gave_up_period_ = 0, period_ = 0;
    double lead_ = 0;                   // the lead as far as taken up
    double window_e_ = 0;               // the error expected after the window's first nudge
    std::uint64_t anchor_ = 0;
    unsigned changes_ = 0;
};

}  // namespace refresh_lock
}  // namespace ps2host

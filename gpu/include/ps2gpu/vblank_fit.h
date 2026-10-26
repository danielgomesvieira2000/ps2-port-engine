/*
 * VblankFit: a screen's vertical blanks as a thread saw them (each a little
 * late, by its wake-up), fitted to a line - the refresh's period and the
 * time of one blank, steadier than any one sample. The display's thread
 * (VulkanDisplay::RefreshTiming) feeds it; the host's vertical blank timer
 * follows what it says (refresh_lock.h).
 *
 * Times are the steady clock's, in nanoseconds: the timer's own. A blank
 * missed between two samples (the thread late by more than a refresh) is
 * counted from the time between them; a sample far from the line (a wake-up
 * a millisecond late) is not used; many in a row, or a long gap (the
 * screen's mode changed, the machine slept), and the fit starts again.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ps2gpu {

class VblankFit {
public:
    static constexpr int kSamples = 240;        // the line's: 1.7 s at 144 Hz, 4 s at 60
    static constexpr int kEnough = 60;          // samples before it says anything
    static constexpr double kFarNs = 1'000'000; // further than this from the line: not used
    static constexpr double kEvenNs = 500'000;  // the samples' spread around it, at most, to be used

    /* A blank seen at `t_ns`. `nominal_ns`: the period the screen's mode
       says (0: not known) - how blanks missed between samples are counted
       before the fit has a period of its own, and, far from the fit's,
       a sign the mode changed. */
    void Add(std::uint64_t t_ns, double nominal_ns) {
        if (count_ >= kEnough && nominal_ns > 0 && std::fabs(period_ - nominal_ns) > 0.02 * nominal_ns) Reset();
        if (!last_) {
            first_ = last_ = t_ns;
            index_ = 0;
            Push(0, 0.0);
            return;
        }
        if (t_ns <= last_) return;
        const double p = count_ >= kEnough ? period_ : nominal_ns;
        const double dt = double(t_ns - last_);
        const long long steps = p > 0 ? std::max(1LL, std::llround(dt / p)) : 1;
        if (steps > 30) {                       // a long gap: again from here
            Reset();
            Add(t_ns, nominal_ns);
            return;
        }
        index_ += steps;
        last_ = t_ns;
        const double at = double(t_ns - first_);
        if (count_ >= kEnough) {
            const double predicted = offset_ + period_ * double(index_);
            if (std::fabs(at - predicted) > kFarNs) {
                if (++far_ > 30) {
                    Reset();
                    Add(t_ns, nominal_ns);
                }
                return;
            }
        }
        far_ = 0;
        Push(index_, at);
    }

    /* Enough samples, close to their line. */
    bool Valid() const { return count_ >= kEnough && period_ > 0 && spread_ < kEvenNs; }
    /* The refresh's period, in nanoseconds. */
    double Period() const { return period_; }
    /* The line's time of the last blank seen. */
    std::uint64_t Anchor() const { return first_ + std::uint64_t(std::llround(offset_ + period_ * double(index_))); }
    /* The samples' root mean square distance from the line. */
    double Spread() const { return spread_; }
    /* The last sample's time (0: none yet). */
    std::uint64_t Last() const { return last_; }

    void Reset() {
        count_ = next_ = far_ = 0;
        first_ = last_ = 0;
        index_ = 0;
        period_ = offset_ = spread_ = 0;
    }

private:
    void Push(long long n, double t) {
        n_[next_] = n;
        t_[next_] = t;
        next_ = (next_ + 1) % kSamples;
        count_ = std::min(count_ + 1, kSamples);
        if (count_ < 2) return;
        // Least squares, about the means.
        double mn = 0, mt = 0;
        for (int i = 0; i < count_; ++i) mn += double(n_[i]), mt += t_[i];
        mn /= count_;
        mt /= count_;
        double nn = 0, nt = 0;
        for (int i = 0; i < count_; ++i) {
            const double dn = double(n_[i]) - mn;
            nn += dn * dn;
            nt += dn * (t_[i] - mt);
        }
        if (nn <= 0) return;
        period_ = nt / nn;
        offset_ = mt - period_ * mn;
        double squares = 0;
        for (int i = 0; i < count_; ++i) {
            const double r = t_[i] - (offset_ + period_ * double(n_[i]));
            squares += r * r;
        }
        spread_ = std::sqrt(squares / count_);
    }

    long long n_[kSamples] = {};
    double t_[kSamples] = {};
    int count_ = 0, next_ = 0, far_ = 0;
    std::uint64_t first_ = 0, last_ = 0;
    long long index_ = 0;
    double period_ = 0, offset_ = 0, spread_ = 0;
};

}  // namespace ps2gpu

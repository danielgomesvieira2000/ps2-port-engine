/*
 * Discord's Rich Presence: what the player is doing, shown on their Discord
 * profile ("Raging in Silver Lake", the time since the race began).
 *
 * Talks to the Discord app on this PC itself, through its local pipe
 * (\\.\pipe\discord-ipc-0 to -9: frames of an op code, a length and JSON) -
 * no Discord SDK, nothing sent anywhere else. On a thread of its own: the
 * game never waits for it. No Discord running, or it closes, and it tries
 * again every few seconds; an update at most every few seconds (Discord's
 * limit), the latest one kept. A game's port decides what is said
 * (Burnout 3: games/burnout3/port/presence.cpp).
 */

#ifndef PS2HOST_PRESENCE_H
#define PS2HOST_PRESENCE_H

#include <string>

namespace ps2host {

/* What the profile shows; an empty string is left out. */
struct PresenceActivity {
    std::string details;            // the first line ("Raging in Silver Lake")
    std::string state;              // the second ("Silver Lake Southbound")
    long long start = 0;            // Unix seconds the clock counts from (0: no clock)
    std::string large_image, large_text;    // an asset's key in the Discord application (or an https URL)
    std::string small_image, small_text;
    std::string button_label, button_url;   // a button under it (a link)

    bool operator==(const PresenceActivity& o) const {
        return details == o.details && state == o.state && start == o.start && large_image == o.large_image &&
               large_text == o.large_text && small_image == o.small_image && small_text == o.small_text &&
               button_label == o.button_label && button_url == o.button_url;
    }
    bool operator!=(const PresenceActivity& o) const { return !(*this == o); }
};

/* Starts talking to Discord as the application `application_id` (its
   Application ID, from Discord's developer portal); again with another
   restarts. Not on Windows, nothing. */
void PresenceStart(const std::string& application_id);
/* Stops: the profile shows nothing of the game any more. */
void PresenceStop();
bool PresenceRunning();

/* What to show from now (sent when it differs from what was, a few seconds
   after the last at most). */
void PresenceSet(const PresenceActivity& activity);
/* Nothing shown (the game still running). */
void PresenceClear();

/* Discord answered the handshake: the profile shows what was set. */
bool PresenceConnected();

/* The Unix time now, in seconds (PresenceActivity::start). */
long long PresenceNow();

}  // namespace ps2host

#endif  /* PS2HOST_PRESENCE_H */

/*
 * Logitech's IOP modules answered with nothing attached: LGDEVW.IRX (its
 * USB devices - wheels; RPC server 0x046D046D, Logitech's USB vendor ID
 * twice) and LGAUD.IRX (its USB headsets; server 0x50494C42, "PILB"). Games
 * ship them on their discs (Burnout 3 does); their EE libraries ask each
 * module's version first and halt on one they do not know, then enumerate
 * devices until one fails - here the first.
 */

#ifndef PS2HOST_LOGITECH_H
#define PS2HOST_LOGITECH_H

#include <cstdint>

namespace ps2host {
namespace logitech {

/* lgDev: procedure 12 answers the version at +4 (the game's library checks
   it: Burnout 3's wants 0x010B1B00), procedure 1 (lgDevEnumerate) no device. */
void RegisterNoDevices(std::uint32_t version = 0x010B1B00u);
/* lgAud: procedure 0x10 answers the version at +8 (Burnout 3's 0x010A0100),
   procedure 1 (lgAudEnumerate) no headset. */
void RegisterNoHeadsets(std::uint32_t version = 0x010A0100u);

}  // namespace logitech
}  // namespace ps2host

#endif  /* PS2HOST_LOGITECH_H */

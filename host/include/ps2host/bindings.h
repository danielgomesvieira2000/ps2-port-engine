/*
 * A player's bindings: which key or controller button holds each of the
 * DualShock 2's inputs (libpad's order: 0 Select ... 15 Square), kept in the
 * PC settings - what a port's own controls are built on (Burnout 3's
 * controls.cpp: the keyboard's and the controller's, racing and in the
 * menus, rebound in PC SETTINGS > CONTROLS).
 *
 * A table is one device's in one of the port's contexts (the port names
 * them: "racing", "menus"); its settings are "bind.<device>.<context>.<input>"
 * = a key's name ("LEFT SHIFT"), a button's (its PlayStation name, "CROSS")
 * or NONE. What a port does with a table - when each context holds, how a
 * trigger's travel becomes pressure - is the port's.
 */

#ifndef PS2HOST_BINDINGS_H
#define PS2HOST_BINDINGS_H

#include <cstdint>
#include <string>
#include <vector>

namespace ps2host {
namespace bindings {

// The DualShock 2's inputs (libpad's order).
constexpr int kSelect = 0, kL3 = 1, kR3 = 2, kStart = 3, kUp = 4, kRight = 5, kDown = 6, kLeft = 7, kL2 = 8,
              kR2 = 9, kL1 = 10, kR1 = 11, kTriangle = 12, kCircle = 13, kCross = 14, kSquare = 15;
constexpr int kInputs = 16;
constexpr std::uint16_t Bit(int input) { return std::uint16_t(1u << input); }
/* An input's name in the settings ("select" ... "square"). */
const char* InputKey(int input);

enum class Device { kKeyboard, kController };
const char* DeviceKey(Device device);       // "keyboard", "controller"

/* The keys a binding may use, by virtual key, and their names as a row
   shows them (in capitals; words for punctuation). F1, F9, F10, F12, Tab
   and Escape are the engine's own (the FPS counter, the PC settings panel,
   the HUD editor, a capture, the texture pack, cancelling) - none bindable. */
const std::vector<int>& BindableKeys();
const char* KeyName(int vk);                // null: not one of them
int KeyByName(const std::string& name);     // 0: none
/* Whether any bindable key, or Escape, is down. */
bool AnyKeyDown();

/* A controller's buttons by their DualShock 2 place, as one of its makers
   names them: "CROSS" or "A". "NONE" for none. */
enum class Style { kPlayStation, kXbox };
const char* ButtonName(int button, Style style);
int ButtonByName(const std::string& playstation_name);     // -1: none

/* One context's bindings: for each input, what holds it - a virtual key (0
   none) or a controller button by its place (-1 none). */
struct Table {
    int physical[kInputs];
};
bool Same(const Table& a, const Table& b);

std::string SettingKey(Device device, const std::string& context, int input);
/* The settings' bindings on top of `defaults` - what a setting names as
   nothing (or not at all) keeps its default. `log`: each setting taken, in
   the log. */
Table Load(Device device, const std::string& context, const Table& defaults, bool log = false);
void Save(Device device, const std::string& context, const Table& table);
/* `input` bound to `physical`: an input that had it gets `input`'s old one
   (each key or button does one thing in a context). */
void Bind(Table& table, int input, int physical);

}  // namespace bindings
}  // namespace ps2host

#endif  /* PS2HOST_BINDINGS_H */

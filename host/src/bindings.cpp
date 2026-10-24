/*
 * A player's bindings (ps2host/bindings.h).
 */

#include "ps2host/bindings.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdio>
#include <cstring>

#include "ps2host/run.h"
#include "ps2host/settings.h"

namespace ps2host {
namespace bindings {
namespace {

const char* const kInputKeys[kInputs] = {"select", "l3", "r3", "start", "up", "right", "down", "left",
                                         "l2", "r2", "l1", "r1", "triangle", "circle", "cross", "square"};

/* The keys and their names (the game fonts have no '-' or '/': words). */
struct Key {
    int vk;
    const char* name;
};
const Key kKeys[] = {
    {'A', "A"}, {'B', "B"}, {'C', "C"}, {'D', "D"}, {'E', "E"}, {'F', "F"}, {'G', "G"}, {'H', "H"}, {'I', "I"},
    {'J', "J"}, {'K', "K"}, {'L', "L"}, {'M', "M"}, {'N', "N"}, {'O', "O"}, {'P', "P"}, {'Q', "Q"}, {'R', "R"},
    {'S', "S"}, {'T', "T"}, {'U', "U"}, {'V', "V"}, {'W', "W"}, {'X', "X"}, {'Y', "Y"}, {'Z', "Z"},
    {'0', "0"}, {'1', "1"}, {'2', "2"}, {'3', "3"}, {'4', "4"}, {'5', "5"}, {'6', "6"}, {'7', "7"}, {'8', "8"},
    {'9', "9"},
    {VK_UP, "UP ARROW"}, {VK_DOWN, "DOWN ARROW"}, {VK_LEFT, "LEFT ARROW"}, {VK_RIGHT, "RIGHT ARROW"},
    {VK_SPACE, "SPACE"}, {VK_RETURN, "ENTER"}, {VK_BACK, "BACKSPACE"},
    {VK_LSHIFT, "LEFT SHIFT"}, {VK_RSHIFT, "RIGHT SHIFT"}, {VK_LCONTROL, "LEFT CTRL"}, {VK_RCONTROL, "RIGHT CTRL"},
    {VK_LMENU, "LEFT ALT"}, {VK_RMENU, "RIGHT ALT"}, {VK_CAPITAL, "CAPS LOCK"},
    {VK_INSERT, "INSERT"}, {VK_DELETE, "DELETE"}, {VK_HOME, "HOME"}, {VK_END, "END"}, {VK_PRIOR, "PAGE UP"},
    {VK_NEXT, "PAGE DOWN"},
    {VK_F2, "F2"}, {VK_F3, "F3"}, {VK_F4, "F4"}, {VK_F5, "F5"}, {VK_F6, "F6"}, {VK_F7, "F7"}, {VK_F8, "F8"},
    {VK_F11, "F11"},
    {VK_NUMPAD0, "NUM 0"}, {VK_NUMPAD1, "NUM 1"}, {VK_NUMPAD2, "NUM 2"}, {VK_NUMPAD3, "NUM 3"},
    {VK_NUMPAD4, "NUM 4"}, {VK_NUMPAD5, "NUM 5"}, {VK_NUMPAD6, "NUM 6"}, {VK_NUMPAD7, "NUM 7"},
    {VK_NUMPAD8, "NUM 8"}, {VK_NUMPAD9, "NUM 9"}, {VK_DECIMAL, "NUM POINT"}, {VK_ADD, "NUM PLUS"},
    {VK_SUBTRACT, "NUM MINUS"}, {VK_MULTIPLY, "NUM TIMES"}, {VK_DIVIDE, "NUM DIVIDE"},
    {VK_OEM_1, "SEMICOLON"}, {VK_OEM_PLUS, "EQUALS"}, {VK_OEM_COMMA, "COMMA"}, {VK_OEM_MINUS, "MINUS"},
    {VK_OEM_PERIOD, "PERIOD"}, {VK_OEM_2, "SLASH"}, {VK_OEM_3, "BACKQUOTE"}, {VK_OEM_4, "LEFT BRACKET"},
    {VK_OEM_5, "BACKSLASH"}, {VK_OEM_6, "RIGHT BRACKET"}, {VK_OEM_7, "APOSTROPHE"},
};

const char* const kPlayStationNames[kInputs] = {"SELECT", "L3", "R3", "START", "DPAD UP", "DPAD RIGHT",
                                                 "DPAD DOWN", "DPAD LEFT", "L2", "R2", "L1", "R1",
                                                 "TRIANGLE", "CIRCLE", "CROSS", "SQUARE"};
const char* const kXboxNames[kInputs] = {"VIEW", "LS", "RS", "MENU", "DPAD UP", "DPAD RIGHT", "DPAD DOWN",
                                         "DPAD LEFT", "LT", "RT", "LB", "RB", "Y", "B", "A", "X"};

}  // namespace

const char* InputKey(int input) { return input >= 0 && input < kInputs ? kInputKeys[input] : "?"; }

const char* DeviceKey(Device device) { return device == Device::kKeyboard ? "keyboard" : "controller"; }

const std::vector<int>& BindableKeys() {
    static const std::vector<int> keys = [] {
        std::vector<int> v;
        for (const Key& k : kKeys) v.push_back(k.vk);
        return v;
    }();
    return keys;
}

const char* KeyName(int vk) {
    for (const Key& k : kKeys)
        if (k.vk == vk) return k.name;
    return nullptr;
}

int KeyByName(const std::string& name) {
    for (const Key& k : kKeys)
        if (name == k.name) return k.vk;
    return 0;
}

bool AnyKeyDown() {
    for (const Key& k : kKeys)
        if (HostKeyDown(k.vk)) return true;
    return HostKeyDown(VK_ESCAPE);
}

const char* ButtonName(int button, Style style) {
    if (button < 0 || button >= kInputs) return "NONE";
    return style == Style::kXbox ? kXboxNames[button] : kPlayStationNames[button];
}

int ButtonByName(const std::string& name) {
    for (int i = 0; i < kInputs; ++i)
        if (name == kPlayStationNames[i]) return i;
    return -1;
}

bool Same(const Table& a, const Table& b) { return std::memcmp(a.physical, b.physical, sizeof a.physical) == 0; }

std::string SettingKey(Device device, const std::string& context, int input) {
    return std::string("bind.") + DeviceKey(device) + "." + context + "." + kInputKeys[input];
}

Table Load(Device device, const std::string& context, const Table& defaults, bool log) {
    Table b = defaults;
    for (int i = 0; i < kInputs; ++i) {
        const std::string value = Setting(SettingKey(device, context, i));
        if (value.empty()) continue;
        if (value == "NONE") b.physical[i] = device == Device::kKeyboard ? 0 : -1;
        else if (device == Device::kKeyboard && KeyByName(value)) b.physical[i] = KeyByName(value);
        else if (device == Device::kController && ButtonByName(value) >= 0) b.physical[i] = ButtonByName(value);
        else continue;
        if (log)
            std::fprintf(stderr, "[controls] %s %s %s: %s (the settings')\n", DeviceKey(device), context.c_str(),
                         kInputKeys[i], value.c_str());
    }
    return b;
}

void Save(Device device, const std::string& context, const Table& b) {
    for (int i = 0; i < kInputs; ++i) {
        const int p = b.physical[i];
        const char* name = device == Device::kKeyboard ? (p ? KeyName(p) : nullptr) : (p >= 0 ? kPlayStationNames[p] : nullptr);
        SetSetting(SettingKey(device, context, i), name ? name : "NONE");
    }
}

void Bind(Table& b, int input, int physical) {
    for (int i = 0; i < kInputs; ++i)
        if (i != input && b.physical[i] == physical) b.physical[i] = b.physical[input];
    b.physical[input] = physical;
}

}  // namespace bindings
}  // namespace ps2host

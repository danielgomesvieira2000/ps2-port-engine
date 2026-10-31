/*
 * Every game port's entry point. The game supplies its translated functions (the
 * table engine/tools/port.py generates) and its overrides; the rest is shared.
 *
 * On Android the port is a library SDL's activity loads (engine/android's
 * PortActivity): its main is SDL_main, run on the activity's own thread,
 * given the app's folder (--android-home) - where the PC port's files beside
 * its executable go - and the player's disc image asked of the activity
 * (its file picker), which keeps the choice.
 */

#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <sys/stat.h>
#include <string>
#include <vector>

#include "ps2host/platform.h"
#ifdef _WIN32
#include <io.h>
#include <commdlg.h>
#else
#include <unistd.h>
#endif
#ifdef __ANDROID__
#include <jni.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>      // main is SDL_main, which the activity calls
#include "ps2gpu/display.h"
#elif defined(__linux__)
#include <SDL3/SDL.h>           // (a player's messages: SDL's message box)
#endif

#include "ps2host/disc.h"
#include "ps2host/host.h"
#include "ps2host/run.h"
#include "ps2host/settings.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

extern const ps2host::FunctionEntry kGuestFunctions[];
extern const std::size_t kGuestFunctionCount;
extern const ps2host::OverlayEntry kGuestOverlays[];
extern const std::size_t kGuestOverlayCount;
#ifdef PS2HOST_PGO_GENERATE
extern "C" int __llvm_profile_write_file(void);   // PS2HOST_PGO=generate (LLVM's profile runtime)
extern "C" void __llvm_profile_set_filename(const char*);
#endif

/* The game's host implementations of guest functions (its port's overrides.cpp). */
void RegisterGameOverrides();

namespace {

/* The port is built for AVX2 (engine/host/CMakeLists.txt; the VU's lanes and
   the rest take its three-operand forms): a processor without it is told so
   rather than stopped by an illegal instruction somewhere. */
bool HasAvx2() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)) && defined(__AVX2__)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuid(info, 1);
    const bool osxsave = (info[2] >> 27) & 1, avx = (info[2] >> 28) & 1;
    if (!osxsave || !avx) return false;
    if ((_xgetbv(0) & 6) != 6) return false;        // the OS saves the YMM registers
    __cpuidex(info, 7, 0);
    return (info[1] >> 5) & 1;
#else
    return true;
#endif
}

std::string g_log;     // the log a run started without arguments writes

#ifdef PS2HOST_GAME_EXECUTABLE
/* Whether `path` is an image of the game's disc: the executable it boots is
   the one the port was translated from (game.json's). */
bool IsGameDisc(const std::string& path) {
    ps2host::Disc disc;
    if (!disc.Open(path)) return false;
    std::string boot = disc.BootPath();
    const std::size_t semicolon = boot.find(';');
    if (semicolon != std::string::npos) boot.resize(semicolon);
    while (!boot.empty() && (boot[0] == '\\' || boot[0] == '/')) boot.erase(0, 1);
    return _stricmp(boot.c_str(), PS2HOST_GAME_EXECUTABLE) == 0;
}

#ifdef __ANDROID__
/* The player's disc image, from the activity (PortActivity.pickDisc: the
   one chosen before, or its file picker's): a path to it, "" if they quit.
   `rejected`: the last one was not the game's (the activity says so and
   forgets it). */
std::string PickDisc(const std::string& rejected) {
    JNIEnv* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
    jobject activity = static_cast<jobject>(SDL_GetAndroidActivity());
    if (!env || !activity) return {};
    jclass type = env->GetObjectClass(activity);
    jmethodID pick = env->GetMethodID(type, "pickDisc", "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;");
    std::string path;
    if (pick) {
        jstring said = env->NewStringUTF(rejected.c_str());
        jstring name = env->NewStringUTF(PS2HOST_GAME_TITLE);
        auto answer = static_cast<jstring>(env->CallObjectMethod(activity, pick, said, name));
        if (env->ExceptionCheck()) env->ExceptionClear();
        if (answer) {
            const char* text = env->GetStringUTFChars(answer, nullptr);
            path = text ? text : "";
            env->ReleaseStringUTFChars(answer, text);
            env->DeleteLocalRef(answer);
        }
        env->DeleteLocalRef(said);
        env->DeleteLocalRef(name);
    }
    env->DeleteLocalRef(type);
    env->DeleteLocalRef(activity);
    return path;
}
#elif defined(_WIN32)
/* The player's disc image, from a file dialog; "" if they cancel. */
std::string PickDisc() {
    char file[MAX_PATH] = {};
    OPENFILENAMEA dialog = {};
    dialog.lStructSize = sizeof dialog;
    dialog.lpstrFilter = "Disc images (*.iso; *.cue; *.bin)\0*.iso;*.cue;*.bin\0All files (*.*)\0*.*\0";
    dialog.lpstrFile = file;
    dialog.nMaxFile = MAX_PATH;
    dialog.lpstrTitle = "Choose your " PS2HOST_GAME_NAME " disc image";
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&dialog) ? std::string(file) : std::string();
}
#elif defined(__linux__)
/* The player's disc image, from the desktop's file dialog (zenity, GNOME's
   and most desktops'; kdialog, KDE's); "" if they cancel or there is none. */
std::string PickDisc() {
    const std::string title = "Choose your " PS2HOST_GAME_NAME " disc image";
    const std::string commands[] = {
        "zenity --file-selection --title=\"" + title + "\" --file-filter=\"Disc images | *.iso *.ISO *.cue *.CUE *.bin *.BIN\" 2>/dev/null",
        "kdialog --getopenfilename \"$HOME\" \"*.iso *.cue *.bin|Disc images\" --title \"" + title + "\" 2>/dev/null",
    };
    for (const std::string& command : commands) {
        const std::string tool = command.substr(0, command.find(' '));
        if (std::system(("command -v " + tool + " >/dev/null 2>&1").c_str()) != 0) continue;
        std::FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) continue;
        std::string path;
        char chunk[512];
        while (std::fgets(chunk, sizeof chunk, pipe)) path += chunk;
        pclose(pipe);
        while (!path.empty() && (path.back() == '\n' || path.back() == '\r')) path.pop_back();
        return path;
    }
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, PS2HOST_GAME_NAME,
                             "Put an image of your " PS2HOST_GAME_NAME " disc (.iso, or .cue with its .bin) in the "
                             "folder beside this program, or start it with --disc <image>.",
                             nullptr);
    return {};
}

/* Where a player's files go: beside the program, as on Windows (a folder
   unpacked from the release), else - where it may not write (installed for
   every user) - $XDG_DATA_HOME/<target>, ~/.local/share/<target>. */
std::filesystem::path PlayerFolder(const std::filesystem::path& program) {
    std::error_code error;
    if (access(program.c_str(), W_OK) == 0) return program;
    const char* data = std::getenv("XDG_DATA_HOME");
    const char* home = std::getenv("HOME");
    std::filesystem::path folder = data && *data ? std::filesystem::path(data)
                                   : std::filesystem::path(home ? home : ".") / ".local" / "share";
    folder /= PS2HOST_GAME_TARGET;
    std::filesystem::create_directories(folder, error);
    return folder;
}
#endif
#endif

/* What the run says, to `log` (no terminal to read): opened once, shared, and
   given to both streams - their lines in the order they were written. */
void LogTo(const std::string& log) {
#ifdef _WIN32
    std::FILE* stream = nullptr;
    freopen_s(&stream, "NUL", "w", stdout);    // streams with descriptors, in a windowed program
    freopen_s(&stream, "NUL", "w", stderr);
    const int file = _open(log.c_str(), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_TEXT, _S_IREAD | _S_IWRITE);
    if (file >= 0) {
        _dup2(file, _fileno(stdout));
        _dup2(file, _fileno(stderr));
        _close(file);
    }
#else
    const int file = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (file >= 0) {
        dup2(file, fileno(stdout));
        dup2(file, fileno(stderr));
        close(file);
    }
#endif
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // a run ended by force loses nothing
    std::setvbuf(stderr, nullptr, _IONBF, 0);
}

#if defined(__ANDROID__) && defined(PS2HOST_GAME_EXECUTABLE)
/* Android's: as PlayArguments below, in the app's folder `home` (the
   activity's --android-home), the disc image the activity's. */
std::vector<std::string> AndroidPlayArguments(const char* self, const std::string& home) {
    const std::filesystem::path here(home);
    std::error_code error;
    std::filesystem::create_directories(here, error);
    std::filesystem::current_path(here, error);
    ps2host::LoadSettings((here / "pc_settings.ini").string());
    g_log = std::string(PS2HOST_GAME_TARGET) + ".log";
    LogTo((here / g_log).string());
    std::string disc = PickDisc("");
    while (!disc.empty() && !IsGameDisc(disc)) {
        std::fprintf(stderr, "[port] %s is not an image of the game's disc (%s)\n", disc.c_str(),
                     ps2host::Disc().Open(disc) ? "another disc" : "cannot be read");
        disc = PickDisc(disc);
    }
    if (disc.empty()) return {};
    std::fprintf(stderr, "[port] disc %s\n", disc.c_str());
    std::vector<std::string> args = {self, "--disc", disc, "--memcard", (here / "memcard").string(),
                                     "--window", "--gpu", "--audio", "--timeout", "315360000"};  // ten years
    // A test's options (a scripted run's, port/runs/<run>.args), pushed with
    // adb to args.txt beside the settings: added after these, so they win
    // (the last --memcard is the one taken).
    if (std::FILE* file = std::fopen((here / "args.txt").string().c_str(), "r")) {
        char word[1024];
        while (std::fscanf(file, "%1023s", word) == 1) args.push_back(word);
        std::fclose(file);
        std::fprintf(stderr, "[port] args.txt: %zu options added\n", args.size() - 10);
    }
    return args;
}
#endif

/* Started without arguments - from Explorer, a shortcut - the port plays its
   game from the player's own disc image: the one chosen before (kept in the
   PC settings, beside the executable), or asked for and checked to be the
   game's disc; the memory card a directory beside it too. In a window, with
   sound, on the GPU, with no time limit (the default is for test runs).
   An empty list: the player gave none. */
std::vector<std::string> PlayArguments(const char* self) {
#if defined(PS2HOST_GAME_EXECUTABLE) && (defined(_WIN32) || (defined(__linux__) && !defined(__ANDROID__)))
#ifdef _WIN32
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    const std::filesystem::path here = std::filesystem::path(exe).parent_path();
#else
    std::error_code error;
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", error);
    const std::filesystem::path program = error ? std::filesystem::current_path() : exe.parent_path();
    const std::filesystem::path here = PlayerFolder(program);
#endif
    ps2host::LoadSettings((here / "pc_settings.ini").string());     // where --memcard puts them
    std::string disc = ps2host::Setting("disc");
    // None chosen yet, or gone: an image of the game's disc beside the
    // executable is taken without asking.
    if (!IsGameDisc(disc)) {
        std::error_code missing;
        for (const auto& entry : std::filesystem::directory_iterator(here, missing)) {
            std::string ext = entry.path().extension().string();
            for (char& ch : ext) ch = char(tolower(static_cast<unsigned char>(ch)));
            if ((ext == ".iso" || ext == ".bin" || ext == ".cue") && IsGameDisc(entry.path().string())) {
                disc = entry.path().string();
                break;
            }
        }
    }
    while (!IsGameDisc(disc)) {
#ifdef _WIN32
        const bool there = !disc.empty() && GetFileAttributesA(disc.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
        std::error_code missing;
        const bool there = !disc.empty() && std::filesystem::exists(disc, missing);
#endif
        if (there) {
            const std::string text = "This is not an image of the " PS2HOST_GAME_NAME " disc "
                                     "(" PS2HOST_GAME_EXECUTABLE "):\n\n" + disc;
#ifdef _WIN32
            MessageBoxA(nullptr, text.c_str(), PS2HOST_GAME_NAME, MB_OK | MB_ICONWARNING);
#else
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_WARNING, PS2HOST_GAME_NAME, text.c_str(), nullptr);
#endif
        }
        disc = PickDisc();
        if (disc.empty()) return {};
    }
    if (disc != ps2host::Setting("disc")) ps2host::SetSetting("disc", disc);
    // No terminal to read: what the run says goes to a log beside it. Opened
    // once, shared (fopen's would lock the file to one stream), and given to
    // both: their lines in the order they were written.
    g_log = std::filesystem::path(exe).stem().string() + ".log";     // (on Linux: the program's name, .log)
    LogTo((here / g_log).string());
    std::fprintf(stderr, "[port] disc %s\n", disc.c_str());
    return {self, "--disc", disc, "--memcard", (here / "memcard").string(), "--window", "--gpu",
            "--audio", "--timeout", "315360000"};     // ten years
#else
    return {self};
#endif
}

}  // namespace

/* A port built as a windowed program has no console of its own. Started from
   a terminal its output goes to that one; redirected (a file, a pipe: the
   tools' runs), where it was sent. */
void AttachTerminal() {
#ifdef _WIN32
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out != nullptr && out != INVALID_HANDLE_VALUE) return;
    if (!AttachConsole(ATTACH_PARENT_PROCESS)) return;
    std::FILE* stream = nullptr;
    freopen_s(&stream, "CONOUT$", "w", stdout);
    freopen_s(&stream, "CONOUT$", "w", stderr);
#endif
}

/* A message for the player (a run without a terminal). */
void Tell(const std::string& text, bool error) {
#ifdef _WIN32
    MessageBoxA(nullptr, text.c_str(), ps2host::GameTitle().empty() ? "PS2 port" : ps2host::GameTitle().c_str(),
                MB_OK | (error ? MB_ICONERROR : MB_ICONWARNING));
#elif defined(__ANDROID__)
    SDL_ShowSimpleMessageBox(error ? SDL_MESSAGEBOX_ERROR : SDL_MESSAGEBOX_WARNING,
                             ps2host::GameTitle().empty() ? "PS2 port" : ps2host::GameTitle().c_str(), text.c_str(),
                             nullptr);
#else
    std::fprintf(stderr, "%s\n", text.c_str());
#endif
}

#ifdef __ANDROID__
/* The activity's on-screen controls (PortActivity.nativeTouchPad /
   nativeTouchShown): one more controller (ps2gpu::SetTouchPad). */
extern "C" JNIEXPORT void JNICALL Java_org_ps2port_PortActivity_nativeTouchPad(JNIEnv*, jclass, jint buttons,
                                                                               jint lx, jint ly, jint rx, jint ry) {
    ps2gpu::Display::PadState state;
    state.buttons = std::uint16_t(buttons);
    state.sticks[0] = std::uint8_t(rx), state.sticks[1] = std::uint8_t(ry);
    state.sticks[2] = std::uint8_t(lx), state.sticks[3] = std::uint8_t(ly);
    state.l2 = (buttons & (1 << 8)) ? 255 : 0;          // L2, R2: a trigger all the way
    state.r2 = (buttons & (1 << 9)) ? 255 : 0;
    ps2gpu::SetTouchPad(state);
}
extern "C" JNIEXPORT void JNICALL Java_org_ps2port_PortActivity_nativeTouchShown(JNIEnv*, jclass, jboolean shown) {
    ps2gpu::ShowTouchPad(shown != JNI_FALSE);
}
#endif

int main(int argc, char** argv) {
    AttachTerminal();
#ifndef _WIN32
    // A send to a peer that went away (a LAN player leaving) an error, not
    // the end of the process.
    std::signal(SIGPIPE, SIG_IGN);
#endif
    std::vector<std::string> play;
    std::vector<char*> play_argv;
#ifdef __ANDROID__
    if (argc == 3 && std::string(argv[1]) == "--android-home") {
#ifdef PS2HOST_GAME_EXECUTABLE
        play = AndroidPlayArguments(argv[0], argv[2]);
#endif
#ifdef PS2HOST_PGO_GENERATE
        // The counts beside the settings, a file a run (android.py pgo pulls them).
        static const std::string counts = std::string(argv[2]) + "/pgo-%p.profraw";
        __llvm_profile_set_filename(counts.c_str());
#endif
        if (play.empty()) return 0;
        for (std::string& arg : play) play_argv.push_back(arg.data());
        play_argv.push_back(nullptr);
        argc = int(play.size());
        argv = play_argv.data();
    }
#endif
    if (argc == 1) {
        play = PlayArguments(argv[0]);
        if (play.empty()) return 0;
        for (std::string& arg : play) play_argv.push_back(arg.data());
        play_argv.push_back(nullptr);
        argc = int(play.size());
        argv = play_argv.data();
    }
    if (!HasAvx2()) {
        std::fprintf(stderr, "This build needs a processor with AVX2 (Intel Haswell, AMD Excavator or later).\n");
        return 70;
    }
#ifdef PS2HOST_GAME_TITLE
    ps2host::SetGameTitle(PS2HOST_GAME_TITLE);      // the window's (game.json's "title")
#endif
#ifdef PS2HOST_GAME_VERSION
    std::fprintf(stderr, "[port] %s, version %s\n", PS2HOST_GAME_TITLE, PS2HOST_GAME_VERSION);
#endif
    ps2host::RegisterFunctions(kGuestFunctions, kGuestFunctionCount);
    ps2host::RegisterOverlays(kGuestOverlays, kGuestOverlayCount);
    RegisterGameOverrides();
    const int code = ps2host::Main(argc, argv);
    if (!play.empty() && code != 0) {         // closing the window ends a run with 0
        std::fflush(nullptr);
        const std::string text = "The game stopped because of an error. What happened is in " +
                                 g_log + ", beside the game.";
        Tell(text, true);
    }
    // What must be finished is (the report written, the settings and the
    // memory card saved as they changed, the WAV's header here); the rest -
    // the window, the GPU device, the threads - goes with the process.
    // Taken down one by one at exit, the Vulkan device's could hang
    // (vkDestroyDevice with the driver's threads and the renderer's images
    // still about), leaving a window that never closed.
    ps2host::CloseAudio();
    ps2host::ReleaseController();
    std::fflush(nullptr);
#ifdef PS2HOST_PGO_GENERATE
    __llvm_profile_write_file();            // (the profile's own writer runs at exit, which this skips)
#endif
#ifdef _WIN32
    TerminateProcess(GetCurrentProcess(), UINT(code));
#else
    std::_Exit(code);
#endif
    return code;
}

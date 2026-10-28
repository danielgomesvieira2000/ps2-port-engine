/*
 * What the platform the port runs on has (ps2host/run.h): a PC's window and
 * keyboard, or a phone's touch controls - for a port's settings, which show
 * only the rows that mean something there. Android's through the activity
 * (PortActivity's methods, over JNI).
 */

#include "ps2host/run.h"

#ifdef __ANDROID__
#include <jni.h>
#include <SDL3/SDL.h>
#include "ps2gpu/display.h"

#include <atomic>
#include <thread>
#endif

namespace ps2host {

#ifdef __ANDROID__
namespace {
std::atomic<bool> g_keyboard{false};    // the activity's nativeKeyboard

/* The activity's setTouchShown(shown), from a thread of its own: the game's
   run on fibers, whose stacks ART takes for an overflow (a StackOverflowError
   left pending, and the next call into Java aborts). */
void TellActivity(bool shown) {
    std::thread([shown] {
        JNIEnv* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
        jobject activity = static_cast<jobject>(SDL_GetAndroidActivity());
        if (!env || !activity) return;
        jclass type = env->GetObjectClass(activity);
        jmethodID method = env->GetMethodID(type, "setTouchShown", "(Z)V");
        if (method) env->CallVoidMethod(activity, method, jboolean(shown));
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->DeleteLocalRef(type);
        env->DeleteLocalRef(activity);
    }).join();
}
}  // namespace

bool OnPhone() { return true; }
bool KeyboardPresent() { return g_keyboard.load(std::memory_order_relaxed); }
bool TouchControlsShown() { return ps2gpu::TouchPadShown(); }
void SetTouchControls(bool shown) {
    ps2gpu::ShowTouchPad(shown);        // at once: the activity's own note comes from its thread
    TellActivity(shown);
}
#else
bool OnPhone() { return false; }
bool KeyboardPresent() { return true; }
bool TouchControlsShown() { return false; }
void SetTouchControls(bool) {}
#endif

}  // namespace ps2host

#ifdef __ANDROID__
extern "C" JNIEXPORT void JNICALL Java_org_ps2port_PortActivity_nativeKeyboard(JNIEnv*, jclass, jboolean present) {
    ps2host::g_keyboard.store(present != JNI_FALSE, std::memory_order_relaxed);
}
#endif

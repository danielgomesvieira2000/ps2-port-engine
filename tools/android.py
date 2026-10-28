#!/usr/bin/env python3
"""Android: a port as an APK - the game's library for each processor, SDL's
activity with the engine's (engine/android: the disc image picked in the
system's file picker, on-screen controls), signed.

    python engine/tools/android.py --game games/<name> build     # the libraries
    python engine/tools/android.py --game games/<name> apk       # ...and the APK
    python engine/tools/android.py --game games/<name> install   # ...onto a device (adb)

No Gradle: CMake with the NDK's toolchain builds lib<target>.so for each ABI
(--abi, default arm64-v8a; x86_64 for the emulator), javac and d8 make the
dex from SDL's Java (its source tree, from the PC build's FetchContent) and
engine/android/java, aapt2 links the manifest and the icon, zipalign and
apksigner finish it: build/android/<Name>-<version>-android.apk.

Needs the Android SDK (ANDROID_HOME, else %LOCALAPPDATA%/Android/Sdk) with an
NDK, a platform and build-tools; a JDK (JAVA_HOME, else Program Files/Java);
and the PC build first (`port.py build`: its glslang compiles the shaders,
its _deps hold SDL's sources). The APK is signed with a key of the machine's
own (PS2_ANDROID_KEYSTORE, else ~/.android/ps2port-release.jks, made the
first time; PS2_ANDROID_KEYSTORE_PASS its password): an update installs over
an earlier APK only when signed with the same key, so keep that file.

game.json may name "android_package" (the app's id; else
io.github.ps2port.<target>) and "android_label" (else its "title").
"""

import argparse
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from ps2decomp import game as game_config  # noqa: E402
import port  # noqa: E402

ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_SDK = 29        # Android 10: Vulkan 1.1 at least in every device's driver (1.3 checked at run time)
TARGET_SDK = 35


def sdk_root() -> str:
    for path in (os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk")):
        if path and os.path.isdir(path):
            return path
    raise SystemExit("android: no Android SDK (set ANDROID_HOME)")


def newest(pattern: str, what: str) -> str:
    found = sorted(glob.glob(pattern), key=lambda p: [int(n) for n in re.findall(r"\d+", os.path.basename(p))])
    if not found:
        raise SystemExit("android: no %s (%s)" % (what, pattern))
    return found[-1]


def java_home() -> str:
    home = os.environ.get("JAVA_HOME")
    if home and os.path.isfile(os.path.join(home, "bin", "javac.exe" if os.name == "nt" else "javac")):
        return home
    return newest(os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Java", "jdk*"), "JDK")


def tools():
    sdk = sdk_root()
    build_tools = newest(os.path.join(sdk, "build-tools", "*"), "build-tools")
    exe = ".exe" if os.name == "nt" else ""
    bat = ".bat" if os.name == "nt" else ""
    jdk = java_home()
    return {
        "sdk": sdk,
        "ndk": newest(os.path.join(sdk, "ndk", "*"), "NDK"),
        "android_jar": os.path.join(newest(os.path.join(sdk, "platforms", "android-*"), "platform"), "android.jar"),
        "aapt2": os.path.join(build_tools, "aapt2" + exe),
        "d8": os.path.join(build_tools, "d8" + bat),
        "zipalign": os.path.join(build_tools, "zipalign" + exe),
        "apksigner": os.path.join(build_tools, "apksigner" + bat),
        "adb": os.path.join(sdk, "platform-tools", "adb" + exe),
        "javac": os.path.join(jdk, "bin", "javac" + exe),
        "keytool": os.path.join(jdk, "bin", "keytool" + exe),
        "jdk": jdk,
    }


def run(command, env=None, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, env=env, **kwargs)
    if result.returncode:
        out = (result.stdout + result.stderr).strip().splitlines()
        errors = [line for line in out if "error" in line.lower()] or out[-40:]
        raise SystemExit("android: %s failed:\n%s" % (os.path.basename(command[0]), "\n".join(errors[:60])))
    return result


def tool_env(t) -> dict:
    env = dict(os.environ)
    env["JAVA_HOME"] = t["jdk"]
    env["SKIP_JDK_VERSION_CHECK"] = "1"
    return env


def target_name(game) -> str:
    return os.path.basename(os.path.normpath(game.directory))


def android_package(game, settings: dict = None) -> str:
    """The app's id: game.json's android_package, else io.github.ps2port.<target>."""
    settings = port._game_json(game) if settings is None else settings
    return settings.get("android_package") or "io.github.ps2port." + re.sub(r"[^a-z0-9_]", "", target_name(game).lower())


def pc_build(game) -> str:
    return game.path(os.path.join(game.build, "port", "cmake"))


def pgo_profile(game) -> str:
    """The Android build's profile (android.py pgo), where one was made."""
    return game.path(os.path.join(game.build, "port", "android-pgo.profdata"))


def build_library(game, generated: str, abi: str, t, pgo: str = "") -> str:
    """lib<target>.so for one ABI; returns its path. `pgo`: "generate" (a
    counting build, a folder of its own), else the profile of android.py pgo
    when there is one."""
    host_glslang = os.path.join(pc_build(game), "_deps", "glslang-build", "StandAlone",
                                "glslang.exe" if os.name == "nt" else "glslang")
    if not os.path.isfile(host_glslang):
        raise SystemExit("android: the PC build first (port.py build): its glslang compiles the shaders")
    binary = game.path(os.path.join(game.build, "port", "android-" + {"arm64-v8a": "arm64"}.get(abi, abi)
                                    + ("-pgo-generate" if pgo == "generate" else "")))
    if not pgo and os.path.isfile(pgo_profile(game)):
        pgo = os.path.abspath(pgo_profile(game))
    os.makedirs(binary, exist_ok=True)
    cmake = port._visual_studio_tool("CMake", "bin", "cmake.exe") if os.name == "nt" else "cmake"
    ninja = port._visual_studio_tool("Ninja", "ninja.exe") if os.name == "nt" else "ninja"
    slash = lambda p: p.replace("\\", "/")
    run([cmake, "-S", game.path("port"), "-B", binary, "-G", "Ninja",
         "-DCMAKE_MAKE_PROGRAM=" + slash(ninja),
         "-DCMAKE_TOOLCHAIN_FILE=" + slash(os.path.join(t["ndk"], "build", "cmake", "android.toolchain.cmake")),
         "-DANDROID_ABI=" + abi, "-DANDROID_PLATFORM=android-%d" % MIN_SDK,
         "-DCMAKE_BUILD_TYPE=Release", "-DPS2_GENERATED_DIR=" + slash(generated),
         "-DPS2HOST_AVX2=OFF", "-DPS2_HOST_GLSLANG=" + slash(host_glslang),
         "-DPS2HOST_LTO_JOBS=%d" % port._build_jobs(), "-DPS2HOST_PGO=" + slash(pgo)])
    run([cmake, "--build", binary, "--target", target_name(game), "-j", str(port._build_jobs())])
    library = os.path.join(binary, "lib%s.so" % target_name(game))
    if not os.path.isfile(library):
        raise SystemExit("android: %s was not built" % library)
    # Symbols off (the release's size: 3-4 times less), kept beside it for a crash's stack.
    strip = newest(os.path.join(t["ndk"], "toolchains", "llvm", "prebuilt", "*", "bin",
                                "llvm-strip" + (".exe" if os.name == "nt" else "")), "llvm-strip")
    stripped = library[:-3] + ".stripped.so"
    run([strip, "--strip-unneeded", "-o", stripped, library])
    print("built %s (%s, %.1f MB stripped)" % (library, abi, os.path.getsize(stripped) / 1e6))
    return stripped


def version_code(version: str) -> int:
    numbers = [int(n) for n in re.findall(r"\d+", version)[:3]] + [0, 0, 0]
    return numbers[0] * 1000000 + numbers[1] * 10000 + numbers[2] * 100


def manifest(game, settings: dict) -> str:
    title = settings.get("title") or settings.get("name") or target_name(game)
    label = settings.get("android_label") or title
    package = android_package(game, settings)
    escape = lambda s: s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    return """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="{package}"
    android:installLocation="auto">
    <!-- Vulkan 1.3 (the renderer's dynamic rendering, synchronization2) -->
    <uses-feature android:name="android.hardware.vulkan.version" android:version="0x403000" android:required="true" />
    <uses-feature android:name="android.hardware.touchscreen" android:required="false" />
    <uses-feature android:name="android.hardware.gamepad" android:required="false" />
    <uses-feature android:name="android.hardware.usb.host" android:required="false" />
    <uses-feature android:name="android.hardware.bluetooth" android:required="false" />
    <uses-feature android:name="android.hardware.type.pc" android:required="false" />
    <uses-permission android:name="android.permission.VIBRATE" />
    <!-- LAN play -->
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
    <uses-permission android:name="android.permission.CHANGE_WIFI_MULTICAST_STATE" />
    <application android:label="{label}"
        android:icon="@mipmap/ic_launcher"
        android:allowBackup="false"
        android:appCategory="game"
        android:isGame="true"
        android:extractNativeLibs="false"
        android:hardwareAccelerated="true"
        android:theme="@android:style/Theme.NoTitleBar.Fullscreen">
        <meta-data android:name="ps2port.library" android:value="{library}" />
        <activity android:name="org.ps2port.PortActivity"
            android:label="{label}"
            android:exported="true"
            android:launchMode="singleTask"
            android:alwaysRetainTaskState="true"
            android:screenOrientation="sensorLandscape"
            android:resizeableActivity="false"
            android:preferMinimalPostProcessing="true"
            android:configChanges="layoutDirection|locale|grammaticalGender|fontScale|fontWeightAdjustment|orientation|uiMode|screenLayout|screenSize|smallestScreenSize|keyboard|keyboardHidden|navigation">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
            <intent-filter>
                <action android:name="android.hardware.usb.action.USB_DEVICE_ATTACHED" />
            </intent-filter>
        </activity>
    </application>
</manifest>
""".format(package=package, label=escape(label), library=target_name(game))


def icon_png(game, size: int) -> bytes:
    """The port's own icon (port/icon.ico: never the disc-made one), as a PNG."""
    from PIL import Image
    image = Image.open(game.path(os.path.join("port", "icon.ico")))
    sizes = sorted(image.info.get("sizes", [image.size]), key=lambda s: -s[0])
    try:
        image.size = sizes[0]
    except Exception:
        pass
    image = image.convert("RGBA").resize((size, size), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def keystore(t) -> tuple:
    path = os.environ.get("PS2_ANDROID_KEYSTORE") or os.path.join(os.path.expanduser("~"), ".android",
                                                                  "ps2port-release.jks")
    password = os.environ.get("PS2_ANDROID_KEYSTORE_PASS", "ps2port-android")
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        run([t["keytool"], "-genkeypair", "-keystore", path, "-alias", "ps2port", "-keyalg", "RSA", "-keysize",
             "4096", "-validity", "36500", "-storepass", password, "-keypass", password,
             "-dname", "CN=ps2-port-engine"])
        print("android: made the signing key %s - keep it: updates must be signed with it" % path)
    return path, password


def apk(game, libraries: dict, t) -> str:
    settings = port._game_json(game)
    version = settings.get("version", "0.0.0")
    work = game.path(os.path.join(game.build, "android"))
    shutil.rmtree(os.path.join(work, "stage"), ignore_errors=True)
    stage = os.path.join(work, "stage")
    os.makedirs(stage)
    # Java: SDL's activity (its sources as the PC build fetched them) and the engine's.
    sdl_java = os.path.join(pc_build(game), "_deps", "sdl3-src", "android-project", "app", "src", "main", "java")
    if not os.path.isdir(sdl_java):
        raise SystemExit("android: no SDL sources in %s (the PC build first)" % sdl_java)
    sources = glob.glob(os.path.join(sdl_java, "**", "*.java"), recursive=True) + \
        glob.glob(os.path.join(ENGINE, "android", "java", "**", "*.java"), recursive=True)
    classes = os.path.join(stage, "classes")
    os.makedirs(classes)
    listing = os.path.join(stage, "sources.txt")
    with open(listing, "w", encoding="utf-8") as handle:
        handle.write("\n".join('"%s"' % s.replace("\\", "/") for s in sources))
    run([t["javac"], "--release", "11", "-nowarn", "-encoding", "UTF-8", "-classpath", t["android_jar"],
         "-d", classes, "@" + listing], env=tool_env(t))
    class_files = glob.glob(os.path.join(classes, "**", "*.class"), recursive=True)
    run([t["d8"], "--release", "--min-api", str(MIN_SDK), "--lib", t["android_jar"], "--output", stage]
        + class_files, env=tool_env(t))
    # Resources: the icon. The manifest.
    res = os.path.join(stage, "res", "mipmap-xxxhdpi")
    os.makedirs(res)
    with open(os.path.join(res, "ic_launcher.png"), "wb") as handle:
        handle.write(icon_png(game, 192))
    compiled = os.path.join(stage, "res.zip")
    run([t["aapt2"], "compile", "--dir", os.path.join(stage, "res"), "-o", compiled])
    with open(os.path.join(stage, "AndroidManifest.xml"), "w", encoding="utf-8") as handle:
        handle.write(manifest(game, settings))
    base = os.path.join(stage, "base.apk")
    run([t["aapt2"], "link", "-o", base, "-I", t["android_jar"], "--manifest",
         os.path.join(stage, "AndroidManifest.xml"), "--min-sdk-version", str(MIN_SDK),
         "--target-sdk-version", str(TARGET_SDK), "--version-code", str(version_code(version)),
         "--version-name", version, compiled])
    # The dex and the libraries: the libraries stored, not compressed (loaded
    # from the APK where they are: page-aligned by zipalign -P 16).
    unaligned = os.path.join(stage, "unaligned.apk")
    shutil.copyfile(base, unaligned)
    with zipfile.ZipFile(unaligned, "a") as z:
        z.write(os.path.join(stage, "classes.dex"), "classes.dex", compress_type=zipfile.ZIP_DEFLATED)
        for abi, library in libraries.items():
            z.write(library, "lib/%s/lib%s.so" % (abi, target_name(game)), compress_type=zipfile.ZIP_STORED)
    name = re.sub(r"[^A-Za-z0-9]+", "", settings.get("title", target_name(game)).split(":")[0]) or target_name(game)
    out = os.path.join(work, "%s-%s-android.apk" % (name, version))
    aligned = os.path.join(stage, "aligned.apk")
    run([t["zipalign"], "-f", "-P", "16", "4", unaligned, aligned])
    path, password = keystore(t)
    run([t["apksigner"], "sign", "--ks", path, "--ks-pass", "pass:" + password, "--ks-key-alias", "ps2port",
         "--out", out, aligned], env=tool_env(t))
    run([t["apksigner"], "verify", out], env=tool_env(t))
    print("apk: %s (%.1f MB; %s; version %s, code %d)" % (out, os.path.getsize(out) / 1e6, ", ".join(libraries),
                                                          version, version_code(version)))
    return out


PGO_RUNS = ("race", "crash")        # the scripted runs (port/runs) the profile is made from


def pgo(game, generated: str, t) -> str:
    """The profile-guided build's profile, made on the emulator: a counting
    x86_64 build installed, the scripted runs played with it (its args.txt),
    the counts pulled and merged (the NDK's llvm-profdata) into
    build/port/android-pgo.profdata, which the builds after use. The same
    sources for every processor, so the emulator's counts are a phone's.
    Needs the emulator (or a device) on adb."""
    import time
    library = build_library(game, generated, "x86_64", t, pgo="generate")
    package = apk(game, {"x86_64": library}, t)
    adb = t["adb"]
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    run([adb, "install", "-r", package], env=env)
    app = android_package(game)
    home = "/storage/emulated/0/Android/data/%s/files" % app
    raw = game.path(os.path.join(game.build, "port", "android-pgo"))
    shutil.rmtree(raw, ignore_errors=True)
    os.makedirs(raw)
    for name in PGO_RUNS:
        args = open(game.path(os.path.join("port", "runs", name + ".args"))).read().strip()
        listed = os.path.join(raw, "args.txt")
        with open(listed, "w", newline=chr(10)) as f:
            f.write(args + " --no-memcard" + chr(10))
        subprocess.run([adb, "shell", "am", "force-stop", app], env=env)
        subprocess.run([adb, "shell", "rm", "-f", home + "/" + target_name(game) + ".log", home + "/pgo-*.profraw"], env=env)
        run([adb, "push", listed, home + "/args.txt"], env=env)
        run([adb, "shell", "am", "start", "-n", app + "/org.ps2port.PortActivity"], env=env)
        print("pgo: %s on the device" % name, flush=True)
        began = time.time()
        while time.time() - began < 3 * 3600:
            time.sleep(20)
            alive = subprocess.run([adb, "shell", "pidof", app], capture_output=True, text=True, env=env).stdout.strip()
            if not alive:
                break
        listing = subprocess.run([adb, "shell", "ls", home], capture_output=True, text=True, env=env).stdout.split()
        for f in listing:
            if f.startswith("pgo-") and f.endswith(".profraw"):
                run([adb, "pull", home + "/" + f, os.path.join(raw, name + "-" + f)], env=env)
        subprocess.run([adb, "shell", "rm", "-f", home + "/args.txt"], env=env)
        print("pgo: %s done in %.0f s" % (name, time.time() - began), flush=True)
    counts = [os.path.join(raw, f) for f in sorted(os.listdir(raw)) if f.endswith(".profraw")]
    if not counts:
        raise SystemExit("pgo: no counts came back from the device")
    merge = newest(os.path.join(t["ndk"], "toolchains", "llvm", "prebuilt", "*", "bin",
                                "llvm-profdata" + (".exe" if os.name == "nt" else "")), "llvm-profdata")
    run([merge, "merge", "-o", pgo_profile(game)] + counts)
    print("pgo: %s from %d runs" % (pgo_profile(game), len(counts)))
    return pgo_profile(game)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("action", nargs="?", default="apk", choices=("build", "apk", "install", "pgo"))
    parser.add_argument("--abi", default="arm64-v8a",
                        help="ABIs, by commas: arm64-v8a (phones), x86_64 (the emulator)")
    args = parser.parse_args(argv)
    game = game_config.from_args(args)
    t = tools()
    generated = port.generate(game)
    if args.action == "pgo":
        pgo(game, generated, t)
        return 0
    libraries = {abi: build_library(game, generated, abi, t) for abi in args.abi.split(",")}
    if args.action == "build":
        return 0
    package = apk(game, libraries, t)
    if args.action == "install":
        run([t["adb"], "install", "-r", package])
        print("installed %s" % package)
    return 0


if __name__ == "__main__":
    sys.exit(main())

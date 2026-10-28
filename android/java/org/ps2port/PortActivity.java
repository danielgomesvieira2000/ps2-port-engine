/*
 * A port's Android activity: SDL's, with the player's disc image chosen in
 * the system's file picker (and kept: asked for once), the files a PC port
 * keeps beside its executable in the app's own folder, and on-screen
 * controls over the picture - a small button shows and hides them.
 *
 * The native side (engine/host/src/main.cpp, __ANDROID__) asks for the disc
 * with pickDisc() from its own thread, and is told what the touch controls
 * hold through nativeTouchPad().
 */
package org.ps2port;

import android.app.AlertDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import android.view.InputDevice;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import org.libsdl.app.SDLActivity;

import java.io.File;

public class PortActivity extends SDLActivity {
    private static final String TAG = "ps2port";
    private static final int PICK_DISC = 0x5032;
    private static final String KEY_DISC = "disc_uri";
    private static final String KEY_TOUCH = "touch_controls";

    private final Object picking = new Object();
    private Uri picked;                 // the picker's answer (null: cancelled)
    private boolean answered;
    private ParcelFileDescriptor disc;  // the image's, kept open while the game runs
    private boolean firstAsk = true;

    private TouchControls touch;

    /* The game's library (SDL built into it). */
    @Override
    protected String[] getLibraries() {
        String library = "main";
        try {
            ApplicationInfo info = getPackageManager().getApplicationInfo(getPackageName(), PackageManager.GET_META_DATA);
            if (info.metaData != null && info.metaData.getString("ps2port.library") != null)
                library = info.metaData.getString("ps2port.library");
        } catch (PackageManager.NameNotFoundException ignored) {
        }
        return new String[] {library};
    }

    /* The folder the port keeps its files in (pc_settings.ini, the memory
       card, the log, a texture pack, custom songs): the app's own on the
       shared storage, which a file manager or a PC over USB can reach. */
    @Override
    protected String[] getArguments() {
        File dir = getExternalFilesDir(null);
        if (dir == null) dir = getFilesDir();
        dir.mkdirs();
        return new String[] {"--android-home", dir.getAbsolutePath()};
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (mLayout == null) return;      // SDL refused to start (its own message shown)
        touch = new TouchControls(this);
        mLayout.addView(touch, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                                                          ViewGroup.LayoutParams.MATCH_PARENT));
        SharedPreferences prefs = getPreferences(MODE_PRIVATE);
        final boolean shown = prefs.getBoolean(KEY_TOUCH, !ControllerPresent());
        touch.setShown(shown);
        nativeTouchShown(shown);
        touch.setListener(new TouchControls.Listener() {
            @Override public void onPad(int buttons, int lx, int ly, int rx, int ry) {
                nativeTouchPad(buttons, lx, ly, rx, ry);
            }
            @Override public void onShownChanged(boolean shown) {
                getPreferences(MODE_PRIVATE).edit().putBoolean(KEY_TOUCH, shown).apply();
                nativeTouchPad(0, 0x80, 0x80, 0x80, 0x80);
                nativeTouchShown(shown);
            }
        });
    }

    /* Whether a game controller is connected (the touch controls start hidden then). */
    private static boolean ControllerPresent() {
        for (int id : InputDevice.getDeviceIds()) {
            InputDevice device = InputDevice.getDevice(id);
            if (device == null || device.isVirtual()) continue;
            int sources = device.getSources();
            if ((sources & InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD
                || (sources & InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK)
                return true;
        }
        return false;
    }

    /* Called by the native side, on its own thread: the disc image as a path
       it can open ("/proc/self/fd/N", the picked document's descriptor), ""
       when the player cancels. `rejected`: the last answer was not the
       game's disc (said to the player, and asked again). The image chosen
       before is answered without asking, while it can still be opened. */
    public String pickDisc(final String rejected, final String gameName) {
        SharedPreferences prefs = getPreferences(MODE_PRIVATE);
        if (rejected != null && !rejected.isEmpty()) {
            prefs.edit().remove(KEY_DISC).apply();
            closeDisc();
            runOnUiThread(() -> Toast.makeText(this, "This is not an image of the " + gameName + " disc.",
                                               Toast.LENGTH_LONG).show());
        } else if (firstAsk) {
            String saved = prefs.getString(KEY_DISC, null);
            if (saved != null) {
                String path = open(Uri.parse(saved));
                if (path != null) { firstAsk = false; return path; }
                prefs.edit().remove(KEY_DISC).apply();
            }
        }
        firstAsk = false;
        synchronized (picking) {
            answered = false;
            picked = null;
            runOnUiThread(() -> {
                if (rejected == null || rejected.isEmpty()) {
                    new AlertDialog.Builder(this)
                        .setTitle(gameName)
                        .setMessage("Choose your own " + gameName + " disc image (.iso) - it stays on your device; "
                                    + "the game reads it from there.")
                        .setCancelable(false)
                        .setPositiveButton("Choose", (d, w) -> launchPicker())
                        .setNegativeButton("Quit", (d, w) -> answer(null))
                        .show();
                } else {
                    launchPicker();
                }
            });
            while (!answered) {
                try { picking.wait(); } catch (InterruptedException e) { return ""; }
            }
        }
        if (picked == null) return "";
        try {
            getContentResolver().takePersistableUriPermission(picked, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (SecurityException e) {
            Log.w(TAG, "no lasting permission for " + picked + ": " + e);
        }
        String path = open(picked);
        if (path == null) return pickDisc("", gameName);
        prefs.edit().putString(KEY_DISC, picked.toString()).apply();
        return path;
    }

    private void launchPicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        try {
            startActivityForResult(intent, PICK_DISC);
        } catch (Exception e) {
            Toast.makeText(this, "No file picker on this device.", Toast.LENGTH_LONG).show();
            answer(null);
        }
    }

    private void answer(Uri uri) {
        synchronized (picking) {
            picked = uri;
            answered = true;
            picking.notifyAll();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == PICK_DISC) {
            answer(resultCode == RESULT_OK && data != null ? data.getData() : null);
            return;
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    /* The document opened for reading, as a path to its descriptor. */
    private String open(Uri uri) {
        closeDisc();
        try {
            disc = getContentResolver().openFileDescriptor(uri, "r");
            if (disc == null) return null;
            Log.i(TAG, "disc " + uri + " open as fd " + disc.getFd() + ", " + disc.getStatSize() + " bytes");
            return "fd:" + disc.getFd();
        } catch (Exception e) {
            Log.w(TAG, "cannot open " + uri + ": " + e);
            return null;
        }
    }

    private void closeDisc() {
        if (disc == null) return;
        try { disc.close(); } catch (Exception ignored) { }
        disc = null;
    }

    /* The touch controls' state: libpad's button bits (Select L3 R3 Start Up
       Right Down Left L2 R2 L1 R1 Triangle Circle Cross Square), the left
       and right sticks 0-255 (0x80 centred). */
    public static native void nativeTouchPad(int buttons, int lx, int ly, int rx, int ry);
    /* Whether they are shown (a controller among the game's then). */
    public static native void nativeTouchShown(boolean shown);
}

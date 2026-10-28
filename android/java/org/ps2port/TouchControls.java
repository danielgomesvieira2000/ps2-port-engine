/*
 * On-screen controls over the game's picture: a DualShock 2's - the d-pad,
 * both sticks, the four face buttons, L1 L2 R1 R2, Select and Start - drawn
 * by the system's compositor over SDL's surface (the game's renderer never
 * sees them) and redrawn only when a touch changes them. A small button at
 * the top shows and hides them; hidden, every touch but that button's goes
 * through to the game (SDL's touch and mouse events).
 */
package org.ps2port;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.view.HapticFeedbackConstants;
import android.view.MotionEvent;
import android.view.View;

public class TouchControls extends View {
    public interface Listener {
        void onPad(int buttons, int lx, int ly, int rx, int ry);
        void onShownChanged(boolean shown);
    }

    // libpad's bits.
    static final int SELECT = 1 << 0, L3 = 1 << 1, R3 = 1 << 2, START = 1 << 3;
    static final int UP = 1 << 4, RIGHT = 1 << 5, DOWN = 1 << 6, LEFT = 1 << 7;
    static final int L2 = 1 << 8, R2 = 1 << 9, L1 = 1 << 10, R1 = 1 << 11;
    static final int TRIANGLE = 1 << 12, CIRCLE = 1 << 13, CROSS = 1 << 14, SQUARE = 1 << 15;

    /* A round or a pill-shaped button: its centre and size as fractions of
       the view's height (x from the left or, `right`, from the right edge). */
    private static final class Button {
        final int bit; final String label; final boolean right; final float fx, fy, fw, fh;
        float cx, cy, w, h;
        Button(int bit, String label, boolean right, float fx, float fy, float fw, float fh) {
            this.bit = bit; this.label = label; this.right = right; this.fx = fx; this.fy = fy; this.fw = fw; this.fh = fh;
        }
        boolean hit(float x, float y, float slack) {
            return Math.abs(x - cx) <= w / 2 * slack && Math.abs(y - cy) <= h / 2 * slack;
        }
    }

    private final Button[] buttons = {
        new Button(L2, "L2", false, 0.17f, 0.08f, 0.26f, 0.11f),
        new Button(L1, "L1", false, 0.17f, 0.22f, 0.26f, 0.11f),
        new Button(R2, "R2", true, 0.17f, 0.08f, 0.26f, 0.11f),
        new Button(R1, "R1", true, 0.17f, 0.22f, 0.26f, 0.11f),
        new Button(TRIANGLE, "", true, 0.30f, 0.43f, 0.15f, 0.15f),
        new Button(CIRCLE, "", true, 0.15f, 0.58f, 0.15f, 0.15f),
        new Button(CROSS, "", true, 0.30f, 0.73f, 0.15f, 0.15f),
        new Button(SQUARE, "", true, 0.45f, 0.58f, 0.15f, 0.15f),
        new Button(SELECT, "SELECT", false, 0.0f, 0.93f, 0.20f, 0.08f),     // (placed from the middle: layout())
        new Button(START, "START", false, 0.0f, 0.93f, 0.20f, 0.08f),
    };

    /* A stick: its centre (fractions of the height, `right` from the right
       edge), its reach, and the touch that holds it. */
    private static final class Stick {
        final boolean right; final float fx, fy, fr;
        float cx, cy, r, dx, dy;
        int pointer = -1;
        Stick(boolean right, float fx, float fy, float fr) { this.right = right; this.fx = fx; this.fy = fy; this.fr = fr; }
        int x() { return Math.max(0, Math.min(255, Math.round(128 + 127 * dx / r))); }
        int y() { return Math.max(0, Math.min(255, Math.round(128 + 127 * dy / r))); }
    }
    private final Stick left = new Stick(false, 0.62f, 0.72f, 0.17f);
    private final Stick rightStick = new Stick(true, 0.70f, 0.84f, 0.11f);

    // The d-pad: its centre and reach.
    private float padX, padY, padR;
    // The show/hide button.
    private final RectF toggle = new RectF();

    private final Paint fill = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint line = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint text = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Path path = new Path();

    private boolean shown = true;
    private int held;                   // the buttons held
    private Listener listener;

    public TouchControls(Context context) {
        super(context);
        line.setStyle(Paint.Style.STROKE);
        text.setTextAlign(Paint.Align.CENTER);
        text.setFakeBoldText(true);
        setFocusable(false);
    }

    public void setListener(Listener listener) { this.listener = listener; }

    public void setShown(boolean on) {
        shown = on;
        held = 0;
        left.pointer = rightStick.pointer = -1;
        left.dx = left.dy = rightStick.dx = rightStick.dy = 0;
        invalidate();
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        final float unit = Math.min(h, w * 0.6f);
        for (Button b : buttons) {
            b.w = b.fw * unit; b.h = b.fh * unit;
            b.cx = b.right ? w - b.fx * unit : b.fx * unit;
            b.cy = b.fy * h;
        }
        buttons[8].cx = w / 2f - 0.13f * unit;      // SELECT, START either side of the middle
        buttons[9].cx = w / 2f + 0.13f * unit;
        for (Stick s : new Stick[] {left, rightStick}) {
            s.r = s.fr * unit;
            s.cx = s.right ? w - s.fx * unit : s.fx * unit;
            s.cy = s.fy * h;
        }
        padR = 0.15f * unit;
        padX = 0.22f * unit;
        padY = 0.50f * h;
        final float tw = 0.16f * unit, th = 0.07f * unit;
        toggle.set(w / 2f - tw / 2, 0.012f * h, w / 2f + tw / 2, 0.012f * h + th);
        line.setStrokeWidth(Math.max(2f, unit * 0.006f));
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        final int action = e.getActionMasked();
        if (action == MotionEvent.ACTION_DOWN || action == MotionEvent.ACTION_POINTER_DOWN) {
            final int i = e.getActionIndex();
            if (toggle.contains(e.getX(i), e.getY(i))) {
                if (action == MotionEvent.ACTION_DOWN || shown) {
                    setShown(!shown);
                    performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY);
                    if (listener != null) listener.onShownChanged(shown);
                }
                return true;
            }
            if (!shown) return false;       // to the game
            // A first touch on none of the controls is the game's too (the
            // menus take a tap as the mouse's click).
            if (action == MotionEvent.ACTION_DOWN && !OnControl(e.getX(i), e.getY(i))) return false;
        }
        if (!shown) return false;
        // The sticks follow the touch that took them; every other touch
        // presses what it is over (sliding from one button to the next).
        if (action == MotionEvent.ACTION_DOWN || action == MotionEvent.ACTION_POINTER_DOWN) {
            final int i = e.getActionIndex();
            for (Stick s : new Stick[] {left, rightStick})
                if (s.pointer < 0 && Math.hypot(e.getX(i) - s.cx, e.getY(i) - s.cy) <= s.r * 1.25f)
                    s.pointer = e.getPointerId(i);
        }
        if (action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_POINTER_UP
            || action == MotionEvent.ACTION_CANCEL) {
            final int gone = action == MotionEvent.ACTION_CANCEL ? -2 : e.getPointerId(e.getActionIndex());
            for (Stick s : new Stick[] {left, rightStick})
                if (s.pointer == gone || gone == -2) { s.pointer = -1; s.dx = s.dy = 0; }
        }
        int now = 0;
        for (int i = 0; i < e.getPointerCount(); ++i) {
            final int id = e.getPointerId(i);
            final boolean lifting = (action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_POINTER_UP)
                                    && i == e.getActionIndex();
            if (lifting || action == MotionEvent.ACTION_CANCEL) continue;
            final float x = e.getX(i), y = e.getY(i);
            Stick taken = id == left.pointer ? left : id == rightStick.pointer ? rightStick : null;
            if (taken != null) {
                float dx = x - taken.cx, dy = y - taken.cy;
                final float d = (float) Math.hypot(dx, dy);
                if (d > taken.r) { dx *= taken.r / d; dy *= taken.r / d; }
                taken.dx = dx; taken.dy = dy;
                continue;
            }
            now |= DpadAt(x, y);
            for (Button b : buttons)
                if (b.hit(x, y, 1.15f)) now |= b.bit;
        }
        if ((now & ~held) != 0) performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY);
        held = now;
        if (listener != null) listener.onPad(held, left.x(), left.y(), rightStick.x(), rightStick.y());
        invalidate();
        return true;
    }

    /* Whether a touch there is on one of the controls (or near enough). */
    private boolean OnControl(float x, float y) {
        if (Math.hypot(x - padX, y - padY) <= padR * 1.3) return true;
        for (Stick s : new Stick[] {left, rightStick})
            if (Math.hypot(x - s.cx, y - s.cy) <= s.r * 1.25f) return true;
        for (Button b : buttons)
            if (b.hit(x, y, 1.15f)) return true;
        return false;
    }

    /* The d-pad's directions under a touch: eight sectors, the diagonals two. */
    private int DpadAt(float x, float y) {
        final float dx = x - padX, dy = y - padY;
        final double d = Math.hypot(dx, dy);
        if (d > padR * 1.3 || d < padR * 0.18) return 0;
        final double angle = Math.toDegrees(Math.atan2(-dy, dx));      // 0 right, 90 up
        int bits = 0;
        if (angle > 22.5 && angle < 157.5) bits |= UP;
        if (angle < -22.5 && angle > -157.5) bits |= DOWN;
        if (angle > -67.5 && angle < 67.5) bits |= RIGHT;
        if (angle > 112.5 || angle < -112.5) bits |= LEFT;
        return bits;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        // The show/hide button, always.
        fill.setColor(0x60000000);
        final float round = toggle.height() / 2;
        canvas.drawRoundRect(toggle, round, round, fill);
        line.setColor(shown ? 0xB0FFFFFF : 0x70FFFFFF);
        canvas.drawRoundRect(toggle, round, round, line);
        DrawPadGlyph(canvas, toggle.centerX(), toggle.centerY(), toggle.height() * 0.30f, shown ? 0xD0FFFFFF : 0x80FFFFFF);
        if (!shown) return;

        // The d-pad.
        final float arm = padR * 0.36f;
        for (int k = 0; k < 4; ++k) {
            final int bit = k == 0 ? UP : k == 1 ? RIGHT : k == 2 ? DOWN : LEFT;
            canvas.save();
            canvas.rotate(90 * k, padX, padY);
            path.reset();
            path.moveTo(padX - arm, padY - padR);
            path.lineTo(padX + arm, padY - padR);
            path.lineTo(padX + arm, padY - arm * 0.9f);
            path.lineTo(padX, padY - arm * 0.1f);
            path.lineTo(padX - arm, padY - arm * 0.9f);
            path.close();
            fill.setColor((held & bit) != 0 ? 0x90FFFFFF : 0x40FFFFFF);
            canvas.drawPath(path, fill);
            line.setColor(0x90FFFFFF);
            canvas.drawPath(path, line);
            canvas.restore();
        }

        // The sticks.
        for (Stick s : new Stick[] {left, rightStick}) {
            fill.setColor(0x30FFFFFF);
            canvas.drawCircle(s.cx, s.cy, s.r, fill);
            line.setColor(0x80FFFFFF);
            canvas.drawCircle(s.cx, s.cy, s.r, line);
            fill.setColor(s.pointer >= 0 ? 0xA0FFFFFF : 0x70FFFFFF);
            canvas.drawCircle(s.cx + s.dx, s.cy + s.dy, s.r * 0.45f, fill);
        }

        // The buttons.
        for (Button b : buttons) {
            final boolean on = (held & b.bit) != 0;
            fill.setColor(on ? 0x90FFFFFF : 0x40FFFFFF);
            line.setColor(0x90FFFFFF);
            final boolean round_ = b.label.isEmpty();
            if (round_) {
                canvas.drawCircle(b.cx, b.cy, b.w / 2, fill);
                canvas.drawCircle(b.cx, b.cy, b.w / 2, line);
                DrawMark(canvas, b);
            } else {
                final RectF r = new RectF(b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2);
                canvas.drawRoundRect(r, b.h / 2, b.h / 2, fill);
                canvas.drawRoundRect(r, b.h / 2, b.h / 2, line);
                text.setColor(0xE0FFFFFF);
                text.setTextSize(b.h * 0.5f);
                canvas.drawText(b.label, b.cx, b.cy + b.h * 0.18f, text);
            }
        }
    }

    /* The face buttons' marks, in their colours. */
    private void DrawMark(Canvas canvas, Button b) {
        final float s = b.w * 0.22f;
        final Paint mark = new Paint(line);
        mark.setStrokeWidth(line.getStrokeWidth() * 1.6f);
        path.reset();
        switch (b.bit) {
            case TRIANGLE:
                mark.setColor(0xF040E0B0);
                path.moveTo(b.cx, b.cy - s);
                path.lineTo(b.cx + s * 1.05f, b.cy + s * 0.75f);
                path.lineTo(b.cx - s * 1.05f, b.cy + s * 0.75f);
                path.close();
                canvas.drawPath(path, mark);
                break;
            case CIRCLE:
                mark.setColor(0xF0FF6070);
                canvas.drawCircle(b.cx, b.cy, s, mark);
                break;
            case CROSS:
                mark.setColor(0xF080A8FF);
                canvas.drawLine(b.cx - s, b.cy - s, b.cx + s, b.cy + s, mark);
                canvas.drawLine(b.cx - s, b.cy + s, b.cx + s, b.cy - s, mark);
                break;
            case SQUARE:
                mark.setColor(0xF0FF80D0);
                canvas.drawRect(b.cx - s * 0.85f, b.cy - s * 0.85f, b.cx + s * 0.85f, b.cy + s * 0.85f, mark);
                break;
        }
    }

    /* A small controller's outline: the show/hide button's sign. */
    private void DrawPadGlyph(Canvas canvas, float cx, float cy, float s, int colour) {
        final Paint glyph = new Paint(line);
        glyph.setColor(colour);
        final RectF body = new RectF(cx - s * 1.6f, cy - s * 0.8f, cx + s * 1.6f, cy + s * 0.8f);
        canvas.drawRoundRect(body, s * 0.8f, s * 0.8f, glyph);
        canvas.drawLine(cx - s * 1.0f, cy, cx - s * 0.4f, cy, glyph);
        canvas.drawLine(cx - s * 0.7f, cy - s * 0.3f, cx - s * 0.7f, cy + s * 0.3f, glyph);
        canvas.drawCircle(cx + s * 0.8f, cy - s * 0.15f, s * 0.12f, glyph);
        canvas.drawCircle(cx + s * 0.5f, cy + s * 0.2f, s * 0.12f, glyph);
    }
}

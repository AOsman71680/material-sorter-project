"""The Tk window for the sorter GUI. It only draws; the decisions are made in src.sorter_gui."""

from __future__ import annotations

import tkinter as tk

import cv2
from PIL import Image, ImageTk

from src.data_collection import CameraError
from src.sorter_gui import (
    BIN_ORDER,
    FRAME_WIDTH,
    MATERIAL_COLOURS,
    NEUTRAL_COLOUR,
    TICK_MS,
    WINDOW_TITLE,
    BinView,
    FrameView,
    MaterialView,
    SorterSession,
    StmSender,
)

BACKGROUND = "#1e1e1e"
TRACK = "#4a4a4a"  # empty part of a bar
TEXT = "#f0f0f0"
MUTED = "#9e9e9e"
BAR_HEIGHT = 44
FONT = ("Segoe UI", 12)
FONT_BOLD = ("Segoe UI", 13, "bold")
FONT_MONO = ("Consolas", 10)


class SorterWindow:
    """Shows the camera picture and the two bars. ``refresh`` pulls one frame and redraws."""

    def __init__(self, root: tk.Tk, session: SorterSession, sender: StmSender) -> None:
        self.root = root
        self.session = session
        self.sender = sender
        self.error: str | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._tick_id: str | None = None
        self._closing = False

        root.title(WINDOW_TITLE)
        root.configure(bg=BACKGROUND)
        root.resizable(False, False)
        root.bind("<q>", lambda _event: self.close())
        root.bind("<Escape>", lambda _event: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.picture = tk.Label(root, bg=BACKGROUND)
        self.picture.pack(padx=12, pady=(12, 8))

        tk.Label(root, text="Detected material", fg=MUTED, bg=BACKGROUND, font=FONT, anchor="w").pack(fill="x", padx=12)
        self.material_bar = tk.Canvas(root, width=FRAME_WIDTH, height=BAR_HEIGHT, bg=TRACK, highlightthickness=0)
        self.material_bar.pack(padx=12, pady=(2, 10))

        self.bin_title = tk.Label(root, text="Colour sensor is over", fg=MUTED, bg=BACKGROUND, font=FONT, anchor="w")
        self.bin_title.pack(fill="x", padx=12)
        self.bin_bar = tk.Canvas(root, width=FRAME_WIDTH, height=BAR_HEIGHT, bg=BACKGROUND, highlightthickness=0)
        self.bin_bar.pack(padx=12, pady=(2, 10))

        self.status = tk.Label(root, text="", fg=TEXT, bg=BACKGROUND, font=FONT_MONO, justify="left", anchor="w")
        self.status.pack(fill="x", padx=12, pady=(0, 12))

    # ---- updates ------------------------------------------------------

    def refresh(self) -> None:
        """Pull one frame from the session and redraw. Raises CameraError if the camera has stopped."""
        self.sender.poll()
        self._draw(self.session.step())

    def tick(self) -> None:
        """Refresh, then schedule the next refresh. A camera failure closes the window with the error."""
        self._tick_id = None
        if self._closing:
            return
        try:
            self.refresh()
        except CameraError as exc:
            self.error = str(exc)
            self.close()
            return
        self._tick_id = self.root.after(TICK_MS, self.tick)

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None
        self.root.destroy()

    def _draw(self, view: FrameView) -> None:
        self._draw_picture(view)
        self._draw_material(view.material)
        self._draw_bins(view.bin)
        self.bin_title.configure(text=f"Colour sensor is over: {view.bin.text}")
        lines = list(view.status) + list(self.sender.log)
        self.status.configure(text="\n".join(lines))

    def _draw_picture(self, view: FrameView) -> None:
        if view.rgb is None:
            return
        height, width = view.rgb.shape[:2]
        scaled = cv2.resize(view.rgb, (FRAME_WIDTH, int(height * FRAME_WIDTH / width)), interpolation=cv2.INTER_AREA)
        self._photo = ImageTk.PhotoImage(Image.fromarray(scaled))
        self.picture.configure(image=self._photo)

    def _draw_material(self, view: MaterialView) -> None:
        canvas = self.material_bar
        canvas.delete("all")
        width = int(canvas.cget("width"))
        fill = MATERIAL_COLOURS.get(view.material or "", NEUTRAL_COLOUR)
        if view.fraction > 0:
            canvas.create_rectangle(0, 0, int(width * view.fraction), BAR_HEIGHT, fill=fill, outline="")
        text_colour = "#111111" if view.material == "ORGANIC" else "#ffffff"
        canvas.create_text(12, BAR_HEIGHT // 2, anchor="w", text=view.text, fill=text_colour, font=FONT_BOLD)

    def _draw_bins(self, view: BinView) -> None:
        canvas = self.bin_bar
        canvas.delete("all")
        width = int(canvas.cget("width"))
        segment = width / len(BIN_ORDER)
        for index, name in enumerate(BIN_ORDER):
            left = int(index * segment)
            right = int((index + 1) * segment) - 2
            active = view.active == index
            fill = MATERIAL_COLOURS[name] if active else TRACK
            canvas.create_rectangle(left, 0, right, BAR_HEIGHT, fill=fill, outline="")
            if active:
                text_colour = "#111111" if name == "ORGANIC" else "#ffffff"
            else:
                text_colour = MUTED
            canvas.create_text((left + right) // 2, BAR_HEIGHT // 2, text=name, fill=text_colour, font=FONT_BOLD if active else FONT)


def run_window(session: SorterSession, sender: StmSender) -> str | None:
    """Open the window and run until it is closed. Returns the camera error, if one stopped it."""
    root = tk.Tk()
    window = SorterWindow(root, session, sender)
    window.tick()
    root.mainloop()
    return window.error

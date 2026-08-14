"""
EPI file viewer.

Dev tool: given an absolute directory path, scan all EPI files in it and open a
GUI to browse them as images. Arrow keys switch images. A button copies the
filename of the currently displayed image.

"""
import os
import struct
import sys
import tkinter as tk
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def read_epi_content_file(file_path: str):
    """Read an EPI content file. Returns (img, [x_min, y_min, x_max, y_max])."""
    with open(file_path, "rb") as file:
        nb_cols, nb_rows = struct.unpack("2i", file.read(8))
        x_min, y_min, x_max, y_max = struct.unpack("4f", file.read(16))

        pixel_count = nb_cols * nb_rows
        pixel_array = struct.unpack(f"{pixel_count}f", file.read(4 * pixel_count))
        img = np.reshape(pixel_array, (nb_rows, nb_cols))

    return img, [x_min, y_min, x_max, y_max]


class Viewer:
    def __init__(self, root, files):
        self.root = root
        self.files = files
        self.index = 0

        root.title("EPI Viewer")

        # top bar: filename label + copy button + counter
        bar = tk.Frame(root)
        bar.pack(side=tk.TOP, fill=tk.X)

        self.name_var = tk.StringVar()
        tk.Label(bar, textvariable=self.name_var, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        tk.Button(bar, text="Copy filename", command=self.copy_filename).pack(
            side=tk.RIGHT
        )

        # matplotlib canvas
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.colorbar = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        root.bind("<Left>", lambda e: self.step(-1))
        root.bind("<Right>", lambda e: self.step(1))

        self.render()

    def step(self, delta):
        self.index = (self.index + delta) % len(self.files)
        self.render()

    def render(self):
        path = self.files[self.index]
        img, (x_min, y_min, x_max, y_max) = read_epi_content_file(str(path))

        self.ax.clear()
        im = self.ax.imshow(
            img,
            extent=(x_min, x_max, y_min, y_max),
            origin="lower",
            cmap="jet",
        )
        self.ax.set_title(path.name)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        if self.colorbar is None:
            self.colorbar = self.fig.colorbar(im, ax=self.ax, label="Pixel value")
        else:
            self.colorbar.update_normal(im)

        self.name_var.set(f"[{self.index + 1}/{len(self.files)}] {path.name}")
        self.canvas.draw_idle()

    def copy_filename(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.files[self.index].name)


def main():

    load_dotenv()
    directory = Path(os.environ.get("FILES"))

    files = sorted(
        p for p in directory.iterdir() if p.is_file() and ".epi" in p.name.lower()
    )

    if not files:
        print(f"No EPI files found in {directory}")
        sys.exit(1)

    root = tk.Tk()
    root.geometry("1000x800")
    Viewer(root, files)
    root.mainloop()


if __name__ == "__main__":
    main()

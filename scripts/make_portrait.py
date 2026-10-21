#!/usr/bin/env python3
"""Converts an image into one of Marin's startup art assets by glyph matching (needs Pillow and numpy).

  make_portrait.py IMAGE OUT.bin --cols 68 --rows 31 [--crop LEFT,TOP,RIGHT,BOTTOM] [--preview OUT.png]

The image is resampled to the pixel size of COLS x ROWS terminal cells. For every cell the script picks the
glyph (quadrant blocks, braille, half and full blocks) and the foreground and background colors that best
reproduce the cell's pixels, scoring each glyph by its real rasterized shape in a monospace font. Thin outlines
and highlights therefore survive as terminal glyph texture instead of becoming one flat color per pixel.

Two modes. "blocks" (wordmarks) is the least-squares fit above. "dots" (portraits) works at braille-dot
resolution, 2x4 dots per cell: the tone-mapped image is error-diffused into lit and unlit dots, so negative space
stays empty, outlines are one dot thin and every lit dot keeps its own color; each cell is the braille glyph of its
lit dots in the mean of their colors on the terminal's own background (background bytes are black).

File format: "MPG1", cols, rows, then per cell 9 bytes: glyph code point (u16 little endian), ink coverage
(0-255), foreground RGB, background RGB.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

FONT = "/usr/share/fonts/TTF/JetBrainsMonoNerdFontMono-Regular.ttf"
CELL_W, CELL_H = 10, 20
HALVES = [" ", "█", "▀", "▄", "▌", "▐"]
QUADRANTS = list("▖▗▘▝▚▞▙▛▜▟")
BRAILLE = [chr(0x2800 + i) for i in range(1, 256)]


def coverage(ch, font):
    cell = Image.new("L", (CELL_W, CELL_H), 0)
    ImageDraw.Draw(cell).text((0, -1), ch, font=font, fill=255)
    return np.asarray(cell).astype(np.float64) / 255.0


def prepare(image, cols, rows, sharpen, gamma, saturation, contrast):
    image = image.resize((cols * CELL_W, rows * CELL_H), Image.Resampling.LANCZOS)
    if sharpen:
        image = image.filter(ImageFilter.UnsharpMask(radius=1.4, percent=int(sharpen * 100), threshold=2))
    image = ImageEnhance.Color(image).enhance(saturation)
    pixels = (np.asarray(image).astype(np.float64) / 255.0) ** gamma
    return np.clip((pixels - 0.5) * contrast + 0.5, 0.0, 1.0) * 255.0


def fit(target, glyphs, cols, rows, font):
    """Least-squares glyph, foreground and background for every cell."""
    shapes = np.stack([coverage(g, font).reshape(-1) for g in glyphs])
    pixels = shapes.shape[1]
    ink = shapes.sum(1)
    ink2 = (shapes**2).sum(1)
    cells = target.reshape(rows, CELL_H, cols, CELL_W, 3).transpose(0, 2, 1, 3, 4).reshape(rows * cols, pixels, 3)
    total = cells.sum(1)
    total2 = (cells**2).sum(1)
    corr = np.einsum("npc,gp->ngc", cells, shapes)
    mean_ink = ink / pixels
    var_ink = ink2 / pixels - mean_ink**2
    flat = var_ink < 1e-6
    slope = (corr / pixels - (total / pixels)[:, None, :] * mean_ink[None, :, None]) / np.maximum(var_ink, 1e-9)[None, :, None]
    bg = total[:, None, :] / pixels - slope * mean_ink[None, :, None]
    fg = bg + slope
    bg = np.where(flat[None, :, None], total[:, None, :] / pixels, bg)
    fg = np.where(flat[None, :, None], bg, fg)
    bg, fg = np.clip(bg, 0, 255), np.clip(fg, 0, 255)
    diff = fg - bg
    error = (
        total2[:, None, :]
        - 2 * (bg * total[:, None, :] + diff * corr)
        + (pixels * bg**2 + 2 * bg * diff * ink[None, :, None] + diff**2 * ink2[None, :, None])
    ).sum(2)
    best = error.argmin(1)
    index = np.arange(rows * cols)
    return best, fg[index, best], bg[index, best], shapes


def dots(image, cols, rows, black, white, gamma, level):
    """Braille glyph per cell from an error-diffused dot grid; returns (glyph codes, fg colors)."""
    w, h = cols * 2, rows * 4
    rgb = np.asarray(image.resize((w, h), Image.Resampling.BOX)).astype(np.float64)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722]) / 255.0
    tone = np.clip((luma - black) / (white - black), 0, 1) ** gamma
    scale = np.where(luma > 1e-6, tone / np.maximum(luma, 1e-6), 0.0)
    color = rgb * scale[..., None] / level
    color *= np.minimum(1.0, 255.0 / np.maximum(color.max(2), 1e-6))[..., None]
    density = np.clip(tone / level, 0, 1)
    lit = np.zeros((h, w), dtype=bool)
    err = density.copy()
    for y in range(h):
        for x in range(w):
            on = err[y, x] >= 0.5
            lit[y, x] = on
            e = err[y, x] - float(on)
            if x + 1 < w: err[y, x + 1] += e * 7 / 16
            if y + 1 < h:
                if x > 0: err[y + 1, x - 1] += e * 3 / 16
                err[y + 1, x] += e * 5 / 16
                if x + 1 < w: err[y + 1, x + 1] += e * 1 / 16
    bit = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]]
    codes, fg = [], []
    for cy in range(rows):
        for cx in range(cols):
            mask, total, count = 0, np.zeros(3), 0
            for dy in range(4):
                for dx in range(2):
                    if lit[cy * 4 + dy, cx * 2 + dx]:
                        mask |= bit[dy][dx]
                        total += color[cy * 4 + dy, cx * 2 + dx]
                        count += 1
            codes.append(0x2800 + mask if mask else 0x20)
            fg.append(total / count if count else total)
    return codes, np.array(fg)


def write_dots(path, cols, rows, codes, fg, font):
    data = bytearray(b"MPG1" + bytes([cols, rows]))
    for code, color in zip(codes, fg):
        ink = 0 if code == 0x20 else int(round(coverage(chr(code), font).mean() * 255))
        data += code.to_bytes(2, "little") + bytes([ink]) + bytes(int(round(v)) for v in color) + bytes(3)
    open(path, "wb").write(data)


def preview_dots(path, cols, rows, codes, fg, font):
    out = np.zeros((rows * CELL_H, cols * CELL_W, 3))
    for i, (code, color) in enumerate(zip(codes, fg)):
        y, x = divmod(i, cols)
        out[y * CELL_H:(y + 1) * CELL_H, x * CELL_W:(x + 1) * CELL_W] = coverage(chr(code), font)[..., None] * color
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path)


def write(path, cols, rows, glyphs, best, fg, bg, shapes):
    data = bytearray(b"MPG1" + bytes([cols, rows]))
    for i, g in enumerate(best):
        data += ord(glyphs[g]).to_bytes(2, "little")
        data += bytes([int(round(shapes[g].mean() * 255))])
        data += bytes(int(round(v)) for v in fg[i]) + bytes(int(round(v)) for v in bg[i])
    open(path, "wb").write(data)


def preview(path, cols, rows, best, fg, bg, shapes):
    out = np.zeros((rows * CELL_H, cols * CELL_W, 3))
    for i, g in enumerate(best):
        y, x = divmod(i, cols)
        c = shapes[g].reshape(CELL_H, CELL_W)[..., None]
        out[y * CELL_H:(y + 1) * CELL_H, x * CELL_W:(x + 1) * CELL_W] = bg[i] * (1 - c) + fg[i] * c
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("out")
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--crop", help="LEFT,TOP,RIGHT,BOTTOM in source pixels")
    parser.add_argument("--sharpen", type=float, default=1.4)
    parser.add_argument("--gamma", type=float, default=0.72, help="below 1 brightens midtones")
    parser.add_argument("--saturation", type=float, default=1.5)
    parser.add_argument("--contrast", type=float, default=1.3)
    parser.add_argument("--mode", choices=["blocks", "dots"], default="blocks")
    parser.add_argument("--black", type=float, default=0.10, help="dots: luminance mapped to empty")
    parser.add_argument("--white", type=float, default=0.80, help="dots: luminance mapped to full")
    parser.add_argument("--level", type=float, default=0.75, help="dots: brightness of a lit dot")
    parser.add_argument("--font", default=FONT)
    parser.add_argument("--preview")
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    if args.crop:
        image = image.crop(tuple(int(v) for v in args.crop.split(",")))
    font = ImageFont.truetype(args.font, 16.67)
    glyphs = HALVES + QUADRANTS + BRAILLE
    if args.mode == "dots":
        graded = prepare(image, args.cols, args.rows, args.sharpen, args.gamma, args.saturation, args.contrast)
        graded = Image.fromarray(graded.astype(np.uint8))
        codes, fg = dots(graded, args.cols, args.rows, args.black, args.white, 1.0, args.level)
        write_dots(args.out, args.cols, args.rows, codes, fg, font)
        if args.preview:
            preview_dots(args.preview, args.cols, args.rows, codes, fg, font)
        return
    target = prepare(image, args.cols, args.rows, args.sharpen, args.gamma, args.saturation, args.contrast)
    best, fg, bg, shapes = fit(target, glyphs, args.cols, args.rows, font)
    write(args.out, args.cols, args.rows, glyphs, best, fg, bg, shapes)
    if args.preview:
        preview(args.preview, args.cols, args.rows, best, fg, bg, shapes)


if __name__ == "__main__":
    main()

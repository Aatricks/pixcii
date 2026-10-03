from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

# Candidate glyphs. The ramp actually used is these, sorted by how much ink each has in the loaded font.
LEVELS = 20
LINE_CHARS = "-_|/\\"  # kept for edges only: as brightness levels they draw fake lines
RAMP_CHARS = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "DejaVuSansMono.ttf",
    "consola.ttf",
    "cour.ttf",
]
@lru_cache
def load_font(size: int = 12) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size)  # proportional, but always there


def cell_size(font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """(width, height) in pixels of one character cell."""
    ascent, descent = font.getmetrics()
    return max(1, round(font.getlength("M"))), ascent + descent


@lru_cache
def glyph_atlas(font_size: int = 12) -> tuple[str, np.ndarray, np.ndarray]:
    """Return glyphs sorted by ink (lightest first), their alpha tiles (G, ch, cw) in 0..1,
    and the indices of the glyphs used as brightness levels.

    Tiles are cropped to the rows where the glyphs have ink: the empty band
    above capitals and below descenders would show as dark stripes between lines.
    Brightness levels keep only glyphs at least 1/LEVELS more inky than the previous
    level: many glyphs have almost the same ink but look very different, and using
    them all turns smooth gradients into bands of unrelated letters.
    """
    font = load_font(font_size)
    cw, ch = cell_size(font)
    tiles = []
    for c in RAMP_CHARS:
        tile = Image.new("L", (cw, ch))
        ImageDraw.Draw(tile).text((0, 0), c, font=font, fill=255)
        tiles.append(np.asarray(tile, np.float32) / 255)
    tiles = np.array(tiles)
    inked = np.flatnonzero(tiles.max(axis=(0, 2)) > 0.2)
    tiles = tiles[:, inked[0] : inked[-1] + 1]
    coverage = tiles.mean(axis=(1, 2))
    order = np.argsort(coverage, kind="stable")
    coverage = coverage[order] / coverage.max()
    chars = "".join(RAMP_CHARS[i] for i in order)
    levels = [0]
    for i, c in enumerate(coverage):
        if chars[i] not in LINE_CHARS and c - coverage[levels[-1]] >= 1 / LEVELS:
            levels.append(i)
    return chars, tiles[order], np.array(levels)


def cell_shape(font_size: int = 12) -> tuple[int, int]:
    """(width, height) in pixels of one rendered cell."""
    tiles = glyph_atlas(font_size)[1]
    return tiles.shape[2], tiles.shape[1]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def adjust_image(image: Image.Image, brightness: float = 1.0, contrast: float = 1.0, gamma: float = 1.0) -> Image.Image:
    if gamma != 1.0:
        image = image.point(lambda x: int(((x / 255) ** (1 / gamma)) * 255))
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    return image


def grid_size(image: Image.Image, columns: int, character_ratio: float) -> tuple[int, int]:
    """Grid (columns, rows) that keeps the image aspect for cells of height/width = character_ratio."""
    width, height = image.size
    return columns, max(1, round(columns * height / (width * character_ratio)))


def is_light(image: Image.Image) -> bool:
    """True when the picture is mostly light, so dark ink on light paper suits it best."""
    return np.asarray(image.convert("L").resize((64, 64))).mean() > 140


def tone_map(lum: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """Spread luminance (0..1) over the full glyph ramp.

    Contrast-limited histogram equalization: tones the picture uses a lot get more
    distinct glyphs (e.g. dark rock under a bright sky), but no tone gets more than
    `clip` times its fair share, so big flat areas do not turn into bands.
    """
    bins = np.minimum((lum * 256).astype(int), 255)
    hist = np.minimum(np.bincount(bins.ravel(), minlength=256), clip * lum.size / 256)
    cdf = np.cumsum(hist)
    cdf = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1)
    return cdf[bins]


def edge_glyphs(lum: np.ndarray, character_ratio: float, threshold: float) -> np.ndarray:
    """Line glyph ('-', '|', '/', '\\') along strong edges of `lum`, '' elsewhere.

    Lines go on the bright side of each edge only, so they stay one cell thick.
    """
    lum = lum.astype(np.float32)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3) / character_ratio  # cells are taller than wide
    edge = (np.hypot(gx, gy) > threshold) & (lum >= cv2.blur(lum, (3, 3)))
    # Edge direction is the gradient turned by 90 degrees; image y points down.
    angle = (np.degrees(np.arctan2(gy, gx)) + 90) % 180
    line = np.select([(angle < 22.5) | (angle >= 157.5), angle < 67.5, angle < 112.5], ["-", "\\", "|"], "/")
    return np.where(edge, line, "")


def ink_colors(rgb: np.ndarray) -> np.ndarray:
    """Push each color to full brightness, keeping its hue and saturation.

    Glyph density already shows how bright a cell is; multiplying by a dark color
    too would make dark areas vanish.
    """
    peak = rgb.max(axis=-1, keepdims=True)
    return rgb / np.maximum(peak, 1e-3)


def retro_colors(rgb: np.ndarray) -> np.ndarray:
    """Quantize hue to 6 steps, saturation to on/off, value to full."""
    hsv = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).convert("HSV")).astype(np.float32)
    hsv[..., 0] = (np.round(hsv[..., 0] / 255 * 6) % 6) * 255 / 6
    hsv[..., 1] = np.where(hsv[..., 1] < 64, 0, 255)
    hsv[..., 2] = 255
    return np.asarray(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"), np.float32) / 255


def convert(
    image: Image.Image,
    columns: int,
    character_ratio: float,
    font_size: int = 12,
    light: bool = False,
    use_retro: bool = False,
    use_bw: bool = False,
    mask: np.ndarray | None = None,
    edge_threshold: float = 0.5,
    tint: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick a glyph and a color for every cell.

    Returns ramp indices (rows, cols), ink colors and paper (cell background) colors,
    both (rows, cols, 3) in 0..1.
    `light` draws dark ink on light paper: density then follows darkness instead of brightness.
    `mask` is a (rows, cols) bool array; True cells are left blank.
    `edge_threshold` is the edge strength above which line glyphs replace ramp glyphs; 0 turns this off.
    `tint` (0..1) fills each cell's background with the picture's color at that strength.
    """
    cols, rows = grid_size(image, columns, character_ratio)
    ramp, tiles, levels = glyph_atlas(font_size)

    rgb = np.asarray(image.resize((cols, rows), Image.Resampling.BOX), np.float32) / 255
    lum = np.asarray(image.convert("L").resize((cols, rows), Image.Resampling.BOX), np.float32) / 255
    if light:
        rgb, lum = 1 - rgb, 1 - lum

    # Nearest brightness level by ink coverage, the densest glyph standing for full brightness.
    coverage = tiles[levels].mean(axis=(1, 2))
    coverage /= coverage[-1]
    idx = levels[np.searchsorted((coverage[1:] + coverage[:-1]) / 2, tone_map(lum))]
    if edge_threshold > 0:
        # Edges on a plain stretch: equalization steepens smooth gradients into false edges.
        lo, hi = np.percentile(lum, (1, 99))
        lines = edge_glyphs(np.clip((lum - lo) / max(hi - lo, 1e-3), 0, 1), character_ratio, edge_threshold)
        for c in "-|/\\":
            idx[lines == c] = ramp.index(c)

    if use_bw:
        colors = np.ones_like(rgb)
    elif use_retro:
        colors = retro_colors(rgb)
    else:
        colors = ink_colors(rgb)
    paper = rgb * tint
    if mask is not None:
        idx[mask] = 0  # the ramp starts with space
        paper[mask] = 0
    if light:
        colors, paper = 1 - colors, 1 - paper
    return idx, colors, paper


def render(idx: np.ndarray, colors: np.ndarray, paper: np.ndarray, font_size: int = 12) -> Image.Image:
    """Draw the glyph grid as an RGB image."""
    tiles = glyph_atlas(font_size)[1]
    rows, cols = idx.shape
    _, ch, cw = tiles.shape
    alpha = tiles[idx].transpose(0, 2, 1, 3).reshape(rows * ch, cols * cw, 1)
    ink = np.repeat(np.repeat(colors, ch, axis=0), cw, axis=1)
    paper = np.repeat(np.repeat(paper, ch, axis=0), cw, axis=1)
    out = paper + alpha * (ink - paper)
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8))


def to_ansi(idx: np.ndarray, colors: np.ndarray, paper: np.ndarray | None = None, font_size: int = 12) -> str:
    """Glyph grid as 24-bit ANSI colored text. Without `paper` the terminal background shows through."""
    ramp = glyph_atlas(font_size)[0]
    fg = (colors * 255 + 0.5).astype(np.uint8)
    bg = None if paper is None else (paper * 255 + 0.5).astype(np.uint8)
    lines = []
    for y, row in enumerate(idx):
        cells = []
        for x, i in enumerate(row):
            code = "38;2;%d;%d;%d" % tuple(fg[y, x])
            if bg is not None:
                code += ";48;2;%d;%d;%d" % tuple(bg[y, x])
            cells.append(f"\x1b[{code}m{ramp[i]}")
        lines.append("".join(cells) + "\x1b[0m")
    return "\n".join(lines)

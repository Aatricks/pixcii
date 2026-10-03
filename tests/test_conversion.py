import numpy as np
from PIL import Image

from pixcii import conversion, minimalistic


def ramp_chars(idx):
    ramp = conversion.glyph_atlas()[0]
    return np.vectorize(lambda i: ramp[i])(idx)


def test_ramp_starts_blank_and_gets_inkier():
    ramp, tiles, levels = conversion.glyph_atlas()
    coverage = tiles.mean(axis=(1, 2))
    assert ramp[0] == " "
    assert np.all(np.diff(coverage) >= 0)
    assert np.all(np.diff(levels) > 0) and len(levels) >= 10


def test_output_keeps_aspect_ratio():
    image = Image.new("RGB", (400, 300), "gray")
    cw, ch = conversion.cell_shape()
    idx, colors, paper = conversion.convert(image, 80, ch / cw)
    out = conversion.render(idx, colors, paper)
    assert abs(out.width / out.height - 400 / 300) < 0.02 * 400 / 300


def test_edges_use_line_glyphs():
    vertical = np.zeros((40, 40), np.uint8)
    vertical[:, 20:] = 255
    for pixels, glyph in ((vertical, "|"), (vertical.T.copy(), "-")):
        idx, _, _ = conversion.convert(Image.fromarray(pixels).convert("RGB"), 40, 1.0)
        assert glyph in ramp_chars(idx)


def test_masked_cells_are_blank():
    image = Image.new("RGB", (20, 20), "white")
    mask = np.zeros((20, 20), bool)
    mask[:, :10] = True
    idx, _, paper = conversion.convert(image, 20, 1.0, mask=mask, tint=0.5)
    assert (ramp_chars(idx)[mask] == " ").all()
    assert (paper[mask] == 0).all()


def test_simple_background_mask():
    mask = minimalistic.create_background_mask(Image.new("RGB", (10, 10), "white"))
    assert mask.mode == "L" and np.asarray(mask).all()

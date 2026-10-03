import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image

from . import conversion, minimalistic

ML_MODELS = ["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime", "sam", "birefnet-general", "birefnet-general-lite", "birefnet-portrait", "birefnet-dis", "birefnet-hrsod", "birefnet-cod", "birefnet-massive"]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert images to ASCII art.")
    parser.add_argument("input_path", help="Path to the input image.")
    parser.add_argument("output_path", nargs="?", help="Path to save the output ASCII art image. If not provided, output to terminal.")
    parser.add_argument("-w", "--width", type=int, help="Output width in characters. Default: 200 for images, terminal width for terminal output.")
    parser.add_argument("--font-size", type=int, default=12, help="Font size in pixels for image output.")
    parser.add_argument("--background", choices=["auto", "dark", "light"], default="auto", help="Canvas color for image output. 'auto' picks light paper for mostly-light pictures.")
    parser.add_argument("--tint", type=float, default=0.0, help="Fill each character's background with the picture's color at this strength (0 to 1). Try 0.3.")
    parser.add_argument("--edge-threshold", type=float, default=0.5, help="Edge strength above which outlines are drawn with line characters (- | / \\). 0 turns them off.")
    parser.add_argument("-m", "--minimalistic", action="store_true", help="Enable minimalistic mode for subject isolation.")
    parser.add_argument("--bg-removal-method", choices=["simple", "ml"], default="ml", help="The method for background removal in minimalistic mode.")
    parser.add_argument("--ml-model", choices=ML_MODELS, default="u2net", help="The ML model to use for background removal.")
    parser.add_argument("--dilation-kernel-size", type=int, default=1, help="Grow the removed background by this many cells.")
    parser.add_argument("--retro", action="store_true", help="Use retro color mode.")
    parser.add_argument("--bw", action="store_true", help="Use black and white mode.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction value.")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness adjustment factor.")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast adjustment factor.")
    parser.add_argument("--character-ratio", type=float, default=2.0, help="Height-to-width ratio of terminal characters. Only for terminal output.")
    return parser.parse_args()


def get_mask(image: Image.Image, grid: tuple[int, int], args: argparse.Namespace) -> np.ndarray | None:
    """Bool (rows, cols) array, True where the background is."""
    if not args.minimalistic:
        return None
    if args.bg_removal_method == "simple":
        small = image.resize(grid, Image.Resampling.BOX)
        mask = minimalistic.create_background_mask(small)
    else:
        # Segment a reasonably sized copy: the character grid is too coarse for the model.
        work = image.copy()
        work.thumbnail((1024, 1024))
        alpha = minimalistic.remove_background_ml(work, args.ml_model).getchannel("A")
        mask = Image.fromarray((np.asarray(alpha.resize(grid, Image.Resampling.BOX)) < 128).astype(np.uint8) * 255)
    return np.asarray(minimalistic.refine_mask(mask, args.dilation_kernel_size)) > 0


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path) if args.output_path else None

    try:
        image = conversion.load_image(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)
    image = conversion.adjust_image(image, args.brightness, args.contrast, args.gamma)

    if output_path is None:
        ratio = args.character_ratio
        if args.width:
            columns = args.width
        else:  # fit the whole picture in the terminal window
            term = shutil.get_terminal_size()
            columns = min(term.columns, int((term.lines - 1) * ratio * image.width / image.height))
        light = args.background == "light"
    else:
        columns = args.width or 200
        cw, ch = conversion.cell_shape(args.font_size)
        ratio = ch / cw
        light = args.background == "light" or (args.background == "auto" and conversion.is_light(image))

    mask = get_mask(image, conversion.grid_size(image, columns, ratio), args)
    idx, colors, paper = conversion.convert(image, columns, ratio, args.font_size, light, args.retro, args.bw, mask, args.edge_threshold, args.tint)

    if output_path is None:
        print(conversion.to_ansi(idx, colors, paper if light or args.tint else None, args.font_size))
        return
    try:
        conversion.render(idx, colors, paper, args.font_size).save(output_path)
    except (IOError, ValueError) as e:
        print(f"Error saving output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

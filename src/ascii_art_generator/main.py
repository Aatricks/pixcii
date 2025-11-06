import argparse
import sys

import numpy as np
from PIL import Image

from . import conversion, minimalistic


def main():
    parser = argparse.ArgumentParser(description="Convert images to ASCII art.")
    parser.add_argument("input_path", help="Path to the input image.")
    parser.add_argument("output_path", nargs='?', help="Path to save the output ASCII art image. If not provided, output to terminal.")
    parser.add_argument("-w", "--width", type=int, default=120, help="The desired width of the output image in characters.")
    parser.add_argument("-m", "--minimalistic", action="store_true", help="Enable minimalistic mode for subject isolation.")
    parser.add_argument("--bg-removal-method", choices=["simple", "ml"], default="ml", help="The method for background removal in minimalistic mode.")
    parser.add_argument("--dilation-kernel-size", type=int, default=3, help="The kernel size for dilation in background removal.")
    parser.add_argument("--blur-kernel-size", type=int, default=5, help="The kernel size for Gaussian blur on the subject mask.")
    parser.add_argument("--edge-threshold", type=float, default=4.0, help="The threshold for edge detection.")
    parser.add_argument("--retro", action="store_true", help="Use retro color mode.")
    parser.add_argument("--bw", action="store_true", help="Use black and white mode.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction value for colors.")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness adjustment factor.")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast adjustment factor.")
    parser.add_argument("--character-ratio", type=float, default=2.0, help="Height-to-width ratio for characters.")
    parser.add_argument("--max-height", type=int, default=48, help="Maximum height in characters.")

    args = parser.parse_args()

    try:
        image = conversion.load_image(args.input_path)
        image = conversion.adjust_image(image, args.brightness, args.contrast, args.gamma)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output_path is None:
        # Terminal output
        resized_image = conversion.resize_image(image, args.width, args.character_ratio)
        conversion.print_ascii_art(resized_image, args.edge_threshold, args.retro, args.bw, args.gamma)
    else:
        # Image output
        resized_image = conversion.resize_image(image, args.width, 1.0)
        ascii_chars = conversion.image_to_ascii_chars(resized_image)

        if args.minimalistic:
            if args.bg_removal_method == "simple":
                background_mask = minimalistic.create_background_mask(resized_image)
                edge_mask = minimalistic.emphasize_edges(resized_image, background_mask, args.blur_kernel_size, args.edge_threshold)
                output_image = conversion.draw_ascii_art(resized_image, ascii_chars, background_mask, edge_mask, args.edge_threshold, args.retro, args.bw, args.gamma)
            else:
                removed_bg_image = minimalistic.remove_background_ml(resized_image)
                background_mask = Image.fromarray((np.array(removed_bg_image.split()[-1]) == 0).astype(np.uint8) * 255)
                refined_background_mask = minimalistic.refine_mask(background_mask, args.dilation_kernel_size)
                edge_mask = minimalistic.emphasize_edges(resized_image, refined_background_mask, args.blur_kernel_size, args.edge_threshold)
                output_image = conversion.draw_ascii_art(resized_image, ascii_chars, refined_background_mask, edge_mask, args.edge_threshold, args.retro, args.bw, args.gamma)
        else:
            output_image = conversion.draw_ascii_art(resized_image, ascii_chars, edge_threshold=args.edge_threshold, use_retro=args.retro, use_bw=args.bw, gamma=args.gamma)

        try:
            output_image.save(args.output_path)
        except IOError:
            print(f"Error: Could not save output file to {args.output_path}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()

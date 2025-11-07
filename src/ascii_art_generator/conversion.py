import colorsys

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

ASCII_CHARS = " .-=+*x#$&X@"

def get_character_ratio():
    font_size = 10
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    bbox = font.getbbox("A")
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]
    return char_height / char_width

def load_image(path):
    """Load an image from a file path."""
    image = Image.open(path)
    return image.convert("RGB")

def adjust_image(image, brightness=1.0, contrast=1.0, gamma=1.0):
    """Adjust brightness, contrast, and gamma of the image."""
    if gamma != 1.0:
        image = Image.eval(image, lambda x: ((x / 255) ** (1 / gamma)) * 255)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    return image

def resize_image(image, new_width=120, character_ratio=1.0):
    """Resize an image to a new width, maintaining aspect ratio with character ratio."""
    width, height = image.size
    new_height = int(new_width * height / (character_ratio * width))
    return image.resize((new_width, new_height))

def get_pixel_color(pixel, use_bw=False, gamma=1.0):
    """Get the color for a pixel."""
    if use_bw:
        if len(pixel) >= 3:
            gray = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
            return (gray, gray, gray)
        else:
            val = int(pixel[0])
            return (val, val, val)
    else:
        if len(pixel) >= 3:
            r = int(((pixel[0] / 255) ** 1.0) * 255)
            g = int(((pixel[1] / 255) ** 1.0) * 255)
            b = int(((pixel[2] / 255) ** 1.0) * 255)
            return (r, g, b)
        else:
            return (pixel[0], pixel[0], pixel[0])

def get_retro_color(pixel):
    """Get retro quantized color."""
    h, s, v = colorsys.rgb_to_hsv(pixel[0]/255, pixel[1]/255, pixel[2]/255)
    v = 1.0
    h = round(h * 6) / 6  # quantize to 60 degrees
    s = 0.0 if s < 0.25 else 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r*255), int(g*255), int(b*255))

def image_to_ascii_chars(image):
    """Convert an image to a list of ASCII characters based on brightness."""
    width, height = image.size
    chars = []
    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            if len(pixel) >= 3:
                # Use luminance for brightness
                gray = (0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]) / 255
            else:
                gray = pixel[0] / 255
            index = int(gray * (len(ASCII_CHARS) - 1))
            chars.append(ASCII_CHARS[min(index, len(ASCII_CHARS)-1)])
    return ''.join(chars)

def draw_ascii_art(image, ascii_chars, background_mask=None, use_retro=False, use_bw=False, gamma=1.0):
    """Draw the ASCII art on a new image canvas with color."""
    width, height = image.size
    font_size = 10
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate the size of the output image
    bbox = font.getbbox("A")
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]
    output_width = width * char_width
    output_height = height * char_height

    output_image = Image.new("RGB", (output_width, output_height), "black")
    draw = ImageDraw.Draw(output_image)

    background_pixels = None
    if background_mask:
        background_pixels = background_mask.getdata()



    for i, char in enumerate(ascii_chars):
        if background_pixels and background_pixels[i] > 0:
            continue

        x = i % width
        y = i // width
        pixel = image.getpixel((x, y))

        if use_bw:
            color = get_pixel_color(pixel, True)
        elif use_retro:
            color = get_retro_color(pixel)
        else:
            color = get_pixel_color(pixel, False, 1.0)

        char_to_draw = char

        draw.text((x * char_width, y * char_height), char_to_draw, font=font, fill=color)

    return output_image

def print_ascii_art(image, use_retro=False, use_bw=False, gamma=1.0):
    """Print ASCII art to terminal with color."""
    width, height = image.size

    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            if use_bw:
                if len(pixel) >= 3:
                    gray = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
                    r = g = b = gray
                    gray_val = gray / 255
                else:
                    gray_val = pixel[0] / 255
                    r = g = b = int(pixel[0])
            elif use_retro:
                r, g, b = get_retro_color(pixel)
                h, s, v = colorsys.rgb_to_hsv(pixel[0]/255, pixel[1]/255, pixel[2]/255)
                gray_val = v ** 2
            else:
                if len(pixel) >= 3:
                    r = int(((pixel[0] / 255) ** 1.0) * 255)
                    g = int(((pixel[1] / 255) ** 1.0) * 255)
                    b = int(((pixel[2] / 255) ** 1.0) * 255)
                    gray_val = (0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]) / 255
                else:
                    gray_val = pixel[0] / 255
                    r = g = b = int(pixel[0])

            index = int(gray_val * (len(ASCII_CHARS) - 1))
            char = ASCII_CHARS[min(index, len(ASCII_CHARS)-1)]

            print(f"\x1b[38;2;{r};{g};{b}m{char}", end='')
        print()
    print("\x1b[0m", end='')

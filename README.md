# Pixcii

A powerful command-line tool to convert images into high-quality, colorized ASCII art. Transform your photos into stylized text-based representations with options for standard conversion, minimalistic "sticker" effects, and various customization settings.

## Features

- **High-Quality Conversion**: Uses a dense set of ASCII characters to represent different brightness levels, preserving image detail.
- **Color Preservation**: Accurately applies original colors and gradients to ASCII characters.
- **Minimalistic Mode**: Isolates subjects against black backgrounds with edge enhancement, supporting both simple color-based and advanced ML-based background removal.
- **Terminal Output**: Display ASCII art directly in the terminal with color support.
- **Image Output**: Save ASCII art as image files for sharing or further use.
- **Customization Options**:
  - Adjustable output width and character ratio.
  - Retro color quantization.
  - Black and white mode.
  - Gamma, brightness, and contrast adjustments.
  - Multiple ML models for background removal (e.g., U2Net, BiRefNet variants).
- **Fast Processing**: Optimized for performance with progress bars.

## Installation

### Prerequisites

- Python 3.11 or later
- pip for package management

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Aatricks/pyscii.git
   cd pyscii
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install .
   ```

This installs the `pyscii` command globally in your environment.

## Usage

### Basic Syntax

```bash
pyscii [OPTIONS] INPUT_PATH [OUTPUT_PATH]
```

- `INPUT_PATH`: Path to the input image (PNG, JPG, etc.).
- `OUTPUT_PATH`: (Optional) Path to save the output image. If omitted, ASCII art is printed to the terminal.

### Examples

#### Standard Conversion to Image

Convert an image to colorized ASCII art and save as a new image:

```bash
pyscii input.png output.png
```

#### Terminal Output

Display ASCII art directly in the terminal:

```bash
pyscii input.png
```

#### Minimalistic Mode

Isolate the subject and create a "sticker" effect:

```bash
pyscii input.png output.png --minimalistic
```

Use ML-based background removal for complex images:

```bash
pyscii input.png output.png --minimalistic --bg-removal-method ml
```

#### Custom Width

Control detail level by setting character width:

```bash
pyscii input.png output.png --width 200
```

#### Retro Style

Apply retro color quantization:

```bash
pyscii input.png output.png --retro
```

#### Black and White

Convert to grayscale ASCII art:

```bash
pyscii input.png output.png --bw
```

#### Adjustments

Fine-tune colors and brightness:

```bash
pyscii input.png output.png --gamma 1.2 --brightness 1.1 --contrast 0.9
```

### Advanced Options

- `--bg-removal-method {simple,ml}`: Background removal method (default: ml).
- `--ml-model MODEL`: ML model for background removal (default: u2net). Options include u2net, birefnet-general, etc.
- `--dilation-kernel-size SIZE`: Kernel size for edge dilation in minimalistic mode (default: 1).
- `--blur-kernel-size SIZE`: Kernel size for mask blurring (default: 5).
- `--character-ratio RATIO`: Height-to-width ratio for terminal output (default: 2.0).
- `--max-height HEIGHT`: Maximum height in characters for terminal output (default: 48).

For a full list of options, run:

```bash
pyscii --help
```

## Examples and Screenshots

![Demo Image](./HomeImage.png)

## Testing

Run the test suite with:

```bash
cd src
pytest
```

Or use the project's linting and testing command:

```bash
cd src; pytest; ruff check .
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Write tests for new functionality.
4. Ensure all tests pass and code follows style guidelines (use `ruff`).
5. Submit a pull request with a clear description.

Refer to the `specs/` directory for detailed feature plans and requirements.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Dependencies

- Pillow: Image processing
- NumPy: Numerical operations
- OpenCV-Python: Computer vision utilities
- rembg: Background removal with ML models
- onnxruntime: ML model inference
- tqdm: Progress bars

## Support

If you encounter issues or have questions, please open an issue on the GitHub repository.

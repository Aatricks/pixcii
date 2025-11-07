# ASCII Art Generator

A command-line tool to convert images into colorized ASCII art.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd ascii-g
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -e .
    ```


![A Demo Image](./HomeImage.png)


## Usage

### Standard Conversion

To convert an image to colorized ASCII art:

```bash
python -m ascii_art_generator.main path/to/input.png path/to/output.png
```

### Minimalistic Mode

To isolate a subject and create a "sticker" effect:

```bash
python -m ascii_art_generator.main path/to/input.png path/to/output.png --minimalistic
```

By default, this uses a simple color-based method for background removal. For more complex images, you can use the ML-based method:

```bash
python -m ascii_art_generator.main path/to/input.png path/to/output.png --minimalistic --bg-removal-method ml
```

### Adjusting Width

You can control the level of detail by setting the output width in characters:

```bash
python -m ascii_art_generator.main input.png output.png --width 200
```

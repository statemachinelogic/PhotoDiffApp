# PhotoDiffApp

PhotoDiffApp is a Python application that allows users to compare two images, detect differences, and highlight them on the second image. It features a modern GUI with zooming, panning, and the ability to save results as a PDF or export images. Built with Tkinter, OpenCV, and PIL, it runs on Windows and Debian Linux.

## Features
- Upload an original and a comparison image.
- Automatically detect and highlight differences with a red overlay.
- Zoom and pan images within the main window or view full-size images in separate windows.
- Save results as a PDF with user details, images, and a list of differences.
- Export original, comparison, and highlighted images as PNG files.
- Keyboard shortcuts for quick actions (Ctrl+O, Ctrl+C, Ctrl+R, Ctrl+S).

## Prerequisites
- **Python 3.8+**: Ensure Python is installed on your system.
- **Operating System**: Tested on Windows and Debian Linux.

### Required Libraries
Install these via `pip`:
```bash
pip install opencv-python pillow tk fpdf numpy

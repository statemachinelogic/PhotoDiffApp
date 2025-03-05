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
```
On Debian Linux, ensure Tkinter is installed (it’s often included with Python, but verify):
```bash
sudo apt install python3-tk
```
Setup Steps
1. Clone or download the repo
```bash
git clone https://github.com/statemachinelogicllc/PhotoDiffApp.git
```
3. Navigate to the directory
```bash
cd PhotoDiffApp
```
4. Install the dependencies
```bash
pip install -r requirements.txt
```
5. Verify Installation
```bash
python photo_diff.py
```
If using Debian, you may have to run the following command first:
```bash
sudo chmod +x photo_diff.py
python3 photo_diff.py
```

Usage Instructions
1. Run the Application:
    Windows: python photo_diff.py
    Debian Linux: python3 photo_diff.py

2. Steps to Compare Images:
    Upload Original: Click "Upload Original" (or press Ctrl+O) and select the first image.
    Upload Comparison: Click "Upload Comparison" (or press Ctrl+C) and select the second image.
    Compare: Click "Compare" (or press Ctrl+R) to analyze and highlight differences.
    View Results:
        Differences are highlighted in red on the "Highlighted Changes" panel.
        A list of detected differences appears in the listbox below.
    Zoom and Pan:
        Use the mouse wheel to zoom in/out.
        Click and drag to pan the image.
        Adjust the zoom slider under each panel for precise control (0.1x to 10x).
    View Full Image: Click "View Full Image" under any panel to open a separate window with scrollbars and zooming.
    Save as PDF:
        Click "Save Result" (or press Ctrl+S).
        Enter your name, image name, summary, and location in the dialog.
        Choose a save location for the PDF, which includes all images and differences.
    Export Images: Click "Export Images" to save original, comparison, and highlighted images as PNGs in a selected directory.

3. Example Workflow:
    Upload photo1.jpg (original) and photo2.jpg (edited with added text).
    Click "Compare" to see photo2.jpg with red highlights on changes.
    Save a PDF report with your details and export the images for further use.

Notes
Image Size: Images larger than 2000x2000 pixels are automatically resized to 2000x2000 to prevent memory issues.
Threshold: Difference detection uses a fixed threshold of 30 (on a 0-100 scale), scaled to OpenCV’s 0-255 range (approximately 77). This balances sensitivity and noise rejection.
Icons: The app looks for optional icon files (upload_icon.png, compare_icon.png, save_icon.png) in the same directory. Without them, buttons display text only.
Reset Behavior: Uploading a new original image clears all previous results (after confirmation).
Performance: Large images or complex comparisons may take a few seconds; a progress bar shows processing status.

Troubleshooting
"No module named...":
    Ensure all libraries are installed (pip install opencv-python pillow tk fpdf numpy).
    Verify Python version compatibility (3.8+).

Images Not Displaying:
    Check file paths and supported formats (PNG, JPG, JPEG, BMP).
    Ensure images aren’t corrupted—test with known good files.

Comparison Fails:
    Verify both images are uploaded and valid.
    Check console output for error details (run from terminal).

PDF Save Errors:
    Ensure write permissions in the save directory.
    Avoid special characters in filenames that might cause issues.

Zoom/Pan Issues:
    If zooming feels unresponsive, try adjusting the slider first, then use the mouse wheel.

Application Crashes:
    Reduce image size if memory errors occur (e.g., edit images to <2000px).
    Report detailed errors in GitHub Issues.

Contributing

Contributions are welcome! To contribute:
    Fork this repository.
    Create a new branch (git checkout -b feature/your-feature).
    Make your changes and commit them (git commit -m "Add your feature").
    Push to your fork (git push origin feature/your-feature).
    Open a Pull Request with a description of your changes.

Please report bugs or suggest features via GitHub Issues.

License

MIT License

Copyright (c) 2025 [Daniel Allnutt]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Acknowledgments
    Built with contributions from the open-source community.
    Developed with assistance from Grok 3, created by xAI.
    Thanks to the developers of Tkinter, OpenCV, PIL, FPDF, and NumPy for their excellent libraries.

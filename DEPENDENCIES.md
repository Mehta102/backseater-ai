# Dependencies

## Required Libraries

This project requires the following Python packages:

### Core Dependencies

- **scikit-learn** - Machine learning library for the Decision Tree classifier
- **opencv-python** - Computer vision library for image processing and board detection
- **numpy** - Numerical computing library for array operations
- **mss** - Fast cross-platform screen capture
- **pyttsx3** - Text-to-speech engine 
- **pillow** - Image processing library for GUI display

## Installation

### Windows
```bash
py -m pip install scikit-learn opencv-python numpy mss pyttsx3 pillow
```

### macOS
```bash
pip3 install scikit-learn opencv-python numpy mss pyttsx3 pillow
```

## Platform-Specific Notes

### macOS
- Text-to-speech uses the native `say` command (no pyttsx3 needed, but install anyway for compatibility)
- First run requires Screen Recording permission (System Preferences → Security & Privacy → Privacy → Screen Recording)

### Windows
- Text-to-speech uses pyttsx3 with SAPI5 engine
- No special permissions required


## Python Version

Tested on Python 3.8+

## Verifying Installation

Run this command to check if all dependencies are installed:

```bash
python -c "import cv2, numpy, mss, pyttsx3, PIL, sklearn; print('All dependencies installed!')"
```

on macOS:
```bash
python3 -c "import cv2, numpy, mss, pyttsx3, PIL, sklearn; print('All dependencies installed!')"
```

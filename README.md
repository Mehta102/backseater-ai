# Backseater AI

**Real-time AI gameplay commentator.** Watches single-player games live, detects what is happening on screen and responds with spoken, context-aware commentary. Final year project, BSc Computer Science with AI.

## What it does

- Captures game frames and extracts events with OpenCV and Tesseract OCR
- Decides which moments deserve a reaction, so it adds to the game and does not talk over it (scikit-learn)
- Generates short, in-character commentary with an LLM
- Speaks it back through text-to-speech

## Engineering challenges

- **Latency:** keeping capture, OCR, the API call and TTS fast enough that commentary lands while the moment is still relevant
- **Signal vs noise:** commentary frequency and repetition control
- **Robustness:** OCR against stylised game fonts

## Tech

Python · OpenCV · Tesseract · scikit-learn · GPT API · text-to-speech

## Status

Built and evaluated. Report phase complete.

## Next

Learned event detection to replace OCR, multi-game support, lower end-to-end latency.

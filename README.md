# Backseater AI

A real-time gameplay commentator. It watches your screen while you play a single-player game, works out what just happened, and talks to you about it like a friend sitting behind you on the sofa.

This was my final year project for BSc Computer Science with AI.

## The idea

Streamers get a lot of their charm from the running commentary. Most people play alone, in silence. I wanted to see if a program could fill that gap without being annoying, which turned out to be the real problem: knowing *when* to speak matters as much as *what* to say.

## How it works

1. **Look.** OpenCV grabs frames from the game and Tesseract reads on-screen text, so the program can tell when something has changed (a kill, a pickup, a menu, a death screen).
2. **Decide.** A small scikit-learn model helps decide whether the moment is worth commenting on, so it is not talking over every single frame.
3. **Think.** The event is passed to a large language model (GPT) which writes a short, in-character reaction.
4. **Speak.** Text-to-speech reads it out loud.

## Built with

- Python
- OpenCV and Tesseract for screen understanding
- scikit-learn
- GPT for the commentary
- Text-to-speech for the voice

## Status

The build is finished and I am in the post-report and evaluation phase, looking at what worked, what was too chatty, and what I would change next time.

## Things I would do differently

- Cut the delay between event and comment further, since late commentary feels worse than none
- Rely less on OCR and more on proper visual detection of events
- Test across more games instead of one or two

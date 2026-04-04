# src/speech.py
# Handles text-to-speech using pyttsx3.
# Runs in a background thread so it never blocks the camera feed.

import pyttsx3
import threading


class SpeechEngine:
    """
    Wrapper around pyttsx3.
    Uses a background thread so speech doesn't freeze the video.
    """

    def __init__(self, rate=150, volume=1.0, gender="female"):
        self.engine  = pyttsx3.init()
        self.lock    = threading.Lock()
        self.running = False

        # Set speed
        self.engine.setProperty('rate', rate)

        # Set volume (0.0 to 1.0)
        self.engine.setProperty('volume', volume)

        # Set voice gender
        voices = self.engine.getProperty('voices')
        if gender == "female" and len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)
        else:
            self.engine.setProperty('voice', voices[0].id)

        print(f"✅ Speech engine ready  (rate={rate}, gender={gender})")

    def speak(self, text):
        """
        Speaks text in a background thread.
        If already speaking, skips — never queues up.
        """
        if not text:
            return

        if self.running:
            return   # don't interrupt current speech

        def _speak():
            with self.lock:
                self.running = True
                self.engine.say(text)
                self.engine.runAndWait()
                self.running = False

        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()

    def speak_letter(self, letter):
        """Speaks a single letter."""
        self.speak(letter)

    def speak_word(self, word):
        """Speaks a full word."""
        self.speak(word)

    def speak_sentence(self, sentence):
        """Speaks a full sentence."""
        self.speak(sentence)
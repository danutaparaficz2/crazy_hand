import speech_recognition as sr
import time
import threading
import queue

class VoiceRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.audio_queue = queue.Queue()
        self.is_listening = True
        
        # Word to number mapping
        self.word_to_number = {
            'ok': 1,
            'thumbs up': 2,
            'one': 3,
            'two': 4
        }
        
        # Adjust for ambient noise
        print("Adjusting for ambient noise... Please be quiet for a moment.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        print("Ready to listen!")
    
    def listen_continuously(self):
        """Continuously listen for audio in a separate thread"""
        with self.microphone as source:
            while self.is_listening:
                try:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    self.audio_queue.put(audio)
                except sr.WaitTimeoutError:
                    pass  # No speech detected, continue listening
    
    def process_audio(self):
        """Process audio from the queue"""
        while self.is_listening:
            if not self.audio_queue.empty():
                audio = self.audio_queue.get()
                try:
                    # Recognize speech using Google's speech recognition
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: '{text}'")
                    
                    # Check if any of our target words are in the recognized text
                    number = self.get_number_from_text(text)
                    if number:
                        print(f"Output: {number}")
                        return number
                    else:
                        print("No matching command found")
                        
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Error with recognition service: {e}")
            
            time.sleep(0.1)  # Small delay to prevent high CPU usage
    
    def get_number_from_text(self, text):
        """Extract number based on recognized words"""
        # Check for exact matches first
        for word, number in self.word_to_number.items():
            if word in text:
                return number
        
        # Check for variations
        if 'thumb' in text and 'up' in text:
            return 2
        
        return None
    
    def start_listening(self):
        """Start the voice recognition system"""
        print("Voice Recognition Started!")
        print("Say: 'ok', 'thumbs up', 'one', or 'two'")
        print("Press Ctrl+C to stop")
        
        # Start listening thread
        listen_thread = threading.Thread(target=self.listen_continuously)
        listen_thread.daemon = True
        listen_thread.start()
        
        try:
            # Process audio in main thread
            while True:
                result = self.process_audio()
                if result:
                    # You can add additional processing here
                    pass
        except KeyboardInterrupt:
            print("\nStopping voice recognition...")
            self.is_listening = False

def main():
    try:
        voice_recognizer = VoiceRecognizer()
        voice_recognizer.start_listening()
    except Exception as e:
        print(f"Error initializing voice recognition: {e}")
        print("Make sure you have a microphone connected and the required packages installed:")
        print("pip install SpeechRecognition pyaudio")

if __name__ == "__main__":
    main()

import pyaudio
import wave
import numpy as np
import threading
import time
import os
import tempfile
from faster_whisper import WhisperModel
import torch


FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1             # Mono audio
RATE = 16000             # Sample rate: 16kHz (Whisper's preferred input)
CHUNK = 1024             # Number of audio frames per buffer
RECORD_SILENCE_THRESHOLD = 500  # Threshold for silence detection (optional, not used in this basic version)

# --- Load Whisper Model ---
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if WHISPER_DEVICE == "cuda" else "int8" # Use int8 for CPU for efficiency

print(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} with {WHISPER_COMPUTE_TYPE} precision...")
try:
    model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    print("Whisper model loaded successfully!")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    print("Please check your GPU setup, CUDA installation, or try a smaller model/CPU inference.")
    exit()

# Global variables for recording control
audio_frames = []
is_recording = False
recording_thread = None
p = None # PyAudio instance

def record_audio():
    """Records audio from the microphone and stores it in audio_frames."""
    global p, is_recording, audio_frames

    audio_frames = []
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("🎙️ Recording started! Press Enter again to stop.")
    is_recording = True

    try:
        while is_recording:
            data = stream.read(CHUNK, exception_on_overflow=False) # exception_on_overflow=False handles potential buffer overflow
            audio_frames.append(data)
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("🔴 Recording stopped.")

def main():
    global is_recording, recording_thread, audio_frames

    print("Press Enter to start recording audio.")
    input() # Wait for user to press Enter to start

    # Start recording in a separate thread
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()

    input() # Wait for user to press Enter to stop
    is_recording = False
    recording_thread.join() # Wait for the recording thread to finish

    if not audio_frames:
        print("No audio was recorded.")
        return

    # Save the recorded audio to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
        wav_filename = temp_wav_file.name
        wf = wave.open(wav_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()

    print(f"Audio saved to temporary file: {wav_filename}")
    print("Transcribing audio...")

    # Transcribe the audio using faster-whisper
    try:
        segments, info = model.transcribe(wav_filename, beam_size=5, vad_filter=True) # Added VAD filter for better results
        print(f"\n🗣️ Detected language: {info.language} with probability {info.language_probability:.2f}")
        print("\n--- Transcription Results ---")
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        print("-----------------------------\n")
    except Exception as e:
        print(f"Error during transcription: {e}")
    finally:
        # Clean up the temporary WAV file
        os.remove(wav_filename)
        print(f"Temporary file {wav_filename} deleted.")

if __name__ == "__main__":
    import torch # Import torch here to use torch.cuda.is_available() for device check
    main()
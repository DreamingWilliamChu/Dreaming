import os
import uuid
import openai
import pyaudio
import wave
import keyboard
import time
import requests
from pydub import AudioSegment
import sounddevice as sd
import numpy as np
import time as tm
from scipy.io.wavfile import write
import threading  # import threading module

openai.api_key = "..."
user_stopped = False

voiceid = "..."
elelabapi_key = "..."


def recording(filename, silence_threshold_sec=3, sample_rate=44100):
    # Set the duration and sample rate
    recording_buffer = []

    # Silence counter and threshold
    silence_counter = 0
    silence_threshold = silence_threshold_sec * sample_rate  # 3 seconds of silence

    # Timer for printing volume
    last_print_time = tm.time()

    # Create a threading event
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        nonlocal silence_counter
        nonlocal last_print_time
        volume_norm = np.linalg.norm(indata) * 10

        # Print volume approximately once per second
        current_time = tm.time()
        if current_time - last_print_time >= 2.0:
            print('|' * int(volume_norm))  # Show volume in terminal
            last_print_time = current_time

        if volume_norm < 0.1:  # If volume below threshold increment silence counter
            silence_counter += frames
            if silence_counter > silence_threshold:  # If 3 seconds of silence detected, stop recording
                print("3 seconds of silence detected, stopping recording")
                stop_event.set()  # Set the stop event
                raise sd.CallbackStop
        else:
            silence_counter = 0  # Reset the silence counter if sound is detected
            recording_buffer.append(indata.copy())  # Append to recording buffer if not silent

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate):
        print('Recording started, speak into the microphone...')
        try:
            while not stop_event.is_set():  # Loop until the stop event is set
                sd.sleep(1000)  # Sleep for 1 second at a time
        except Exception as e:
            print('Recording failed: ', e)

    if recording_buffer:  # Check if there is any data in the buffer before saving
        write(filename, sample_rate, np.concatenate(recording_buffer))  # Save as WAV file
        print('Recording saved as: ' + filename)
    else:
        print('No audio data recorded')


# Speech to text -- recording
def record_audio(filename, rate=44100, channels=1, chunk=1024, format=pyaudio.paInt16):
    global user_stopped
    user_stopped = False
    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Press 'space' to start recording."
          "\nPress 'space' again to stop."
          "\nPress 'q' to terminate the conversation.")

    # Wait for the space key to be pressed to start recording
    while True:
        if keyboard.is_pressed('space'):
            break
        elif keyboard.is_pressed('q'):
            print("The conversation has been terminated!")
            stream.stop_stream()
            stream.close()
            p.terminate()
            user_stopped = True  # Update the global variable
            return
    time.sleep(0.5)  # Introduce a short delay to allow the key to be released

    print("Recording...\n Press 'space' to stop")

    frames = []
    while not keyboard.is_pressed('space'):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the final combined audio
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


# text to speech engine (elevenlabs streaming)
def stream_tts(text, voiceid, api_key):
    CHUNK_SIZE = 64
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voiceid}/stream"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }

    response = requests.post(url, json=data, headers=headers, stream=True)

    if response.status_code == 200:
        # Print response status code and headers
        # print(f"Status code: {response.status_code}")
        # print(f"Headers: {response.headers}")

        # Create temporary file names for the MP3 and raw audio data
        temp_mp3_file_name = f"{uuid.uuid4().hex}.mp3"
        temp_raw_file_name = f"{uuid.uuid4().hex}.raw"

        # Save the MP3 data to the temporary MP3 file
        with open(temp_mp3_file_name, 'wb') as temp_mp3_file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    temp_mp3_file.write(chunk)

        # Extract audio parameters from the MP3 file and convert the MP3 data to raw audio data using ffmpeg
        mp3_audio = AudioSegment.from_file(temp_mp3_file_name, format="mp3")
        channels = mp3_audio.channels
        frame_rate = mp3_audio.frame_rate
        mp3_audio.export(temp_raw_file_name, format="raw")

        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Open the PyAudio stream
        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=frame_rate,
                        output=True)

        # Play the raw audio data using PyAudio
        with open(temp_raw_file_name, 'rb') as temp_raw_file:
            while True:
                chunk = temp_raw_file.read(CHUNK_SIZE)
                if not chunk:
                    break
                stream.write(chunk)

        # Close the PyAudio stream and terminate PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Remove the temporary MP3 and raw audio files
        os.remove(temp_mp3_file_name)
        os.remove(temp_raw_file_name)
    else:
        print(f"Error: {response.status_code}, {response.text}")


def chat():
    global user_stopped
    user_input_count = 0
    max_user_inputs = 20
    # user_stopped = False

    # system message
    messages = [{"role": "system", "content": "You're Rachel."
                                              "Keep answer short."
                                              "You're a native English speaker just chillin' with a friend, so keep the convo real casual."
                                              "Use informal language, slang, abbrevs, and contractions to sound natural."
                                              "Remember, no examples, symbols, code, formulas, technical terms or anything like that, got it?"
                                              "And avoid showing how to do stuff, just give general advice. Stick to text in your replies."}
                ]

    while user_input_count < max_user_inputs:
        # Load the audio file as a binary file
        recording('recording.wav')
        if user_stopped:  # Check the global variable
            break
        with open("recording.wav", "rb") as audio_file:
            # Transcribe the audio
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            user_input = transcript['text']

        # append the message to the list
        messages.append({"role": "user", "content": user_input})

        # create result to hold final response
        result = ""
        print("Assistant: ")
        for chunk in openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 3.5-turbo
            messages=messages,
            stream=True
        ):
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content is not None:
                print(content, end='')
                result += content

        stream_tts(result, voiceid, elelabapi_key)

        # append the AI response as the context
        messages.append({"role": "assistant", "content": result})

        # count number of messages
        user_input_count += 1

    if not user_stopped:
        print("You have reached the maximum input limit. The conversation will now end!")

if __name__ == '__main__':
    chat()
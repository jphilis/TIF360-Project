import librosa
import os
import soundfile as sf


def filter_bat_clips(input_folder):
    threshold = 0.01  # Adjust this threshold based on the sensitivity needed
    files = os.listdir(input_folder)
    for file in files:
        file_path = os.path.join(input_folder, file)
        if file_path.endswith(".wav"):  # Ensure we process only WAV files
            y, sr = librosa.load(file_path, sr=None)
            envelope = librosa.onset.onset_strength(y=y, sr=sr)
            if max(envelope) < threshold:
                print(f"Deleting {file} due to low activity")
                os.remove(file_path)
            else:
                print(f"Keeping {file}")


# Example usage
filter_bat_clips("/path/to/your/folder")

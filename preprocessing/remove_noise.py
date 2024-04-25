import librosa
import os
import soundfile as sf
import numpy as np


def remove_low_db_clips(input_folder):
    # removes clips that are low audio, and assumes they are no bats
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


def remove_no_peaks_clips(input_folder):
    # removes clips that does not have a peak, and assume they are no bats
    pre_max = 1  # Number of samples before a peak to consider for peak picking
    post_max = 1  # Number of samples after a peak to consider for peak picking
    pre_avg = (
        1  # Number of samples before a peak to compute the average for normalization
    )
    post_avg = (
        1  # Number of samples after a peak to compute the average for normalization
    )
    delta = 0.1  # The threshold for picking peaks, relative to the surrounding data
    wait = 0  # The minimum number of samples between successive peaks
    files = os.listdir(input_folder)
    for file in files:
        file_path = os.path.join(input_folder, file)
        if file_path.endswith(".wav"):  # Ensure we process only WAV files
            y, sr = librosa.load(file_path, sr=None)
            # Compute the short-term Fourier transform (STFT) and get the magnitude
            D = np.abs(librosa.stft(y))
            # Aggregate the magnitude over time
            mag = np.sum(D, axis=1)
            # Normalize the magnitude
            norm_mag = librosa.util.normalize(mag)
            # Find peaks
            peaks = librosa.util.peak_pick(
                norm_mag, pre_max, post_max, pre_avg, post_avg, delta, wait
            )

            if len(peaks) == 0:
                print(f"Deleting {file} due to no peaks")
                # os.remove(file_path)
            else:
                print(f"Keeping {file}")

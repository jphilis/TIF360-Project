import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, variance
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

print(torch.__version__)
print(torchaudio.__version__)
import os

current_dir = os.path.dirname(__file__)
# parent_dir = os.path.dirname(current_dir)
main_folder = (
    r"C:\Users\jonat\OneDrive\chalmers\Advanced neural networks\project\dataset"
)
input_data = f"{main_folder}/labeled_dataset"
output_data = f"{main_folder}/preprocessed_audio"
if not os.path.exists(output_data):
    os.makedirs(output_data)


def list_files():
    sample_files = {}
    for folder in os.listdir(input_data):
        folder_path = os.path.join(input_data, folder)
        # take 10 random samples from anywhere in thsi folder, or in a subfolder
        for root, dirs, files in os.walk(folder_path):
            if len(files) > 0:
                if files[0].endswith(".wav"):
                    nr_files = len(files)
                    files = np.random.choice(files, 10, replace=True)
                    sample_files[folder] = [os.path.join(root, file) for file in files]
    return sample_files


def plot_mel_spectrogram_torchaudio(waveform, sample_rate, folder, filename):
    # waveform, sample_rate = torchaudio.load(wav_file)
    transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128, f_max=120000)
    mel_spectrogram = transform(waveform)

    fig = plt.figure(figsize=(10, 4), frameon=True)

    plt.imshow(
        mel_spectrogram.log2()[0, :, :].numpy(),
        cmap="hot",
        aspect="auto",
        origin="lower",
    )
    # ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # plt.title("Mel Spectrogram (Torchaudio)")
    # plt.xlabel("Time")
    # plt.ylabel("Frequency")
    # plt.colorbar(format="%+2.0f dB")
    # plt.show()
    random_nr = np.random.randint(1000)
    filename_no_extension = filename.split("//").split(".")[0]
    filename = f"mel_spectrograms/{folder}/{filename_no_extension}.png"
    if not os.path.exists(f"mel_spectrograms/{folder}"):
        os.makedirs(f"mel_spectrograms/{folder}")
    # resample image to 224x224 before saving
    fig.savefig(filename)
    print("saved", filename)


def resample(signal, sr, target_sample_rate):
    resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
    signal = resampler(signal)
    return signal


def reshape(waveform, target_size):
    clips = []
    # Create clips
    num_samples = waveform.shape[1]
    start_sample = 0

    while start_sample < num_samples:
        end_sample = start_sample + target_size
        clip = waveform[:, start_sample:end_sample]  # Slice the waveform
        clips.append(clip)
        start_sample += target_size
    if clips[-1].shape[1] < target_size:
        num_missing_samples = target_size - clips[-1].shape[1]
        last_dim_padding = (0, num_missing_samples)
        clips[-1] = torch.nn.functional.pad(clips[-1], last_dim_padding)
    return clips


import torch
import torchaudio
from torchaudio.functional import bandpass_biquad


def apply_bandpass_filter(waveform, sample_rate, low_freq, high_freq):
    """
    Applies a bandpass filter to the given waveform.

    Parameters:
        waveform (Tensor): The input waveform.
        sample_rate (int): The sample rate of the waveform.
        low_freq (int): The lower boundary of the bandpass filter.
        high_freq (int): The upper boundary of the bandpass filter.

    Returns:
        Tensor: The filtered waveform.
    """
    # First apply a highpass filter to remove frequencies below low_freq
    waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, low_freq)

    # Then apply a lowpass filter to remove frequencies above high_freq
    waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, high_freq)

    return waveform


def main():

    target_sr = 300000  # parameter, may be changed
    target_size = 2 * target_sr  # parameter, may be changed
    sample_rates = []
    data_lengths = []
    sample_files = list_files()
    img_nr = 0
    failed_files = 0
    for folder, files in sample_files.items():
        print(folder)
        print(len(files))
        for file in files:
            # if not os.path.exists(f"audio_clips/{folder}"):
            #     os.makedirs(f"audio_clips/{folder}")
            try:
                waveform, sample_rate = torchaudio.load(file)
            except Exception as e:
                print("error loading file", file, e)
                continue
            # if sample_rate != target_sr:
            waveform = resample(waveform, sample_rate, target_sr)
            sample_rate = target_sr
            waveform = apply_bandpass_filter(waveform, sample_rate, 20000, 120000)
            clips = reshape(waveform, target_size)
            # save audio clips
            for waveform in clips:

                # save audio waveform with same name as before
                filename = file.split("\\")[-1]
                # print("filename", filename)
                filepath = f"{output_data}/{folder}/{filename}"
                if not os.path.exists(f"{output_data}/{folder}"):
                    os.makedirs(f"{output_data}/{folder}")
                try:
                    torchaudio.save(filepath, waveform, sample_rate)
                    img_nr += 1
                except Exception as e:
                    print("error saving file", filename, e)
                    failed_files += 1
        print("failed files", failed_files)
        print("success files", img_nr)


if __name__ == "__main__":
    main()

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
main_raw_data = r"C:\Users\jonat\OneDrive\chalmers\Advanced neural networks\project\dataset\raw data"


def list_files():
    sample_files = {}
    for folder in os.listdir(main_raw_data):
        folder_path = os.path.join(main_raw_data, folder)
        # take 10 random samples from anywhere in thsi folder, or in a subfolder
        for root, dirs, files in os.walk(folder_path):
            if len(files) > 0:
                if files[0].endswith(".wav"):
                    nr_files = len(files)
                    # files = np.random.choice(files, int(nr_files / 100), replace=True)
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


import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os


# def plot_mel_spectrogram_torchaudio_new(waveform, sample_rate, folder):
#     # Set the n_fft and hop_length for high-resolution spectrograms
#     n_fft = 2048  # Larger FFT size for better frequency resolution
#     hop_length = 512  # Smaller hop length for finer temporal resolution
#     f_min = 60000  # Minimum frequency in Hz (60 kHz)
#     f_max = 150000  # Maximum frequency in Hz (150 kHz)

#     # Transform waveform to Mel Spectrogram
#     transform = T.MelSpectrogram(
#         sample_rate=sample_rate,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         n_mels=128,
#         f_min=f_min,
#         f_max=f_max,
#         power=2.0,
#         normalized=False,
#     )
#     mel_spectrogram = transform(waveform)

#     # Convert to decibels
#     mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

#     # Plotting
#     plt.figure(figsize=(10, 4))
#     plt.imshow(
#         mel_spectrogram_db.log2()[0, :, :].numpy(),
#         cmap="hot",
#         aspect="auto",
#         origin="lower",
#     )
#     plt.title("Mel Spectrogram (Torchaudio)")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Mel Frequency")
#     plt.colorbar(format="%+2.0f dB")

#     # Ensure the directory exists
#     directory_path = f"mel_spectrograms/{folder}"
#     if not os.path.exists(directory_path):
#         os.makedirs(directory_path)

#     # Save the figure
#     random_nr = np.random.randint(1000)
#     filename = f"{directory_path}/{random_nr}.png"
#     plt.savefig(filename)
#     plt.close()


def resample(signal, sr, target_sample_rate):
    resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
    signal = resampler(signal)
    return signal


def reshape(signal, target_size):
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


target_sr = 300000  # parameter, may be changed
target_size = 2 * target_sr  # parameter, may be changed
sample_rates = []
data_lengths = []
sample_files = list_files()
img_nr = 0
for folder, files in sample_files.items():
    print(folder)
    print(len(files))
    for file in files:
        if not os.path.exists(f"audio_clips/{folder}"):
            os.makedirs(f"audio_clips/{folder}")
        try:
            waveform, sample_rate = torchaudio.load(file)
        except Exception as e:
            print("error loading file", file, e)
            continue
        if sample_rate != target_sr:
            waveform = resample(waveform, sample_rate, target_sr)
            sample_rate = target_sr
            clips = reshape(waveform, target_size)
        # save audio clips
        for waveform in clips:
            # save audio waveform with same name as before
            filename = file.split("\\")[-1]
            print("filename", filename)
            try:
                torchaudio.save(
                    f"audio_clips/{folder}/img{img_nr}.wav", waveform, sample_rate
                )
                img_nr += 1
            except Exception as e:
                print("error saving file", filename, e)
            # plot_mel_spectrogram_torchaudio(waveform, sample_rate, folder, file)
        """sample_rates.append(sample_rate)
        num_channels, num_samples = wave_form.shape
        data_lengths.append(num_samples)
average_sample_rate = mean(sample_rates)
variance_sample = variance(sample_rates)
max_sample_rate = max(sample_rates)
min_sample_rate = min(sample_rates)
average_length = mean(data_lengths)
variance_lengths = variance(data_lengths)
max_data_lengths = max(data_lengths)
min_data_lengths = min(data_lengths)
print(average_sample_rate)
print(np.sqrt(variance_sample))
print("max", max_sample_rate)
print("min", min_sample_rate)
print(average_length)
print(np.sqrt(variance_lengths))
print("max", max_data_lengths)
print("min", min_data_lengths)"""
# plot_mel_spectrogram_torchaudio(one_wav_file)

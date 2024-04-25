import argparse
from pathlib import Path
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


def save_mel_spectrogram(waveform, sample_rate, filename):
    # waveform, sample_rate = torchaudio.load(wav_file)
    transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128, f_max=120000)
    mel_spectrogram = transform(waveform)

    fig = plt.figure(figsize=(10, 4), frameon=False)

    plt.imshow(
        mel_spectrogram.log2()[0, :, :].numpy(),
        cmap="hot",
        aspect="auto",
        origin="lower",
    )

    # plt.plot(mel_spectrogram.log2()[0, :, :].numpy(), cmap="hot", aspect="auto2", origin="lower")
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    # ax.plot(mel_spectrogram.log2()[0, :, :].numpy(), cmap="hot", aspect="auto", origin="lower")
    fig.add_axes(ax)
    plt.axis("off")
    # plt.title("Mel Spectrogram (Torchaudio)")
    # plt.xlabel("Time")
    # plt.ylabel("Frequency")
    # plt.colorbar(format="%+2.0f dB")
    # plt.show()
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

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument("destination_folder", help="Path to the destination folder")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    destination_folder = Path(args.destination_folder)

    print(f"\n\nInput folder: {input_folder.absolute()}")
    print(f"\n\nDestination folder: {destination_folder.absolute()}")
    i = input("Continue? y/n \n")
    if i.lower() != "y":
        print("Exiting...")
        return

    target_sr = 300000  # parameter, may be changed
    target_size = 2 * target_sr  # parameter, may be changed

    #Iterate thorough all folders

    for bat in input_folder.iterdir():
        print(bat.stem)

        for file in bat.iterdir():
            print(file.absolute())
            waveform, sample_rate = torchaudio.load(file.absolute())
            waveform = resample(waveform, sample_rate, target_sr)
            waveform = apply_bandpass_filter(waveform, target_sr, 20000, 120000)

            bat_path = Path(destination_folder / bat.stem)

            if not bat_path.exists():
                bat_path.mkdir(parents=True, exist_ok=True)

            save_destination = Path( destination_folder / bat.stem / file.stem)
            save_mel_spectrogram(waveform, target_sr, save_destination.absolute())


    # sample_rates = []
    # data_lengths = []
    # sample_files = list_files()
    # img_nr = 0
    # failed_files = 0
    # for folder, files in sample_files.items():
    #     print(folder)
    #     print(len(files))
    #     for file in files:
    #         # if not os.path.exists(f"audio_clips/{folder}"):
    #         #     os.makedirs(f"audio_clips/{folder}")
    #         try:
    #             waveform, sample_rate = torchaudio.load(file)
    #         except Exception as e:
    #             print("error loading file", file, e)
    #             continue
    #         # if sample_rate != target_sr:
    #         waveform = resample(waveform, sample_rate, target_sr)
    #         sample_rate = target_sr
    #         waveform = apply_bandpass_filter(waveform, sample_rate, 20000, 120000)
    #         clips = reshape(waveform, target_size)
    #         # save audio clips
    #         for waveform in clips:

    #             # save audio waveform with same name as before
    #             filename = file.split("\\")[-1]
    #             # print("filename", filename)
    #             filepath = f"{output_data}/{folder}/{filename}"
    #             if not os.path.exists(f"{output_data}/{folder}"):
    #                 os.makedirs(f"{output_data}/{folder}")
    #             try:
    #                 torchaudio.save(filepath, waveform, sample_rate)
    #                 img_nr += 1
    #             except Exception as e:
    #                 print("error saving file", filename, e)
    #                 failed_files += 1
    #     print("failed files", failed_files)
    #     print("success files", img_nr)



#Read files



#Split files


#Save files



if __name__ == "__main__":
    main()

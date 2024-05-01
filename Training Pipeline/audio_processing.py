from pathlib import Path
import torch
import torchaudio
import torchaudio


def exceeds_energy_threshold(waveform, absolute_threshold, max_count):

    waveform_1d = waveform[0] if waveform.dim() > 1 else waveform

    # Take the absolute value of the waveform
    waveform_abs = torch.abs(waveform_1d)

    # Count the number of points exceeding the absolute threshold
    exceed_count = torch.sum(waveform_abs > absolute_threshold)

    if exceed_count >= 15:
        return True
    else:
        return False


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
        if clip.shape[1] < target_size:
            num_missing_samples = target_size - clip.shape[1]
            last_dim_padding = (0, num_missing_samples)
            clip = torch.nn.functional.pad(clip, last_dim_padding)
        clips.append(clip)
        start_sample += target_size
    return clips


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
    """parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument("destination_folder", help="Path to the destination folder")
    args = parser.parse_args()"""
    target_sr = 300000  # parameter, may be changed
    target_size = 2 * target_sr  # parameter, may be changed
    low_freq = 20000
    high_freq = 120000

    # for noise/no noise
    absolute_threshold = 0.0035
    max_count = 15

    script_directory = Path(__file__).resolve().parent

    # Define input folder and destination folder paths (relative to the script directory)
    # input_folder = script_directory.parent / 'data' / 'chirovox' / 'all' #github data
    input_folder = script_directory.parent.parent / "dataset" / "labeled_dataset"
    # destination_folder = script_directory / 'training_data'
    destination_folder = script_directory.parent.parent / "dataset" / "training_data"

    # Print the absolute paths of the input folder and destination folder
    print(f"\n\nInput folder: {input_folder.resolve()}")
    print(f"Destination folder: {destination_folder.resolve()}")
    i = input("Continue? y/n \n")
    if i.lower() != "y":
        print("Exiting...")
        return

    # Iterate thorough all folders

    for bat in input_folder.iterdir():
        for file in bat.iterdir():
            waveform, sample_rate = torchaudio.load(file.absolute())
            waveform = resample(waveform, sample_rate, target_sr)
            waveform_clips = reshape(waveform, target_size)
            for i, clip in enumerate(waveform_clips):
                clip = apply_bandpass_filter(clip, target_sr, low_freq, high_freq)

                if exceeds_energy_threshold(clip, absolute_threshold, max_count):
                    bat_path = Path(destination_folder / bat.stem)
                    if not bat_path.exists():
                        bat_path.mkdir(parents=True, exist_ok=True)
                else:
                    path = "noise"
                    bat_path = Path(destination_folder / path)
                    if not bat_path.exists():
                        bat_path.mkdir(parents=True, exist_ok=True)

                clip_filename = file.stem + f"_{i}.wav"
                save_destination = Path(bat_path / clip_filename)
                torchaudio.save(uri=save_destination, src=clip, sample_rate=target_sr)


if __name__ == "__main__":
    main()

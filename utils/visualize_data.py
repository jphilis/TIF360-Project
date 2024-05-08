import matplotlib.pyplot as plt

from pathlib import Path

import numpy as np
import pandas as pd


def visualize_raw_data(data_path: Path):
    classes = [d for d in data_path.iterdir() if d.is_dir()]
    class_names = [bat_class.name.replace("_", " ") for bat_class in classes]

    dataset_files = {
        "chriovox": [],
        "bavaria": [],
        "stefan-nyman": [],
        "xeno-canto": [],
        "batcalls-com": [],
        "bat-recordings": [],
        "thomas-johanssen": [],
    }

    sizes = []

    fig = plt.figure(figsize=(10, 10))

    y_pos = np.arange(len(class_names))

    ax = fig.add_subplot(111)
    for bat_class in classes:
        size = get_directory_size(bat_class)
        sizes.append(size)
        for dataset in dataset_files.keys():
            files = list(bat_class.glob(f"*{dataset}.wav"))
            # print(files)
            file_sizes = sum([f.stat().st_size / (10**9) for f in files])
            dataset_files[dataset].append(file_sizes)

    # print(dataset_files)

    for dataset, sizes in dataset_files.items():
        hbars = ax.barh(y_pos, sizes, align="center", label=dataset)
        # ax.bar_label(hbars, fmt='%.3f GB')

    # hbars = ax.barh(y_pos, sizes, align='center', label="Raw data")
    ax.set_yticks(y_pos, labels=class_names)

    ax.invert_yaxis()
    ax.legend()
    ax.set_ylabel("Class")
    ax.set_xlabel("Size in GB")
    ax.set_title("Size of data per class")
    fig.tight_layout()
    plt.show()


def visualize_preprocessed_data():

    raise NotImplementedError()


def get_directory_size(root_directory: Path):
    total = sum(f.stat().st_size for f in root_directory.glob("**/*") if f.is_file())
    return total / (10**9)  # in GB


if __name__ == "__main__":
    current_file_path = Path(__file__).resolve()
    data_path = (
        current_file_path.parent.parent.parent / "dataset" / "training_data_2" / "train"
    )
    print("data path: ", data_path)
    visualize_raw_data(data_path)

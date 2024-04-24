import argparse
from pathlib import Path
import os
import shutil
import pandas as pd
import re


species_list = [
    "Myotis daubentonii",
    "Myotis dasycneme",   
    "Myotis brandtii",
    "Myotis mystacinus",
    "Myotis nattereri",
    "Myotis bechsteinii",
    "Myotis myotis",
    "Pipistrellus pygmaeus",
    "Pipistrellus pipistrellus",
    "Pipistrellus nathusii",
    "Nyctalus noctula",
    "Nyctalus leisleri",
    "Eptesicus serotinus",
    "Eptesicus nilssonii",
    "Vespertilio murinus",
    "Barbastella barbastellus",
    "Plecotus auritus",
    "Plecotus austriacus",
    "Myotis alcathoe"
]


def main():
    parser = argparse.ArgumentParser(description='Reorder files in the input folder.')
    parser.add_argument('input_folder', help='Path to the input folder')
    parser.add_argument('destination_folder', help='Path to the destination folder')
    parser.add_argument('dataset_name', help='Name of the dataset to be appended to the file names.')
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    destination_folder = Path(args.destination_folder)

    print(f"\n\nInput folder: {input_folder.absolute()}")
    print(f"\n\Destination folder: {destination_folder.absolute()}")
    i = input("Continue? y/n \n")
    if i.lower() != 'y':
        print("Exiting...")
        return
    
    #Look for dataset
    match args.dataset_name:
        case "chriovox":
            space_or_underscore(input_folder, destination_folder, args.dataset_name)
        case "bavaria":
            space_or_underscore(input_folder, destination_folder, args.dataset_name)
        case "stefan-nyman":
            space_or_underscore(input_folder, destination_folder, args.dataset_name)
        case "xeno-canto":
            space_or_underscore(input_folder, destination_folder, args.dataset_name)
        case "batcalls-com":
            space_or_underscore(input_folder, destination_folder, args.dataset_name)
        case "bat-recordings":
            space_or_underscore(input_folder, destination_folder, args.dataset_name)
        case "thomas-johanssen":
            johanssen(input_folder, destination_folder, args.dataset_name)

        case _:
            print("Dataset not found")
            return
    

    # Rest of your code goes here


def johanssen(input_folder: Path, destination_folder: Path, dataset_name: str) -> None:
    
    df = pd.read_excel("utils\Data_Faxe_2023_TWJohansen Thomas W. Johansen.xlsx")
    for i, row in df.iterrows():
        #Folder
        one = row["boks"].replace(" ", "_").lower()
        two = row["projekt"]
        three = "#" 
        four = row["Lokalitet"]
        five = str(row["lat"]) .replace(".", ",")
        six = str(row["lon"]).replace(".", ",")

        #Filename
        seven = row["dato"]
        eight = row["tid"]
        nine = row["m.sek"]
        if nine == 0:
            nine = "000"

        folder_name = f"{one}_{two}_{three}_{four}_{five}_{six}"
        file_name = f"{folder_name}_{seven}_{eight}_{nine}.wav"

        art = str(row["art3"]).replace(" ", "_").lower()
        print(art)
    
        path = f"{folder_name}/{file_name}"
        print(path)

        bat_path = Path(destination_folder / art)
        bat_path.mkdir(parents=True, exist_ok=True)

        new_file_path = Path(bat_path / file_name)
        shutil.copy2(input_folder / path, new_file_path)
    

def space_or_underscore(input_folder: Path, destination_folder: Path, dataset_name: str) -> None:

    # print(input_folders)

    for folder in input_folder.iterdir():
        if (not folder.is_dir()):
            continue
        folder_name = "_".join(re.split(r'[_\s]+', folder.name)[:2]).lower()
        if folder_name not in [s.replace(" ", "_").lower() for s in species_list]:
            print(f"Folder {folder_name} not in species list")
            continue

        print(folder_name)

        bat_path = Path(destination_folder / folder_name)
        bat_path.mkdir(parents=True, exist_ok=True)

        for i, file in enumerate(folder.glob("*.wav")):
            new_name = f"{file.stem}_{dataset_name}.wav"
            # new_path = destination_folder.joinpath(folder_name).joinpath(new_name)

            new_file_path = Path(bat_path / new_name)

            shutil.copy2(file.absolute(), new_file_path.absolute())

            print(f"Renaming {file.absolute()} to {new_name} in {new_file_path.absolute()}")


if __name__ == "__main__":
    main()

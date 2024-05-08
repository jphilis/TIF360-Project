import argparse
from pathlib import Path
import os
import shutil
import pandas as pd
import re
from tqdm import tqdm


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
    "Myotis alcathoe",
]


def main():
    parser = argparse.ArgumentParser(description="Reorder files in the input folder.")
    parser.add_argument("input_folder", help="Path to the input folder")
    parser.add_argument("destination_folder", help="Path to the destination folder")
    parser.add_argument(
        "dataset_name", help="Name of the dataset to be appended to the file names."
    )
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    destination_folder = Path(args.destination_folder)

    print(f"\n\nInput folder: {input_folder.absolute()}")
    print(f"\n\Destination folder: {destination_folder.absolute()}")
    i = input("Continue? y/n \n")
    if i.lower() != "y":
        print("Exiting...")
        return

    # Look for dataset
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
            space_or_underscore(input_folder, destination_folder, args.dataset_name)
            # johanssen(input_folder, destination_folder, args.dataset_name)

        case _:
            print("Dataset not found")
            return

    # Rest of your code goes here


def johanssen(input_folder: Path, destination_folder: Path, excel_path: Path,) -> None:

    df = pd.read_excel(excel_path)
    # df["art3"] = str(df["art3"]).replace("/", "").lower()
    suceeded_files = 0
    failed_files = 0
    folder_that_failed = {}
    all_file_paths = []
    for i, row in df.iterrows():
        # Folder
        one = row["boks"]
        two = row["projekt"]
        three = "#"
        four = row["Lokalitet"]
        five = str(row["lat"]).replace(".", ",")
        six = str(row["lon"]).replace(".", ",")

        # Filename
        seven = row["dato"]
        eight = row["tid"]
        eight = f"{eight:06}"
        nine = f"{row["m.sek"]:03}"
        if one == "gamma":
            four = "51a"
        if one == "twj-05":
            four = "51b"
        

        folder_name = f"{one}_{two}_{three}_{four}_{five}_{six}"
        file_name = f"{folder_name}_{seven}_{eight}_{nine}.wav"

        art = str(row["art3"]).replace(" ", "_").replace("/", "").lower()

        # print(art)

        bat_path = Path(destination_folder / art)
        bat_path.mkdir(parents=True, exist_ok=True)
        new_file_path = Path(bat_path / file_name)
        try:
            path = f"{folder_name}/{file_name}"
            # print("path orig", path)
            if not (input_folder / path).exists():
                folder_name = f"{one}_{two}_{three}_ekstra 1_{five}_{six}"
                file_name = f"{folder_name}_{seven}_{eight}_{nine}.wav"
            # if not (input_folder / path).exists():
            #     folder_name = f"{one}_{two}_{three}_{four}_{five}_{six}"
            #     file_name = f"{folder_name}_{seven}_{eight}_{nine}.wav"
            path = f"{folder_name}/{file_name}"
            # print("path new", path)
            shutil.copy2(input_folder / path, new_file_path)

            all_file_paths.append((file_name, new_file_path))
            suceeded_files += 1
        except Exception as e:
            print(f"Failed to copy {path} to {new_file_path}")
            print(e)
            failed_files += 1
            folder_that_failed[folder_name] = path
            # raise e
    print(f"Succeeded files: {suceeded_files}")
    print(f"Failed files: {failed_files}")
    print("folders that failed", folder_that_failed)
    return all_file_paths

def look_for_duplicates(all_file_paths):
    duplicates = []
    for i, file_tuple in enumerate(all_file_paths):
        file_name, new_file_path = file_tuple
        for j, file_tuple2 in enumerate(all_file_paths):
            file_name2, new_file_path2 = file_tuple2
            if j <= i:
                continue
            if file_name == file_name2:
                duplicates.extend((new_file_path, new_file_path2))
                #dleete the new_file_path2
                try:
                    os.remove(new_file_path)
                    os.remove(new_file_path2)
                
                except Exception as e:
                    pass
    print("Duplicates", duplicates)
    print("len duplicates", len(duplicates))


from pathlib import Path

def rename_folders(input_folder: Path):
    bat_species_to_scientific = {
        "brunflagermus": "myotis_brandtii",
        "sydflagermus": "nyctalus_leisleri",
        "vandflagermus": "myotis_daubentonii",
        "dværgflagermus": "pipistrellus_pipistrellus",
        "brun-/skimmel-/sydflagermus": "nyctalus_noctula",
        "troldflagermus": "plecotus_auritus",
        "damflagermus": "myotis_dasycneme",
        "frynseflagermus": "myotis_nattereri",
        "myotis_sp": "myotis_sp",
        "skimmelflagermus": "myotis_bechsteinii",
        "dam-/vandflagermus": "myotis_dasycneme",
        "bredøret_flagermus": "plecotus_auritus",
        "brandts-/bechsteins-/skægflagermus": "myotis_brandtii_/_myotis_bechsteinii_/_myotis_mystacinus",
        "støj": "noise",
        "ubestemt_flagermus": "undetermined_bat",
        "brun_langøre": "plecotus_austriacus"
    }
    for folder_path in input_folder.iterdir():
        folder_name = folder_path.name.lower()
        print("folder_name", folder_name)
        if folder_name in bat_species_to_scientific:
            new_folder_name = bat_species_to_scientific[folder_name]
            print("rename folder", folder_name, "to", new_folder_name)
            folder_path.rename(folder_path.parent / new_folder_name)


def space_or_underscore(
    input_folder: Path, destination_folder: Path, dataset_name: str
) -> None:

    # print(input_folders)

    for folder in input_folder.iterdir():
        if not folder.is_dir():
            continue
        folder_name = "_".join(re.split(r"[_\s]+", folder.name)[:2]).lower()
        if folder_name not in [s.replace(" ", "_").lower() for s in species_list]:
            print(f"Folder {folder_name} not in species list")
            continue

        print(folder_name)

        bat_path = Path(destination_folder / folder_name)
        bat_path.mkdir(parents=True, exist_ok=True)

        for i, file in tqdm(enumerate(folder.glob("*.wav"))):
            
            new_name = f"{file.stem}_{dataset_name}.wav"
            # new_path = destination_folder.joinpath(folder_name).joinpath(new_name)

            new_file_path = Path(bat_path / new_name)

            shutil.copy2(file.absolute(), new_file_path.absolute())

            print(
                f"Renaming {file.absolute()} to {new_name} in {new_file_path.absolute()}"
            )


if __name__ == "__main__":
    # main()
    source_folder = r"C:\Users\jonat\OneDrive\chalmers\Advanced neural networks\project\dataset\raw data\Thomas W johansen_4"
    destination_folder = r"C:\Users\jonat\OneDrive\chalmers\Advanced neural networks\project\dataset\raw data\Thomas W johansen_sorted_4"
    excel_path = f"{source_folder}/excel_data.xlsx"
    main()
    # all_file = johanssen(Path(source_folder), Path(destination_folder), Path(excel_path))
    # rename_folders(Path(destination_folder))


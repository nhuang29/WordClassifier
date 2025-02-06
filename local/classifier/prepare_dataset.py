import librosa
import os
import json

# Setting paths
DATASET_PATH = "../dataset"
JSON_PATH = "data.json"

# 22050 is 1 sec worth of sound
SAMPLES_TO_CONSIDER = 22050

# Hop-Length is division of audio file into 512 frames
def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    # data dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # Loop through all the sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # We need to ensure we are not at a root level.
        if dirpath is not dataset_path:

            # update the mappings
            category = dirpath.split("/")[-1]  # splits: dataset/down --> ["dataset", "down", ...]
            data["mappings"].append(category)
            print(f"Processing {category}")

            # loop through all the file names and extract MFCCs
            for f in filenames:

                if not f.endswith('.wav'):
                    continue

                # get the file path
                file_path = os.path.join(dirpath, f)

                # load audio file
                signal, sr = librosa.load(file_path)

                # ensure the audio file is at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # enfroce 1 second long signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract the MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal,
                                                 n_mfcc=n_mfcc,
                                                 hop_length=hop_length,
                                                 n_fft=n_fft)

                    # store data
                    data["labels"].append(i - 1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)

                    print(f"{file_path}: {i - 1}")

    # store in a json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)
    print("Finished Processing All Files.")
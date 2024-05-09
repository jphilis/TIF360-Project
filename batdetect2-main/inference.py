from batdetect2 import api
import os

current_data_path = os.path.dirname(os.path.abspath(__file__))
parent_data_path = os.path.dirname(current_data_path)
parent_parent_data_path = os.path.dirname(parent_data_path)


path_to_data = os.path.join(current_data_path, "example_data", "audio")
# path_to_data = r"C:\Users\jonat\OneDrive\chalmers\Advanced_neural_networks\project\dataset\training_data_short\train\myotis_alcathoe"
# audio file should be the first .wav file in the folder
AUDIO_FILE = os.path.join(path_to_data, os.listdir(path_to_data)[1])

# Process a whole file
results = api.process_file(AUDIO_FILE)

# Or, load audio and compute spectrograms
audio = api.load_audio(AUDIO_FILE)
spec = api.generate_spectrogram(audio)
print("Spectrogram shape:", spec.shape)
print("Spectrogram dtype:", spec.dtype)
# plot the spec
import matplotlib.pyplot as plt

# Invalid shape (1, 1, 128, 32) for image data

plt.imshow(spec[0, 0, :, :])
plt.show()

# And process the audio or the spectrogram with the model
detections, features, spec = api.process_audio(audio)
detections, features = api.process_spectrogram(spec)

print(detections, "detections")

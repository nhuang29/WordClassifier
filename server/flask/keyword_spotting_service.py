import keras as keras
import numpy as np
import librosa

MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050


# A singleton class (only can have 1 instance)
class _Keyword_Spotting_Service:
    model = None
    _mappings = [
        "right",
        "go",
        "no",
        "left",
        "stop",
        "up",
        "down",
        "yes",
        "on",
        "off"
    ]
    _instance = None  # Python has no singleton so you need this variable

    def predict(self, file_path):
        # extract the data and create the MFCC of the audio file
        # Expecetd output --> (# segments, # of coefficients)
        MFCCs = self.preprocess(file_path)

        # convert 2d MFCC array into 4d array -> (# samples, # segments, # coefficients, # channels)
        # The added front dimension specifies the number of samples in the batch
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make predictions
        # Predict returns 2D array: [[0.1, 0.6, ...], [], []], the internal array represents the number of samples
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, hop_length=512, n_fft=2048):
        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(y=signal,
                                     n_mfcc=n_mfcc,
                                     hop_length=hop_length,
                                     n_fft=n_fft)
        return MFCCs.T


def Keyword_Spotting_Service():
    # ensure that we only have 1 instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance

if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    keyword1 = kss.predict("test/down.wav")

    print(f"Predicted keywords: {keyword1}")
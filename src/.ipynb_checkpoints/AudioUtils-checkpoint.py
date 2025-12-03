import tensorflow as tf
import numpy as np
import librosa

class AudioUtils:
    def load_wav_16k_mono(self, filename):
    audio, _ = librosa.load(filename, sr=16000, mono=True)
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    return audio
import numpy as np
import librosa
import random

class AudioMutations:

    def enforce_one_second(self, data, sr):
        target_length = sr
        if len(data) == target_length: return data
        if len(data) > target_length:
            start = np.random.randint(0, len(data) - target_length + 1)
            return data[start : start + target_length]
        else:
            return np.pad(data, (0, target_length - len(data)), mode='constant')
        
    def add_white_noise(self, data, noise_factor=0.005):
        """
        Adds random static noise to the audio.
        Great for making the model ignore background hiss.
        """
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        # Cast back to same data type
        return augmented_data.astype(type(data[0]))

    def time_stretch(self, data, rate=1.0):
        """
        Speeds up or slows down the audio without changing pitch.
        Rate > 1.0 speeds up, Rate < 1.0 slows down.
        """
        return librosa.effects.time_stretch(y=data, rate=rate)

    def pitch_shift(self, data, sampling_rate, n_steps=0.0):
        """
        Changes the pitch without changing the speed.
        n_steps: fraction of a semitone. Positive is higher, negative is lower.
        """
        return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=n_steps)

    def shift_time(self, data, shift_max=0.2, sampling_rate=16000):
        """
        Shifts the audio start time (rolling). 
        It moves the end to the start or vice versa.
        shift_max: max seconds to shift.
        """
        shift = np.random.randint(sampling_rate * shift_max)
        return np.roll(data, shift)

    def change_volume(self, data, factor=1.0):
        """
        Makes audio louder or quieter.
        factor > 1 is louder, < 1 is quieter.
        """
        return data * factor
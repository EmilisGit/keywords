import numpy as np
import librosa
import random
import soundfile as sf # Added for saving the audio file

class AudioMutations:

    def _load_and_mutate(self, file_path, mutate_func, output_path=None, **kwargs):
        try:
            data, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None, None
        
        augmented_data = mutate_func(data, sr, **kwargs)
        if output_path:
            try:
                sf.write(output_path, augmented_data, sr)
                print(f"Mutated audio saved to: {output_path}")
            except Exception as e:
                print(f"Error saving file to {output_path}: {e}")
        
        # 4. Return the modified data and sampling rate
        return augmented_data, sr
    
    # --- Mutation Logic Functions (Internal) ---
    # These functions now only contain the core mutation logic
    
    def _enforce_one_second_logic(self, data, sr):
        """Core logic for enforce_one_second."""
        target_length = sr
        if len(data) == target_length: 
            return data
        if len(data) > target_length:
            start = np.random.randint(0, len(data) - target_length + 1)
            return data[start : start + target_length]
        else:
            return np.pad(data, (0, target_length - len(data)), mode='constant')
        
    def _add_white_noise_logic(self, data, sr, noise_factor):
        """Core logic for add_white_noise."""
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        # Cast back to same data type
        return augmented_data.astype(data.dtype)

    def _time_stretch_logic(self, data, sr, rate):
        """Core logic for time_stretch."""
        return librosa.effects.time_stretch(y=data, rate=rate)

    def _pitch_shift_logic(self, data, sr, n_steps):
        """Core logic for pitch_shift."""
        # 'sr' is now available from the loaded file
        return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

    def _shift_time_logic(self, data, sr, shift_max):
        """Core logic for shift_time."""
        # 'sr' is now available from the loaded file
        shift = np.random.randint(sr * shift_max)
        return np.roll(data, shift)

    def _change_volume_logic(self, data, sr, factor):
        """Core logic for change_volume."""
        return data * factor

    # --- Public Refactored Methods (Accepting file_path) ---

    def enforce_one_second(self, file_path, output_path=None):
        return self._load_and_mutate(
            file_path, 
            self._enforce_one_second_logic, 
            output_path=output_path
        )

    def add_white_noise(self, file_path, noise_factor=0.005, output_path=None):
        return self._load_and_mutate(
            file_path, 
            self._add_white_noise_logic, 
            output_path=output_path,
            noise_factor=noise_factor
        )

    def time_stretch(self, file_path, rate=1.0, output_path=None):
        return self._load_and_mutate(
            file_path, 
            self._time_stretch_logic, 
            output_path=output_path,
            rate=rate
        )

    def pitch_shift(self, file_path, n_steps=0.0, output_path=None):
        # sampling_rate is now automatically loaded and passed via 'sr'
        return self._load_and_mutate(
            file_path, 
            self._pitch_shift_logic, 
            output_path=output_path,
            n_steps=n_steps
        )

    def shift_time(self, file_path, shift_max=0.2, output_path=None):
        # sampling_rate is now automatically loaded and passed via 'sr'
        return self._load_and_mutate(
            file_path, 
            self._shift_time_logic, 
            output_path=output_path,
            shift_max=shift_max
        )

    def change_volume(self, file_path, factor=1.0, output_path=None):
        return self._load_and_mutate(
            file_path, 
            self._change_volume_logic, 
            output_path=output_path,
            factor=factor
        )
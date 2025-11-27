import numpy as np
import os
from rich import print
import matplotlib.pyplot as plt
import wave

def read_clip(filePath):
    audio = wave.open(filePath, "rb") 
    sample_freq = audio.getframerate()
    n_samples = audio.getnframes() 
    signal_wave = audio.readframes(-1) 

    audio.close()

    t_audio = n_samples / sample_freq

    signal_array = np.frombuffer(signal_wave, dtype=np.int16)
    times = np.linspace(0, t_audio, num=n_samples)
    return signal_array, times

def plot_wave(files : list[str], figsize: tuple[int, int] = (15, 8)):
    plt.figure(figsize=figsize)
    for i, file in enumerate(files):
        signal_array, times = read_clip(file)
        plt.subplot(2, 3, i+1) 
        plt.plot(times, signal_array)
        plt.title(file)
        plt.ylabel("Signal Wave")
        plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    files = os.listdir("../Alarms")
    plt.figure(figsize=(16, 10))

    for i, file in enumerate(files):
        signal_array, times = read_clip(f'../Alarms/{file}')
        plt.subplot(2, 3, i+1) 
        plt.plot(times, signal_array)
        plt.title(file)
        plt.ylabel("Signal Wave")
        plt.xlabel("Time (s)")

    plt.show()
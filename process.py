import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    audio_file = "audio-data.wav"
    plotWaveform = False
    plotSpectrum = False
    plotSpectogram = False
    plotMFCC = False

    '''
    Generate Waveform
    '''
    # signal = sr * T = 22050 * 30
    signal, sr = librosa.load(audio_file, sr=22050)

    # Plot Waveform
    if plotWaveform:
        librosa.display.waveplot(signal, sr=sr)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

    '''
    Waveform -> [FFT] -> Spectrum
    (time domain -> frequency domain)
    '''
    fft = np.fft.fft(signal)
    magnitude = np.abs(fft)
    frequency = np.linspace(0, sr, len(magnitude))

    # Crop the symmetric right half [redundant information]
    mid = int(len(frequency)/2)
    left_frequency = frequency[:mid]
    left_magnitude = magnitude[:mid]

    # Plot Spectrum
    if plotSpectrum:
        plt.plot(left_frequency, left_magnitude)
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.show()

    '''
    Waveform -> [STFT] -> Spectogram
    (time domain -> frequency domain)
    '''
    n_fft = 2048
    hop_length = 512

    stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    spectogram = np.abs(stft)
    lot_spectogram = librosa.amplitude_to_db(spectogram)

    # Plot Spectogram
    if plotSpectogram:
        librosa.display.specshow(spectogram, sr=sr, hop_length=hop_length)
        librosa.display.specshow(lot_spectogram, sr=sr, hop_length=hop_length)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.show()

    '''
    Waveform -> [MFCCs] -> MFCCs (Mel-frequency cepstrum)
    '''
    MFCCs = librosa.feature.mfcc(
        signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

    # Plot MFCCs
    if plotMFCC:
        librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
        plt.xlabel("Time")
        plt.ylabel("MFCC")
        plt.colorbar()
        plt.show()

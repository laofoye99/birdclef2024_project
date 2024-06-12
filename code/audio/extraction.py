# _*_ coding: utf-8 _*_
from typing import Tuple
import numpy as np
import librosa

def compute_stft(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform (STFT) of audio signal
    Args:
        audio (np.ndarray): Audio signal

    Returns:
        Tuple[np.ndarray, np.ndarray]: Magnitude and phase of STFT
    """
    stft_audio = librosa.stft(audio)
    magnitude_audio = np.mean(np.abs(stft_audio))
    phase_audio = np.mean(np.angle(stft_audio))
    return magnitude_audio, phase_audio

def extract_mfccs(audio: np.ndarray, sample_rate: int, n_mfcc=13) -> np.ndarray:
    """
    Extract Mel-frequency cepstral coefficients (MFCCs) from audio signal
    Args:
        audio (np.ndarray): Audio signal
        sample_rate (int): Sample rate of audio signal
        n_mfcc (int): Number of MFCCs to extract
    Returns:
        mfccs (): MFCCs
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs)

def extract_chroma(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extract chroma feature from audio signal
    Args:
        audio (np.ndarray): Audio signal
        sample_rate (int): Sample rate of audio signal
    Returns:
        chroma (np.ndarray): Chroma feature
    """
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    return np.mean(chroma)

def extract_mel_spectrogram(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extract mel spectrogram from audio signal
    Args:
        audio (np.ndarray): Audio signal
        sample_rate (int): Sample rate of audio signal
    Returns:
        mel_spectrogram (np.ndarray): Mel spectrogram
    """
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    return np.mean(mel_spectrogram)

def extract_audio(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract audio features
    Args:
        audio (np.ndarray): Audio signal
        sample_rate (int): Sample rate of audio signal
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Magnitude, phase, MFCCs, Chroma, Mel spectrogram
    """
    magnitude_audio, phase_audio = compute_stft(audio)
    mfccs = extract_mfccs(audio, sample_rate)
    chroma = extract_chroma(audio, sample_rate)
    mel_spectrogram = extract_mel_spectrogram(audio, sample_rate)
    return magnitude_audio, phase_audio, mfccs, chroma, mel_spectrogram
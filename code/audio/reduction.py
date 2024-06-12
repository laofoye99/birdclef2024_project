# _*_ coding: utf-8 _*_
import numpy as np
import librosa

def spectral_gate(audio: np.ndarray, noise_reduce_factor: float=0.02) -> np.ndarray:
    """
    Spectral gating for noise reduction
    Args:
        audio (np.ndarray): Audio data
        noise_reduce_factor (float, optional): factor to reduce noise. Defaults to 0.02.

    Returns:
        audio_denoised (np.ndarray): Denoised audio
    """
    # Compute the STFT of the audio signal
    stft = librosa.stft(audio, n_fft=1024, hop_length=512, win_length=1024)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate the noise power
    noise_power = np.mean(magnitude[:, :5], axis=1, keepdims=True)
    gated_magnitude = np.maximum(magnitude - noise_reduce_factor * noise_power, 0.0)
    
    # Reconstruct the STFT
    gated_stft = gated_magnitude * np.exp(1j * phase)
    audio_denoised = librosa.istft(gated_stft, hop_length=512, win_length=1024)
    
    return audio_denoised

def energy_based_silence_removal(audio: np.ndarray, sample_rate: int, energy_threshold: float=0.01) -> np.ndarray:
    """
    Remove silence from audio based on energy threshold
    Args:
        audio (np.ndarray): Audio data
        sample_rate (int): Sample rate
        energy_threshold (float, optional): Energy threshold. Defaults to 0.01.

    Returns:
        audio[mask] (np.ndarray): Cleaned audio
    """
    frame_length = int(sample_rate * 0.02)
    frame_step = int(sample_rate * 0.01)
    energy = np.array([
        sum(abs(audio[i:i+frame_length]**2))
        for i in range(0, len(audio), frame_step)
    ])
    
    mask = energy > energy_threshold
    mask = np.repeat(mask, frame_step)
    mask = mask[:len(audio)]
    
    return audio[mask]

def trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Trim silence from audio
    Args:
        audio (np.ndarray): Audio data
        sample_rate (int): Sample rate

    Returns:
        np.ndarray: Trimmed audio
    """
    y_trimmed, _ = librosa.effects.trim(audio, top_db=15)
    return y_trimmed

def clean_audio(audio: np.ndarray, sample_rate: int, noise_reduce_factor: float=0.02, energy_threshold: float=0.01) -> np.ndarray:
    """
    Clean audio by reducing noise and removing silence
    Args:
        audio (np.ndarray): Audio data
        sample_rate (int): Sample rate
        noise_reduce_factor (float, optional): factor to reduce noise. Defaults to 0.02.
        energy_threshold (float, optional): Energy threshold. Defaults to 0.01.
    Returns:
        audio_clean (np.ndarray): Cleaned audio
    """
    audio_denoised = spectral_gate(audio, noise_reduce_factor)
    audio_clean = energy_based_silence_removal(audio_denoised, sample_rate, energy_threshold)
    
    return audio_clean

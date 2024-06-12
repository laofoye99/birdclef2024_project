# _*_ coding: utf-8 _*_
import random
import numpy as np
import librosa

def add_white_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """
    Add white noise to audio signal
    Args:
        audio (np.ndarray): Audio signal
        noise_factor (float, optional): Factor to control the amount of noise to add. Defaults to 0.005.

    Returns:
        np.ndarray: Augmented audio signal
    """
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def shift_time(audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
    """
    Shift audio in time domain
    Args:
        audio (np.ndarray): Audio signal
        shift_max (float, optional): Maximum shift value. Defaults to 0.2.

    Returns:
        np.ndarray: Shifted audio signal
    """
    shift = np.random.randint(int(len(audio) * shift_max))
    return np.roll(audio, shift)

def change_pitch(audio: np.ndarray, sample_rate: int, pitch_factor: float = 2.0) -> np.ndarray:
    """
    Change pitch of audio signal
    Args:
        audio (np.ndarray): Audio signal
        sample_rate (int): Sample rate of audio signal
        pitch_factor (float, optional): Factor to control the pitch change. Defaults to 2.0.

    Returns:
        np.ndarray: Augmented audio signal
    """
    return librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=pitch_factor)

def change_speed(audio: np.ndarray, speed_factor: float = 1.5) -> np.ndarray:
    """
    Change speed of audio signal
    Args:
        audio (np.ndarray): Audio signal
        speed_factor (float, optional): Factor to control the speed change. Defaults to 1.5.

    Returns:
        np.ndarray: Augmented audio signal
    """
    return librosa.effects.time_stretch(y=audio, rate=speed_factor)

def time_stretch(audio: np.ndarray, stretch_factor: float = 0.5) -> np.ndarray:
    """
    Time stretch audio signal
    Args:
        audio (np.ndarray): Audio signal
        stretch_factor (float, optional): Factor to control the time stretch. Defaults to 1.2.

    Returns:
        np.ndarray: Augmented audio signal
    """
    return librosa.effects.time_stretch(y=audio, rate=stretch_factor)

def change_volume(audio: np.ndarray, volume_factor: float = 1.5) -> np.ndarray:
    """
    Change volume of audio signal
    Args:
        audio (np.ndarray): Audio signal
        volume_factor (float, optional): Factor to control the volume change. Defaults to 1.5.

    Returns:
        np.ndarray: Augmented audio signal
    """
    return audio * volume_factor

def augment_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Augment audio signal
    Args:
        audio (np.ndarray): Audio signal
        sample_rate (int): Sample rate of audio signal

    Returns:
        np.ndarray: Augmented audio signal
    """
    augmentations = [
        add_white_noise,
        shift_time,
        lambda x: change_pitch(x, sample_rate),
        change_speed,
        time_stretch,
        change_volume
    ]
    augmented_audio = audio.copy()
    chosen_augmentations = random.sample(augmentations, random.randint(1, len(augmentations)))
    for augmentation in chosen_augmentations:
        augmented_audio = augmentation(augmented_audio)
    return augmented_audio

# _*_ coding: utf-8 _*_
from typing import Tuple
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_io as tfio

def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file
    Args:
        file_path (str): Path to the audio file
    Returns:
        Tuple containing:
            - audio (np.ndarray): containing audio data
            - sample_rate (int): Sample rate of audio
    """
    audio, sample_rate = librosa.load(file_path, sr=None)
    return audio, sample_rate

def _get_audio_length(audio: np.ndarray, sample_rate: int) -> int:
    """
    Get audio duration
    Args:
        audio (np.ndarray): Audio data
        sample_rate (int): Sample rate of audio
    Returns:
        duration (int): Duration of audio in seconds
    """
    duration = audio.shape[0]/sample_rate
    return duration

def resample_audio(audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """
    Change audio sample rate
    Args:
        audio: AudioIOTensor object
        target_sample_rate: Target sample rate
    Returns:
        audio_resample: Resampled audio tensor
    """
    if sample_rate != target_sample_rate:
        audio_resample = tfio.audio.resample(audio, rate_in=sample_rate, rate_out=target_sample_rate, name='resampled_audio')
        return audio_resample.numpy()
    else:
        return audio
    
def crop_audio(audio: np.ndarray, sample_rate:int, target_length: int) -> tf.Tensor:
    """
    Crop audio to the target length
    Args:
        audio: Audio tensor
        length: Target length
    Returns:
        audio_crop: Cropped audio tensor
    """
    audio_length = _get_audio_length(audio, sample_rate)
    target_samples = target_length * sample_rate
    if audio_length > target_length:
        audio_crop = audio[:target_samples]
    else: 
        repeat_count = int(np.ceil(target_samples / audio.shape[0]))
        audio_repeated = np.tile(audio, repeat_count)
        audio_crop = audio_repeated[:target_samples]
    return audio_crop

def standardize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Standardize audio
    Args:
        audio (np.ndarray): Audio data
    Returns:
        audio_standardize (np.ndarray): Standardized audio data
    """
    mean = np.mean(audio)
    std = np.std(audio)
    audio_standardize = np.where((mean == 0 and std == 1), audio - mean, (audio - mean) / std)
    return audio_standardize

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio
    Args:
        audio (np.ndarray): Audio data
    Returns:
        audio_normalize (np.ndarray): Normalized audio data
    """
    max_val = np.max(np.abs(audio))
    audio_normalize = np.where(max_val == 1.0, audio, audio / max_val)
    return audio_normalize

def process_audio(file_path: str, target_sample_rate: int, target_length: int) -> Tuple[np.ndarray, int]:
    """
    Process audio data
    Args:
        audio (np.ndarray): Audio data
        sample_rate (int): Sample rate of audio
        target_sample_rate (int): Target sample rate
        target_length (int): Target length of audio
    Returns:
        audio_processed (np.ndarray): Processed audio data
    """
    audio, sample_rate = load_audio(file_path)
    audio_resample = resample_audio(audio, sample_rate, target_sample_rate)
    audio_crop = crop_audio(audio_resample, target_sample_rate, target_length)
    audio_standardize = standardize_audio(audio_crop)
    audio_normalize = normalize_audio(audio_standardize)
    return audio_normalize, sample_rate

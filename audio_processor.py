"""
Audio Processing Module for Speech-to-Text System
Handles audio loading, preprocessing, feature extraction, and augmentation
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Optional, Union
import warnings

from config import AudioConfig, get_config

warnings.filterwarnings('ignore')


class AudioProcessor:
    """
    Comprehensive audio processing pipeline for ASR systems
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize audio processor
        
        Args:
            config: AudioConfig instance with processing parameters
        """
        self.config = config or get_config().audio
    
    def load_audio(
        self, 
        audio_path: Union[str, Path],
        sr: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file from disk
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate (use self.config.sample_rate if None)
            mono: Convert to mono if True
            
        Returns:
            audio_data: Audio time series
            sr: Sample rate
        """
        sr = sr or self.config.sample_rate
        
        try:
            audio_data, sample_rate = librosa.load(
                audio_path,
                sr=sr,
                mono=mono
            )
            return audio_data, sample_rate
        except Exception as e:
            raise RuntimeError(f"Error loading audio from {audio_path}: {str(e)}")
    
    def save_audio(
        self,
        audio_data: np.ndarray,
        output_path: Union[str, Path],
        sr: int
    ) -> None:
        """Save audio data to file"""
        sf.write(str(output_path), audio_data, sr)
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio: Audio time series
            
        Returns:
            Normalized audio
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def trim_silence(
        self,
        audio: np.ndarray,
        sr: int,
        threshold_db: float = -40
    ) -> np.ndarray:
        """
        Trim leading and trailing silence
        
        Args:
            audio: Audio time series
            sr: Sample rate
            threshold_db: Threshold for silence detection
            
        Returns:
            Trimmed audio
        """
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=abs(threshold_db)
        )
        return trimmed
    
    def extract_mfcc(
        self,
        audio: np.ndarray,
        sr: int,
        n_mfcc: Optional[int] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients)
        
        Args:
            audio: Audio time series
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            MFCC features (n_mfcc, time_steps)
        """
        n_mfcc = n_mfcc or self.config.n_mfcc
        n_fft = n_fft or self.config.n_fft
        hop_length = hop_length or self.config.hop_length
        
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        return mfcc
    
    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        sr: int,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_mels: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract Mel-scale spectrogram
        
        Args:
            audio: Audio time series
            sr: Sample rate
            n_fft: FFT window size
            hop_length: Hop length
            n_mels: Number of Mel bands
            
        Returns:
            Mel spectrogram (n_mels, time_steps)
        """
        n_fft = n_fft or self.config.n_fft
        hop_length = hop_length or self.config.hop_length
        n_mels = n_mels or self.config.mel_bins
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_zero_crossing_rate(
        self,
        audio: np.ndarray,
        hop_length: Optional[int] = None
    ) -> np.ndarray:
        """Extract zero crossing rate"""
        hop_length = hop_length or self.config.hop_length
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            hop_length=hop_length
        )
        return zcr
    
    def extract_spectral_centroid(
        self,
        audio: np.ndarray,
        sr: int,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None
    ) -> np.ndarray:
        """Extract spectral centroid"""
        n_fft = n_fft or self.config.n_fft
        hop_length = hop_length or self.config.hop_length
        
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        return spectral_centroids
    
    def data_augmentation(
        self,
        audio: np.ndarray,
        sr: int,
        augmentation_type: str = "pitch_shift"
    ) -> np.ndarray:
        """
        Apply data augmentation to audio
        
        Args:
            audio: Audio time series
            sr: Sample rate
            augmentation_type: Type of augmentation
                - "pitch_shift": Shift pitch
                - "time_stretch": Stretch time
                - "dynamic_range": Compress dynamic range
                - "gaussian_noise": Add Gaussian noise
                
        Returns:
            Augmented audio
        """
        if augmentation_type == "pitch_shift":
            # Shift pitch by 1-3 semitones
            steps = np.random.randint(1, 4)
            augmented = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
        
        elif augmentation_type == "time_stretch":
            # Time stretch by 0.9-1.1x
            rate = np.random.uniform(0.9, 1.1)
            augmented = librosa.effects.time_stretch(audio, rate=rate)
        
        elif augmentation_type == "dynamic_range":
            # Compress dynamic range
            S = librosa.feature.melspectrogram(y=audio, sr=sr)
            S_db = librosa.power_to_db(S, ref=np.max)
            # Reduce dynamic range
            augmented = librosa.db_to_power(S_db * 0.7)
            # Convert back to time domain (simplified)
            augmented = audio * 0.95 + np.random.randn(len(audio)) * 0.05
        
        elif augmentation_type == "gaussian_noise":
            # Add Gaussian noise
            noise = np.random.randn(len(audio)) * 0.005
            augmented = audio + noise
        
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")
        
        return augmented
    
    def preprocess_pipeline(
        self,
        audio_path: Union[str, Path],
        normalize: bool = True,
        trim: bool = True,
        sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Complete preprocessing pipeline
        
        Args:
            audio_path: Path to audio file
            normalize: Normalize audio
            trim: Trim silence
            sr: Sample rate
            
        Returns:
            Preprocessed audio and sample rate
        """
        # Load audio
        audio, sr = self.load_audio(audio_path, sr=sr)
        
        # Trim silence
        if trim:
            audio = self.trim_silence(audio, sr)
        
        # Normalize
        if normalize:
            audio = self.normalize_audio(audio)
        
        return audio, sr
    
    def chunk_audio(
        self,
        audio: np.ndarray,
        sr: int,
        chunk_duration: float = 1.0,
        overlap: float = 0.0
    ) -> List[np.ndarray]:
        """
        Split audio into overlapping chunks
        
        Args:
            audio: Audio time series
            sr: Sample rate
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks (0-1)
            
        Returns:
            List of audio chunks
        """
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap * chunk_samples)
        step = chunk_samples - overlap_samples
        
        chunks = []
        for start in range(0, len(audio) - chunk_samples + 1, step):
            chunk = audio[start:start + chunk_samples]
            chunks.append(chunk)
        
        # Handle last chunk
        if start + step < len(audio):
            last_chunk = audio[-(chunk_samples):]
            chunks.append(last_chunk)
        
        return chunks
    
    def get_audio_duration(self, audio: np.ndarray, sr: int) -> float:
        """Get audio duration in seconds"""
        return len(audio) / sr
    
    def pad_audio(
        self,
        audio: np.ndarray,
        sr: int,
        target_duration: float
    ) -> np.ndarray:
        """Pad audio to target duration"""
        target_samples = int(target_duration * sr)
        if len(audio) < target_samples:
            padding = target_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        return audio[:target_samples]


def demonstrate_audio_processor():
    """Demonstrate audio processor capabilities"""
    print("\n" + "="*60)
    print("AUDIO PROCESSOR DEMONSTRATION")
    print("="*60)
    
    processor = AudioProcessor()
    
    # Create synthetic audio (sin wave)
    sr = 16000
    duration = 2
    t = np.linspace(0, duration, sr * duration)
    
    # Mix of frequencies: 440 Hz (A4) + 880 Hz (A5)
    frequency1 = 440
    frequency2 = 880
    audio = 0.3 * np.sin(2 * np.pi * frequency1 * t)
    audio += 0.2 * np.sin(2 * np.pi * frequency2 * t)
    
    print("\n✅ Generated synthetic audio (2 seconds, two frequencies)")
    print(f"  • Sample Rate: {sr} Hz")
    print(f"  • Duration: {duration}s")
    print(f"  • Samples: {len(audio)}")
    
    # Normalize
    normalized = processor.normalize_audio(audio)
    print("\n✅ Normalized audio")
    print(f"  • Min: {normalized.min():.4f}, Max: {normalized.max():.4f}")
    
    # Extract features
    mfcc = processor.extract_mfcc(audio, sr)
    print(f"\n✅ Extracted MFCC features: {mfcc.shape}")
    
    mel_spec = processor.extract_mel_spectrogram(audio, sr)
    print(f"✅ Extracted Mel Spectrogram: {mel_spec.shape}")
    
    zcr = processor.extract_zero_crossing_rate(audio)
    print(f"✅ Extracted Zero Crossing Rate: {zcr.shape}")
    
    spec_centroid = processor.extract_spectral_centroid(audio, sr)
    print(f"✅ Extracted Spectral Centroid: {spec_centroid.shape}")
    
    # Data augmentation
    augmented = processor.data_augmentation(audio, sr, "pitch_shift")
    print(f"\n✅ Applied pitch shift augmentation")
    
    # Chunking
    chunks = processor.chunk_audio(audio, sr, chunk_duration=0.5, overlap=0.1)
    print(f"✅ Chunked audio: {len(chunks)} chunks of 0.5s")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    demonstrate_audio_processor()

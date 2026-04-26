"""
Configuration file for Speech-to-Text and Gesture Recognition System
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000  # 16 kHz
    duration: float = 30.0  # Maximum audio duration in seconds
    n_mfcc: int = 40  # Number of MFCC features
    n_fft: int = 2048  # FFT window size
    hop_length: int = 512  # Hop length for STFT
    mel_bins: int = 128  # Number of Mel frequency bins
    chunk_size: int = 16000  # Samples per chunk (1 second at 16kHz)


@dataclass
class ASRModelConfig:
    """Automatic Speech Recognition Model Configuration"""
    model_name: str = "openai/whisper-base"  # Hugging Face model ID
    # Alternative models:
    # "openai/whisper-tiny"  - Fastest
    # "openai/whisper-small"  - Balanced
    # "openai/whisper-medium" - Accurate
    # "openai/whisper-large"  - Most accurate
    # "facebook/wav2vec2-base" - Alternative architecture
    
    device: str = "cuda"  # "cuda" or "cpu"
    precision: str = "float32"  # "float32" or "float16"
    batch_size: int = 8
    max_input_length: int = 30  # Seconds
    language: str = "en"  # English
    task: str = "transcribe"  # "transcribe" or "translate"


@dataclass
class GestureRecognitionConfig:
    """Gesture Recognition Configuration"""
    model_type: str = "mediapipe"  # "mediapipe" or "tensorflow"
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False
    num_hands: int = 2  # Maximum number of hands to detect
    
    # Gesture classification
    use_hand_landmarks: bool = True
    num_classes: int = 10  # Number of gesture classes
    gesture_threshold: float = 0.8


@dataclass
class TrainingConfig:
    """Training Configuration"""
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    eval_steps: int = 100
    save_steps: int = 500
    
    # Data
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = 4
    
    # Model
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    
    # Optimization
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Checkpointing
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"  # Word Error Rate


@dataclass
class EvaluationConfig:
    """Evaluation Configuration"""
    # Metrics
    calculate_wer: bool = True  # Word Error Rate
    calculate_cer: bool = True  # Character Error Rate
    calculate_bleu: bool = False  # BLEU score
    
    # Thresholds
    wer_threshold: float = 20.0  # Target WER percentage
    cer_threshold: float = 15.0  # Target CER percentage
    
    # Output
    save_predictions: bool = True
    save_figures: bool = True
    verbose: bool = True


class SystemConfig:
    """Main system configuration"""
    
    def __init__(self):
        # Sub-configurations
        self.audio = AudioConfig()
        self.asr = ASRModelConfig()
        self.gesture = GestureRecognitionConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        
        # Paths
        self.data_dir = DATA_DIR
        self.models_dir = MODELS_DIR
        self.results_dir = RESULTS_DIR
        self.log_dir = PROJECT_ROOT / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # System settings
        self.random_seed: int = 42
        self.num_workers: int = 4
        self.debug: bool = False
        

# Global configuration instance
config = SystemConfig()


def get_config() -> SystemConfig:
    """Get the global configuration"""
    return config


def print_config():
    """Print configuration for verification"""
    print("\n" + "="*60)
    print("SYSTEM CONFIGURATION")
    print("="*60)
    
    print("\n📌 Audio Configuration:")
    print(f"  • Sample Rate: {config.audio.sample_rate} Hz")
    print(f"  • Duration: {config.audio.duration}s")
    print(f"  • MFCC Features: {config.audio.n_mfcc}")
    print(f"  • Mel Bins: {config.audio.mel_bins}")
    
    print("\n🤖 ASR Model Configuration:")
    print(f"  • Model: {config.asr.model_name}")
    print(f"  • Device: {config.asr.device}")
    print(f"  • Precision: {config.asr.precision}")
    print(f"  • Batch Size: {config.asr.batch_size}")
    
    print("\n👋 Gesture Recognition Configuration:")
    print(f"  • Type: {config.gesture.model_type}")
    print(f"  • Detection Confidence: {config.gesture.min_detection_confidence}")
    print(f"  • Max Hands: {config.gesture.num_hands}")
    print(f"  • Gesture Classes: {config.gesture.num_classes}")
    
    print("\n📚 Training Configuration:")
    print(f"  • Epochs: {config.training.num_epochs}")
    print(f"  • Learning Rate: {config.training.learning_rate}")
    print(f"  • Train Batch Size: {config.training.train_batch_size}")
    
    print("\n📊 Evaluation Configuration:")
    print(f"  • Calculate WER: {config.evaluation.calculate_wer}")
    print(f"  • WER Target: {config.evaluation.wer_threshold}%")
    print(f"  • Calculate CER: {config.evaluation.calculate_cer}")
    
    print("\n📁 Directories:")
    print(f"  • Data: {config.data_dir}")
    print(f"  • Models: {config.models_dir}")
    print(f"  • Results: {config.results_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_config()

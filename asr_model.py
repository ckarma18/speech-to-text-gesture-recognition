"""
Automatic Speech Recognition (ASR) Model
Transformer-based speech-to-text using Hugging Face Transformers
"""

import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import warnings

try:
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Wav2Vec2Processor,
        Wav2Vec2ForCTC,
        AutoProcessor,
        AutoModelForSpeechSeq2Seq
    )
except ImportError:
    print("⚠️ Warning: transformers library not installed")

from config import ASRModelConfig, AudioConfig, get_config

warnings.filterwarnings('ignore')


class ASRModel:
    """
    Transformer-based Automatic Speech Recognition Model
    Supports Whisper and other transformer-based ASR architectures
    """
    
    def __init__(
        self,
        config: Optional[ASRModelConfig] = None,
        audio_config: Optional[AudioConfig] = None
    ):
        """
        Initialize ASR model
        
        Args:
            config: ASRModelConfig instance
            audio_config: AudioConfig instance
        """
        self.config = config or get_config().asr
        self.audio_config = audio_config or get_config().audio
        
        # Set device
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        print(f"🔧 Using device: {self.device}")
        
        # Load model and processor
        self.model = None
        self.processor = None
        self._load_model()
        
        # Initialize generation config if available
        self.generation_config = None
    
    def _load_model(self):
        """Load pre-trained transformer model and processor"""
        try:
            print(f"📥 Loading model: {self.config.model_name}")
            
            # Determine model type
            if "whisper" in self.config.model_name.lower():
                self._load_whisper_model()
            elif "wav2vec" in self.config.model_name.lower():
                self._load_wav2vec_model()
            else:
                self._load_generic_model()
            
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def _load_whisper_model(self):
        """Load Whisper model"""
        self.processor = WhisperProcessor.from_pretrained(self.config.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.config.model_name
        )
        self.model.to(self.device)
        self.model_type = "whisper"
    
    def _load_wav2vec_model(self):
        """Load Wav2Vec2 model"""
        self.processor = Wav2Vec2Processor.from_pretrained(self.config.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model_type = "wav2vec"
    
    def _load_generic_model(self):
        """Load generic speech-to-speech model"""
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.model_name
        )
        self.model.to(self.device)
        self.model_type = "generic"
    
    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Try using torchaudio first
            waveform, sample_rate = torchaudio.load(str(audio_path))
            audio = waveform.numpy()
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = np.mean(audio, axis=0, keepdims=True)
            
            audio = audio.squeeze()
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    sample_rate,
                    16000
                )
                audio_tensor = torch.from_numpy(audio).float()
                audio = resampler(audio_tensor).numpy()
                sample_rate = 16000
            
            return audio, sample_rate
            
        except Exception as e:
            print(f"Error loading audio: {str(e)}")
            raise
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Preprocess audio for model input
        
        Args:
            audio: Audio time series
            sr: Sample rate
            
        Returns:
            Processed audio tensor
        """
        # Normalize audio
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Process with model processor
        if self.model_type == "whisper":
            inputs = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            )
        elif self.model_type == "wav2vec":
            inputs = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )
        else:
            inputs = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            )
        
        return inputs
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en' for English)
            
        Returns:
            Dictionary with transcription results
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Preprocess audio
        inputs = self.preprocess_audio(audio, sr)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate transcription
        with torch.no_grad():
            if self.model_type == "whisper":
                # Whisper: seq2seq model
                if language:
                    inputs["language"] = language
                
                generated_ids = self.model.generate(
                    **inputs,
                    language=language or "en"
                )
                transcription = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
            
            elif self.model_type == "wav2vec":
                # Wav2Vec: CTC model
                logits = self.model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
            
            else:
                # Generic model
                generated_ids = self.model.generate(**inputs)
                transcription = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
        
        return {
            "text": transcription,
            "audio_path": str(audio_path),
            "model": self.config.model_name,
            "language": language or "en"
        }
    
    def batch_transcribe(
        self,
        audio_paths: List[Union[str, Path]],
        language: Optional[str] = None,
        verbose: bool = True
    ) -> List[Dict[str, str]]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            language: Language code
            verbose: Print progress
            
        Returns:
            List of transcription results
        """
        results = []
        
        for i, audio_path in enumerate(audio_paths):
            if verbose:
                print(f"Transcribing {i+1}/{len(audio_paths)}: {Path(audio_path).name}")
            
            try:
                result = self.transcribe(audio_path, language)
                results.append(result)
            except Exception as e:
                print(f"  ❌ Error transcribing {audio_path}: {str(e)}")
                results.append({
                    "text": "",
                    "audio_path": str(audio_path),
                    "error": str(e)
                })
        
        return results
    
    def transcribe_long_audio(
        self,
        audio_path: Union[str, Path],
        chunk_duration: float = 30.0,
        overlap: float = 2.0
    ) -> str:
        """
        Transcribe long audio by chunking
        
        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            
        Returns:
            Complete transcription
        """
        import librosa
        
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Calculate chunk size
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap * sr)
        step = chunk_samples - overlap_samples
        
        transcriptions = []
        
        # Process chunks
        for start in range(0, len(audio), step):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            
            # Preprocess chunk
            inputs = self.preprocess_audio(chunk, sr)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                if self.model_type == "whisper":
                    generated_ids = self.model.generate(**inputs)
                    text = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )[0]
                else:
                    # For other models
                    generated_ids = self.model.generate(**inputs)
                    text = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )[0]
            
            if text.strip():
                transcriptions.append(text)
        
        # Combine transcriptions
        complete_text = " ".join(transcriptions)
        return complete_text
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": self.config.model_name,
            "model_type": self.model_type if self.model else "unknown",
            "device": str(self.device),
            "language": self.config.language,
            "task": self.config.task,
            "precision": self.config.precision
        }
    
    def save_model(self, save_path: Union[str, Path]):
        """Save model and processor"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.model:
            self.model.save_pretrained(str(save_path / "model"))
        if self.processor:
            self.processor.save_pretrained(str(save_path / "processor"))
        
        print(f"✅ Model saved to {save_path}")
    
    def load_custom_model(self, model_path: Union[str, Path]):
        """Load custom fine-tuned model"""
        model_path = Path(model_path)
        
        if self.model_type == "whisper":
            self.model = WhisperForConditionalGeneration.from_pretrained(
                str(model_path / "model")
            )
            self.processor = WhisperProcessor.from_pretrained(
                str(model_path / "processor")
            )
        
        self.model.to(self.device)
        print(f"✅ Custom model loaded from {model_path}")


def demonstrate_asr_model():
    """Demonstrate ASR model capabilities"""
    print("\n" + "="*60)
    print("AUTOMATIC SPEECH RECOGNITION MODEL DEMONSTRATION")
    print("="*60)
    
    try:
        # Initialize model
        asr = ASRModel()
        
        print("\n✅ ASR Model initialized")
        
        # Print model info
        info = asr.get_model_info()
        print("\n📊 Model Information:")
        for key, value in info.items():
            print(f"  • {key}: {value}")
        
        # Create synthetic audio example
        print("\n🎵 Creating synthetic audio example...")
        sr = 16000
        duration = 3
        t = np.linspace(0, duration, sr * duration)
        
        # Simple tone to verify loading
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        print(f"  • Sample Rate: {sr} Hz")
        print(f"  • Duration: {duration}s")
        print(f"  • Audio Shape: {audio.shape}")
        
        print("\n✅ ASR Model ready for transcription!")
        print("  Use model.transcribe(audio_path) to transcribe audio files")
        
    except Exception as e:
        print(f"⚠️  Note: Full model loading requires:")
        print("   • torch and transformers library")
        print("   • Internet connection for downloading models")
        print(f"   • Error: {str(e)}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    demonstrate_asr_model()

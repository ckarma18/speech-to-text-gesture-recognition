"""
Main Application for Speech-to-Text with Gesture Recognition
Integrates ASR, gesture recognition, and audio processing into one system
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List
import time
import json
from datetime import datetime

from config import SystemConfig, get_config, print_config
from audio_processor import AudioProcessor
from gesture_recognizer import GestureRecognizer, GestureResult
from asr_model import ASRModel
from evaluation import Evaluator


class SpeechToTextSystem:
    """
    Integrated Speech-to-Text system with gesture recognition
    """
    
    def __init__(self):
        """Initialize the complete system"""
        print("\n" + "="*70)
        print("🎤 SPEECH-TO-TEXT TRANSCRIPTION WITH GESTURE RECOGNITION")
        print("="*70)
        
        self.config = get_config()
        
        # Initialize components
        print("\n🔧 Initializing system components...")
        
        try:
            self.audio_processor = AudioProcessor(self.config.audio)
            print("  ✅ Audio Processor initialized")
        except Exception as e:
            print(f"  ⚠️  Audio Processor error: {e}")
            self.audio_processor = None
        
        try:
            self.gesture_recognizer = GestureRecognizer(self.config.gesture)
            print("  ✅ Gesture Recognizer initialized")
        except Exception as e:
            print(f"  ⚠️  Gesture Recognizer error: {e}")
            self.gesture_recognizer = None
        
        try:
            self.asr_model = ASRModel(self.config.asr, self.config.audio)
            print("  ✅ ASR Model initialized")
        except Exception as e:
            print(f"  ⚠️  ASR Model error: {e}")
            print("     Note: Ensure transformers and torch are installed")
            self.asr_model = None
        
        self.evaluator = Evaluator()
        print("  ✅ Evaluator initialized")
        
        print("\n✅ System ready!\n")
    
    def transcribe_audio(
        self,
        audio_path: str,
        save_result: bool = True
    ) -> Dict:
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            save_result: Save result to file
            
        Returns:
            Dictionary with transcription result
        """
        print(f"\n📥 Processing: {Path(audio_path).name}")
        
        start_time = time.time()
        
        result = {
            "audio_path": str(audio_path),
            "timestamp": datetime.now().isoformat(),
            "transcription": "",
            "status": "error",
            "processing_time": 0.0
        }
        
        try:
            if self.asr_model:
                # Transcribe audio
                transcription_result = self.asr_model.transcribe(audio_path)
                result["transcription"] = transcription_result["text"]
                result["status"] = "success"
                
                print(f"  📝 Transcription: {result['transcription'][:100]}...")
            else:
                result["error"] = "ASR model not initialized"
                print("  ❌ ASR model not available")
        
        except Exception as e:
            result["error"] = str(e)
            print(f"  ❌ Error transcribing audio: {e}")
        
        result["processing_time"] = time.time() - start_time
        print(f"  ⏱️  Processing time: {result['processing_time']:.2f}s")
        
        # Save result
        if save_result and result["status"] == "success":
            self._save_result(result)
        
        return result
    
    def batch_transcribe(
        self,
        audio_paths: List[str],
        save_results: bool = True
    ) -> List[Dict]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            save_results: Save results to file
            
        Returns:
            List of transcription results
        """
        print(f"\n📥 Batch processing {len(audio_paths)} audio files...")
        
        results = []
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\n[{i}/{len(audio_paths)}] Processing: {Path(audio_path).name}")
            
            result = self.transcribe_audio(audio_path, save_result=False)
            results.append(result)
            
            if result["status"] == "success":
                successful += 1
            else:
                failed += 1
        
        total_time = time.time() - start_time
        
        # Print summary
        print(f"\n" + "="*70)
        print("📊 BATCH PROCESSING SUMMARY")
        print("="*70)
        print(f"  • Total files: {len(audio_paths)}")
        print(f"  • Successful: {successful}")
        print(f"  • Failed: {failed}")
        print(f"  • Total time: {total_time:.2f}s")
        print(f"  • Average time per file: {total_time/len(audio_paths):.2f}s")
        print("="*70 + "\n")
        
        # Save all results
        if save_results:
            self._save_batch_results(results)
        
        return results
    
    def evaluate_transcriptions(
        self,
        hypotheses: List[str],
        references: List[str],
        save_results: bool = True
    ) -> Dict:
        """
        Evaluate transcription accuracy
        
        Args:
            hypotheses: List of predicted transcriptions
            references: List of reference transcriptions
            save_results: Save evaluation results
            
        Returns:
            Evaluation metrics
        """
        print(f"\n📊 Evaluating {len(hypotheses)} transcriptions...")
        
        results = Evaluator.evaluate_batch(hypotheses, references, verbose=True)
        
        # Additional metrics
        accuracy = Evaluator.calculate_accuracy(hypotheses, references)
        results['accuracy'] = accuracy
        
        if save_results:
            output_path = self.config.results_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            Evaluator.save_evaluation_results(results, str(output_path))
        
        return results
    
    def process_with_gesture(
        self,
        audio_path: str,
        video_path: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio with synchronized gesture recognition
        
        Args:
            audio_path: Path to audio file
            video_path: Optional video file for gesture recognition
            
        Returns:
            Combined results
        """
        print(f"\n🎤👋 Processing audio with gesture recognition...")
        
        result = {
            "audio_path": str(audio_path),
            "video_path": str(video_path) if video_path else None,
            "transcription": "",
            "gestures": [],
            "status": "processing"
        }
        
        # Transcribe audio
        transcription_result = self.transcribe_audio(audio_path, save_result=False)
        result["transcription"] = transcription_result.get("transcription", "")
        
        # Process video for gestures
        if video_path and self.gesture_recognizer:
            print(f"  📹 Processing video: {Path(video_path).name}")
            try:
                gestures = self.gesture_recognizer.process_video(video_path)
                result["gestures"] = gestures
            except Exception as e:
                print(f"  ⚠️  Error processing video: {e}")
        
        result["status"] = "success"
        return result
    
    def get_system_info(self) -> Dict:
        """Get system information"""
        info = {
            "system": "Speech-to-Text with Gesture Recognition",
            "components": {
                "audio_processor": self.audio_processor is not None,
                "gesture_recognizer": self.gesture_recognizer is not None,
                "asr_model": self.asr_model is not None,
                "evaluator": True
            },
            "config": {
                "audio": {
                    "sample_rate": self.config.audio.sample_rate,
                    "n_mfcc": self.config.audio.n_mfcc,
                    "mel_bins": self.config.audio.mel_bins
                },
                "asr": {
                    "model_name": self.config.asr.model_name,
                    "device": self.config.asr.device,
                    "language": self.config.asr.language
                },
                "gesture": {
                    "model_type": self.config.gesture.model_type if self.gesture_recognizer else "unavailable",
                    "num_hands": self.config.gesture.num_hands if self.gesture_recognizer else 0
                }
            }
        }
        return info
    
    def _save_result(self, result: Dict):
        """Save single transcription result"""
        output_path = self.config.results_dir / f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"  💾 Result saved to: {output_path}")
    
    def _save_batch_results(self, results: List[Dict]):
        """Save batch transcription results"""
        output_path = self.config.results_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Batch results saved to: {output_path}")


def demonstrate_system():
    """Demonstrate system capabilities"""
    print("\n" + "="*70)
    print("SYSTEM DEMONSTRATION")
    print("="*70)
    
    # Initialize system
    system = SpeechToTextSystem()
    
    # Print system info
    info = system.get_system_info()
    print("\n✅ System Information:")
    print(f"  • Audio Processor: {info['components']['audio_processor']}")
    print(f"  • Gesture Recognizer: {info['components']['gesture_recognizer']}")
    print(f"  • ASR Model: {info['components']['asr_model']}")
    
    # Demonstrate evaluation
    print("\n" + "-"*70)
    print("EVALUATION DEMONSTRATION")
    print("-"*70)
    
    # Example transcriptions
    hypotheses = [
        "the quick brown fox",
        "hello world test",
        "automatic speech recognition"
    ]
    
    references = [
        "the quick brown fox",
        "hello world tests",
        "automatic speech recognition"
    ]
    
    # Evaluate
    eval_results = system.evaluate_transcriptions(
        hypotheses,
        references,
        save_results=False
    )
    
    print(f"\n✅ Demonstration complete!")
    print(f"\nTo use the system:")
    print(f"  1. Place audio files in: {system.config.data_dir}")
    print(f"  2. Call: system.transcribe_audio('path/to/audio.wav')")
    print(f"  3. Results saved to: {system.config.results_dir}")
    
    return system


if __name__ == "__main__":
    # Show configuration
    print_config()
    
    # Run demonstration
    system = demonstrate_system()
    
    print("\n" + "="*70)
    print("✅ SYSTEM READY FOR USE")
    print("="*70 + "\n")

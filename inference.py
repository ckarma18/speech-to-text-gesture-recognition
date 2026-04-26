"""
Inference Script for Speech-to-Text System
Run predictions on audio files using trained models
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
import argparse
from datetime import datetime

from config import get_config
from asr_model import ASRModel
from audio_processor import AudioProcessor
from gesture_recognizer import GestureRecognizer
from main import SpeechToTextSystem


class InferenceEngine:
    """
    High-level inference engine for transcribing audio
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize inference engine
        
        Args:
            use_gpu: Use GPU if available
        """
        self.config = get_config()
        
        # Set device preference
        if not use_gpu:
            self.config.asr.device = "cpu"
        
        # Initialize components
        self.system = SpeechToTextSystem()
    
    def transcribe_file(self, audio_path: str) -> Dict:
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription result
        """
        return self.system.transcribe_audio(audio_path)
    
    def transcribe_directory(
        self,
        directory: str,
        pattern: str = "*.wav",
        recursive: bool = False
    ) -> List[Dict]:
        """
        Transcribe all audio files in a directory
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            recursive: Search recursively
            
        Returns:
            List of transcription results
        """
        dir_path = Path(directory)
        
        if recursive:
            audio_files = list(dir_path.rglob(pattern))
        else:
            audio_files = list(dir_path.glob(pattern))
        
        audio_files = sorted([f for f in audio_files if f.is_file()])
        
        print(f"\n📁 Found {len(audio_files)} audio files in {directory}")
        
        if not audio_files:
            print("⚠️  No audio files found")
            return []
        
        # Transcribe all files
        return self.system.batch_transcribe([str(f) for f in audio_files])
    
    def get_model_info(self) -> Dict:
        """Get current model information"""
        return self.system.get_system_info()


def create_argument_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Speech-to-Text Inference Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe a single file
  python inference.py --audio path/to/audio.wav
  
  # Transcribe a directory
  python inference.py --directory ./audio_files --pattern "*.wav"
  
  # Transcribe with custom model
  python inference.py --audio audio.wav --model openai/whisper-large
  
  # Save results
  python inference.py --audio audio.wav --output results.json
        """
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to single audio file"
    )
    
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory containing audio files"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.wav",
        help="File pattern for directory (default: *.wav)"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directory recursively"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-base",
        help="Model name from Hugging Face"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Do not use GPU"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser


def main():
    """Main inference function"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not args.audio and not args.directory:
        parser.print_help()
        print("\n❌ Error: Specify either --audio or --directory")
        sys.exit(1)
    
    # Initialize inference engine
    print("\n🔧 Initializing inference engine...")
    engine = InferenceEngine(use_gpu=not args.no_gpu)
    
    # Show model info
    if args.verbose:
        info = engine.get_model_info()
        print("\n📊 System Information:")
        print(f"  Model: {info['config']['asr']['model_name']}")
        print(f"  Device: {info['config']['asr']['device']}")
        print(f"  Language: {info['config']['asr']['language']}")
    
    results = []
    
    # Process audio file
    if args.audio:
        audio_path = Path(args.audio)
        
        if not audio_path.exists():
            print(f"\n❌ Error: File not found: {args.audio}")
            sys.exit(1)
        
        print(f"\n🎤 Transcribing: {audio_path.name}")
        result = engine.transcribe_file(str(audio_path))
        results = [result]
    
    # Process directory
    elif args.directory:
        results = engine.transcribe_directory(
            args.directory,
            pattern=args.pattern,
            recursive=args.recursive
        )
    
    # Save results
    if results:
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {output_path}")
        
        # Print results
        print("\n" + "="*70)
        print("TRANSCRIPTION RESULTS")
        print("="*70)
        
        for i, result in enumerate(results, 1):
            print(f"\n📄 Result {i}:")
            print(f"  File: {Path(result['audio_path']).name}")
            print(f"  Status: {result['status']}")
            
            if result['status'] == 'success':
                print(f"  Transcription: {result['transcription']}")
                print(f"  Time: {result['processing_time']:.2f}s")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "="*70)


def example_usage():
    """Show example usage"""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║         SPEECH-TO-TEXT INFERENCE ENGINE - USAGE EXAMPLES              ║
╚═══════════════════════════════════════════════════════════════════════╝

1. SINGLE FILE TRANSCRIPTION
   ─────────────────────────
   python inference.py --audio path/to/audio.wav

2. BATCH PROCESSING
   ─────────────────
   python inference.py --directory ./audio_files

3. CUSTOM MODEL
   ────────────
   python inference.py --audio audio.wav --model openai/whisper-large

4. SAVE RESULTS
   ────────────
   python inference.py --audio audio.wav --output results.json

5. CPU ONLY
   ────────
   python inference.py --audio audio.wav --no-gpu

6. VERBOSE OUTPUT
   ──────────────
   python inference.py --audio audio.wav --verbose

7. RECURSIVE DIRECTORY SEARCH
   ──────────────────────────
   python inference.py --directory ./data --recursive

SUPPORTED MODELS:
  • openai/whisper-tiny    (fastest)
  • openai/whisper-base    (default, balanced)
  • openai/whisper-small   (better accuracy)
  • openai/whisper-medium  (more accurate)
  • openai/whisper-large   (most accurate)

OUTPUT FORMAT (JSON):
  [
    {{
      "audio_path": "path/to/audio.wav",
      "transcription": "the transcribed text",
      "status": "success",
      "processing_time": 2.34,
      "timestamp": "2026-04-08T10:30:45.123456"
    }}
  ]
    """)


if __name__ == "__main__":
    # Check if running with --help or no arguments
    if "--help" in sys.argv or "-h" in sys.argv or len(sys.argv) == 1:
        if len(sys.argv) == 1:
            example_usage()
        parser = create_argument_parser()
        parser.print_help()
    else:
        main()

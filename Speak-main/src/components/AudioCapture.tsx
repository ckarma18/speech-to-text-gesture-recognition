import { useRef, useState, useEffect } from "react";
import type { SpeechRecognition } from "../types";

interface TranscriptionResponse {
  transcription: string;
  confidence: number;
  processing_time: number;
  language?: string;
  model?: string;
  model_type?: "pretrained" | "custom";
  input_mode?: "live" | "upload";
}

interface AudioCaptureProps {
  onTranscriptionResult: (result: TranscriptionResponse) => void;
  onError: (message: string) => void;
  onClear: () => void;
  isTranscribing: boolean;
  setIsTranscribing: (value: boolean) => void;
}

export default function AudioCapture({
  onTranscriptionResult,
  onError,
  onClear,
  isTranscribing,
  setIsTranscribing,
}: AudioCaptureProps) {
  const [mode, setMode] = useState<"live" | "upload">("live");
  const [speechModel, setSpeechModel] = useState<"pretrained" | "custom">("pretrained");
  const [isRecording, setIsRecording] = useState(false);
  const [audioError, setAudioError] = useState<string>("");
  const [status, setStatus] = useState("Ready");

  const recognitionRef = useRef<SpeechRecognition | null>(null);

  const clearError = () => {
    setAudioError("");
    onError("");
  };

  const resetSession = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    setIsRecording(false);
    setStatus("Ready");
    clearError();
    onClear();
  };

  const startLiveRecognition = async () => {
    clearError();

    try {
      const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;

      if (!SpeechRecognition) {
        throw new Error(
          "Speech recognition is not supported in this browser. Please use Chrome or Edge."
        );
      }

      const recognition = new SpeechRecognition();
      recognitionRef.current = recognition;

      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";

      recognition.onstart = () => {
        setIsRecording(true);
        setStatus("Listening live...");
      };

      recognition.onresult = (event) => {
        let finalTranscript = "";
        let interimTranscript = "";

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }

        const liveText = (finalTranscript + interimTranscript).trim();

        if (liveText) {
          onTranscriptionResult({
            transcription: liveText,
            confidence: 0.95,
            processing_time: 0.0,
            language: "en",
            model: "Browser SpeechRecognition",
            model_type: "pretrained",
            input_mode: "live",
          });
        }
      };

      recognition.onerror = (event) => {
        const errorMessage = `Speech recognition error: ${event.error}`;
        setAudioError(errorMessage);
        onError(errorMessage);
        setIsRecording(false);
        setStatus("Live recognition failed");
      };

      recognition.onend = () => {
        setIsRecording(false);
        setStatus("Live recognition stopped");
      };

      recognition.start();
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Unable to start live speech recognition.";
      setAudioError(message);
      onError(message);
      setStatus("Live recognition unavailable");
    }
  };

  const stopLiveRecognition = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsRecording(false);
      setStatus("Stopped");
    }
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    clearError();

    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setIsTranscribing(true);
      setStatus("Uploading audio to backend...");

      const formData = new FormData();
      formData.append("audio", file);

      const endpoint =
        speechModel === "custom"
          ? "http://localhost:8000/api/transcribe-custom"
          : "http://localhost:8000/api/transcribe";

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to transcribe audio");
      }

      onTranscriptionResult({
        transcription: data.transcription || "",
        confidence: data.confidence || 0,
        processing_time: data.processing_time || 0,
        language: data.language || "en",
        model:
          data.model ||
          (speechModel === "custom"
            ? "Custom KNN Speech Model"
            : "openai/whisper-base"),
        model_type: data.model_type || speechModel,
        input_mode: "upload",
      });

      setStatus("Transcription complete");
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Failed to upload/transcribe audio.";
      setAudioError(message);
      onError(message);
      setStatus("Transcription failed");
    } finally {
      setIsTranscribing(false);
      event.target.value = "";
    }
  };

  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  return (
    <div className="bg-white rounded-2xl shadow-2xl p-8 min-h-[620px]">
      <h2 className="text-3xl font-bold mb-6 text-gray-800 flex items-center gap-2">
        🎤 Live Speech Recognition
      </h2>

      <div className="space-y-6">
        <div className="flex gap-3">
          <button
            onClick={() => setMode("live")}
            className={`px-5 py-2 rounded-lg font-semibold ${
              mode === "live"
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-700"
            }`}
          >
            🎙️ Live Voice
          </button>

          <button
            onClick={() => setMode("upload")}
            className={`px-5 py-2 rounded-lg font-semibold ${
              mode === "upload"
                ? "bg-purple-600 text-white"
                : "bg-gray-100 text-gray-700"
            }`}
          >
            📁 Upload Audio
          </button>
        </div>

        <div className="p-4 rounded-lg bg-indigo-50 border-2 border-indigo-200">
          <p className="text-xs font-bold text-gray-600 uppercase mb-2">
            Speech Model Selection
          </p>
          <div className="flex gap-3">
            <button
              onClick={() => setSpeechModel("pretrained")}
              className={`px-4 py-2 rounded-lg font-semibold ${
                speechModel === "pretrained"
                  ? "bg-indigo-600 text-white"
                  : "bg-white text-gray-700 border"
              }`}
            >
              Pretrained Whisper
            </button>

            <button
              onClick={() => setSpeechModel("custom")}
              disabled={mode === "live"}
              className={`px-4 py-2 rounded-lg font-semibold ${
                speechModel === "custom"
                  ? "bg-emerald-600 text-white"
                  : "bg-white text-gray-700 border"
              } ${mode === "live" ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              Custom Trained Speech
            </button>
          </div>
          {mode === "live" && (
            <p className="text-xs text-gray-500 mt-2">
              Custom trained speech is available only in Upload Audio mode.
            </p>
          )}
        </div>

        <div className="flex items-center gap-3 p-4 rounded-lg bg-blue-50 border-2 border-blue-200">
          {isRecording && (
            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
          )}
          {isTranscribing && (
            <div className="w-3 h-3 bg-yellow-500 rounded-full animate-pulse"></div>
          )}
          <span className="text-gray-700 font-semibold">{status}</span>
        </div>

        {mode === "live" ? (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <button
                onClick={startLiveRecognition}
                disabled={isRecording}
                className={`px-6 py-3 rounded-lg font-bold text-white transition-all ${
                  isRecording
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-gradient-to-r from-green-500 to-emerald-500 hover:shadow-lg"
                }`}
              >
                🎙️ Start Live Voice
              </button>

              <button
                onClick={stopLiveRecognition}
                disabled={!isRecording}
                className={`px-6 py-3 rounded-lg font-bold text-white transition-all ${
                  !isRecording
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-gradient-to-r from-red-500 to-pink-500 hover:shadow-lg"
                }`}
              >
                ⏹️ Stop Live Voice
              </button>

              <button
                onClick={resetSession}
                className="px-6 py-3 rounded-lg font-bold text-white bg-gradient-to-r from-slate-500 to-slate-700 hover:shadow-lg"
              >
                🔄 New Session
              </button>
            </div>

            <div className="p-4 rounded-lg bg-gray-50 border-l-4 border-blue-500">
              <p className="text-sm text-gray-700">
                <strong>Live Voice Mode:</strong> Uses browser speech recognition
                for real-time transcription.
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <label className="block w-full">
              <div className="w-full px-6 py-4 rounded-lg font-bold text-center text-white bg-gradient-to-r from-purple-600 to-indigo-600 hover:shadow-lg cursor-pointer">
                📁 Choose Audio File
              </div>
              <input
                type="file"
                accept=".wav,.mp3,.flac,.ogg,.m4a,.webm"
                className="hidden"
                onChange={handleFileUpload}
              />
            </label>

            <button
              onClick={resetSession}
              className="w-full px-6 py-3 rounded-lg font-bold text-white bg-gradient-to-r from-slate-500 to-slate-700 hover:shadow-lg"
            >
              🔄 Clear Result / New Session
            </button>

            <div className="p-4 rounded-lg bg-gray-50 border-l-4 border-purple-500">
              <p className="text-sm text-gray-700">
                <strong>Upload Audio Mode:</strong> Sends audio to either the pretrained
                Whisper backend or your custom trained speech model.
              </p>
            </div>
          </div>
        )}

        {audioError && (
          <div className="p-4 rounded-lg bg-red-50 border-2 border-red-200 text-red-800">
            <p className="font-semibold">Audio Error</p>
            <p className="mt-2 text-sm leading-relaxed">{audioError}</p>
          </div>
        )}
      </div>
    </div>
  );
}
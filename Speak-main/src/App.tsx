import { useEffect, useState } from "react";
import Header from "./components/Header";
import AudioCapture from "./components/AudioCapture";
import ResultsPanel from "./components/ResultsPanel";
import GestureDetection from "./components/GestureDetection";

interface SpeechResult {
  transcription: string;
  confidence: number;
  processing_time: number;
  language?: string;
  model?: string;
  model_type?: "pretrained" | "custom";
  input_mode?: "live" | "upload";
}

function App() {
  const [transcription, setTranscription] = useState<string>("");
  const [isTranscribing, setIsTranscribing] = useState<boolean>(false);
  const [confidence, setConfidence] = useState<number>(0);
  const [processingTime, setProcessingTime] = useState<number>(0);
  const [error, setError] = useState<string>("");
  const [modelName, setModelName] = useState<string>("openai/whisper-base");
  const [language, setLanguage] = useState<string>("en");
  const [modelType, setModelType] = useState<"pretrained" | "custom">("pretrained");
  const [inputMode, setInputMode] = useState<"live" | "upload">("live");

  const [currentDetectedGesture, setCurrentDetectedGesture] = useState<string>("");
  const [showGestureMode, setShowGestureMode] = useState<boolean>(false);
  const [backendStatus, setBackendStatus] = useState<"connected" | "disconnected" | "checking">("checking");

  const checkBackendConnection = async (): Promise<boolean> => {
    try {
      const response = await fetch("http://localhost:8000/api/health", {
        method: "GET",
      });

      if (response.ok) {
        setBackendStatus("connected");
        return true;
      }

      setBackendStatus("disconnected");
      return false;
    } catch {
      setBackendStatus("disconnected");
      return false;
    }
  };

  useEffect(() => {
    let retryTimer: number | undefined;

    const initBackendCheck = async () => {
      const connected = await checkBackendConnection();

      if (!connected) {
        retryTimer = window.setInterval(async () => {
          const success = await checkBackendConnection();
          if (success && retryTimer) {
            window.clearInterval(retryTimer);
          }
        }, 3000);
      }
    };

    initBackendCheck();

    return () => {
      if (retryTimer) window.clearInterval(retryTimer);
    };
  }, []);

  const handleAudioTranscriptionResult = (result: SpeechResult) => {
    setError("");
    setTranscription(result.transcription || "");
    setConfidence((result.confidence || 0) * 100);
    setProcessingTime(result.processing_time || 0);
    setModelName(result.model || "openai/whisper-base");
    setLanguage(result.language || "en");
    setModelType(result.model_type || "pretrained");
    setInputMode(result.input_mode || "live");
  };

  const handleAudioError = (message: string) => {
    setError(message);
  };

  const handleClearAudio = () => {
    setTranscription("");
    setConfidence(0);
    setProcessingTime(0);
    setError("");
    setLanguage("en");
    setModelName("openai/whisper-base");
    setModelType("pretrained");
    setInputMode("live");
  };

  const handleGestureDetected = (gesture: string) => {
    setCurrentDetectedGesture(gesture);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      <Header />

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex justify-center gap-4 mb-8">
          <button
            onClick={() => setShowGestureMode(false)}
            className={`px-8 py-3 rounded-lg font-bold transition-all ${
              !showGestureMode
                ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg"
                : "bg-white text-gray-700 border-2 border-gray-200"
            }`}
          >
            🎤 Audio Transcription
          </button>

          <button
            onClick={() => setShowGestureMode(true)}
            className={`px-8 py-3 rounded-lg font-bold transition-all ${
              showGestureMode
                ? "bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg"
                : "bg-white text-gray-700 border-2 border-gray-200"
            }`}
          >
            👋 Gesture Recognition
          </button>
        </div>

        <div
          className={`mb-6 p-3 rounded-lg text-sm font-semibold text-center ${
            backendStatus === "connected"
              ? "bg-green-100 text-green-800"
              : backendStatus === "checking"
              ? "bg-yellow-100 text-yellow-800"
              : "bg-red-100 text-red-800"
          }`}
        >
          {backendStatus === "checking" && "Checking backend connection..."}
          {backendStatus === "connected" && "✅ Backend Connected"}
          {backendStatus === "disconnected" && "❌ Backend Disconnected"}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            {!showGestureMode ? (
              <AudioCapture
                onTranscriptionResult={handleAudioTranscriptionResult}
                onError={handleAudioError}
                onClear={handleClearAudio}
                isTranscribing={isTranscribing}
                setIsTranscribing={setIsTranscribing}
              />
            ) : (
              <GestureDetection onGestureDetected={handleGestureDetected} />
            )}
          </div>

          <div>
            {!showGestureMode ? (
              <ResultsPanel
                transcription={transcription}
                confidence={confidence}
                processingTime={processingTime}
                error={error}
                isLoading={isTranscribing}
                modelName={modelName}
                language={language}
                modelType={modelType}
                inputMode={inputMode}
              />
            ) : (
              <div className="bg-white rounded-2xl shadow-2xl p-8 min-h-[620px]">
                <h2 className="text-3xl font-bold mb-6 text-gray-800 flex items-center gap-2">
                  👋 Detected Gesture
                </h2>

                <div className="p-4 rounded-lg bg-emerald-50 border-2 border-emerald-200 mb-6">
                  <p className="text-xs font-bold text-gray-600 uppercase">
                    Gesture Model Source
                  </p>
                  <p className="text-lg font-bold text-emerald-700 mt-1">
                    Custom Trained Model
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    KNN classifier using MediaPipe hand landmarks
                  </p>
                </div>

                {currentDetectedGesture ? (
                  <div
                    className={`p-8 rounded-2xl border-2 text-3xl font-bold text-center ${
                      currentDetectedGesture.includes("NO MATCHING")
                        ? "bg-yellow-100 border-yellow-300 text-yellow-800"
                        : currentDetectedGesture.includes("SHOW HAND")
                        ? "bg-gray-100 border-gray-300 text-gray-700"
                        : "bg-purple-100 border-purple-200 text-purple-800"
                    }`}
                  >
                    {currentDetectedGesture}
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-center text-gray-500 text-lg">
                    <div>
                      No gesture detected yet.
                      <br />
                      Show one clear gesture to the camera. 👐
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="mt-12 border-t border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-4 py-6 text-center text-sm text-gray-600">
          <p>
            💡 Backend:
            <code className="bg-gray-100 px-3 py-1 rounded font-mono ml-2">
              http://localhost:8000
            </code>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
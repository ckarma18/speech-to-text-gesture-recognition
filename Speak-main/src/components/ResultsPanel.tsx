interface ResultsPanelProps {
  transcription: string;
  confidence: number;
  processingTime: number;
  error: string;
  isLoading: boolean;
  modelName: string;
  language: string;
  modelType: "pretrained" | "custom";
  inputMode: "live" | "upload";
}

export default function ResultsPanel({
  transcription,
  confidence,
  processingTime,
  error,
  isLoading,
  modelName,
  language,
  modelType,
  inputMode,
}: ResultsPanelProps) {
  const showConfidence = inputMode === "live" && confidence > 0;

  const formattedTranscription =
    transcription && transcription.length > 0
      ? transcription.charAt(0).toUpperCase() + transcription.slice(1)
      : "";

  return (
    <div className="bg-white rounded-2xl shadow-2xl p-8 min-h-[620px]">
      <h2 className="text-3xl font-bold mb-6 text-gray-800 flex items-center gap-2">
        📝 Transcription Result
      </h2>

      <div className="space-y-6">
        {/* Model Comparison Header */}
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-gray-500 mb-3">
            Model Comparison
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Pretrained Model Box */}
            <div
              className={`p-5 rounded-xl border-2 transition-all ${
                modelType === "pretrained"
                  ? "bg-blue-50 border-blue-300 shadow-md"
                  : "bg-gray-50 border-gray-200"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs font-bold uppercase text-gray-500">
                  Pretrained Model
                </p>
                {modelType === "pretrained" && (
                  <span className="text-xs font-bold bg-blue-600 text-white px-2 py-1 rounded-full">
                    ACTIVE
                  </span>
                )}
              </div>

              <p
                className={`text-2xl font-bold ${
                  modelType === "pretrained" ? "text-blue-700" : "text-gray-500"
                }`}
              >
                Whisper
              </p>

              <p className="text-sm text-gray-500 mt-1">
                {modelName || "openai/whisper-base"}
              </p>
            </div>

            {/* Custom Model Box */}
            <div
              className={`p-5 rounded-xl border-2 transition-all ${
                modelType === "custom"
                  ? "bg-emerald-50 border-emerald-300 shadow-md"
                  : "bg-gray-50 border-gray-200"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs font-bold uppercase text-gray-500">
                  Custom Trained Model
                </p>
                {modelType === "custom" && (
                  <span className="text-xs font-bold bg-emerald-600 text-white px-2 py-1 rounded-full">
                    ACTIVE
                  </span>
                )}
              </div>

              <p
                className={`text-2xl font-bold ${
                  modelType === "custom" ? "text-emerald-700" : "text-gray-500"
                }`}
              >
                KNN + MFCC
              </p>

              <p className="text-sm text-gray-500 mt-1">
                Your trained speech model
              </p>
            </div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="p-4 rounded-lg bg-red-50 border-2 border-red-200">
            <p className="text-red-800 font-semibold">⚠️ Error</p>
            <p className="text-red-700 text-sm mt-1">{error}</p>
          </div>
        )}

        {/* Loading */}
        {isLoading && (
          <div className="flex items-center justify-center gap-3 p-6 rounded-lg bg-blue-50 border-2 border-blue-200">
            <div className="animate-spin w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full"></div>
            <span className="text-blue-800 font-semibold">
              Processing transcription...
            </span>
          </div>
        )}

        {/* Transcription Result */}
        {formattedTranscription && !isLoading && (
          <div className="p-8 rounded-xl bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200 min-h-[240px]">
            <p className="text-sm font-semibold text-gray-500 mb-3 uppercase tracking-wide">
              Transcribed Text
            </p>

            <p className="text-3xl text-gray-800 leading-relaxed font-semibold">
              {formattedTranscription}
            </p>

            <p className="text-sm text-gray-500 mt-4 leading-relaxed">
              {modelType === "pretrained"
                ? "Using advanced pretrained Whisper model for general speech recognition."
                : "Using custom trained MFCC + KNN model trained on your project dataset."}
            </p>
          </div>
        )}

        {/* Waiting State */}
        {!formattedTranscription && !isLoading && !error && (
          <div className="p-8 rounded-xl bg-gray-50 border-2 border-gray-200 min-h-[240px] flex items-center justify-center">
            <div className="text-center">
              <p className="text-gray-400 text-xl mb-2">🎤 Ready to transcribe</p>
              <p className="text-gray-500 text-sm">
                Start Live Voice or upload audio to see the result here
              </p>
            </div>
          </div>
        )}

        {/* Metrics */}
        {formattedTranscription && !isLoading && (
          <div
            className={`grid gap-4 ${
              showConfidence ? "grid-cols-1 md:grid-cols-3" : "grid-cols-1 md:grid-cols-2"
            }`}
          >
            {showConfidence && (
              <div className="p-4 rounded-xl bg-green-50 border-2 border-green-200">
                <p className="text-xs font-bold text-gray-600 uppercase">
                  Confidence
                </p>
                <p className="text-2xl font-bold text-green-700 mt-1">
                  {confidence.toFixed(1)}%
                </p>
              </div>
            )}

            <div className="p-4 rounded-xl bg-purple-50 border-2 border-purple-200">
              <p className="text-xs font-bold text-gray-600 uppercase">
                Processing Time
              </p>
              <p className="text-2xl font-bold text-purple-700 mt-1">
                {processingTime.toFixed(2)}s
              </p>
            </div>

            <div className="p-4 rounded-xl bg-cyan-50 border-2 border-cyan-200">
              <p className="text-xs font-bold text-gray-600 uppercase">
                Language
              </p>
              <p className="text-2xl font-bold text-cyan-700 mt-1">
                {language || "en"}
              </p>
            </div>
          </div>
        )}

        {/* Copy Button */}
        {formattedTranscription && (
          <button
            onClick={() => navigator.clipboard.writeText(formattedTranscription)}
            className="w-full px-6 py-4 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-600 to-purple-600 hover:shadow-lg transition-all transform hover:scale-[1.01]"
          >
            📋 Copy to Clipboard
          </button>
        )}
      </div>
    </div>
  );
}
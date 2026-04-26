export default function Header() {
  return (
    <header className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="text-4xl">🎤👋</div>
            <div>
              <h1 className="text-4xl font-bold">Speech-to-Text with Gesture Recognition</h1>
              <p className="text-blue-100 mt-1 text-lg">
                Real-time transcription using Transformers and gesture detection
              </p>
            </div>
          </div>
          <div className="text-right text-blue-100">
            <p className="text-sm">Powered by OpenAI Whisper + MediaPipe</p>
          </div>
        </div>
      </div>
    </header>
  );
}
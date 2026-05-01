import { useEffect, useRef, useState } from "react";
import { useGestureRecognition } from "../hooks/useGestureRecognition";
import { GestureOverlay } from "./GestureOverlay";

interface GestureDetectionProps {
  onGestureDetected: (gesture: string) => void;
}

export default function GestureDetection({
  onGestureDetected,
}: GestureDetectionProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");

  const { detectedHands, currentGesture, isProcessing } = useGestureRecognition(
    videoRef.current,
    isCameraActive
  );

  useEffect(() => {
    if (currentGesture) {
      onGestureDetected(
        `${currentGesture.gesture} (${Math.round(currentGesture.confidence * 100)}%)`
      );
    }
  }, [currentGesture, onGestureDetected]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      alert("Unable to access camera. Please check permissions.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      setIsCameraActive(false);
    }
  };

  const switchCamera = () => {
    stopCamera();
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
  };

  useEffect(() => {
    if (isCameraActive) {
      stopCamera();
      startCamera();
    }
  }, [facingMode]);

  return (
    <div className="bg-white rounded-2xl shadow-2xl p-8 min-h-[620px]">
      <h2 className="text-3xl font-bold mb-6 text-gray-800 flex items-center gap-2">
        👋 Gesture Recognition
      </h2>

      <div className="space-y-6">
        <div className="relative w-full bg-black rounded-xl overflow-hidden shadow-lg h-[340px]">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={`w-full h-full object-cover ${isCameraActive ? "block" : "hidden"}`}
          />

          {isCameraActive && (
            <GestureOverlay
              detectedHands={detectedHands}
              videoElement={videoRef.current}
            />
          )}

          {!isCameraActive && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <p className="text-white text-center px-4 text-lg">
                Camera is off. Click "Start Camera" to begin.
              </p>
            </div>
          )}
        </div>

        <div className="grid grid-cols-3 gap-3">
          <button
            onClick={startCamera}
            disabled={isCameraActive}
            className={`px-6 py-4 rounded-lg font-bold text-white transition-all ${
              isCameraActive
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-gradient-to-r from-green-500 to-emerald-500 hover:shadow-lg"
            }`}
          >
            📹 Start Camera
          </button>

          <button
            onClick={stopCamera}
            disabled={!isCameraActive}
            className={`px-6 py-4 rounded-lg font-bold text-white transition-all ${
              !isCameraActive
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-gradient-to-r from-red-500 to-pink-500 hover:shadow-lg"
            }`}
          >
            ⏹️ Stop Camera
          </button>

          <button
            onClick={switchCamera}
            disabled={!isCameraActive}
            className={`px-6 py-4 rounded-lg font-bold text-white transition-all ${
              !isCameraActive
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-gradient-to-r from-blue-500 to-indigo-500 hover:shadow-lg"
            }`}
          >
            🔄 Switch
          </button>
        </div>

        <div className="flex items-center gap-3 p-4 rounded-lg bg-blue-50 border-2 border-blue-200">
          {isCameraActive && (
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
          )}
          <span className="text-gray-700 font-semibold text-lg">
            {isCameraActive ? "Camera is live" : "Camera is off"}
          </span>
          {isProcessing && (
            <span className="text-sm text-gray-600">Detecting hand...</span>
          )}
        </div>

        {currentGesture && (
  <div
    className={`p-4 rounded-lg border-2 ${
      currentGesture.gesture.includes("NO MATCHING")
        ? "bg-yellow-100 border-yellow-400"
        : currentGesture.gesture.includes("SHOW HAND")
        ? "bg-gray-100 border-gray-400"
        : "bg-green-100 border-green-400"
    }`}
  >
    <p className="text-sm font-semibold text-gray-600">Current Gesture</p>
    <p className="text-2xl font-bold mt-2">{currentGesture.gesture}</p>
  </div>
)}

        <div className="p-4 rounded-lg bg-gray-50 border-l-4 border-blue-500">
          <p className="text-sm text-gray-700">
            <strong>Instructions:</strong> Show one clear gesture at a time.
            <br />
            Use distinct poses:
            <br />
            👍 GOOD, 👎 BAD, ✋ STOP, 👉 GO, 👋 COME, 👌 OK, 📞 CALL, 🚻 TOILET, 🤞 GOOD LUCK, 🤫 SILENT
          </p>
        </div>
      </div>
    </div>
  );
}
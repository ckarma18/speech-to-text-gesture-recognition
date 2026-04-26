import { useEffect, useRef, useState, useCallback } from 'react';
import { Hands } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';
import { DetectedHand, GestureResult } from '../types';

export const useGestureRecognition = (
  videoElement: HTMLVideoElement | null,
  isActive: boolean
) => {
  const [detectedHands, setDetectedHands] = useState<DetectedHand[]>([]);
  const [currentGesture, setCurrentGesture] = useState<GestureResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handsRef = useRef<Hands | null>(null);
  const cameraRef = useRef<Camera | null>(null);
  const lastGestureTime = useRef<number>(0);

  const sendToBackend = async (landmarks: number[]) => {
    try {
      const res = await fetch("http://127.0.0.1:8000/api/predict-gesture", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ landmarks }),
      });

      if (!res.ok) {
        throw new Error(`Backend returned ${res.status}`);
      }

      return await res.json();
    } catch (err) {
      console.error("Backend error:", err);
      return null;
    }
  };

  const onResults = useCallback(async (results: any) => {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const hands: DetectedHand[] = results.multiHandLandmarks.map(
        (landmarks: any, index: number) => ({
          landmarks,
          handedness: results.multiHandedness?.[index]?.label || 'Unknown',
          score: results.multiHandedness?.[index]?.score || 0
        })
      );

      setDetectedHands(hands);

      const now = Date.now();

      if (now - lastGestureTime.current > 500) {
        const row: number[] = [];

        hands[0].landmarks.forEach((lm: any) => {
          row.push(lm.x);
          row.push(lm.y);
        });

        if (row.length === 42) {
          const result = await sendToBackend(row);

          if (result) {
            let displayGesture = "";

            if (result.gesture === "no_matching") {
              displayGesture = "NO MATCHING ❓";
            } else if (result.gesture === "no_hand") {
              displayGesture = "SHOW HAND TO DETECT 🖐️";
            } else {
              displayGesture = `${result.gesture.toUpperCase()} ${result.emoji || ""}`.trim();
            }

            setCurrentGesture({
              gesture: displayGesture,
              confidence: result.confidence ?? 1.0,
              timestamp: Date.now()
            });

            lastGestureTime.current = now;
          }
        }
      }
    } else {
      setDetectedHands([]);
      setCurrentGesture({
        gesture: "SHOW HAND TO DETECT 🖐️",
        confidence: 0,
        timestamp: Date.now()
      });
    }
  }, []);

  useEffect(() => {
    if (!videoElement || !isActive) {
      if (cameraRef.current) {
        cameraRef.current.stop();
      }
      return;
    }

    const initializeHands = async () => {
      try {
        setIsProcessing(true);

        handsRef.current = new Hands({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        handsRef.current.setOptions({
          maxNumHands: 1,
          modelComplexity: 1,
          minDetectionConfidence: 0.7,
          minTrackingConfidence: 0.5
        });

        handsRef.current.onResults(onResults);

        cameraRef.current = new Camera(videoElement, {
          onFrame: async () => {
            if (handsRef.current && videoElement) {
              await handsRef.current.send({ image: videoElement });
            }
          },
          width: 1280,
          height: 720
        });

        await cameraRef.current.start();
      } catch (error) {
        console.error('Failed to initialize hand detection:', error);
      } finally {
        setIsProcessing(false);
      }
    };

    initializeHands();

    return () => {
      if (cameraRef.current) {
        cameraRef.current.stop();
      }
    };
  }, [videoElement, isActive, onResults]);

  return {
    detectedHands,
    currentGesture,
    isProcessing
  };
};
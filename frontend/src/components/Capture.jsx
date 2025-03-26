import { useEffect, useRef, useState } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs-core";
import JSZip from "jszip";
import { saveAs } from "file-saver";
import config from "../config.js";
import axios from "axios";

const captures = [
  {
    // Lumbar Side Flexion (Left)
    instruction:
      "Stand upright, hands by your side. Keeping your back straight and knees locked, slide your LEFT hand down your leg as far as comfortable and hold briefly.",
    provideFeedback: true,
    requiredKeypoints: [
      "left_shoulder",
      "left_wrist",
      "left_hip",
      "left_knee",
      "left_ankle",
      "nose",
      "right_ankle",
    ],
  },
  {
    // Lumbar Side Flexion (Right)
    instruction: "Repeat on the RIGHT side.",
    provideFeedback: true,
    requiredKeypoints: [
      "right_shoulder",
      "right_wrist",
      "right_hip",
      "right_knee",
      "right_ankle",
      "nose",
      "left_ankle",
    ],
  },
  {
    // Intermalleolar Distance (Attempt 1)
    instruction:
      "Stand upright with your feet close together. Keeping your legs straight and knees not bent, slowly move your feet apart sideways as far as possible. Hold briefly.",
    provideFeedback: true,
    requiredKeypoints: [
      "nose",
      "left_ankle",
      "right_ankle",
      "left_hip",
      "right_hip",
      "left_knee",
      "right_knee",
    ],
  },
  {
    // Intermalleolar Distance (Attempt 2)
    instruction: "Repeat again.",
    provideFeedback: true,
    requiredKeypoints: [
      "nose",
      "left_ankle",
      "right_ankle",
      "left_hip",
      "right_hip",
      "left_knee",
      "right_knee",
    ],
  },
  {
    // Cervical Rotation (Left)
    instruction:
      "Stand upright. Slowly turn your head as far as possible to the LEFT without moving your shoulders. Hold briefly.",
    provideFeedback: true,
    requiredKeypoints: [
      "nose",
      "left_ear",
      "right_ear",
      "left_shoulder",
      "right_shoulder",
    ],
  },
  {
    // Cervical Rotation (Right)
    instruction: "Repeat, turning your head to the RIGHT.",
    provideFeedback: true,
    requiredKeypoints: [
      "nose",
      "left_ear",
      "right_ear",
      "left_shoulder",
      "right_shoulder",
    ],
  },
  {
    // Tragus-to-Wall (Left)
    instruction:
      "Position the camera to their LEFT side. Stand with your back against the wall (ensuring your heels, hips, and shoulders touch the wall). Slowly pull your head back towards the wall without tilting up or down. Hold briefly.",
    provideFeedback: true,
    requiredKeypoints: ["left_ear", "left_shoulder"],
  },
  {
    // Tragus-to-Wall (Right)
    instruction: "Repeat with the camera on their RIGHT side.",
    provideFeedback: true,
    requiredKeypoints: ["right_ear", "right_shoulder"],
  },
];
const detectorConfig = {
  modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
  enableSmoothing: true,
};

export default function Capture({ user, userHeight, onLogout }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunks = useRef([]);
  const capturedVideosRef = useRef([]);
  const lastKeypointsRef = useRef(null);
  const feedbackRef = useRef(null);
  const feedbackStreak = useRef(0);

  const [feedback, setFeedback] = useState("Position yourself...");
  const [currentCapture, setCurrentCapture] = useState(0);
  const [cameraFacingMode, setCameraFacingMode] = useState("environment");
  const [capturing, setCapturing] = useState(false);
  const [capturedVideos, setCapturedVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");

  useEffect(() => {
    let detector;
    let animationFrameId;

    async function setupCamera() {
      try {
        if (!videoRef.current) return;
        if (videoRef.current.srcObject) {
          videoRef.current.srcObject
            .getTracks()
            .forEach((track) => track.stop());
        }

        videoRef.current.srcObject = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { exact: cameraFacingMode },
            width: 1280,
            height: 720,
          },
        });
        await new Promise((resolve) => {
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            const canvas = canvasRef.current;
            const video = videoRef.current;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            resolve();
          };
        });
      } catch (err) {
        if (err.name !== "AbortError") {
          // Can safely be ignored
          alert("Error accessing camera: " + err.message);
        }
      }
    }

    async function setupPoseDetection() {
      await tf.setBackend("webgl");
      await tf.ready();

      detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        detectorConfig
      );

      setLoading(false);
      detectPose();
    }

    async function detectPose() {
      if (videoRef.current.readyState < 2) {
        animationFrameId = requestAnimationFrame(detectPose);
        return;
      }
      const poses = await detector.estimatePoses(videoRef.current, {
        flipHorizontal: cameraFacingMode === "user",
      });
      drawKeypoints(poses);
      handleFeedback(poses);
      animationFrameId = requestAnimationFrame(detectPose);
    }

    function handleFeedback(poses) {
      const { provideFeedback, requiredKeypoints } = captures[currentCapture];
      if (!provideFeedback) {
        setFeedback("");
        return;
      }
      if (!poses.length) {
        updateFeedback("Move back");
        return;
      }
      const keypoints = poses[0].keypoints.filter((kp) =>
        requiredKeypoints.includes(kp.name)
      );
      const fullyVisible = keypoints.every((kp) => kp.score > 0.5);
      updateFeedback(fullyVisible ? "Ready" : "Move back");
    }

    function updateFeedback(newStatus) {
      const current = feedbackRef.current;

      if (current === null && newStatus !== "Position yourself...") {
        feedbackRef.current = newStatus;
        setFeedback(newStatus);
        feedbackStreak.current = 0;
        return;
      }

      if (newStatus === current) {
        if (feedback !== current) setFeedback(current);
        feedbackStreak.current = 0;
        return;
      }

      feedbackStreak.current++;
      const requiredStreak = newStatus === "Ready" ? 2 : 30;

      if (feedbackStreak.current >= requiredStreak) {
        feedbackRef.current = newStatus;
        setFeedback(newStatus);
        feedbackStreak.current = 0;
      }
    }

    function drawKeypoints(poses) {
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

      if (!poses.length) return;
      const newKeypoints = poses[0].keypoints;
      if (!lastKeypointsRef.current) lastKeypointsRef.current = newKeypoints;
      const smoothed = newKeypoints.map((kp, i) => {
        const last = lastKeypointsRef.current[i];
        const x = last.x * 0.7 + kp.x * 0.3;
        const y = last.y * 0.7 + kp.y * 0.3;
        const score = kp.score;
        return { x, y, score };
      });
      lastKeypointsRef.current = smoothed;
      smoothed.forEach(({ x, y, score }) => {
        if (score > 0.5) {
          ctx.fillStyle = "lime";
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    }

    setupCamera();
    setupPoseDetection();

    return () => {
      cancelAnimationFrame(animationFrameId);
      if (detector) detector.dispose();
    };
  }, [cameraFacingMode, currentCapture]);

  function toggleCapture() {
    if (capturing) {
      mediaRecorderRef.current.stop();
    } else {
      const stream = videoRef.current.srcObject;
      mediaRecorderRef.current = new MediaRecorder(stream);

      chunks.current = [];
      mediaRecorderRef.current.ondataavailable = (event) =>
        chunks.current.push(event.data);

      mediaRecorderRef.current.onstop = async () => {
        const videoBlob = new Blob(chunks.current, { type: "video/mp4" });
        capturedVideosRef.current.push(videoBlob);
        setCapturedVideos([...capturedVideosRef.current]);
        await nextCapture();
      };
      mediaRecorderRef.current.start();
    }
    setCapturing(!capturing);
  }

  async function nextCapture() {
    if (currentCapture < captures.length - 1) {
      setCurrentCapture((prev) => prev + 1);
      feedbackRef.current = null;
      feedbackStreak.current = 0;
      setFeedback("Position yourself...");
    } else {
      await handleDownloadZip();
    }
  }

  async function handleDownloadZip() {
    setUploading(true);
    setUploadMessage("Processing...");
    const zip = new JSZip();

    capturedVideosRef.current.forEach((videoBlob, idx) => {
      zip.file(`${idx + 1}.mp4`, videoBlob);
    });

    const metadata = {
      username: user,
      datetime: new Date().toISOString(),
      height: userHeight,
    };
    zip.file("metadata.json", JSON.stringify(metadata, null, 2));

    /*zip.generateAsync({type: 'blob'})
        .then((content) => {
          saveAs(content, `${user}.zip`);
          onLogout();
        })
        .catch((err) => {
          alert("Error creating ZIP: " + err.message);
        });*/
    const zipBlob = await zip.generateAsync({ type: "blob" });
    const formData = new FormData();
    formData.append("file", zipBlob, `${user}.zip`);

    try {
      setUploadMessage("Uploading...");
      const response = await axios.post(
        `${config.API_BASE_URL}/api/upload_zip/`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          onUploadProgress: (progressEvent) => {
            const percent = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setUploadMessage(`Uploading... ${percent}%`);
          },
        }
      );

      if (response.data.success) {
        alert("Upload successful!");
        onLogout();
      } else {
        alert("Upload failed: " + response.data.error);
      }
    } catch (error) {
      alert("Upload error: " + error.message);
      console.error(error);
    } finally {
      setUploading(false);
      setUploadMessage("");
    }
  }

  function goBack() {
    if (currentCapture === 0 || capturing) return;

    if (
      !window.confirm(
        "Are you sure you want to go back? The last video will be deleted."
      )
    )
      return;

    capturedVideosRef.current = capturedVideosRef.current.slice(0, -1);
    setCapturedVideos([...capturedVideosRef.current]);

    feedbackRef.current = null;
    feedbackStreak.current = 0;
    setFeedback("Position yourself...");

    setCurrentCapture((prev) => prev - 1);
  }

  return (
    <div className="relative h-screen w-full flex items-center justify-center bg-gray-100 overflow-hidden">
      {loading && (
        <div
          className="absolute inset-0 flex items-center justify-center bg-opacity-70 z-50"
          style={{ backgroundColor: "#0092CA" }}
        >
          <div className="text-white text-lg font-semibold animate-pulse">
            Loading Model...
          </div>
        </div>
      )}
      {uploading && (
        <div
          className="absolute inset-0 bg-opacity-70 flex items-center justify-center z-50"
          style={{ backgroundColor: "#0092CA" }}
        >
          <div className="flex flex-col items-center gap-4">
            <div className="text-center text-white text-lg font-semibold animate-pulse">
              {uploadMessage}
            </div>
            <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-white"></div>
          </div>
        </div>
      )}
      <video
        ref={videoRef}
        className={`absolute top-0 left-0 h-full w-full object-cover ${
          cameraFacingMode === "user" ? "transform scale-x-[-1]" : ""
        }`}
        playsInline
        muted
      />
      <canvas
        ref={canvasRef}
        className={`absolute top-0 left-0 h-full w-full object-cover pointer-events-none ${
          cameraFacingMode === "user" ? "transform scale-x-[-1]" : ""
        }`}
      />
      {captures[currentCapture].provideFeedback && (
        <div
          className={`absolute top-8 text-white p-2 rounded font-semibold ${
            feedback === "Ready" ? "bg-lime-500" : "bg-red-500"
          }`}
        >
          {feedback}
        </div>
      )}
      <button
        onClick={() =>
          setCameraFacingMode((prev) =>
            prev === "environment" ? "user" : "environment"
          )
        }
        className="absolute top-4 left-4 bg-white bg-opacity-70 rounded-full p-2 shadow-lg"
      >
        🔄
      </button>
      <button
        onClick={onLogout}
        className="absolute top-4 right-4 bg-white bg-opacity-70 rounded-full p-2 shadow-lg"
      >
        ✖️
      </button>

      <div className="absolute bottom-0 w-full p-4 bg-white rounded-t-lg shadow-lg">
        <div className="text-center mb-2 font-bold">
          CAPTURE {currentCapture + 1} OF {captures.length}
        </div>
        <div className="text-center text-sm">
          {captures[currentCapture].instruction
            .split(/(LEFT|RIGHT)/)
            .map((part, i) =>
              part === "LEFT" || part === "RIGHT" ? (
                <b className="font-bold animate-pulse" key={i}>
                  {part}
                </b>
              ) : (
                <span key={i}>{part}</span>
              )
            )}
        </div>
        <div className="mt-4 flex gap-2 items-stretch">
          <button
            onClick={goBack}
            className={`flex px-4 py-3 rounded items-center justify-center font-medium ${
              currentCapture === 0 || capturing
                ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                : "bg-black text-white hover:bg-gray-800"
            }`}
            disabled={currentCapture === 0 || capturing}
          >
            ↩️
          </button>
          <button
            onClick={toggleCapture}
            className={`flex-1 font-medium text-white p-3 rounded ${
              capturing ? "bg-red-600" : "bg-black hover:bg-gray-800"
            }`}
          >
            {capturing ? "Capturing..." : "CAPTURE"}
          </button>
        </div>
        <div className="mt-2 text-xs flex justify-between">
          <span>
            User: <b>{user}</b>
          </span>
          <span>
            Date: <b>{new Date().toLocaleDateString()}</b>
          </span>
        </div>
      </div>
    </div>
  );
}
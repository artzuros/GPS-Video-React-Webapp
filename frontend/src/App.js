import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import UploadForm from "./components/UploadForm";
import VideoMap from "./components/VideoMap";
import "./App.css";

export default function App() {
  const [videos, setVideos] = useState([]);
  const [videoData, setVideoData] = useState(null);
  const [mainVideoSrc, setMainVideoSrc] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [inferenceStatus, setInferenceStatus] = useState("idle");
  const [progress, setProgress] = useState(0);
  const [showHeatmap, setShowHeatmap] = useState(false);

  const videoRef = useRef(null);
  const heatmapRef = useRef(null);
  const isSeekingRef = useRef(false);

  // --- Fetch all uploaded videos ---
  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const res = await axios.get("http://localhost:8000/videos");
        setVideos(res.data);
      } catch (err) {
        console.error("Failed to fetch videos list:", err);
      }
    };
    fetchVideos();
  }, []);

  // --- Load video details ---
  const handleUpload = async (id) => {
    try {
      const res = await axios.get(`http://localhost:8000/video/${id}`);
      setVideoData(res.data);
      setCurrentTime(0);
      if (videoRef.current) videoRef.current.currentTime = 0;
    } catch (err) {
      console.error("Failed to fetch video data:", err);
    }
  };

  // --- Keep video + heatmap in sync ---
  const handleTimeUpdate = () => {
    if (!videoRef.current || isSeekingRef.current) return;
    const t = videoRef.current.currentTime;
    setCurrentTime(t);
    if (heatmapRef.current) {
      const diff = Math.abs(heatmapRef.current.currentTime - t);
      if (diff > 0.3) heatmapRef.current.currentTime = t;
    }
  };

  const handleSeek = (time) => {
    if (!videoRef.current) return;
    isSeekingRef.current = true;
    videoRef.current.currentTime = time;
    if (heatmapRef.current) heatmapRef.current.currentTime = time;
    setCurrentTime(time);
    setTimeout(() => (isSeekingRef.current = false), 300);
  };

  // --- Toggle heatmap visibility ---
  const toggleHeatmap = () => {
    setShowHeatmap((prev) => !prev);
  };

  // --- Run inference + monitor progress ---
  const handleRunInference = async () => {
    if (!videoData) return;
    try {
      setInferenceStatus("running");
      setProgress(0);

      await axios.post(
        `http://localhost:8000/api/videos/${videoData.id}/inference?generate_heatmap=true`
      );

      const interval = setInterval(async () => {
        const res = await axios.get(
          `http://localhost:8000/api/videos/${videoData.id}/progress`
        );
        setProgress(res.data.progress);

        if (res.data.status === "done" || res.data.progress >= 100) {
          clearInterval(interval);
          setInferenceStatus("done");

          const refreshed = await axios.get(
            `http://localhost:8000/video/${videoData.id}`
          );
          setVideoData(refreshed.data);
        }
      }, 1000);
    } catch (err) {
      console.error("Inference failed:", err);
      setInferenceStatus("idle");
      alert("Inference failed. Check console for details.");
    }
  };

  // --- Determine correct video sources ---
  useEffect(() => {
    if (!videoData) return;

    const getMainVideoSrc = async () => {
      const lastInference = videoData.inferences?.at(-1);
      if (lastInference?.inference_results_path) {
        const inferenceName = videoData.name.replace(".mp4", "_inference.mp4");
        const inferenceUrl = `http://localhost:8000/uploads/${encodeURIComponent(
          inferenceName
        )}`;

        try {
          await axios.head(inferenceUrl);
          return inferenceUrl; // ✅ Inference video exists
        } catch {
          // fallback
          return `http://localhost:8000/uploads/${encodeURIComponent(videoData.name)}`;
        }
      }

      return `http://localhost:8000/uploads/${encodeURIComponent(videoData.name)}`;
    };

    (async () => {
      const src = await getMainVideoSrc();
      setMainVideoSrc(src);
    })();
  }, [videoData]);

  const getHeatmapSrc = () => {
    const lastInference = videoData?.inferences?.at(-1);
    return lastInference?.heatmap_path
      ? `http://localhost:8000/${lastInference.heatmap_path}`
      : null;
  };

  // --- UI ---
  return (
    <div className="app-container">
      <h1 className="app-title">RAHI WebApp</h1>

      {/* Upload Form */}
      <UploadForm onUpload={handleUpload} />

      {/* Thumbnails */}
      <div className="thumbnails-row">
        {videos.map((v) => (
          <button
            key={v.id}
            className="thumbnail-btn"
            onClick={() => handleUpload(v.id)}
            title={v.name}
          >
            <video
              className="thumbnail-video"
              src={`http://localhost:8000/uploads/${encodeURIComponent(v.name)}`}
              muted
              preload="metadata"
            />
            <div className="thumbnail-label">{v.name}</div>
          </button>
        ))}
      </div>

      {videoData && (
        <div className="content-wrapper">
          {/* --- Video --- */}
          <div className="video-wrapper">
            <h3>
              {videoData.inferences?.length
                ? "Inference Result"
                : "Original Video"}
            </h3>

            <div className="video-overlay-wrapper">
              <video
                ref={videoRef}
                className="base-video"
                src={mainVideoSrc || ""}
                controls
                onTimeUpdate={handleTimeUpdate}
                onPlay={() => heatmapRef.current && heatmapRef.current.play()}
                onPause={() => heatmapRef.current && heatmapRef.current.pause()}
              />

              {showHeatmap && getHeatmapSrc() && (
                <video
                  ref={heatmapRef}
                  className="heatmap-video-overlay"
                  src={getHeatmapSrc()}
                  muted
                />
              )}
            </div>

            {getHeatmapSrc() && (
              <div className="toggle-btn-wrapper">
                <button className="toggle-heatmap-btn" onClick={toggleHeatmap}>
                  {showHeatmap
                    ? "Hide Heatmap Overlay"
                    : "Show Heatmap Overlay"}
                </button>
              </div>
            )}
          </div>

          {/* --- Map --- */}
          <div className="map-wrapper">
            <VideoMap
              gpsPoints={videoData.gps_points}
              currentTime={currentTime}
              onSeek={handleSeek}
            />
          </div>

          {/* --- Inference Panel --- */}
          <div className="inference-panel">
            <h3>Inference</h3>
            <button
              onClick={handleRunInference}
              disabled={inferenceStatus === "running"}
              className="inference-button"
            >
              {inferenceStatus === "running"
                ? "Running Inference..."
                : "Run Inference + Heatmap"}
            </button>

            {inferenceStatus === "running" && (
              <div className="progress-container">
                <div className="progress-label">Processing...</div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {videoData.inferences?.length > 0 ? (
              <ul className="inference-list">
                {videoData.inferences.map((inf) => (
                  <li key={inf.id}>
                    📄{" "}
                    <a
                      href={`http://localhost:8000/${inf.inference_results_path}`}
                      download
                    >
                      CSV
                    </a>
                    {inf.heatmap_path && (
                      <>
                        {" | "}
                        🎥{" "}
                        <a
                          href={`http://localhost:8000/${inf.heatmap_path}`}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          Heatmap
                        </a>
                      </>
                    )}
                    <div className="inference-timestamp">
                      Generated at:{" "}
                      {new Date(inf.created_at).toLocaleString()}
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p>No inference results yet.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

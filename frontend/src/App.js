import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import UploadForm from './components/UploadForm';
import VideoMap from './components/VideoMap';
import './App.css';

export default function App() {
  const [videos, setVideos] = useState([]);
  const [videoData, setVideoData] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const videoRef = useRef(null);
  const heatmapRef = useRef(null);
  const [inferenceStatus, setInferenceStatus] = useState('idle'); // idle | running | done
  const isSeekingRef = useRef(false);
  const lastManualSeekRef = useRef(0);
  const [progress, setProgress] = useState(0);


  // Fetch all uploaded videos
  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const res = await axios.get('http://localhost:8000/videos');
        setVideos(res.data);
      } catch (err) {
        console.error('Failed to fetch videos list:', err);
      }
    };
    fetchVideos();
  }, []);

  // Fetch full video info when selected
  const handleUpload = async (id) => {
    try {
      const res = await axios.get(`http://localhost:8000/video/${id}`);
      setVideoData(res.data);
      setCurrentTime(0);
      if (videoRef.current) videoRef.current.currentTime = 0;
    } catch (err) {
      console.error('Failed to fetch video data:', err);
    }
  };

  const handleTimeUpdate = () => {
    if (!videoRef.current || isSeekingRef.current) return;
    const t = videoRef.current.currentTime;
    setCurrentTime(t);

    // Keep heatmap synced smoothly
    if (heatmapRef.current) {
      const diff = Math.abs(heatmapRef.current.currentTime - t);
      if (diff > 0.3) {
        heatmapRef.current.currentTime = t;
      }
    }
  };

  const handleSeek = (time) => {
    if (!videoRef.current) return;
    isSeekingRef.current = true;
    lastManualSeekRef.current = Date.now();

    // Jump both videos instantly
    videoRef.current.currentTime = time;
    if (heatmapRef.current) heatmapRef.current.currentTime = time;
    setCurrentTime(time);

    // Turn off seek lock shortly after (300ms)
    setTimeout(() => {
      isSeekingRef.current = false;
    }, 300);
  };


  const handleRunInference = async () => {
    if (!videoData) return;
    try {
      setInferenceStatus('running');
      setProgress(0);
      await axios.post(
        `http://localhost:8000/api/videos/${videoData.id}/inference?generate_heatmap=true`
      );

      // Poll for progress
      const interval = setInterval(async () => {
        const res = await axios.get(`http://localhost:8000/api/videos/${videoData.id}/progress`);
        setProgress(res.data.progress);

        if (res.data.status === 'done' || res.data.progress >= 100) {
          clearInterval(interval);
          setInferenceStatus('done');

          // Refresh data
          const refreshed = await axios.get(`http://localhost:8000/video/${videoData.id}`);
          setVideoData(refreshed.data);
        }
      }, 1000);
    } catch (err) {
      console.error('Inference failed:', err);
      setInferenceStatus('idle');
      alert('Inference failed. Check console for details.');
    }
  };


  return (
    <div className="app-container">
      <h1 className="app-title">RAHI WebApp</h1>

      {/* Upload Form */}
      <UploadForm onUpload={handleUpload} />

      {/* Thumbnails */}
      <div className="thumbnails-row">
        {videos.map((v) => (
          <button
            type="button"
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

      {/* Main Video + Map + Inference Panel */}
      {videoData && (
        <div className="content-wrapper">

          {/* 🧠 VIDEO + HEATMAP SIDE BY SIDE */}
          <div className="video-section">
            {/* Original Video */}
            <div className="video-wrapper">
              <h3>Original Video</h3>
              <video
                ref={videoRef}
                src={`http://localhost:8000/uploads/${encodeURIComponent(videoData.name)}`}
                controls
                onTimeUpdate={handleTimeUpdate}
                onPlay={() => {
                  if (heatmapRef.current && heatmapRef.current.paused) {
                    heatmapRef.current.play();
                  }
                }}
                onPause={() => {
                  if (heatmapRef.current && !heatmapRef.current.paused) {
                    heatmapRef.current.pause();
                  }
                }}
              />
            </div>

            {/* Heatmap Video (if exists) */}
            {videoData?.inferences?.length > 0 &&
              videoData.inferences[videoData.inferences.length - 1].heatmap_path && (
                <div className="video-wrapper">
                  <h3>Heatmap</h3>
                  <video
                    ref={heatmapRef}
                    src={`http://localhost:8000/${videoData.inferences[
                      videoData.inferences.length - 1
                    ].heatmap_path}`}
                    controls
                    muted
                    onPlay={() => {
                      if (videoRef.current && videoRef.current.paused) videoRef.current.play();
                    }}
                    onPause={() => {
                      if (videoRef.current && !videoRef.current.paused) videoRef.current.pause();
                    }}

                  />
                </div>
              )}
          </div>

          {/* Map */}
          <div className="map-wrapper">
            <VideoMap
              gpsPoints={videoData.gps_points}
              currentTime={currentTime}
              onSeek={handleSeek}
            />
          </div>

          {/* Inference Panel */}
          <div className="inference-panel">
            <h3>Inference</h3>

            <button
              onClick={handleRunInference}
              disabled={inferenceStatus === 'running'}
              className="inference-button"
            >
              {inferenceStatus === 'running'
                ? 'Running Inference...'
                : 'Run Inference + Heatmap'}
            </button>

            {/* 🔄 Progress bar */}
            {inferenceStatus === 'running' && (
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

            {/* List of past inferences */}
            {videoData.inferences && videoData.inferences.length > 0 ? (
              <ul className="inference-list">
                {videoData.inferences.map((inf) => (
                  <li key={inf.id}>
                    📄{' '}
                    <a
                      href={`http://localhost:8000/${inf.inference_results_path}`}
                      download
                    >
                      CSV
                    </a>
                    {inf.heatmap_path && (
                      <>
                        {' | '}
                        🎥{' '}
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
                      Generated at: {new Date(inf.created_at).toLocaleString()}
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

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
  const [inferenceStatus, setInferenceStatus] = useState('idle'); // idle | running | done


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
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleSeek = (time) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const handleRunInference = async () => {
    if (!videoData) return;
    try {
      setInferenceStatus('running');
      const res = await axios.post(
        `http://localhost:8000/api/videos/${videoData.id}/inference?generate_heatmap=true`
      );
      setInferenceStatus('done');
      const refreshed = await axios.get(`http://localhost:8000/video/${videoData.id}`);
      setVideoData(refreshed.data);
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
          {/* Video */}
          <div className="video-wrapper">
            <video
              ref={videoRef}
              src={`http://localhost:8000/uploads/${encodeURIComponent(videoData.name)}`}
              controls
              onTimeUpdate={handleTimeUpdate}
            />
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
              {inferenceStatus === 'running' ? 'Running Inference...' : 'Run Inference + Heatmap'}
            </button>

            {inferenceStatus === 'running' && (
              <div className="inference-status">🔄 Please wait, processing video...</div>
            )}

            {/* List of past inferences */}
            {videoData.inferences && videoData.inferences.length > 0 ? (
              <ul className="inference-list">
                {videoData.inferences.map((inf) => (
                  <li key={inf.id}>
                    📄{' '}
                    <a href={`http://localhost:8000/${inf.inference_results_path}`} download>
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

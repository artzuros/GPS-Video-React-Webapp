import React, { useState } from 'react';
import axios from 'axios';
import './UploadForm.css';

export default function UploadForm({ onUpload }) {
  const [video, setVideo] = useState(null);
  const [csv, setCsv] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleSubmit = async () => {
    if (!video) return alert("Please select a video.");

    setIsUploading(true);
    const formData = new FormData();
    formData.append("video", video);
    if (csv) formData.append("csv_file", csv);

    try {
      const res = await axios.post("http://localhost:8000/upload/", formData);
      onUpload(res.data.video_id);
    } catch (err) {
      console.error("Upload failed:", err);
      alert("Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="upload-form">
      <input type="file" accept="video/*" onChange={(e) => setVideo(e.target.files[0])} />
      <input type="file" accept=".csv" onChange={(e) => setCsv(e.target.files[0])} />
      <button onClick={handleSubmit} disabled={isUploading}>
        {isUploading ? 'Uploading...' : 'Upload'}
      </button>
    </div>
  );
}

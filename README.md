# 🛣️ RAHI – Road Analysis with Human In The Loop

> **RAHI** (Road Analysis with Human In The Loop) is an AI-driven web application for intelligent roadway monitoring and video analysis.
> This project is made alongside MoRD (Ministry of Rural Development).
> It enables synchronized playback of road inspection videos with GPS traces, annotation of bad road segments, and backend support for future AI-based detection models.

---

## Features

### Video Management

* Upload, view, and store road inspection videos.
* Maintain structured metadata (location, progress, timestamps, etc.).

### GPS Path Tracking

* Visualize GPS traces alongside video playback.
* Highlight road quality sections (e.g., good/bad segments).
* Sync GPS points with corresponding video timestamps.

### AI Model Integration (Pluggable)

* Modular support for deep learning models.
* Integrates **CBAM-enhanced ResNet** for binary road condition classification.
* Built for easy extension with other CV models (e.g., YOLO, LoFTR, LightGlue).

### Interactive Visualization

* React-based frontend with:

  * Integrated video player
  * Interactive map (Leaflet.js)
  * GPS path overlays and synchronized markers

---

## Tech Stack

| Layer            | Technology                             |
| ---------------- | -------------------------------------- |
| **Frontend**     | React.js, Axios            |
| **Backend**      | FastAPI, SQLAlchemy                    |
| **Database**     | PostgreSQL                             |
| **AI/ML Models** | PyTorch, ResNet-CBAM                   |
| **Dev Tools**    | Docker (optional), Uvicorn, Conda/venv |

---

## Project Structure

```
RAHI/
├── backend/
│   ├── BinaryClassification/
│   │   ├── CBAM/
│   │   │   ├── gradcam_inference.py
│   │   │   ├── inference.py
│   │   │   ├── resnet_cbam.py
│   │   │   └── weights/
│   │   └── ...
│   ├── crud.py
│   ├── database.py
│   ├── inference_utils.py
│   ├── models.py
│   ├── main.py
│   ├── uploads/
│   └── video_routes.py
│
├── db/
│   └── init.sql                # Database init script
│
└── frontend/
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── components/
    │   │   ├── RunInference.js
    │   │   ├── UploadForm.js
    │   │   └── VideoMap.js
    │   ├── App.js
    │   └── index.js
    ├── package.json
    └── example.env
```

---

## Backend Setup (FastAPI)

### Create and activate a Python environment

```bash
conda create -n rahi python=3.10
conda activate rahi
```

### Install dependencies

```bash
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic torch torchvision opencv-python
```

### Setup PostgreSQL

Make sure PostgreSQL is running, and create a database:

```bash
psql -U postgres
CREATE DATABASE rahi;
```

Initialize tables:

```bash
psql -U postgres -d rahi -f db/init.sql
```

### Run the backend

```bash
uvicorn backend.main:app --reload
```

Server will start at:
 `http://127.0.0.1:8000`

### 5️⃣ API Routes Overview

| Endpoint         | Method | Description                    |
| ---------------- | ------ | ------------------------------ |
| `/upload-video/` | POST   | Upload video file              |
| `/videos/`       | GET    | List all uploaded videos       |
| `/gps/`          | GET    | Fetch GPS coordinates          |
| `/inference/`    | POST   | Run AI model on uploaded video |

---

## Frontend Setup (React)

### Move into the frontend folder

```bash
cd frontend
```

### Install dependencies

```bash
npm install
```

### Configure environment

Copy `.env` file:

```bash
cp example.env .env
```

Update backend API URL if needed:

```
REACT_APP_API_URL=http://127.0.0.1:8000
```

### Run the frontend

```bash
npm start
```

Frontend will start at:
`http://localhost:3000`

---

## Workflow

1. **Upload a road inspection video** via the web UI.
The video along with a .csv file like the example below
    ```
    lon,lat,highlight,timestamp
    77.33395,28.64751,false,5
    77.33453,28.64781,true,10
    77.33489,28.64756,true,15
    77.33511,28.64737,false,20
    ```
    The highlight = false indicates good road (This is just a placeholder for now, will only need gps and timestamps soon.)
2. **GPS trace** is fetched or uploaded.
3. **Map view** displays GPS route.
4. **Video player** syncs with GPS markers.
5. **AI model (CBAM ResNet)** performs classification on video frames.
6. Results and highlights update dynamically on the map.

---

## Model Overview

The backend currently integrates a **ResNet with CBAM (Convolutional Block Attention Module)** for binary road classification.

* Detects whether the road segment is in good or bad condition.
* Produces GradCAM visualizations to interpret predictions.

Model path:
`backend/BinaryClassification/CBAM/weights/`

Google Drive Link for model weights : 
`https://drive.google.com/drive/folders/15gkcbGOrtMzxruEzMWsNPcBhidLNJNvX?usp=drive_link`

---

## 🗺️ Frontend Overview

**Key Components:**

* `UploadForm.js` → Upload new video
* `RunInference.js` → Run AI analysis
* `VideoMap.js` → Interactive map with GPS traces
* `App.js` → Root app component

**Key Features:**

* Map-Video synchronization
* Video annotation & visualization
* Responsive, modular design


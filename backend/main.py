import os, csv
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import cv2, math

from . import crud, models, database
from .video_routes import router as video_router
from .inference_utils import transcode_to_h264

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Database ---
models.Base.metadata.create_all(bind=database.engine)

# --- FastAPI app ---
app = FastAPI()
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database session dependency ---
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Upload Endpoint ---
@app.post("/upload/")
async def upload_files(
    video: UploadFile = File(...),
    csv_file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    filename = Path(video.filename).name
    video_path = UPLOAD_DIR / filename

    # Save uploaded file
    with video_path.open("wb") as f:
        f.write(await video.read())
    print(f"📥 Uploaded file: {video_path.name}")

    # --- Always Transcode ---
    try:
        transcoded_path = Path(transcode_to_h264(str(video_path)))
        print(f"✅ Transcoded to H.264: {transcoded_path.name}")

        # Remove original file if different
        if transcoded_path != video_path and video_path.exists():
            os.remove(video_path)
            print(f"🧹 Removed original: {video_path.name}")

        video_path = transcoded_path
    except Exception as e:
        print(f"⚠️ Transcoding failed: {e}")

    # --- Compute Duration ---
    video_cap = cv2.VideoCapture(str(video_path))
    if not video_cap.isOpened():
        duration = 0.0
        print(f"⚠️ Unable to open video: {video_path}")
    else:
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0.0
        duration = math.floor(duration)
    video_cap.release()

    # --- Save DB Record ---
    new_video = crud.create_video(db, name=video_path.name, file_path=str(video_path), duration=duration)
    print(f"✅ Saved video record: {new_video.name} ({duration}s)")

    # --- Optional GPS CSV Upload ---
    if csv_file:
        contents = (await csv_file.read()).decode("utf-8").splitlines()
        reader = csv.DictReader(contents)
        gps_data = [
            {"lat": float(row["lat"]), "lon": float(row["lon"]), "timestamp": float(row["timestamp"])}
            for row in reader
        ]
        crud.create_gps_points(db, video_id=new_video.id, gps_data=gps_data)
        print(f"📍 Added {len(gps_data)} GPS points for video {new_video.id}")

    return {"video_id": new_video.id}


# --- Get Single Video ---
@app.get("/video/{video_id}")
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = crud.get_video_with_gps(db, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    inference_results = crud.get_inference_results_by_video(db, video_id)
    return {
        "id": video.id,
        "name": video.name,
        "path": video.file_path,
        "duration": video.duration,
        "gps_points": [
            {"lat": p.lat, "lon": p.lon, "highlight": p.highlight, "timestamp": p.timestamp}
            for p in video.gps_points
        ],
        "inferences": [
            {
                "id": inf.id,
                "inference_results_path": inf.inference_results_path,
                "heatmap_path": inf.heatmap_path,
                "created_at": inf.created_at,
            }
            for inf in inference_results
        ],
    }


# --- List All Videos ---
@app.get("/videos/")
def list_videos(db: Session = Depends(get_db)):
    videos = crud.get_all_videos(db)
    return [
        {"id": v.id, "name": v.name, "gps_ids": [p.id for p in v.gps_points]}
        for v in videos
    ]


# --- Delete Video ---
@app.delete("/video/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(video_id: int, db: Session = Depends(get_db)):
    video = crud.get_video_with_gps(db, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    crud.delete_video(db, video_id)
    print(f"🗑️ Deleted video {video_id}")
    return


# --- Delete GPS Point ---
@app.delete("/gps_point/{gps_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_gps_point(gps_id: int, db: Session = Depends(get_db)):
    gps_point = crud.get_gps_point(db, gps_id)
    if gps_point is None:
        raise HTTPException(status_code=404, detail="GPS point not found")
    crud.delete_gps_point(db, gps_id)
    print(f"🗑️ Deleted GPS point {gps_id}")
    return


# --- Include Video Inference Routes ---
app.include_router(video_router, prefix="/api")

import os, csv
from fastapi import FastAPI, UploadFile, File, Form, Depends
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2, math
from sqlalchemy.orm import Session
from . import crud, models, database

from .video_routes import router as video_router

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/upload/")
async def upload_files(
    video: UploadFile = File(...),
    csv_file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    filename = Path(video.filename).name
    video_path = UPLOAD_DIR / filename
    with video_path.open("wb") as f:
        f.write(await video.read())

    # Dummy duration
    video_cap = cv2.VideoCapture(str(video_path))
    if not video_cap.isOpened():
        duration = 0.0
    else:
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0.0
        duration = math.floor(duration)
    video_cap.release()

    new_video = crud.create_video(db, name=filename, file_path=str(video_path), duration=duration)

    if csv_file:
        contents = (await csv_file.read()).decode('utf-8').splitlines()
        reader = csv.DictReader(contents)
        gps_data = [
            {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "highlight": row["highlight"].lower() == 'true',
                "timestamp": float(row["timestamp"])
            }
            for row in reader
        ]
        crud.create_gps_points(db, video_id=new_video.id, gps_data=gps_data)

    return { "video_id": new_video.id }

from fastapi import HTTPException
from fastapi import status
    
@app.get("/video/{video_id}")
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = crud.get_video_with_gps(db, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    # Fetch inference history
    inference_results = crud.get_inference_results_by_video(db, video_id)
    return {
        "id": video.id,
        "name": video.name,
        "path": video.file_path,
        "duration": video.duration,
        "gps_points": [{
            "lat": p.lat,
            "lon": p.lon,
            "highlight": p.highlight,
            "timestamp": p.timestamp
        } for p in video.gps_points],
        "inferences": [{
            "id": inf.id,
            "inference_results_path": inf.inference_results_path,
            "heatmap_path": inf.heatmap_path,
            "created_at": inf.created_at
        } for inf in inference_results]
    }

@app.get("/videos/")
def list_videos(db: Session = Depends(get_db)):
    videos = crud.get_all_videos(db)
    result = []
    for video in videos:
        gps_ids = [p.id for p in video.gps_points]
        result.append({
            "id": video.id,
            "name": video.name,
            "gps_ids": gps_ids
        })
    return result

@app.delete("/video/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(video_id: int, db: Session = Depends(get_db)):
    video = crud.get_video_with_gps(db, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    crud.delete_video(db, video_id)
    return

@app.delete("/gps_point/{gps_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_gps_point(gps_id: int, db: Session = Depends(get_db)):
    gps_point = crud.get_gps_point(db, gps_id)
    if gps_point is None:
        raise HTTPException(status_code=404, detail="GPS point not found")
    crud.delete_gps_point(db, gps_id)
    return


app.include_router(video_router, prefix="/api")

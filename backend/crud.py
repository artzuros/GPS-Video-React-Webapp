from sqlalchemy.orm import Session
from . import models
import os
import subprocess
from .utils.transcode import transcode_to_h264

def create_video(db: Session, name: str, file_path: str, duration: float):
    # Ensure upload directory exists
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    # Transcode to H.264
    transcoded_path = transcode_to_h264(file_path)
    # Update name and path for DB
    name = os.path.basename(transcoded_path)
    video = models.Video(name=name, file_path=transcoded_path, duration=duration)

    db.add(video)
    db.commit()
    db.refresh(video)
    return video

def create_gps_points(db: Session, video_id: int, gps_data: list):
    """
    gps_data: list of dicts with keys lat, lon, timestamp
    highlight will be set after inference
    """
    for row in gps_data:
        point = models.GPSPoint(
            video_id=video_id,
            lat=row["lat"],
            lon=row["lon"],
            timestamp=row["timestamp"],
            highlight=row.get("highlight", None),  # optional
        )
        db.add(point)
    db.commit()

def update_gps_points_with_inference(
    db: Session, video_id: int, frame_timestamps: list, raw_probs: list, smoothed_probs: list = None
):
    """
    Match video frame predictions to GPS points based on timestamp.
    Assign binary highlight = True if smoothed_prob > 0.5 (or raw if smoothed is None)
    """
    gps_points = db.query(models.GPSPoint).filter(models.GPSPoint.video_id == video_id).all()
    for point in gps_points:
        # Find nearest frame timestamp
        nearest_idx = min(range(len(frame_timestamps)), key=lambda i: abs(frame_timestamps[i] - point.timestamp))
        prob = smoothed_probs[nearest_idx] if smoothed_probs is not None else raw_probs[nearest_idx]
        point.highlight = True if prob > 0.5 else False
        print(point.highlight)
    db.commit()


def get_video_with_gps(db: Session, video_id: int):
    return db.query(models.Video).filter(models.Video.id == video_id).first()

def get_all_videos(db: Session):
    return db.query(models.Video).all()

def delete_video(db: Session, video_id: int):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if video:
        db.delete(video)
        db.commit()
        return True
    return False

def get_gps_point(db: Session, gps_point_id: int):
    return db.query(models.GPSPoint).filter(models.GPSPoint.id == gps_point_id).first()

def delete_gps_point(db: Session, gps_point_id: int):
    gps_point = db.query(models.GPSPoint).filter(models.GPSPoint.id == gps_point_id).first()
    if gps_point:
        db.delete(gps_point)
        db.commit()
        return True
    return False

def get_inference_results_by_video(db: Session, video_id: int):
    return db.query(models.InferenceResult).filter(models.InferenceResult.video_id == video_id).all()

def create_inference_result(
    db: Session,
    video_id: int,
    inference_results_path: str,
    heatmap_path: str = None,
    created_at: str = None
):
    result = models.InferenceResult(
        video_id=video_id,
        inference_results_path=inference_results_path,
        heatmap_path=heatmap_path,
        created_at=created_at
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result

def delete_video(db: Session, video_id: int):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        return False

    # Delete associated inference files
    inferences = db.query(models.InferenceResult).filter(models.InferenceResult.video_id == video_id).all()
    for inf in inferences:
        if inf.inference_results_path and os.path.exists(inf.inference_results_path):
            os.remove(inf.inference_results_path)
        if inf.heatmap_path and os.path.exists(inf.heatmap_path):
            os.remove(inf.heatmap_path)

    # InferenceResult rows will be deleted due to ON DELETE CASCADE
    db.delete(video)
    db.commit()
    return True

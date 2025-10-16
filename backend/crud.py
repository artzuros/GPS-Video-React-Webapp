from sqlalchemy.orm import Session
from . import models
import os
import subprocess

def create_video(db: Session, name: str, file_path: str, duration: float):
    # Define upload directory and output file path
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    output_file = os.path.join(upload_dir, os.path.basename(file_path).replace('.mp4', '_h264.mp4'))

    # FFmpeg command to transcode and force even dimensions
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', file_path,
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264', '-profile:v', 'baseline', '-level', '3.0',
        '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
        output_file
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    name = name.replace('.mp4', '_h264.mp4')
    # Save video record with new file path
    video = models.Video(name=name, file_path=output_file, duration=duration)
    db.add(video)
    db.commit()
    db.refresh(video)
    return video

def create_gps_points(db: Session, video_id: int, gps_data: list):
    for row in gps_data:
        point = models.GPSPoint(
            video_id=video_id,
            lat=row["lat"],
            lon=row["lon"],
            highlight=row["highlight"],
            timestamp=row["timestamp"]
        )
        db.add(point)
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

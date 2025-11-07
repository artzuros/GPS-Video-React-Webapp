# /backend/video_routes.py

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from . import crud, models, database
from .inference_utils import run_inference_on_video_async, progress_tracker
from .utils.transcode import transcode_to_h264
import os
import asyncio

router = APIRouter()

@router.post("/videos/{video_id}/inference")
async def infer_on_video(
    video_id: int,
    background_tasks: BackgroundTasks,
    generate_heatmap: bool = Query(default=False),
    smoothing: str = Query(default="ema", description="Smoothing method: 'ema' or 'moving_average'"),
    db: Session = Depends(database.get_db),
):
    video = crud.get_video_with_gps(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "BinaryClassification", "CBAM", "weights", "model_best.pth.tar",
    )

    # ✅ Always ensure uploaded video is in H.264 before inference
    try:
        transcoded_path = transcode_to_h264(video.file_path)
        if transcoded_path != video.file_path:
            video.file_path = transcoded_path
            db.commit()
    except Exception as e:
        print(f"⚠️ Transcoding before inference failed: {e}")

    # Initialize progress
    progress_tracker[str(video_id)] = {"current": 0, "total": 1, "status": "starting"}

    async def inference_job():
        try:
            results = await run_inference_on_video_async(
                video_path=video.file_path,
                video_id=str(video_id),
                model_path=model_path,
                generate_heatmap=generate_heatmap,
                smoothing=smoothing,
            )

            crud.update_gps_points_with_inference(
                db=db,
                video_id=video_id,
                frame_timestamps=results["frame_timestamps"],
                raw_probs=results["raw_probs"],
                smoothed_probs=results.get("smoothed_probs"),
            )

            crud.create_inference_result(
                db=db,
                video_id=video_id,
                inference_results_path=results["csv_output"],
                heatmap_path=results["heatmap_video"] if generate_heatmap else None,
                created_at=results["created_at"],
            )

            progress_tracker[str(video_id)]["status"] = "done"

        except Exception as e:
            progress_tracker[str(video_id)]["status"] = f"error: {str(e)}"
            print(f"[ERROR] Inference failed for video {video_id}: {e}")

    def run_async_task(coro_func):
        asyncio.run(coro_func())

    # ✅ Launch inference asynchronously
    background_tasks.add_task(run_async_task, inference_job)

    return {"message": "Inference started", "status": "running"}


@router.get("/videos/{video_id}/progress")
async def get_progress(video_id: str):
    progress = progress_tracker.get(str(video_id), {"current": 0, "total": 1, "status": "idle"})
    percent = (progress["current"] / progress["total"]) * 100 if progress["total"] > 0 else 0
    return {"progress": percent, "status": progress["status"]}


@router.get("/videos/{video_id}/inference")
def get_inference_history(video_id: int, db: Session = Depends(database.get_db)):
    video = crud.get_video_with_gps(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    inference_results = crud.get_inference_results_by_video(db, video_id)
    return [
        {
            "id": inf.id,
            "inference_results_path": inf.inference_results_path,
            "heatmap_path": inf.heatmap_path,
            "created_at": inf.created_at
        }
        for inf in inference_results
    ]

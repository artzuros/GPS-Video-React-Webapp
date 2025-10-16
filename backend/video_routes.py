# /backend/video_routes.py

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from . import crud, models, database
from . inference_utils import run_inference_on_video
import os

router = APIRouter()
@router.post("/videos/{video_id}/inference")
def infer_on_video(
    video_id: int,
    generate_heatmap: bool = Query(default=False, description="Whether to generate heatmap overlay"),
    db: Session = Depends(database.get_db)
):
    video = crud.get_video_with_gps(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BinaryClassification", "CBAM", "weights", "model_best.pth.tar")

    try:
        results = run_inference_on_video(
            video_path=video.file_path,
            model_path=model_path,
            generate_heatmap=generate_heatmap
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    crud.create_inference_result(
        db=db,
        video_id=video_id,
        inference_results_path=results["csv_output"],
        heatmap_path=results["heatmap_video"] if generate_heatmap else None
    )

    return {
        "message": "Inference completed successfully",
        "output_video": results["output_video"],
        "csv_output": results["csv_output"],
        "heatmap_video": results["heatmap_video"] if generate_heatmap else None
    }


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

import asyncio
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
import csv
import numpy as np
from collections import deque
from tqdm import tqdm
from backend.BinaryClassification.CBAM.gradcam import GradCAM
import backend.BinaryClassification.CBAM.resnet_cbam as resnet_cbam
from .utils.transcode import transcode_to_h264

progress_tracker = {}

# --- Temporal smoothing ---
def apply_moving_average(probs, window_size=7):
    smoothed = []
    dq = deque(maxlen=window_size)
    for p in probs:
        dq.append(p)
        smoothed.append(np.mean(dq))
    return smoothed

def apply_ema(probs, alpha=0.3):
    smoothed = []
    for i, p in enumerate(probs):
        if i == 0:
            smoothed.append(p)
        else:
            smoothed.append(alpha * p + (1 - alpha) * smoothed[-1])
    return smoothed


# --- Main async inference wrapper ---
async def run_inference_on_video_async(
    video_path: str,
    video_id: str,
    model_path: str,
    generate_heatmap: bool = True,
    smoothing: str = "moving_average"
):
    """
    Async wrapper that runs inference in a thread pool
    and updates progress_tracker for UI polling.
    """

    def _run_inference():
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load model
        model = resnet_cbam.resnet101_cbam(pretrained=False)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(DEVICE)
        model.eval()

        gradcam = GradCAM(model, model.layer4[-1].conv3) if generate_heatmap else None

        transform = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress_tracker[str(video_id)] = {"current": 0, "total": total_frames, "status": "running"}

        base_dir = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(base_dir, f"{base_name}_inference.mp4")
        heatmap_video_path = os.path.join(base_dir, f"{base_name}_heatmap.mp4")
        csv_output_path = os.path.join(base_dir, f"{base_name}_predictions.csv")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        out_heatmap = cv2.VideoWriter(heatmap_video_path, fourcc, fps, (frame_width, frame_height)) if generate_heatmap else None

        raw_probs, frames, timestamps = [], [], []

        with tqdm(total=total_frames, desc=f"Inference on {base_name}", unit="frame") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_sec = frame_idx / fps
                timestamps.append(timestamp_sec)
                frames.append(frame.copy())

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.sigmoid(output).item()

                raw_probs.append(prob)
                pbar.update(1)
                progress_tracker[str(video_id)]["current"] = frame_idx + 1

        cap.release()

        # --- Temporal smoothing ---
        if smoothing == "moving_average":
            smoothed_probs = apply_moving_average(raw_probs)
        elif smoothing == "ema":
            smoothed_probs = apply_ema(raw_probs)
        else:
            smoothed_probs = raw_probs

        # --- Write results ---
        with open(csv_output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Frame', 'Timestamp_sec', 'Raw_Probability', 'Smoothed_Probability', 'Predicted_Label'])

            for idx, (frame, raw_p, smooth_p, ts) in enumerate(zip(frames, raw_probs, smoothed_probs, timestamps)):
                pred_label = 1 if smooth_p > 0.5 else 0
                label_text = 'Good' if pred_label else 'Bad'
                color = (0, 255, 0) if pred_label else (0, 0, 255)

                cv2.putText(frame, f"{label_text} ({smooth_p:.2f})", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                out.write(frame)

                if generate_heatmap:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
                    heatmap = gradcam.generate(input_tensor, class_idx=0)
                    heatmap = cv2.resize(heatmap, (frame_width, frame_height))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    out_heatmap.write(heatmap_color)

                writer.writerow([idx, f"{ts:.2f}", f"{raw_p:.4f}", f"{smooth_p:.4f}", label_text])

        out.release()
        if out_heatmap:
            out_heatmap.release()
            gradcam.remove_hooks()

        # ✅ Overwrite both videos in place (no suffixes)
        transcode_to_h264(output_video_path)
        if generate_heatmap:
            transcode_to_h264(heatmap_video_path)

        progress_tracker[str(video_id)]["status"] = "done"

        import datetime
        infer_time = str(datetime.datetime.now())

        return {
            "output_video": f"uploads/{os.path.basename(output_video_path)}",
            "csv_output": f"uploads/{os.path.basename(csv_output_path)}",
            "heatmap_video": f"uploads/{os.path.basename(heatmap_video_path)}" if generate_heatmap else None,
            "created_at": infer_time,
            "frame_timestamps": timestamps,
            "raw_probs": raw_probs,
            "smoothed_probs": smoothed_probs
        }

    # --- Run the blocking job in background thread ---
    return await asyncio.to_thread(_run_inference)

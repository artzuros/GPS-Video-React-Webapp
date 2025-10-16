import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
import csv
import numpy as np
from backend.BinaryClassification.CBAM.gradcam import GradCAM
import backend.BinaryClassification.CBAM.resnet_cbam as resnet_cbam

def run_inference_on_video(video_path: str, model_path: str, generate_heatmap: bool = False):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = resnet_cbam.resnet101_cbam(pretrained=False)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(DEVICE)
    model.eval()

    # Grad-CAM setup if needed
    gradcam = None
    target_layer = model.layer4[-1].conv3
    if generate_heatmap:
        gradcam = GradCAM(model, target_layer)

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

    base_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(base_dir, f"{base_name}_inference.mp4")
    heatmap_video_path = os.path.join(base_dir, f"{base_name}_heatmap.mp4")
    csv_output_path = os.path.join(base_dir, f"{base_name}_predictions.csv")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    out_heatmap = None
    if generate_heatmap:
        out_heatmap = cv2.VideoWriter(heatmap_video_path, fourcc, fps, (frame_width, frame_height))

    with open(csv_output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Timestamp_sec', 'Predicted Label', 'Probability'])

        frame_idx = 0
        from tqdm import tqdm

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with open(csv_output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Frame', 'Timestamp_sec', 'Predicted Label', 'Probability'])

            for frame_idx in tqdm(range(total_frames), desc="Running Inference", unit="frame"):
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_sec = frame_idx / fps
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.sigmoid(output).item()
                    pred_label = 1 if prob > 0.5 else 0
                    label_text = 'Good' if pred_label == 1 else 'Bad'

                display_text = f"{label_text} ({prob:.2f})"
                color = (0, 255, 0) if pred_label == 1 else (0, 0, 255)
                cv2.putText(frame, display_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                writer.writerow([frame_idx, f"{timestamp_sec:.2f}", label_text, f"{prob:.4f}"])
                out.write(frame)

                if generate_heatmap:
                    heatmap = gradcam.generate(input_tensor, class_idx=0)
                    heatmap = cv2.resize(heatmap, (frame_width, frame_height))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
                    cv2.putText(overlay, display_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    out_heatmap.write(overlay)

            if generate_heatmap:
                heatmap = gradcam.generate(input_tensor, class_idx=0)
                heatmap = cv2.resize(heatmap, (frame_width, frame_height))
                heatmap = np.uint8(255 * heatmap)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
                cv2.putText(overlay, display_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                out_heatmap.write(overlay)

            frame_idx += 1

    cap.release()
    out.release()
    if out_heatmap:
        out_heatmap.release()
        gradcam.remove_hooks()
    
    import datetime
    return {
        "output_video": f"uploads/{base_name}_inference.mp4",
        "csv_output": f"uploads/{base_name}_predictions.csv",
        "heatmap_video": f"uploads/{base_name}_heatmap.mp4" if generate_heatmap else None,
        "created_at": str(datetime.datetime.now())
    }


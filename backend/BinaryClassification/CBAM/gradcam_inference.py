import cv2
import torch
from torchvision import transforms
from PIL import Image
import csv
import os
import numpy as np
import resnet_cbam  # Your model import
from gradcam import GradCAM

# Configs
VIDEO_PATH = './media/myvideo_480p_h264.mp4'  # Change this
OUTPUT_VIDEO_PATH = './media/output.mp4'
HEATMAP_VIDEO_PATH = './media/output_heatmap.mp4'
CSV_OUTPUT_PATH = './results/video_frame_predictions.csv'
MODEL_PATH = './weights/model_best.pth.tar'  # Change this to your model checkpoint path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HYPERPARAMETERS
FRAME_INTERVAL = 1
THRESHOLD = 0.5

# Define transforms (same as used during training)
transform = transforms.Compose([
    transforms.Resize((512, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Load model
model = resnet_cbam.resnet101_cbam(pretrained=False)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(DEVICE)
model.eval()

# Identify the target layer for Grad-CAM (usually last conv layer)
# target_layer = model.layer4[-1].conv2
target_layer = model.layer4[-1].conv3

gradcam = GradCAM(model, target_layer)

# Open video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error opening video file {VIDEO_PATH}")
    exit()

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps}, Total frames: {frame_count}")

# Define video writers for output videos
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
out_heatmap = cv2.VideoWriter(HEATMAP_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

# Prepare CSV file
os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
csv_file = open(CSV_OUTPUT_PATH, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Timestamp_sec', 'Predicted Label', 'Probability'])

frame_idx = 0
frame_interval = 1 if FRAME_INTERVAL is None else FRAME_INTERVAL  # process every frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp_sec = frame_idx / fps

    if frame_idx % frame_interval == 0:
        # Convert frame to PIL RGB image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Apply transforms
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        # Predict & generate heatmap
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            pred_label = 1 if prob > THRESHOLD else 0
            label_text = 'Good' if pred_label == 1 else 'Bad'

        # Generate Grad-CAM heatmap (requires gradients, so no torch.no_grad here)
        heatmap = gradcam.generate(input_tensor, class_idx=0) # only 1 neuron at model end, thats why not using pred_label

        # Resize heatmap to frame size
        heatmap = cv2.resize(heatmap, (frame_width, frame_height))

        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay heatmap on original frame
        overlayed_frame = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

        # Write prediction text on original frame
        display_text = f"{label_text} ({prob:.2f})"
        color = (0, 255, 0) if pred_label == 1 else (0, 0, 255)  # Green or Red
        cv2.putText(frame, display_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(overlayed_frame, display_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Write CSV entry
        csv_writer.writerow([frame_idx, f"{timestamp_sec:.2f}", label_text, f"{prob:.4f}"])

        # Write frames
        out.write(frame)
        out_heatmap.write(overlayed_frame)

    else:
        # For frames without processing, write original frames to both videos
        out.write(frame)
        out_heatmap.write(frame)

    frame_idx += 1

# Clean up
cap.release()
out.release()
out_heatmap.release()
csv_file.close()
gradcam.remove_hooks()

print(f"Processing complete.")
print(f"Output video saved to {OUTPUT_VIDEO_PATH}")
print(f"Heatmap overlay video saved to {HEATMAP_VIDEO_PATH}")
print(f"Predictions saved to {CSV_OUTPUT_PATH}")

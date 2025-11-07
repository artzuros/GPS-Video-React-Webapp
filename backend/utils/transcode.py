import subprocess
from pathlib import Path
import tempfile
import os

def transcode_to_h264(input_path: str) -> str:
    """
    Transcodes a video to H.264 (.mp4) format and overwrites the input file.
    This ensures consistent H.264 encoding across all uploads and inference outputs
    without creating redundant '_h264' suffixed files.
    Returns the final (same) path after successful transcoding.
    """
    input_path = Path(input_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    # Create a temp file in the same directory
    tmp_file = Path(tempfile.mktemp(dir=input_path.parent, suffix=".mp4"))

    print(f"🎬 Transcoding {input_path.name} → (overwrite in place)")

    cmd = [
        "ffmpeg", "-y",                # Overwrite if temp file exists
        "-i", str(input_path),
        "-c:v", "libx264",             # H.264 codec
        "-preset", "fast",             # Speed-quality tradeoff
        "-crf", "23",                  # Quality (lower = better)
        "-c:a", "aac",                 # Audio codec
        "-movflags", "+faststart",     # Enable faster web playback
        str(tmp_file)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.replace(tmp_file, input_path)  # Atomically overwrite original
        print(f"✅ Transcoding complete: {input_path.name}")
        return str(input_path)

    except FileNotFoundError:
        print("❌ FFmpeg not found. Please install ffmpeg and ensure it's in PATH.")
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)
        return str(input_path)

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error during transcoding: {e}")
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)
        return str(input_path)

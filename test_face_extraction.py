import os
import ffmpeg
import cv2
import numpy as np
from PIL import Image

import comfy.utils
import comfy.sd
import folder_paths


class VideoFrameExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "timestamp_sec": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "step": 0.1
                }),
                "auto_select_best_frame": ("BOOL", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT")
    RETURN_NAMES = ("frame", "metadata")
    FUNCTION = "extract_frame"
    CATEGORY = "Video Tools"

    def extract_frame(self, video_path, timestamp_sec, auto_select_best_frame):

        if not os.path.exists(video_path):
            raise Exception(f"Video file not found: {video_path}")

        # ---------------------------------------------------------
        # 1) Extract frame using OpenCV
        # ---------------------------------------------------------
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise Exception("Could not open video file.")

        if auto_select_best_frame:
            # Simple heuristic: pick the frame at 20% of video duration
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_index = int(total_frames * 0.20)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            # Use timestamp
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_index = int(timestamp_sec * fps)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = video.read()
        video.release()

        if not success:
            raise Exception("Failed to extract frame at desired timestamp.")

        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to ComfyUI image format (tensor)
        pil_img = Image.fromarray(frame_rgb)
        image_tensor = np.array(pil_img).astype(np.float32) / 255.0
        image_tensor = np.expand_dims(image_tensor, axis=0)

        # Metadata
        metadata = {
            "video_path": video_path,
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
        }

        return (image_tensor, metadata)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VideoFrameExtractor": VideoFrameExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameExtractor": "Video Frame Extractor"
}
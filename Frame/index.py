import os
import cv2
import subprocess
import numpy as np
from insightface.app import FaceAnalysis

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
VIDEO_PATH = "input.mp4"
FRAMES_DIR = "frames_raw"
BEST_DIR = "frames_best"
FPS_TO_EXTRACT = 1     # extract 1 frame per second
TOP_K = 10             # number of best frames to keep
MIN_FACE_SIZE = 120    # skip frames where face is too small
# ----------------------------------------------------------

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

# ----------------------------------------------------------
# 1. Extract frames using ffmpeg
# ----------------------------------------------------------
print("Extracting frames with ffmpeg...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", VIDEO_PATH,
    "-vf", f"fps={FPS_TO_EXTRACT}",
    f"{FRAMES_DIR}/frame_%06d.png"
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

frame_paths = sorted([os.path.join(FRAMES_DIR, f) for f in os.listdir(FRAMES_DIR)])
print(f"Extracted {len(frame_paths)} frames.")

# ----------------------------------------------------------
# 2. Initialize InsightFace model for detection & landmarks
# ----------------------------------------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

def variance_of_laplacian(image):
    """Sharpness score (higher = sharper)."""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def frontal_score(landmarks):
    """
    Simple frontalness score:
    distance between left/right eyes vs jaw width.
    More equal = more frontal.
    """
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    jaw_left  = landmarks[3]
    jaw_right = landmarks[4]

    eye_diff = abs(left_eye[1] - right_eye[1])
    jaw_diff = abs(jaw_left[1] - jaw_right[1])
    return 1.0 / (1 + eye_diff + jaw_diff)

# ----------------------------------------------------------
# 3. Score frames
# ----------------------------------------------------------
results = []

print("Scoring frames...")
for fp in frame_paths:
    img = cv2.imread(fp)
    if img is None:
        continue

    faces = app.get(img)
    if len(faces) == 0:
        continue  # no face

    # pick the largest face
    faces = sorted(faces, key=lambda f: f.bbox[2] - f.bbox[0], reverse=True)
    face = faces[0]

    # skip tiny faces
    face_width = face.bbox[2] - face.bbox[0]
    if face_width < MIN_FACE_SIZE:
        continue

    # scores
    sharp = variance_of_laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    front = frontal_score(face.landmark_2d_106) if hasattr(face, "landmark_2d_106") else 1.0
    size_score = face_width

    final = sharp * 0.6 + front * 0.2 + size_score * 0.2

    results.append((final, fp, sharp, front, face_width))

# ----------------------------------------------------------
# 4. Keep the top K best frames
# ----------------------------------------------------------
results = sorted(results, reverse=True, key=lambda x: x[0])
best = results[:TOP_K]

print("\nTop frames selected:")
for score, fp, sharp, front, fw in best:
    print(f"{fp}  | score={score:.2f} | sharp={sharp:.2f} | frontal={front:.2f} | face={fw}")

# Save them
for i, (_, fp, _, _, _) in enumerate(best):
    out = os.path.join(BEST_DIR, f"best_{i:02d}.png")
    img = cv2.imread(fp)
    cv2.imwrite(out, img)

print(f"\nSaved {len(best)} best frames to {BEST_DIR}/")
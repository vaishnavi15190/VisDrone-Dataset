import os
import time
import cv2
import pandas as pd
from ultralytics import YOLO

# ==========================
# ABSOLUTE PATHS (YOUR PATHS)
# ==========================
INPUT_DIR = "data/raw_images/final_images"          
OUTPUT_DIR = "outputs/rfdetr_raw/yolo26"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# MODEL WEIGHTS
# ==========================
# If yolo26n.pt exists in your folder, use it.
# Otherwise keep yolov8n.pt (guaranteed to work).
MODEL_WEIGHTS = "yolov8n.pt"
# MODEL_WEIGHTS = "yolo26n.pt"   # use only if you actually have it

model = YOLO(MODEL_WEIGHTS)

results_data = []

for img_name in os.listdir(INPUT_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, img_name)

    start = time.time()
    pred = model(img_path, conf=0.25)
    end = time.time()

    t_ms = (end - start) * 1000

    annotated = pred[0].plot()
    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, annotated)

    results_data.append({
        "image": img_name,
        "detections": len(pred[0].boxes),
        "inference_time_ms": round(t_ms, 2)
    })

    print(f"{img_name} | detections: {len(pred[0].boxes)} | {round(t_ms,2)} ms")

# Save timing CSV in output folder
df = pd.DataFrame(results_data)
df.to_csv(os.path.join(OUTPUT_DIR, "yolo26_timing.csv"), index=False)

print("\n✅ Done. Saved boxed images + yolo26_timing.csv")

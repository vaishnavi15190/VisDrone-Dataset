import os
import time
import csv
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

# ==========================
# PATH SETTINGS (ABSOLUTE PATHS - FIXED)
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = "data/raw_images/final_images"          
OUTPUT_DIR = "outputs/rfdetr_raw/sam_output"

# ==========================
# ROBOFLOW SETTINGS
# ==========================
API_URL = "https://serverless.roboflow.com"
API_KEY = "N4uzZ7XWRds5C4zj6I70"
MODEL_ID = "sam-yssdo/2"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# --------------------------
# DRAW SEGMENTATION MASKS
# --------------------------
def draw_masks(image, predictions):
    
    overlay = image.copy()

    for p in predictions:

        if "points" not in p:
            continue

        # Convert Roboflow dict format → numpy polygon
        polygon = np.array(
            [[pt["x"], pt["y"]] for pt in p["points"]],
            dtype=np.int32
        )

        color = np.random.randint(0, 255, 3).tolist()

        cv2.fillPoly(overlay, [polygon], color)

        cls = p.get("class", "obj")
        conf = float(p.get("confidence", 0.0))

        x, y = polygon[0]

        cv2.putText(
            overlay,
            f"{cls} {conf:.2f}",
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    alpha = 0.7
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)



def main():

    ensure_dir(INPUT_DIR)     # Auto create if missing
    ensure_dir(OUTPUT_DIR)

    print("INPUT_DIR :", INPUT_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(image_files) == 0:
        print("\n❌ No images found inside input_raw1 folder.")
        print("👉 Please add images to:", INPUT_DIR)
        return

    client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

    csv_path = os.path.join(OUTPUT_DIR, "timing_rfseg.csv")

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["image_name", "inference_time_ms", "num_masks", "status"])

        for img_name in image_files:

            img_path = os.path.join(INPUT_DIR, img_name)

            img = cv2.imread(img_path)

            if img is None:
                writer.writerow([img_name, "", "", "cv2_failed"])
                continue

            try:
                start = time.time()
                result = client.infer(img_path, model_id=MODEL_ID)
                end = time.time()
            except Exception as e:
                print(f"[ERROR] {img_name} → {e}")
                writer.writerow([img_name, "", "", "inference_failed"])
                continue

            inference_ms = (end - start) * 1000
            preds = result.get("predictions", [])

            img = draw_masks(img, preds)

            out_path = os.path.join(OUTPUT_DIR, img_name)
            cv2.imwrite(out_path, img)

            writer.writerow([img_name, f"{inference_ms:.2f}", len(preds), "ok"])

            print(f"[OK] {img_name} | masks={len(preds)} | {inference_ms:.2f} ms")

    print("\n✅ DONE! Output saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

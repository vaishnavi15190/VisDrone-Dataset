import os
import time
import csv
import cv2

from inference_sdk import InferenceHTTPClient

# ==========================
# CHANGE THESE LATER
# ==========================
INPUT_DIR = "data/raw_images/final_images"          # later: "data/final_images"
OUTPUT_DIR = "outputs/rfdetr_raw/rfdetr_final"      # later: "outputs/rfdetr_final"

# ==========================
# ROBOFLOW INFERENCE SETTINGS
# ==========================
API_URL = "https://serverless.roboflow.com"
API_KEY = "N4uzZ7XWRds5C4zj6I70"
MODEL_ID = "visdrone-lsbps/8"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def draw_predictions(image, predictions):
    for p in predictions:
        x = int(p["x"])
        y = int(p["y"])
        w = int(p["width"])
        h = int(p["height"])
        cls = p.get("class", "obj")
        conf = float(p.get("confidence", 0.0))

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{cls} {conf:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return image


def main():
    ensure_dir(OUTPUT_DIR)

    # Debug: print absolute paths
    print("INPUT_DIR (abs) :", os.path.abspath(INPUT_DIR))
    print("OUTPUT_DIR (abs):", os.path.abspath(OUTPUT_DIR))

    # Check input dir exists
    if not os.path.exists(INPUT_DIR):
        print("\n❌ ERROR: INPUT_DIR does not exist.")
        return

    # Collect images
    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_files.sort()

    print("\nFound images:", len(image_files))
    print("First few:", image_files[:5])

    if len(image_files) == 0:
        print("\n❌ ERROR: No images found in INPUT_DIR.")
        return

    # Setup Roboflow client
    client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

    csv_path = os.path.join(OUTPUT_DIR, "timing_rfdetr.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "inference_time_ms", "num_detections", "status"])

        for img_name in image_files:
            img_path = os.path.join(INPUT_DIR, img_name)

            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"[SKIP] Cannot read image: {img_name}")
                writer.writerow([img_name, "", "", "cv2_read_failed"])
                continue

            # Run inference safely
            try:
                start = time.time()
                result = client.infer(img_path, model_id=MODEL_ID)
                end = time.time()
            except Exception as e:
                print(f"[ERROR] Inference failed on {img_name}: {e}")
                writer.writerow([img_name, "", "", "inference_failed"])
                continue

            inference_ms = (end - start) * 1000
            preds = result.get("predictions", [])

            # Draw boxes
            img = draw_predictions(img, preds)

            # Save output
            out_path = os.path.join(OUTPUT_DIR, img_name)
            cv2.imwrite(out_path, img)

            writer.writerow([img_name, f"{inference_ms:.2f}", len(preds), "ok"])
            print(f"[OK] {img_name} | det={len(preds)} | {inference_ms:.2f} ms")

    print("\n✅ DONE!")
    print("Saved annotated images to:", OUTPUT_DIR)
    print("Saved timing CSV to:", csv_path)


if __name__ == "__main__":
    main()

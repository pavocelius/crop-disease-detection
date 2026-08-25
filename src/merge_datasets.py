import os
import shutil
import csv
import cv2

# ── Paths (relative to project root — run this script from there) ──
FIELDPLANT_DIR = "data/raw/FieldPlant"
PLANTVILLAGE_DIR = "data/raw/PlantVillage/PlantVillage"
OUT_DIR = "data/raw/merged"
ANNOTATIONS_CSV = os.path.join(FIELDPLANT_DIR, "_annotations.csv")

# ── Step 1: see what class names FieldPlant actually uses, and load all rows ──
found_classes = set()
rows = []
with open(ANNOTATIONS_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        found_classes.add(row["class"])
        rows.append(row)

print("FieldPlant classes found:", sorted(found_classes))

# ── Step 2: map FieldPlant's names to your 15 canonical PlantVillage-style names ──
# Fill this in based on the printed list above. Anything not listed here gets skipped.
CLASS_MAP = {
    # Clear, safe matches
    "Tomato healthy": "Tomato_healthy",
    "Tomato leaf mosaic virus": "Tomato__Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",

    # Deliberately left unmapped — see note below
    # "Tomato blight leaf": ambiguous, no Early vs Late distinction in FieldPlant
    # "Tomato bacterial wilt": different disease from "Bacterial_spot", not a safe match
    # "Tomato Brown Spots": too vague to confidently map to Septoria/Target Spot
    # All Cassava* and Corn* classes: no matching crop in your 15-class model
}

# ── Step 3: crop each labeled box into the merged folder ──
if CLASS_MAP:
    cropped_count = 0
    for i, row in enumerate(rows):
        fp_label = row["class"]
        canonical_label = CLASS_MAP.get(fp_label)
        if canonical_label is None:
            continue

        img_path = os.path.join(FIELDPLANT_DIR, row["filename"])
        img = cv2.imread(img_path)
        if img is None:
            continue

        xmin, ymin = int(row["xmin"]), int(row["ymin"])
        xmax, ymax = int(row["xmax"]), int(row["ymax"])
        crop = img[ymin:ymax, xmin:xmax]

        out_folder = os.path.join(OUT_DIR, canonical_label)
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f"fieldplant_{i}_{row['filename']}")
        cv2.imwrite(out_path, crop)
        cropped_count += 1

    print(f"FieldPlant images cropped and merged: {cropped_count}")
else:
    print("CLASS_MAP is empty — fill it in using the printed class list above, then run this script again.")

# ── Step 4: copy PlantVillage images in unchanged ──
if os.path.isdir(PLANTVILLAGE_DIR):
    for class_folder in os.listdir(PLANTVILLAGE_DIR):
        src = os.path.join(PLANTVILLAGE_DIR, class_folder)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(OUT_DIR, class_folder)
        os.makedirs(dst, exist_ok=True)
        for fname in os.listdir(src):
            shutil.copy(os.path.join(src, fname), os.path.join(dst, fname))
    print("PlantVillage images copied into merged folder.")
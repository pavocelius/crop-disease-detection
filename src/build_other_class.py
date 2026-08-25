import os
import csv
import cv2

# ── Builds the 16th "Other" class from FieldPlant classes that were
#    deliberately left OUT of CLASS_MAP in merge_datasets.py (all the
#    Corn/Cassava classes, plus ambiguous Tomato labels). These become
#    real negative examples: leaves that are NOT one of your 15 trained
#    disease classes. ──

FIELDPLANT_DIR = "data/raw/FieldPlant"
ANNOTATIONS_CSV = os.path.join(FIELDPLANT_DIR, "_annotations.csv")
OUT_DIR = "data/raw/merged"
OTHER_FOLDER = os.path.join(OUT_DIR, "Other")

# These are the ONLY FieldPlant classes that got mapped to real disease
# classes in merge_datasets.py — everything else becomes "Other".
MAPPED_FIELDPLANT_CLASSES = {
    "Tomato healthy",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
}

# Cap per class so no single unused class dominates "Other"
MAX_PER_CLASS = 300

os.makedirs(OTHER_FOLDER, exist_ok=True)

rows = []
with open(ANNOTATIONS_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Count how many we've taken per class so far, to respect MAX_PER_CLASS
counts = {}
copied = 0

for i, row in enumerate(rows):
    fp_label = row["class"]
    if fp_label in MAPPED_FIELDPLANT_CLASSES:
        continue  # skip — these are already used as real disease classes

    counts.setdefault(fp_label, 0)
    if counts[fp_label] >= MAX_PER_CLASS:
        continue

    img_path = os.path.join(FIELDPLANT_DIR, row["filename"])
    img = cv2.imread(img_path)
    if img is None:
        continue

    xmin, ymin = int(row["xmin"]), int(row["ymin"])
    xmax, ymax = int(row["xmax"]), int(row["ymax"])
    crop = img[ymin:ymax, xmin:xmax]

    safe_label = fp_label.replace(" ", "_")
    out_path = os.path.join(OTHER_FOLDER, f"{safe_label}_{i}_{row['filename']}")
    cv2.imwrite(out_path, crop)

    counts[fp_label] += 1
    copied += 1

print(f"'Other' class built: {copied} images copied from {len(counts)} unused FieldPlant classes.")
for label, count in sorted(counts.items()):
    print(f"  {label}: {count}")

print(f"\nNext: add ~100-200 genuinely non-leaf photos (random objects, rooms, people, etc.)")
print(f"into this same folder: {OTHER_FOLDER}")
print(f"This makes 'Other' mean BOTH 'wrong plant' AND 'not a leaf at all' — important for real-world use.")
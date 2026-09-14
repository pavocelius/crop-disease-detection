import os
import random
import shutil

# ── Caps oversized classes down to a target ceiling, moving excess
#    images to a backup folder (not deleting) so the dataset is more
#    balanced across all 16 classes before training. ──

MERGED_DIR = "data/raw/merged"
BACKUP_DIR = "data/raw/excess_backup"
CEILING = 1800  # roughly matches your largest "normal" classes (Tomato_Bacterial_spot: 2127, Tomato_Late_blight: 1909)

os.makedirs(BACKUP_DIR, exist_ok=True)

for class_name in os.listdir(MERGED_DIR):
    class_path = os.path.join(MERGED_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    files = os.listdir(class_path)
    if len(files) <= CEILING:
        continue  # already within limit, leave it alone

    random.shuffle(files)
    to_move = files[CEILING:]

    class_backup = os.path.join(BACKUP_DIR, class_name)
    os.makedirs(class_backup, exist_ok=True)

    for fname in to_move:
        shutil.move(os.path.join(class_path, fname), os.path.join(class_backup, fname))

    print(f"{class_name}: capped from {len(files)} to {CEILING} (moved {len(to_move)} to backup)")

print("\nDone. Final counts:")
for class_name in sorted(os.listdir(MERGED_DIR)):
    class_path = os.path.join(MERGED_DIR, class_name)
    if os.path.isdir(class_path):
        print(f"  {class_name}: {len(os.listdir(class_path))}")
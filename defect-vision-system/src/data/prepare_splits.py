"""
prepare_splits.py
------------------
Splits the raw per-class image folders into train/val/test directories
using a stratified split, ready for tf.keras.utils.image_dataset_from_directory.
"""
import shutil
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, get_logger

logger = get_logger("prepare_splits")
ROOT = Path(__file__).resolve().parents[2]


def main():
    config = load_config()
    raw_dir = ROOT / config["data"]["raw_dir"]
    processed_dir = ROOT / config["data"]["processed_dir"]
    classes = config["data"]["classes"]
    random_state = config["data"]["random_state"]

    # clean previous split
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    for split in ["train", "val", "test"]:
        for cls in classes:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        files = sorted((raw_dir / cls).glob("*.png"))
        train_files, temp_files = train_test_split(files, test_size=0.3, random_state=random_state)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=random_state)

        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            for f in split_files:
                shutil.copy(f, processed_dir / split_name / cls / f.name)

        logger.info(f"{cls}: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    logger.info(f"Split complete -> {processed_dir}")


if __name__ == "__main__":
    main()

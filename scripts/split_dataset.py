import json
import os

from sklearn.model_selection import train_test_split


DATA_PATH = "data/raw/alarm_data.json"
OUTPUT_DIR = "data/processed"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # First split: train and temp
    train_data, temp_data = train_test_split(
        dataset,
        test_size=(1 - TRAIN_RATIO),
        random_state=42,
        shuffle=True,
    )

    # Second split: validation and test
    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_size),
        random_state=42,
        shuffle=True,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_json(train_data, f"{OUTPUT_DIR}/train.json")
    save_json(val_data, f"{OUTPUT_DIR}/val.json")
    save_json(test_data, f"{OUTPUT_DIR}/test.json")

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")


if __name__ == "__main__":
    main()
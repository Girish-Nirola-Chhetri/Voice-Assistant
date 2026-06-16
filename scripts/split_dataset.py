import json
import os

from sklearn.model_selection import train_test_split

# Input dataset path
DATA_PATH = "data/raw/alarm_data.json"

# Folder to save split datasets
OUTPUT_DIR = "data/processed"

# Dataset split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def save_json(data, path):
    """Save data as a formatted JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    # Load the dataset
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Split into training (80%) and temporary (20%)
    train_data, temp_data = train_test_split(
        dataset,
        test_size=(1 - TRAIN_RATIO),
        random_state=42,  # Ensures reproducible splits
        shuffle=True,     # Randomize data before splitting
    )

    # Compute validation ratio within the temporary set
    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    # Split temporary data into validation and test sets
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_size),
        random_state=42,
        shuffle=True,
    )

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save split datasets
    save_json(train_data, f"{OUTPUT_DIR}/train.json")
    save_json(val_data, f"{OUTPUT_DIR}/val.json")
    save_json(test_data, f"{OUTPUT_DIR}/test.json")

    # Print split statistics
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")


# Run the script directly
if __name__ == "__main__":
    main()
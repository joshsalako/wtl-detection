import os
import sys
import json
import glob
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval_utils.config import RESULTS_DIR
from eval_utils.spatial_filter import apply_spatial_filter


def main():
    print(f"Applying spatial filter to all raw JSONs in {RESULTS_DIR}...")

    # Find all raw.json files
    raw_files = glob.glob(os.path.join(RESULTS_DIR, "*", "*_raw.json"))

    for raw_file in tqdm(raw_files, desc="Filtering predictions"):
        filtered_file = raw_file.replace("_raw.json", "_filtered.json")

        if os.path.exists(filtered_file):
            continue

        with open(raw_file, "r") as f:
            try:
                results = json.load(f)
            except Exception as e:
                print(f"Error loading {raw_file}: {e}")
                continue

        if not results:
            continue

        # apply_spatial_filter modifies the predictions in place and returns them
        filtered_results = apply_spatial_filter(results)

        with open(filtered_file, "w") as f:
            json.dump(filtered_results, f, indent=4)

    print("Done applying spatial filter.")


if __name__ == "__main__":
    main()

# dataset_json.py

import json

def load_json_dataset(path):
    """
    Loads your syllogism dataset stored in JSON file:
    [
      { "id": ..., "syllogism": "...", "validity": true/false, "plausibility": true/false }
    ]
    """
    with open(path, "r") as f:
        data = json.load(f)

    formatted = []
    for item in data:
        formatted.append({
            "id": item["id"],
            "text": item["syllogism"],
            "label": "valid" if item["validity"] else "invalid",
            "plausibility": "plausible" if item["plausibility"] else "implausible"
        })

    return formatted

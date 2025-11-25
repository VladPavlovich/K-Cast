import json

def load_json_dataset(path):
    data = json.load(open(path, "r"))
    processed = []

    for item in data:
        syll = item["syllogism"]

        # CLEAN consistent minimal prompt
        prompt = (
            f"Syllogism:\n{syll}\n\n"
            "Answer with exactly one word: valid or invalid.\n"
            "Label:"
        )

        processed.append({
            "text": prompt,
            "label": "valid" if item["validity"] else "invalid",
            "plausibility": "plausible" if item["plausibility"] else "implausible"
        })

    return processed

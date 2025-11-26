import json

def load_json_dataset(path):
    raw = json.load(open(path, "r"))

    # ---- Determine format: dict or list ----
    if isinstance(raw, dict):
        items = raw.values()   # extract the dict entries
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("JSON must be a dict or list")

    processed = []

    for item in items:

        # ---- Handle your parsed format (Premise A, Premise B, Conclusion) ----
        if "Premise A" in item and "Premise B" in item and "Conclusion" in item:
            syll = f"{item['Premise A']}. {item['Premise B']}. {item['Conclusion']}."
            validity = item["Validity"]
            plausibility = item.get("Plausibility", False)

        # ---- Handle old format (syllogism) ----
        else:
            syll = item["syllogism"]
            validity = item["validity"]
            plausibility = item.get("plausibility", False)

        prompt = (
            "Syllogism:\n"
            + syll.strip()
            + "\n\nClassify as 'valid' or 'invalid':"
        )

        processed.append({
            "text": prompt,
            "label": "valid" if validity else "invalid",
            "plausibility": "plausible" if plausibility else "implausible"
        })

    return processed

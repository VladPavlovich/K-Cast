import json

def parse_uuid_dataset(filepath):
    """
    Parses the dataset whether it is a Dictionary (UUID keys) or a List.
    Returns: [{"text": "...", "label": "valid"}, ...]
    """
    try:
        with open(filepath, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file {filepath}")
        return []

    parsed_data = []

    # 1. Normalize data to a list of items
    if isinstance(raw_data, dict):
        items_list = raw_data.values()
    elif isinstance(raw_data, list):
        items_list = raw_data
    else:
        print(f"Error: Unknown JSON format (Type: {type(raw_data)})")
        return []

    # 2. Iterate and Format
    for content in items_list:
        
        # KEY CHECK: Try multiple common key formats for Premises
        p1 = content.get("Premise A") or content.get("Premise 1") or content.get("premise1")
        p2 = content.get("Premise B") or content.get("Premise 2") or content.get("premise2")
        conc = content.get("Conclusion") or content.get("conclusion")
        
        if not p1 or not p2:
            continue # Skip malformed entries

        prompt = (
            f"Premise 1: {p1}\n"
            f"Premise 2: {p2}\n"
            f"Conclusion: {conc}\n\n"
            f"Evaluate if the conclusion logically follows from the premises.\n"
            f"The argument is"
        )

        # LABEL CLEANING logic
        label_raw = content.get("Validity", content.get("label", "invalid"))
        
        # Convert everything to string first, then lowercase
        label_str = str(label_raw).lower()
        
        # Map synonyms to "valid" or "invalid"
        if label_str in ["true", "valid", "1"]:
            final_label = "valid"
        else:
            final_label = "invalid"

        parsed_data.append({
            "text": prompt,
            "label": final_label
        })

    print(f"Loaded {len(parsed_data)} examples from {filepath}")
    
    # DEBUG PRINT: Show distribution
    v_count = sum(1 for d in parsed_data if d["label"] == "valid")
    i_count = sum(1 for d in parsed_data if d["label"] == "invalid")
    print(f"  > Valid: {v_count}, Invalid: {i_count}")
    
    return parsed_data
import torch
from sklearn.metrics import accuracy_score
from collections import defaultdict

def predict_label_from_output(text):
    """
    Very simple parser: you can replace with regex or more robust version.
    Assumes model outputs something like:
        "Yes, it is valid." or "No, this is invalid."
    """
    t = text.lower()
    if "valid" in t and "invalid" not in t:
        return "valid"
    if "invalid" in t:
        return "invalid"
    # fallback: assume invalid
    return "invalid"


def evaluate_model(data, run_fn):
    """
    data: list of {text, label, plausibility}
    run_fn: function that returns model-generated text for the input
            (e.g. run_kcast(prompt))
    """
    gold = []
    pred = []

    buckets = defaultdict(lambda: {"gold": [], "pred": []})

    for item in data:
        text = item["text"]
        gold_label = item["label"]
        plaus = item.get("plausibility", "unknown")

        output = run_fn(text)
        pred_label = predict_label_from_output(output)

        gold.append(gold_label)
        pred.append(pred_label)

        # Record for CE buckets
        key = f"{gold_label}_{plaus}"
        buckets[key]["gold"].append(gold_label)
        buckets[key]["pred"].append(pred_label)

    overall_accuracy = accuracy_score(gold, pred)

    # Compute CE buckets
    def bucket_acc(name):
        if name not in buckets or len(buckets[name]["gold"]) == 0:
            return 0.0
        return accuracy_score(
            buckets[name]["gold"],
            buckets[name]["pred"]
        )

    Vp = bucket_acc("valid_plausible")
    Vi = bucket_acc("valid_implausible")
    Ip = bucket_acc("invalid_plausible")
    Ii = bucket_acc("invalid_implausible")

    CE = ((Vp - Vi) + (Ii - Ip)) / 2.0

    # Intra-plausibility effects
    CE_plausible = (Vp - Ip)
    CE_implausible = (Ii - Vi)

    return {
        "accuracy": overall_accuracy,
        "CE": CE,
        "CE_plausible": CE_plausible,
        "CE_implausible": CE_implausible,
        "bucket_stats": {
            "Vp": Vp,
            "Vi": Vi,
            "Ip": Ip,
            "Ii": Ii
        }
    }

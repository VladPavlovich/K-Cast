from sklearn.metrics import accuracy_score
from collections import defaultdict


def predict_label_from_output(output_text):
    """
    Very simple heuristic: looks for 'valid' or 'invalid' in output.
    Replace with regex if needed.
    """
    t = output_text.lower()

    if "invalid" in t:
        return "invalid"
    if "valid" in t:
        return "valid"
    return "invalid"  # fallback


def evaluate_model(dataset, run_fn):
    """
    dataset: list of {text, label, plausibility}
    run_fn: function that runs the model and returns a string output
    """
    gold, pred = [], []
    buckets = defaultdict(lambda: {"gold": [], "pred": []})

    for item in dataset:
        text = item["text"]
        label = item["label"]
        plaus = item["plausibility"]

        output = run_fn(text)
        pred_label = predict_label_from_output(output)

        gold.append(label)
        pred.append(pred_label)

        key = f"{label}_{plaus}"
        buckets[key]["gold"].append(label)
        buckets[key]["pred"].append(pred_label)

    # Overall accuracy
    accuracy = accuracy_score(gold, pred)

    # Bucket accuracies
    def bucket_acc(name):
        if name not in buckets or len(buckets[name]["gold"]) == 0:
            return 0.0
        return accuracy_score(buckets[name]["gold"], buckets[name]["pred"])

    Vp = bucket_acc("valid_plausible")
    Vi = bucket_acc("valid_implausible")
    Ip = bucket_acc("invalid_plausible")
    Ii = bucket_acc("invalid_implausible")

    CE = ((Vp - Vi) + (Ii - Ip)) / 2.0
    CE_plausible = (Vp - Ip)
    CE_implausible = (Ii - Vi)

    return {
        "accuracy": accuracy,
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

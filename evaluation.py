from sklearn.metrics import accuracy_score
from collections import defaultdict
import textwrap


def evaluate_model(dataset, run_fn):
    gold, pred = [], []
    buckets = defaultdict(lambda: {"gold": [], "pred": []})

    # print("\n==================== BEGIN DETAILED EVALUATION ====================\n")

    for i, item in enumerate(dataset, start=1):
        text = item["text"]
        label = item["label"]
        plaus = item["plausibility"]

        # ----- COMMENTED OUT to reduce verbosity -----
        # print(f"[{i}/{len(dataset)}] Running model on example...")
        # print(f"Gold label: {label}, Plausibility: {plaus}")
        # print(f"Prompt:\n{textwrap.indent(text, '  ')}\n")

        output = run_fn(text).strip()

        # print(f"--- MODEL RAW OUTPUT ---\n{output}\n")

        predicted = output.lower().strip()
        if predicted not in ["valid", "invalid"]:
            # print("WARNING: model returned something unexpected → forcing 'invalid'")
            predicted = "invalid"

        # print(f"Predicted label: {predicted}")
        # print("----------------------------------------------------------------------\n")
        # --------------------------------------------------------------

        gold.append(label)
        pred.append(predicted)

        key = f"{label}_{plaus}"
        buckets[key]["gold"].append(label)
        buckets[key]["pred"].append(predicted)

    # print("===================== END DETAILED EVALUATION =====================\n")

    # -----------------------------
    # Compute metrics
    # -----------------------------
    accuracy = accuracy_score(gold, pred)

    def bucket_acc(name):
        if len(buckets[name]["gold"]) == 0:
            return 0.0
        return accuracy_score(buckets[name]["gold"], buckets[name]["pred"])

    Vp = bucket_acc("valid_plausible")
    Vi = bucket_acc("valid_implausible")
    Ip = bucket_acc("invalid_plausible")
    Ii = bucket_acc("invalid_implausible")

    CE = ((Vp - Vi) + (Ii - Ip)) / 2
    CE_plausible = Vp - Ip
    CE_implausible = Ii - Vi

    # Summary output (kept)
    print("============================== SUMMARY ==============================")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"CE: {CE:.4f}")
    print(f"CE (plausible): {CE_plausible:.4f}")
    print(f"CE (implausible): {CE_implausible:.4f}")
    print("============================== END SUMMARY ===========================\n")

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

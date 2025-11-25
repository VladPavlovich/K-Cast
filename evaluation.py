from sklearn.metrics import accuracy_score
from collections import defaultdict

def evaluate_model(dataset, run_fn):
    gold, pred = [], []
    buckets = defaultdict(lambda: {"gold": [], "pred": []})

    for item in dataset:
        text = item["text"]
        label = item["label"]
        plaus = item["plausibility"]

        out = run_fn(text).strip().lower()

        if out not in ["valid", "invalid"]:
            out = "invalid"

        gold.append(label)
        pred.append(out)

        key = f"{label}_{plaus}"
        buckets[key]["gold"].append(label)
        buckets[key]["pred"].append(out)

    accuracy = accuracy_score(gold, pred)

    def bucket_acc(name):
        return 0.0 if len(buckets[name]["gold"]) == 0 else accuracy_score(
            buckets[name]["gold"], buckets[name]["pred"]
        )

    Vp = bucket_acc("valid_plausible")
    Vi = bucket_acc("valid_implausible")
    Ip = bucket_acc("invalid_plausible")
    Ii = bucket_acc("invalid_implausible")

    CE = ((Vp - Vi) + (Ii - Ip)) / 2
    CE_plausible = Vp - Ip
    CE_implausible = Ii - Vi

    print("============================== SUMMARY ==============================")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"CE: {CE:.4f}")
    print(f"CE (plausible): {CE_plausible:.4f}")
    print(f"CE (implausible): {CE_implausible:.4f}")
    print("====================================================================")

    return {
        "accuracy": accuracy,
        "CE": CE,
        "CE_plausible": CE_plausible,
        "CE_implausible": CE_implausible,
        "bucket_stats": {"Vp": Vp, "Vi": Vi, "Ip": Ip, "Ii": Ii}
    }

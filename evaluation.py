from sklearn.metrics import accuracy_score

def evaluate_model(dataset, run_fn):
    gold, pred = [], []

    for item in dataset:
        text = item["text"]
        label = item["label"]

        out = run_fn(text).strip().lower()

        if out not in ["valid", "invalid"]:
            out = "invalid"   # fallback

        gold.append(label)
        pred.append(out)

    # Compute accuracy
    accuracy = accuracy_score(gold, pred)

    print("============================== SUMMARY ==============================")
    print(f"Accuracy: {accuracy:.4f}")
    print("   (No CE metrics because 'plausibility' is not in this dataset)")
    print("====================================================================")

    return {
        "accuracy": accuracy
    }

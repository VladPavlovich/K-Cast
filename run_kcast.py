from dataset import generate_dataset
from activations import ActivationExtractor
from kcast import KCASTDatastore, KCASTSteerer
import torch

MODEL = "meta-llama/Llama-3.2-1B"
extractor = ActivationExtractor(MODEL)

print("Generating dataset...")
data = generate_dataset(500)

# -------- BUILD TRAINING DATASTORE --------
datastore = KCASTDatastore()

print("Extracting training activations...")
for d in data[:300]:    # sample
    phi = extractor.phi(d["text"])
    datastore.add(phi, d["label"])

# -------- COMPUTE STEERING VECTOR (pos - neg) --------
print("Computing steering vector...")
valid_phis = [extractor.phi(d["text"]) for d in data if d["label"]=="valid"][:100]
invalid_phis = [extractor.phi(d["text"]) for d in data if d["label"]=="invalid"][:100]

mu_valid = torch.stack(valid_phis).mean(0)
mu_invalid = torch.stack(invalid_phis).mean(0)

delta = mu_valid - mu_invalid
torch.save(delta, "kcast_delta.pt")

# -------- RUN K-CAST INFERENCE --------
test_prompt = "All dogs are mammals. All mammals are animals. Therefore, all dogs are animals. Is this valid?"

layer = extractor.model.model.layers[extractor.layer_idx]
hook = layer.register_forward_hook(KCASTSteerer(delta, datastore, alpha=1.0))

inputs = extractor.tokenizer(test_prompt, return_tensors="pt").to(extractor.device)
with torch.no_grad():
    output_ids = extractor.model.generate(**inputs, max_new_tokens=50)

hook.remove()

print("K-CAST OUTPUT:")
print(extractor.tokenizer.decode(output_ids[0], skip_special_tokens=True))
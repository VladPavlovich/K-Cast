import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ActivationExtractor:
    def __init__(self, model_name, layer_idx=-7):
        """
        Loads HF model + tokenizer and registers a forward hook
        to capture hidden activations at the selected layer.
        """
        print(f"Loading model {model_name}...")
        print("number of layers:", layer_idx)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.layer_idx = layer_idx
        self.captured = {}

        # Register activation hook
        layer = self.model.model.layers[self.layer_idx]
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        """
        Stores activations from the chosen layer.
        output shape: (batch, seq_len, hidden_dim)
        """
        self.captured["acts"] = output.detach()

    def phi(self, text):
        """
        Returns the activation vector φ(x) for the LAST token.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs)

        acts = self.captured["acts"][:, -1, :]
        return acts.squeeze().cpu()

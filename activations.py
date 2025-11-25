import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ActivationExtractor:
    """
    Loads model and returns φ(x) from chosen layer index.
    """

    def __init__(self, model_name, layer_idx=-5):
        print(f"Loading model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.layer_idx = layer_idx
        self.captured = {}

        # Register hook to capture last token hidden state
        layer = self.model.model.layers[self.layer_idx]
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.captured["acts"] = output.detach()

    def phi(self, text):
        """
        Returns activation for last token: φ(x) ∈ R^d
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            _ = self.model(**inputs)

        acts = self.captured["acts"][:, -1, :]  # last token
        return acts.squeeze().cpu()

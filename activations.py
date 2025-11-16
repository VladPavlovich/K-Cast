import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ActivationExtractor:
    def __init__(self, model_name, layer_idx=-4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=torch.float16 if self.device=="cuda" else torch.float32
        ).to(self.device)

        self.layer_idx = layer_idx
        self.captured = {}

        layer = self.model.model.layers[self.layer_idx]
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        # output: (batch, seq, hidden)
        self.captured["acts"] = output.detach()

    def phi(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs)
        vec = self.captured["acts"][:, -1, :]
        return vec.squeeze().cpu()

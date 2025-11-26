import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ActivationExtractor:
    def __init__(self, model_name, layer_idx, device="cuda"):
        print(f"Loading model: {model_name}")
        self.device = device
        self.layer_idx = layer_idx
        
        # padding_side="left" is critical for extracting the last token correctly
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )

    def get_phi(self, text):
        """
        Runs forward pass and extracts activation from the specific layer
        at the last token position.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Force output_hidden_states=True here to ignore config warnings
            out = self.model(**inputs, output_hidden_states=True)
        
        # out.hidden_states is a tuple of (layer_0, layer_1, ... layer_N)
        layer_acts = out.hidden_states[self.layer_idx]
        
        # Grab last token: [Batch, Seq, Dim] -> [Dim]
        # We detach it to save memory since we only need the vector
        return layer_acts[0, -1, :].detach()
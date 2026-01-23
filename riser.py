import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random

class RiserRouter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Using a deeper bottleneck with GELU for better non-linear mapping
        self.net = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Softplus ensures the router only ever adds 'positive' steering 
        # towards the target concept.
        return F.softplus(self.net(x))

class RiserSteerer:
    def __init__(self, delta_base, router, global_alpha=1.0):
        # IMPORTANT: Normalize delta. This makes 'alpha' represent 
        # standard deviations/fixed units in latent space.
        self.delta_unit = delta_base / (torch.norm(delta_base) + 1e-6)
        self.router = router
        self.global_alpha = global_alpha

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            is_tuple = True
        else:
            hidden_states = output
            is_tuple = False

        # Target the last token only (standard for classification tasks)
        last_token_vec = hidden_states[:, -1, :]
        
        # Cast to router precision
        router_dtype = next(self.router.parameters()).dtype
        with torch.no_grad():
            # dynamic_scale: [batch, 1]
            dynamic_scale = self.router(last_token_vec.to(router_dtype))
        
        # Sync types and devices
        dynamic_scale = dynamic_scale.to(hidden_states.dtype)
        delta = self.delta_unit.to(hidden_states.device).to(hidden_states.dtype)
        
        # Calculate update: Scale * Alpha * Normalized Direction
        update = (self.global_alpha * dynamic_scale) * delta
        
        # Apply the steer
        new_hidden = hidden_states.clone()
        new_hidden[:, -1, :] += update

        if is_tuple:
            return (new_hidden,) + output[1:]
        return new_hidden

def train_riser(extractor, delta_c, train_data, epochs=8, lr=5e-5):
    """
    Trains the router to distinguish when the model is 'failing' 
    and needs steering intervention.
    """
    d_model = delta_c.shape[0]
    router = RiserRouter(d_model).to(extractor.device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=0.01)
    
    print(f"\n--- Training RISER: Dynamic Intensity Mapping ---")
    router.train()
    
    for epoch in range(epochs):
        total_loss = 0
        # Shuffle training data each epoch to prevent ordering bias
        random.shuffle(train_data)
        
        for item in tqdm(train_data, desc=f"Epoch {epoch+1}"):
            # 1. Get hidden state
            phi = extractor.get_phi(item["text"]).unsqueeze(0).to(torch.float32)
            
            # 2. Predict steering necessity
            predicted_intensity = router(phi)
            
            # 3. Define target: 
            # If the model is wrong (invalid), we want HIGH intensity (e.g., 2.0-5.0)
            # If the model is right (valid), we want LOW intensity (0.0)
            target = torch.tensor([[4.0 if item["label"] == "invalid" else 0.0]], 
                                 device=extractor.device)
            
            loss = F.mse_loss(predicted_intensity, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")
    
    router.eval()
    return router
import torch
import math
import torch.nn.functional as F

class ToDoPatchModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "ratio": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, ratio):
        if ratio == 1:
            return (model, )
        
        ratio_sqr = math.sqrt(ratio)

        @torch.inference_mode() # idk if this helps
        def D(x):
            # x is of shape [batch_size, num_keys, key_dim]

            x = x.permute(0, 2, 1)  # Change to [batch_size, key_dim, num_keys]
            x = F.interpolate(x, scale_factor=ratio, mode='nearest')
            x = x.permute(0, 2, 1)  # Change back

            return x
        
        @torch.inference_mode()
        def todo_patch(q, k, v, extra_options):
            # Compute scaled dot-product of Q and D(K)^T
            q = torch.matmul(q, D(k).transpose(-2, -1)) / ratio_sqr

            # Apply softmax to scaled dot-product
            q = F.softmax(q, dim=-1)

            # Multiply attention scores with D(V)
            q = torch.matmul(q, D(v))

            return q, k, v
        
        m = model.clone()
        m.set_model_attn1_patch(todo_patch)
        return (m, )
    
NODE_CLASS_MAPPINGS = {
    "ToDoPatchModel": ToDoPatchModel,
}

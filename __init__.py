import torch
import math
import torch.nn.functional as F

class ToDoPatchModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "scale_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10, "step": 0.01}),
            "sqrt_scale_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10000000, "step": 0.01}),
            "resize_mode": (['nearest', 'linear', 'area', 'nearest-exact'], {"default": "nearest"}),
            "use_downscaled_kv": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, scale_factor, sqrt_scale_factor, resize_mode, use_downscaled_kv):

        sqrt_scale_factor = math.sqrt(sqrt_scale_factor)
        
        @torch.inference_mode()
        def todo_patch(q, k, v, extra_options):
            orig_k = k
            orig_v = v
            # q and k is of shape [batch_size, num_keys, key_dim]

            # change to [batch_size, key_dim, num_keys] but keep as is to avoid redundant double transpose
            k = F.interpolate(v.transpose(-2, -1), scale_factor=scale_factor, mode=resize_mode)

            # change to [batch_size, key_dim, num_keys] and change back
            v = F.interpolate(v.transpose(-2, -1), scale_factor=scale_factor, mode=resize_mode).transpose(-2, -1)

            # compute scaled dot-product of Q and D(K)^T
            q = torch.matmul(q, k) / sqrt_scale_factor
            
            # apply softmax to scaled dot-product
            q = F.softmax(q, dim=-1)

            # multiply attention scores with D(V)
            q = torch.matmul(q, v)

            if not use_downscaled_kv:
                return q, orig_k, orig_v

            # change k back to [batch_size, num_keys, key_dim]
            k = k.transpose(-2, -1)

            return q, k, v
        
        m = model.clone()
        m.set_model_attn1_patch(todo_patch)
        return (m, )
    
NODE_CLASS_MAPPINGS = {
    "ToDoPatchModel": ToDoPatchModel,
}

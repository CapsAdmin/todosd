import torch
import math
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention_for_device

class ToDoPatchModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "downscale_factor": ("INT", {"default": 8, "min": 1, "max": 128}),
            "resize_mode": (['nearest', 'linear', 'area', 'nearest-exact'], {"default": "nearest"}),
            "downscale_input": ("BOOLEAN", {"default": True}),
            "downscale_middle": ("BOOLEAN", {"default": True}),
            "downscale_output": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, downscale_factor, resize_mode, downscale_input, downscale_middle, downscale_output):
        scale = 1/downscale_factor
        def downsample(x, scale_factor, resize_mode):
            x = x.transpose(-2, -1)
            x = F.interpolate(x, scale_factor=scale_factor, mode=resize_mode)
            x = x.transpose(-2, -1)
            return x

        @torch.inference_mode()
        def todo(q, k, v, extra_options):
            if extra_options["block"][0] == "input" and not downscale_input: return q,k,v
            if extra_options["block"][0] == "middle" and not downscale_middle: return q,k,v
            if extra_options["block"][0] == "output" and not downscale_output: return q,k,v

            k = downsample(k, scale, resize_mode)
            v = downsample(v, scale, resize_mode)

            # TODO, we still need to multiply K by "1/sqrt(dk)" in the attention function which is done later
            # it was suggested by comfyanonymous that we could just scale K by a constant, but in my experience
            # this did not work very well.
            # either I don't know what dk is or it doesn't work

            return q, k, v

        m = model.clone()
        m.set_model_attn1_patch(todo)
        return (m, )
    
NODE_CLASS_MAPPINGS = {
    "ToDoPatchModel": ToDoPatchModel,
}

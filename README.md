An attempt to implement "ToDo: Token Downsampling for Efficient Generation of High-Resolution Images" for comfyui

https://arxiv.org/abs/2402.13573

It works, but I'm not sure how to modify the internal attention function in comfyui properly to scale the "query @ key.transpose(-2, -1) * scale_factor" part in pytorch scaled_dot_product_attention (though I could pass a custom scale but then I'd need to modify comfyui)

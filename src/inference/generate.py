import torch
from typing import Optional
import torch.nn.functional as F

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50, 
                  temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0):
    """
    Generate text using the model
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
    # Generate
    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Decode the output
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_text
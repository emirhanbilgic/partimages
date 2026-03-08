import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

def test_unrolling():
    model_name = "openai/clip-vit-base-patch16"
    model = CLIPModel.from_pretrained(model_name, attn_implementation="eager")
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Dummy Image and Text
    image = Image.new("RGB", (224, 224), (255, 0, 0))
    text = "a photo of a red square"
    
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        image_embeds = model.get_image_features(pixel_values=inputs.pixel_values)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        text_embeds = model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        standard_score = (image_embeds @ text_embeds.T).item()
        
    print(f"Standard CLIP Score: {standard_score}")

    vision_model = model.vision_model
    hidden_states = outputs.vision_model_output.hidden_states 
    attentions = outputs.vision_model_output.attentions 
    
    L = len(vision_model.encoder.layers)
    num_heads = vision_model.config.num_attention_heads
    head_dim = vision_model.config.hidden_size // num_heads
    
    # The visual projection P in Hugging Face is applied after layer norm and pooler
    # image_embeds = P(LayerNorm(Z^L_cls))
    # By linearity of the projection (P is just a linear layer without bias in HF?)
    visual_projection = model.visual_projection
    
    text_emb = text_embeds[0] # [512]
    
    # We want to reconstruct LayerNorm(Z^L_cls) exactly.
    # Actually, in transformers modeling_clip.py:
    # pooled_output = hidden_states[:, 0, :]
    # pooled_output = self.post_layernorm(pooled_output)
    # image_features = self.visual_projection(pooled_output)
    
    # This post_layernorm is NOT linear. So the exact decomposition requires 
    # taking the first order Taylor expansion, OR we can just ignore it for the 
    # relative score weighting as many works do, or freeze LayerNorm parameters.
    # The paper says: "If we collect the small terms into a residual \epsilon, we obtain the compact decomposition"
    # Wait, the paper says: m_{i,l,h} = P W_O^{l,h} (alpha_{cls,i}^{l,h} v_{i}^{l,h})
    # This implies they applied P directly to the out_proj components, entirely ignoring the final layer norm!
    
    # Let's extract m_{i,l,h}
    layer_scores = 0
    all_m = []

    for l in range(L):
        layer = vision_model.encoder.layers[l]
        z_prev = hidden_states[l] # [1, 197, 768]
        
        z_norm = layer.layer_norm1(z_prev)
        v = layer.self_attn.v_proj(z_norm)
        bsz, tgt_len, embed_dim = v.size()
        v_heads = v.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2) # [1, 12, 197, 64]
        
        attn = attentions[l] # [1, 12, 197, 197]
        alpha_cls = attn[0, :, 0, :] # [12, 197]
        
        W_o = layer.self_attn.out_proj.weight # [768, 768]
        
        for h in range(num_heads):
            # [197, 64] * [197, 1] = [197, 64]
            v_h = v_heads[0, h] * alpha_cls[h].unsqueeze(-1)
            
            padded_v = torch.zeros(tgt_len, embed_dim, device=v.device, dtype=v.dtype)
            padded_v[:, h*head_dim : (h+1)*head_dim] = v_h
            
            # W_o applies to the padded v
            # [197, 768]
            msa_out = torch.matmul(padded_v, W_o.t())
            
            # P is visual_projection.weight [512, 768]
            # m_{i,l,h} = P @ msa_out^T => [197, 512]
            m = torch.matmul(msa_out, visual_projection.weight.t())
            
            # A_{i,l,h} = \langle m, M_{text} \rangle
            # [197]
            A = torch.matmul(m, text_emb)
            
            layer_scores += A.sum().item()
            all_m.append(A.detach().cpu().numpy())
            
    print(f"Sum of A_{{i,l,h}} over all layers and heads: {layer_scores}")
    print(f"This should somehow correlate or match the standard score without \epsilon")

if __name__ == '__main__':
    test_unrolling()

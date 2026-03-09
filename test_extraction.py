import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
import numpy as np

def test_unrolling():
    model_name = "ViT-B-16"
    pretrained = "laion2b_s34b_b88k"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

    # Dummy Image and Text
    image = Image.new("RGB", (224, 224), (255, 0, 0))
    text = "a photo of a red square"

    img_tensor = preprocess(image).unsqueeze(0)
    tok = tokenizer([text])

    with torch.no_grad():
        image_embeds = model.encode_image(img_tensor, normalize=True)
        text_embeds = model.encode_text(tok, normalize=True)
        standard_score = (image_embeds @ text_embeds.T).item()

    print(f"Standard CLIP Score: {standard_score}")

    visual = model.visual

    with torch.no_grad():
        # Replicate vision forward pass
        x = visual.conv1(img_tensor)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        x = torch.cat([
            visual.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1).to(x.dtype),
            x,
        ], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)

        if hasattr(visual, 'patch_dropout'):
            x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        num_layers = len(visual.transformer.resblocks)
        embed_dim = x.shape[-1]
        num_heads = visual.transformer.resblocks[0].attn.num_heads
        head_dim = embed_dim // num_heads

        proj = visual.proj  # [embed_dim, proj_dim]
        text_emb = text_embeds[0]  # [proj_dim]

        layer_scores = 0
        all_m = []

        for l, block in enumerate(visual.transformer.resblocks):
            z_norm = block.ln_1(x)
            seq_len = z_norm.shape[0]

            qkv = F.linear(z_norm, block.attn.in_proj_weight, block.attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(seq_len, -1, num_heads, head_dim).permute(1, 2, 0, 3)
            k = k.view(seq_len, -1, num_heads, head_dim).permute(1, 2, 0, 3)
            v_heads = v.view(seq_len, -1, num_heads, head_dim).permute(1, 2, 0, 3)

            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
            attn_weights = torch.softmax(attn_weights, dim=-1)

            alpha_cls = attn_weights[0, :, 0, :]  # [num_heads, seq_len]

            W_o = block.attn.out_proj.weight

            for h in range(num_heads):
                v_h = v_heads[0, h] * alpha_cls[h].unsqueeze(-1)

                padded_v = torch.zeros(seq_len, embed_dim, device=v.device, dtype=v.dtype)
                padded_v[:, h * head_dim:(h + 1) * head_dim] = v_h

                msa_out = torch.matmul(padded_v, W_o.t())
                m = torch.matmul(msa_out, proj)
                A = torch.matmul(m, text_emb)

                layer_scores += A.sum().item()
                all_m.append(A.detach().cpu().numpy())

            x = block(x)

    print(f"Sum of A_{{i,l,h}} over all layers and heads: {layer_scores}")
    print(f"This should somehow correlate or match the standard score without \\epsilon")

if __name__ == '__main__':
    test_unrolling()

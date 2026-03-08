import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndimage
import cv2
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import average_precision_score, jaccard_score
from transformers import CLIPModel, CLIPProcessor
from pycocotools.coco import COCO

def omp_sparse_residual(x_1x: torch.Tensor, D: torch.Tensor, max_atoms: int = 8, tol: float = 1e-6, return_indices: bool = False):
    """
    Simple Orthogonal Matching Pursuit to compute sparse coding residual without training.
    x_1x: [1, d], assumed L2-normalized
    D: [K, d], atom rows, L2-normalized
    Returns residual r (L2-normalized): [1, d]
    If max_atoms <= 0 or D is empty, this is a no-op and just returns the original x_1x.
    """
    if D is None or D.numel() == 0 or max_atoms is None or max_atoms <= 0:
        if return_indices:
            return F.normalize(x_1x, dim=-1), []
        return F.normalize(x_1x, dim=-1)
    # Force CPU execution for OMP loop to avoid CUDA lazy wrapper / context errors
    device = x_1x.device
    dtype = x_1x.dtype
    x = x_1x.clone().cpu().float()  # [1, d]
    D_cpu = D.cpu().float()         # [K, d]
    
    K = D_cpu.shape[0]
    max_atoms = int(min(max_atoms, K))
    selected = []
    r = x.clone()  # residual starts as x
    
    for _ in range(max_atoms):
        # correlations with residual
        c = (r @ D_cpu.t()).squeeze(0)  # [K]
        c_abs = c.abs()
        # mask already selected
        if len(selected) > 0:
            c_abs[selected] = -1.0
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        selected.append(idx)
        # Solve least squares on selected atoms: s = argmin ||x - s^T D_S||^2
        D_S = D_cpu[selected, :]  # [t, d]
        G = D_S @ D_S.t()     # [t, t]
        b = (D_S @ x.t())     # [t, 1]
        
        # Use simple solve on CPU (stable enough for small matrices)
        I = torch.eye(G.shape[0])
        # Try-catch specifically for singular matrix cases
        try:
            L = torch.linalg.cholesky(G + 1e-6 * I)
            s = torch.cholesky_solve(b, L)
        except RuntimeError:
             # Fallback to slower but robust lstsq if Cholesky fails
            s = torch.linalg.lstsq(G + 1e-6 * I, b).solution
            
        x_hat = (s.t() @ D_S)  # [1, d]
        r = (x - x_hat)
        
        # Early stop if residual very small
        if float(torch.norm(r) <= tol):
            break
            
    # Return normalized residual (fallback to x if degenerate)
    # Move back to original device/dtype
    r = r.to(device=device, dtype=dtype)
    
    final_res = F.normalize(x_1x, dim=-1) if torch.norm(r) <= tol else F.normalize(r, dim=-1)

    if return_indices:
        return final_res, selected
    return final_res

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Hyperparameters
ALPHA = 5.0
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "openai/clip-vit-base-patch16"

print(f"Loading model {MODEL_NAME} on {DEVICE}...")
try:
    model = CLIPModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(DEVICE)
except TypeError:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

def extract_chili_activations(image, text_query=None, text_emb=None):
    if text_emb is None:
        inputs = processor(text=[text_query], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
            text_embeds = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            text_emb = text_embeds[0]
            
        vision_model = model.vision_model
        hidden_states = outputs.vision_model_output.hidden_states 
        attentions = outputs.vision_model_output.attentions 
    else:
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"], output_attentions=True, output_hidden_states=True)
            
        vision_model = model.vision_model
        hidden_states = vision_outputs.hidden_states 
        attentions = vision_outputs.attentions 
    
    L = len(vision_model.encoder.layers)
    num_heads = vision_model.config.num_attention_heads
    head_dim = vision_model.config.hidden_size // num_heads
    
    visual_projection = model.visual_projection
    
    # Store A_{l,h} -> shape [L, H, 14, 14]
    # CLIP patch16 with 224x224 image has 14x14 grid
    A_lh = torch.zeros((L, num_heads, 14, 14), device=DEVICE)
    
    for l in range(L):
        layer = vision_model.encoder.layers[l]
        z_prev = hidden_states[l] 
        
        z_norm = layer.layer_norm1(z_prev)
        v = layer.self_attn.v_proj(z_norm)
        bsz, tgt_len, embed_dim = v.size()
        v_heads = v.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2) 
        
        attn = attentions[l]
        alpha_cls = attn[0, :, 0, 1:] # [12, 196] (skip CLS self-attention)
        
        W_o = layer.self_attn.out_proj.weight 
        
        for h in range(num_heads):
            # [196, 64] * [196, 1] = [196, 64]
            v_h = v_heads[0, h, 1:] * alpha_cls[h].unsqueeze(-1)
            
            padded_v = torch.zeros(tgt_len - 1, embed_dim, device=v.device, dtype=v.dtype)
            padded_v[:, h*head_dim : (h+1)*head_dim] = v_h
            
            # W_o applies to the padded v -> [196, 768]
            msa_out = torch.matmul(padded_v, W_o.t())
            
            # P -> [196, 512]
            m = torch.matmul(msa_out, visual_projection.weight.t())
            
            # A_i = <m, M_text> -> [196]
            A = torch.matmul(m, text_emb.to(DEVICE))
            
            A_lh[l, h] = A.view(14, 14)
            
    return A_lh.detach().cpu().numpy()


def get_fm(A_lh):
    # Apply median filter 3x3 on spatial dimensions for each layer and head
    fm_A_lh = np.zeros_like(A_lh)
    for l in range(A_lh.shape[0]):
        for h in range(A_lh.shape[1]):
            fm_A_lh[l, h] = ndimage.median_filter(A_lh[l, h], size=3)
    return fm_A_lh

def get_hm(fm_A_lh):
    # Threshold fm_A_lh by its mean
    hm_A_lh = np.zeros_like(fm_A_lh)
    for l in range(fm_A_lh.shape[0]):
        for h in range(fm_A_lh.shape[1]):
            mean_val = np.mean(fm_A_lh[l, h])
            hm_A_lh[l, h] = (fm_A_lh[l, h] > mean_val).astype(float)
    return hm_A_lh

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def get_image_parts(coco, img_id):
    """Get all unique parts (categories) annotated in an image."""
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    category_ids = list(set([ann['category_id'] for ann in anns]))
    categories = coco.loadCats(category_ids)
    return categories

def get_binary_mask(coco, img_id, cat_id, target_size):
    """Get binary mask for a specific category in an image."""
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)
    
    if len(anns) == 0:
        return np.zeros(target_size, dtype=np.uint8)
        
    mask = np.zeros((coco.imgs[img_id]['height'], coco.imgs[img_id]['width']), dtype=np.uint8)
    for ann in anns:
        mask = np.maximum(mask, coco.annToMask(ann))
        
    mask_img = Image.fromarray(mask * 255)
    mask_img = mask_img.resize(target_size, Image.NEAREST)
    arr = np.array(mask_img)
    return (arr > 128).astype(np.uint8)

def get_random_calibration_set(coco, ds_dir, k=50):
    img_ids = coco.getImgIds()
    np.random.seed(42)
    sample_ids = np.random.choice(img_ids, min(k, len(img_ids)), replace=False)
    
    calib_data = []
    
    for img_id in sample_ids:
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        
        if len(anns) == 0:
            continue
            
        img_path = os.path.join(ds_dir, "images", img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        # Pick one random annotation to probe
        ann = anns[np.random.randint(len(anns))]
        cat = coco.loadCats([ann['category_id']])[0]
        text_query = f"a photo of a {cat['name']}"
        
        gt_mask = coco.annToMask(ann)
        # Resize gt_mask to 14x14 to match features
        gt_mask_14 = cv2.resize(gt_mask, (14, 14), interpolation=cv2.INTER_NEAREST)
        
        calib_data.append({
            "image": image,
            "text": text_query,
            "gt_mask": gt_mask_14
        })
        
    return calib_data

def calibrate_weights(calib_data):
    print("Calibrating CHILI weights...")
    L, H = 12, 12
    iou_sums = np.zeros((L, H))
    
    for i, data in enumerate(tqdm(calib_data)):
        A_lh = extract_chili_activations(data["image"], data["text"])
        fm_A = get_fm(A_lh)
        hm_A = get_hm(fm_A)
        
        for l in range(L):
            for h in range(H):
                iou = calculate_iou(hm_A[l, h], data["gt_mask"])
                iou_sums[l, h] += iou
                
    iou_means = iou_sums / len(calib_data)
    w_lh = 1.0 - np.exp(-ALPHA * iou_means)
    return w_lh

def compute_metrics(heatmap_np, gt_mask, thr):
    import torch
    import torch.nn.functional as F
    
    H_gt, W_gt = gt_mask.shape
    H_hm, W_hm = heatmap_np.shape
    
    heatmap_tensor = torch.from_numpy(heatmap_np).float().unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(
        heatmap_tensor,
        size=(H_gt, W_gt),
        mode='bilinear',
        align_corners=False,
    ).squeeze()
    
    # Thresholding
    Res_1 = (heatmap_resized > thr).float()
    Res_0 = (heatmap_resized <= thr).float()
    output_tensor = torch.stack([Res_0, Res_1], dim=0)       # [2,H,W]
    output_AP     = torch.stack([1.0 - heatmap_resized, heatmap_resized], dim=0)

    gt_tensor = torch.from_numpy(gt_mask).long()
    
    # Implementing batch_intersection_union equivalent
    predict = output_tensor.argmax(0)
    target = gt_tensor
    
    inter = np.zeros(2)
    union = np.zeros(2)
    for c in range(2):
        predict_c = (predict == c)
        target_c = (target == c)
        intersection = (predict_c & target_c).sum().item()
        area_union = (predict_c | target_c).sum().item()
        inter[c] = intersection
        union[c] = area_union
        
    # Implementing batch_pix_accuracy equivalent
    correct_pixels = (predict == target).sum().item()
    labeled_pixels = target.numel()
    
    # Implementing get_ap_scores equivalent
    target_expand = gt_tensor.unsqueeze(0).expand_as(output_AP)
    target_expand_numpy = target_expand.data.cpu().numpy().reshape(-1)
    
    # One-hot encoding
    x = torch.zeros_like(target_expand)
    t = gt_tensor.unsqueeze(0).clamp(min=0).long()
    target_1hot = x.scatter_(0, t, 1)
    
    predict_flat = output_AP.data.cpu().numpy().reshape(-1)
    target_flat = target_1hot.data.cpu().numpy().reshape(-1)
    
    p = predict_flat[target_expand_numpy != -1]
    t_filtered = target_flat[target_expand_numpy != -1]
    
    ap = np.nan_to_num(average_precision_score(t_filtered, p))
        
    return inter, union, correct_pixels, labeled_pixels, ap

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help="Run on a small subset of N images")
    parser.add_argument('--visualize', type=int, default=0, help="Number of visualizations to save")
    parser.add_argument('--image_size', type=int, default=224, help="Fixed image size to evaluate at (default 224)")
    parser.add_argument('--max_dict_cos_sim', type=float, default=0.65)
    parser.add_argument('--max_atoms', type=int, default=3, help='Maximum number of atoms for OMP')
    args = parser.parse_args()

    ds_dir = "partimagenet_1000_subset"
    ann_file = os.path.join(ds_dir, "subset_annotations.json")
    
    print(f"Loading COCO annotations from {ann_file}...")
    coco = COCO(ann_file)
    
    calib_data = get_random_calibration_set(coco, ds_dir, k=5 if args.limit else 50)
    w_lh = calibrate_weights(calib_data)
    
    print(f"\nEvaluating on {'subset of ' + str(args.limit) + ' ' if args.limit else ''}dataset...")
    
    eval_methods = ['baseline', 'omp']
    accumulators = {
        method: {
            'total_inter': np.zeros(2),
            'total_union': np.zeros(2),
            'total_correct': 0,
            'total_labeled': 0,
            'all_aps': []
        }
        for method in eval_methods
    }
    
    img_ids = coco.getImgIds()
    if args.limit:
        img_ids = img_ids[:args.limit]
    
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        
        if len(anns) == 0:
            continue
            
        img_path = os.path.join(ds_dir, "images", img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        image = image.resize((args.image_size, args.image_size), Image.BICUBIC)
        
        categories = get_image_parts(coco, img_id)
        if len(categories) < 2:
            continue
            
        cat_names = [cat['name'] for cat in categories]
        prompts = [f"a photo of a {name}" for name in cat_names]
        
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
            
        for i, target_cat in enumerate(categories):
            target_emb = text_embs[i:i+1]
            
            # Dictionary is everything ELSE in the image
            dictionary_embs = torch.cat([text_embs[:i], text_embs[i+1:]], dim=0)
            
            if dictionary_embs.shape[0] > 0 and 0.0 < args.max_dict_cos_sim < 1.0:
                sim = (dictionary_embs @ target_emb.t()).squeeze(-1).abs()
                keep = sim < args.max_dict_cos_sim
                dictionary_embs = dictionary_embs[keep]
                
            num_negativas = dictionary_embs.shape[0]
            atoms = min(args.max_atoms, num_negativas)
            
            sparse_emb = omp_sparse_residual(target_emb, dictionary_embs, max_atoms=atoms)
            
            gt_mask = get_binary_mask(coco, img_id, target_cat['id'], (args.image_size, args.image_size))
            if gt_mask.sum() == 0:
                continue
                
            heatmaps = {}
            for method in eval_methods:
                working_emb = target_emb[0] if method == 'baseline' else sparse_emb[0]
                A_lh = extract_chili_activations(image, text_emb=working_emb)
                fm_A = get_fm(A_lh)
                A_obj_lh = np.zeros_like(fm_A)
                for l in range(12):
                    for h in range(12):
                        A_obj_lh[l, h] = w_lh[l, h] * fm_A[l, h]
                        
                A_obj = A_obj_lh.sum(axis=(0, 1))
                heatmaps[method] = A_obj
            
            for method in eval_methods:
                A_obj = heatmaps[method]
                if A_obj.max() > A_obj.min():
                    A_obj_norm = (A_obj - A_obj.min()) / (A_obj.max() - A_obj.min())
                else:
                    A_obj_norm = A_obj
                    
                try:
                    inter, union, c_p, l_p, ap = compute_metrics(A_obj_norm, gt_mask, 0.5)
                    accumulators[method]['total_inter'] += inter
                    accumulators[method]['total_union'] += union
                    accumulators[method]['total_correct'] += c_p
                    accumulators[method]['total_labeled'] += l_p
                    accumulators[method]['all_aps'].append(ap)
                except Exception as e:
                    print(f"Error computing metrics for img {img_id}: {e}")
                
                # Visualization code excluded for simplicity in OMP script
                pass
                
    print(f"\n--- Final Results (CHILI + OMP) ---")
    for method in eval_methods:
        accums = accumulators[method]
        if len(accums['all_aps']) > 0:
            iou = accums['total_inter'].astype(np.float64) / (accums['total_union'].astype(np.float64) + 1e-10)
            miou = 100.0 * iou.mean()
            acc = 100.0 * accums['total_correct'] / (accums['total_labeled'] + 1e-10)
            map_score = np.mean(accums['all_aps']) * 100 if accums['all_aps'] else 0.0

            print(f"\n{method.upper()}:")
            print(f"mIoU: {miou:.2f}")
            print(f"Pixel Accuracy: {acc:.2f}")
            print(f"mAP: {map_score:.2f}")
            print(f"Number of target parts evaluated: {len(accums['all_aps'])}")
        else:
            print(f"{method.upper()}: No valid annotations found.")

if __name__ == "__main__":
    main()

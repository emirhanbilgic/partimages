#!/usr/bin/env python3
"""
Optimize CHILI + OMP hyperparameters on PartImageNet subset using Optuna.

Tunes:
- max_atoms (integer: 1 to 5)
- max_dict_cos_sim (float: 0.5 to 1.0)
- sparse_threshold (float: 0.1 to 0.9)

Objective: Maximize combined metric on a subset of images (e.g., 50).
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndimage
import cv2
from PIL import Image
from sklearn.metrics import average_precision_score
from transformers import CLIPModel, CLIPProcessor
from pycocotools.coco import COCO
import optuna

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Re-use sparse encoding implementation
def omp_sparse_residual(x_1x: torch.Tensor, D: torch.Tensor, max_atoms: int = 8, tol: float = 1e-6, return_indices: bool = False):
    if D is None or D.numel() == 0 or max_atoms is None or max_atoms <= 0:
        if return_indices:
            return F.normalize(x_1x, dim=-1), []
        return F.normalize(x_1x, dim=-1)
    device = x_1x.device
    dtype = x_1x.dtype
    x = x_1x.clone().cpu().float()
    D_cpu = D.cpu().float()
    
    K = D_cpu.shape[0]
    max_atoms = int(min(max_atoms, K))
    selected = []
    r = x.clone()
    
    for _ in range(max_atoms):
        c = (r @ D_cpu.t()).squeeze(0)
        c_abs = c.abs()
        if len(selected) > 0:
            c_abs[selected] = -1.0
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        selected.append(idx)
        D_S = D_cpu[selected, :]
        G = D_S @ D_S.t()
        b = (D_S @ x.t())
        
        I = torch.eye(G.shape[0])
        try:
            L = torch.linalg.cholesky(G + 1e-6 * I)
            s = torch.cholesky_solve(b, L)
        except RuntimeError:
            s = torch.linalg.lstsq(G + 1e-6 * I, b).solution
            
        x_hat = (s.t() @ D_S)
        r = (x - x_hat)
        if float(torch.norm(r) <= tol):
            break
            
    r = r.to(device=device, dtype=dtype)
    final_res = F.normalize(x_1x, dim=-1) if torch.norm(r) <= tol else F.normalize(r, dim=-1)

    if return_indices:
        return final_res, selected
    return final_res

# Helper functions for dataset and evaluation
def get_image_parts(coco, img_id):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    category_ids = list(set([ann['category_id'] for ann in anns]))
    categories = coco.loadCats(category_ids)
    return categories

def get_binary_mask(coco, img_id, cat_id, target_size):
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

def compute_metrics(heatmap_np, gt_mask, thr):
    H_gt, W_gt = gt_mask.shape
    heatmap_tensor = torch.from_numpy(heatmap_np).float().unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(
        heatmap_tensor,
        size=(H_gt, W_gt),
        mode='bilinear',
        align_corners=False,
    ).squeeze()
    
    Res_1 = (heatmap_resized > thr).float()
    Res_0 = (heatmap_resized <= thr).float()
    output_tensor = torch.stack([Res_0, Res_1], dim=0)
    output_AP = torch.stack([1.0 - heatmap_resized, heatmap_resized], dim=0)
    gt_tensor = torch.from_numpy(gt_mask).long()
    
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
        
    correct_pixels = (predict == target).sum().item()
    labeled_pixels = target.numel()
    
    target_expand = gt_tensor.unsqueeze(0).expand_as(output_AP)
    target_expand_numpy = target_expand.data.cpu().numpy().reshape(-1)
    x = torch.zeros_like(target_expand)
    t = gt_tensor.unsqueeze(0).clamp(min=0).long()
    target_1hot = x.scatter_(0, t, 1)
    
    predict_flat = output_AP.data.cpu().numpy().reshape(-1)
    target_flat = target_1hot.data.cpu().numpy().reshape(-1)
    p = predict_flat[target_expand_numpy != -1]
    t_filtered = target_flat[target_expand_numpy != -1]
    
    ap = np.nan_to_num(average_precision_score(t_filtered, p))
    return inter, union, correct_pixels, labeled_pixels, ap

# Core CHILI Functions
def extract_chili_activations(model, processor, image, text_query=None, text_emb=None):
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
    
    A_lh = torch.zeros((L, num_heads, 14, 14), device=DEVICE)
    
    for l in range(L):
        layer = vision_model.encoder.layers[l]
        z_prev = hidden_states[l] 
        z_norm = layer.layer_norm1(z_prev)
        v = layer.self_attn.v_proj(z_norm)
        bsz, tgt_len, embed_dim = v.size()
        v_heads = v.view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2) 
        
        attn = attentions[l]
        alpha_cls = attn[0, :, 0, 1:] 
        W_o = layer.self_attn.out_proj.weight 
        
        for h in range(num_heads):
            v_h = v_heads[0, h, 1:] * alpha_cls[h].unsqueeze(-1)
            padded_v = torch.zeros(tgt_len - 1, embed_dim, device=v.device, dtype=v.dtype)
            padded_v[:, h*head_dim : (h+1)*head_dim] = v_h
            msa_out = torch.matmul(padded_v, W_o.t())
            m = torch.matmul(msa_out, visual_projection.weight.t())
            A = torch.matmul(m, text_emb.to(DEVICE))
            A_lh[l, h] = A.view(14, 14)
            
    return A_lh.detach().cpu().numpy()

def get_fm(A_lh):
    fm_A_lh = np.zeros_like(A_lh)
    for l in range(A_lh.shape[0]):
        for h in range(A_lh.shape[1]):
            fm_A_lh[l, h] = ndimage.median_filter(A_lh[l, h], size=3)
    return fm_A_lh

def get_hm(fm_A_lh):
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

# Calibration caching avoids repeating the costly calibration process locally
def get_random_calibration_set(coco, ds_dir, k=50):
    img_ids = coco.getImgIds()
    np.random.seed(42)
    sample_ids = np.random.choice(img_ids, min(k, len(img_ids)), replace=False)
    
    calib_data = []
    for img_id in sample_ids:
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 0: continue
            
        img_path = os.path.join(ds_dir, "images", img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        ann = anns[np.random.randint(len(anns))]
        cat = coco.loadCats([ann['category_id']])[0]
        text_query = f"a photo of a {cat['name']}"
        gt_mask = coco.annToMask(ann)
        gt_mask_14 = cv2.resize(gt_mask, (14, 14), interpolation=cv2.INTER_NEAREST)
        
        calib_data.append({"image": image, "text": text_query, "gt_mask": gt_mask_14})
    return calib_data

def calibrate_weights(model, processor, calib_data):
    L, H = 12, 12
    iou_sums = np.zeros((L, H))
    
    for i, data in enumerate(calib_data):
        A_lh = extract_chili_activations(model, processor, data["image"], data["text"])
        fm_A = get_fm(A_lh)
        hm_A = get_hm(fm_A)
        for l in range(L):
            for h in range(H):
                iou = calculate_iou(hm_A[l, h], data["gt_mask"])
                iou_sums[l, h] += iou
                
    iou_means = iou_sums / len(calib_data)
    w_lh = 1.0 - np.exp(-5.0 * iou_means)
    return w_lh

# Core evaluation logic parameterized
def evaluate_config(coco, images_dir, img_ids, model, processor, w_lh, max_atoms_cap, sparse_threshold, max_dict_cos_sim, image_size):
    total_inter = np.zeros(2)
    total_union = np.zeros(2)
    total_correct = 0
    total_labeled = 0
    all_aps = []
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
            image = image.resize((image_size, image_size), Image.BICUBIC)
            
            categories = get_image_parts(coco, img_id)
            if len(categories) < 2:
                continue
                
            cat_names = [cat['name'] for cat in categories]
            prompts = [f"a photo of a {name}." for name in cat_names]
            
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
            with torch.no_grad():
                text_embs = model.get_text_features(**text_inputs)
                text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
                
            for i, target_cat in enumerate(categories):
                target_emb = text_embs[i:i+1]
                dictionary_embs = torch.cat([text_embs[:i], text_embs[i+1:]], dim=0)
                
                if dictionary_embs.shape[0] > 0 and 0.0 < max_dict_cos_sim < 1.0:
                    sim = (dictionary_embs @ target_emb.t()).squeeze(-1).abs()
                    keep = sim < max_dict_cos_sim
                    dictionary_embs = dictionary_embs[keep]
                    
                num_negativas = dictionary_embs.shape[0]
                atoms = min(max_atoms_cap, num_negativas)
                
                sparse_emb = omp_sparse_residual(target_emb, dictionary_embs, max_atoms=atoms)
                
                gt_mask = get_binary_mask(coco, img_id, target_cat['id'], (image_size, image_size))
                if gt_mask.sum() == 0:
                    continue
                    
                A_lh = extract_chili_activations(model, processor, image, text_emb=sparse_emb[0])
                fm_A = get_fm(A_lh)
                A_obj_lh = np.zeros_like(fm_A)
                for l in range(12):
                    for h in range(12):
                        A_obj_lh[l, h] = w_lh[l, h] * fm_A[l, h]
                        
                A_obj = A_obj_lh.sum(axis=(0, 1))
                
                if A_obj.max() > A_obj.min():
                    A_obj_norm = (A_obj - A_obj.min()) / (A_obj.max() - A_obj.min())
                else:
                    A_obj_norm = A_obj
                    
                inter, union, c_p, l_p, ap = compute_metrics(A_obj_norm, gt_mask, sparse_threshold)
                total_inter += inter
                total_union += union
                total_correct += c_p
                total_labeled += l_p
                all_aps.append(ap)
                
        except Exception as e:
            # print(f"Error on image {img_id}: {e}")
            continue

    iou = total_inter.astype(np.float64) / (total_union.astype(np.float64) + 1e-10)
    miou = 100.0 * iou.mean()
    acc = 100.0 * total_correct / (total_labeled + 1e-10)
    map_score = np.mean(all_aps) * 100 if all_aps else 0.0
    
    return miou, acc, map_score


def main():
    parser = argparse.ArgumentParser(description='Optimize CHILI+OMP hyperparameters with Optuna')
    parser.add_argument('--dataset_dir', type=str, default='partimagenet_1000_subset')
    parser.add_argument('--limit', type=int, default=50, help='Number of images to evaluate per trial')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of optimization trials')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch16')
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()

    annotations_file = os.path.join(args.dataset_dir, 'subset_annotations.json')
    images_dir = os.path.join(args.dataset_dir, 'images')

    print(f"Loading model {args.model_name} on {DEVICE}...")
    try:
        model = CLIPModel.from_pretrained(args.model_name, attn_implementation="eager").to(DEVICE)
    except TypeError:
        # Fallback for older transformers versions where "eager" was the default and the argument didn't exist
        model = CLIPModel.from_pretrained(args.model_name).to(DEVICE)
        
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model.eval()

    coco = COCO(annotations_file)
    all_img_ids = coco.getImgIds()
    
    # We fix the subset of images evaluated across all trials so comparisons are apples-to-apples.
    np.random.seed(42)
    img_ids_subset = np.random.choice(all_img_ids, size=min(args.limit, len(all_img_ids)), replace=False).tolist()
    
    print(f"Loaded dataset. Optimizing on fixed subset of {len(img_ids_subset)} images.")
    
    # Run a single static CHILI calibration sequence for the model. 
    # Technically since calibration doesn't involve OMP (only isolated query text), it is static.
    calib_set = get_random_calibration_set(coco, args.dataset_dir, k=min(10, args.limit))
    w_lh = calibrate_weights(model, processor, calib_set)
    print("Static CHILI calibration applied successfully.")

    def objective(trial):
        sparse_threshold = trial.suggest_float('sparse_threshold', 0.1, 0.9, step=0.025)
        max_dict_cos_sim = trial.suggest_float('max_dict_cos_sim', 0.5, 1.0, step=0.05)
        max_atoms_cap = trial.suggest_int('max_atoms', 1, 5)
        
        miou, acc, map_score = evaluate_config(
            coco=coco,
            images_dir=images_dir,
            img_ids=img_ids_subset,
            model=model,
            processor=processor,
            w_lh=w_lh,
            max_atoms_cap=max_atoms_cap,
            sparse_threshold=sparse_threshold,
            max_dict_cos_sim=max_dict_cos_sim,
            image_size=args.image_size
        )
        
        trial.set_user_attr('miou', miou)
        trial.set_user_attr('acc', acc)
        trial.set_user_attr('map', map_score)
        
        combined_score = (miou + acc + map_score) / 3.0
        
        print(f"Trial Params -> max_atoms: {max_atoms_cap}, threshold: {sparse_threshold:.3f}, max_dict_cos_sim: {max_dict_cos_sim:.2f}")
        print(f"Scores -> mIoU: {miou:.2f}, Pixel Accuracy: {acc:.2f}, mAP: {map_score:.2f}")
        
        return combined_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("\n--- Optuna Optimization Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Average Score): {trial.value:.2f}%")
    print(f"  mIoU: {trial.user_attrs.get('miou', 0.0):.2f}%")
    print(f"  Pixel Accuracy: {trial.user_attrs.get('acc', 0.0):.2f}%")
    print(f"  mAP: {trial.user_attrs.get('map', 0.0):.2f}%")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    output = {
        'best_avg_score': float(trial.value),
        'best_miou': float(trial.user_attrs.get('miou', 0.0)),
        'best_acc': float(trial.user_attrs.get('acc', 0.0)),
        'best_map': float(trial.user_attrs.get('map', 0.0)),
        'best_params': trial.params
    }
    out_file = "chili_omp_optuna_best.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Saved optimized parameters to {out_file}")

if __name__ == '__main__':
    main()

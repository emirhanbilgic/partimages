#!/usr/bin/env python3
"""
Find and visualize the images where CHILI+OMP improves the most over vanilla CHILI.
Uses the best hyperparameters from Optuna tuning.
"""

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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import open_clip
from pycocotools.coco import COCO

import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
ALPHA = 5.0
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"

SPARSE_THRESHOLD = 0.5
MAX_DICT_COS_SIM = 0.7
MAX_ATOMS = 4
IMAGE_SIZE = 224
TOP_K = 10

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading model {MODEL_NAME} ({PRETRAINED}) on {DEVICE}...")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model.eval()

# ── OMP ───────────────────────────────────────────────────────────────────────
def omp_sparse_residual(x_1x, D, max_atoms=8, tol=1e-6):
    if D is None or D.numel() == 0 or max_atoms is None or max_atoms <= 0:
        return F.normalize(x_1x, dim=-1)
    device, dtype = x_1x.device, x_1x.dtype
    x = x_1x.clone().cpu().float()
    D_cpu = D.cpu().float()
    K = D_cpu.shape[0]
    max_atoms = int(min(max_atoms, K))
    selected = []
    r = x.clone()
    for _ in range(max_atoms):
        c = (r @ D_cpu.t()).squeeze(0)
        c_abs = c.abs()
        if selected:
            c_abs[selected] = -1.0
        idx = int(torch.argmax(c_abs).item())
        if c_abs[idx].item() <= tol:
            break
        selected.append(idx)
        D_S = D_cpu[selected, :]
        G = D_S @ D_S.t()
        b = D_S @ x.t()
        I = torch.eye(G.shape[0])
        try:
            L = torch.linalg.cholesky(G + 1e-6 * I)
            s = torch.cholesky_solve(b, L)
        except RuntimeError:
            s = torch.linalg.lstsq(G + 1e-6 * I, b).solution
        r = x - s.t() @ D_S
        if float(torch.norm(r)) <= tol:
            break
    r = r.to(device=device, dtype=dtype)
    return F.normalize(x_1x, dim=-1) if torch.norm(r) <= tol else F.normalize(r, dim=-1)

# ── CHILI core ────────────────────────────────────────────────────────────────
def extract_chili_activations(image, text_emb):
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        visual = model.visual
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
        x = x.permute(1, 0, 2)

        num_layers = len(visual.transformer.resblocks)
        embed_dim = x.shape[-1]
        num_heads = visual.transformer.resblocks[0].attn.num_heads
        head_dim = embed_dim // num_heads
        proj = visual.proj

        A_lh = torch.zeros((num_layers, num_heads, 14, 14), device=DEVICE)
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
            alpha_cls = attn_weights[0, :, 0, 1:]
            W_o = block.attn.out_proj.weight
            for h in range(num_heads):
                v_h = v_heads[0, h, 1:] * alpha_cls[h].unsqueeze(-1)
                padded_v = torch.zeros(seq_len - 1, embed_dim, device=x.device, dtype=x.dtype)
                padded_v[:, h * head_dim:(h + 1) * head_dim] = v_h
                msa_out = torch.matmul(padded_v, W_o.t())
                m = torch.matmul(msa_out, proj)
                A = torch.matmul(m, text_emb.to(DEVICE))
                A_lh[l, h] = A.view(14, 14)
            x = block(x)
    return A_lh.detach().cpu().numpy()


def get_fm(A_lh):
    fm = np.zeros_like(A_lh)
    for l in range(A_lh.shape[0]):
        for h in range(A_lh.shape[1]):
            fm[l, h] = ndimage.median_filter(A_lh[l, h], size=3)
    return fm


def get_hm(fm):
    hm = np.zeros_like(fm)
    for l in range(fm.shape[0]):
        for h in range(fm.shape[1]):
            hm[l, h] = (fm[l, h] > np.mean(fm[l, h])).astype(float)
    return hm


def calculate_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union > 0 else 0.0


def get_image_parts(coco, img_id):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    cat_ids = list(set(a['category_id'] for a in anns))
    return coco.loadCats(cat_ids)


def get_binary_mask(coco, img_id, cat_id, target_size):
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id])
    anns = coco.loadAnns(ann_ids)
    if not anns:
        return np.zeros(target_size, dtype=np.uint8)
    mask = np.zeros((coco.imgs[img_id]['height'], coco.imgs[img_id]['width']), dtype=np.uint8)
    for ann in anns:
        mask = np.maximum(mask, coco.annToMask(ann))
    mask_img = Image.fromarray(mask * 255).resize(target_size, Image.NEAREST)
    return (np.array(mask_img) > 128).astype(np.uint8)


def heatmap_to_pred(heatmap_14x14, w_lh, gt_shape, threshold):
    fm = get_fm(heatmap_14x14)
    weighted = np.zeros_like(fm)
    for l in range(12):
        for h in range(12):
            weighted[l, h] = w_lh[l, h] * fm[l, h]
    combined = weighted.sum(axis=(0, 1))
    if combined.max() > combined.min():
        norm = (combined - combined.min()) / (combined.max() - combined.min())
    else:
        norm = combined
    resized = cv2.resize(norm, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_LINEAR)
    return resized, (resized > threshold).astype(np.uint8)


def compute_miou(pred_bin, gt_mask):
    """Per-sample mIoU over 2 classes (bg + fg)."""
    ious = []
    for c in range(2):
        p = (pred_bin == c)
        t = (gt_mask == c)
        inter = (p & t).sum()
        union = (p | t).sum()
        ious.append(inter / union if union > 0 else 0.0)
    return np.mean(ious) * 100.0


# ── Calibration ───────────────────────────────────────────────────────────────
def calibrate(coco, ds_dir, k=50):
    img_ids = coco.getImgIds()
    np.random.seed(42)
    sample_ids = np.random.choice(img_ids, min(k, len(img_ids)), replace=False)
    calib_data = []
    for img_id in sample_ids:
        info = coco.loadImgs([img_id])[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        if not anns:
            continue
        img = Image.open(os.path.join(ds_dir, "images", info['file_name'])).convert("RGB")
        ann = anns[np.random.randint(len(anns))]
        cat = coco.loadCats([ann['category_id']])[0]
        gt14 = cv2.resize(coco.annToMask(ann), (14, 14), interpolation=cv2.INTER_NEAREST)
        calib_data.append({"image": img, "text": f"a photo of a {cat['name']}", "gt_mask": gt14})

    print(f"Calibrating CHILI weights on {len(calib_data)} samples...")
    L, H = 12, 12
    iou_sums = np.zeros((L, H))
    for d in tqdm(calib_data):
        tok = tokenizer([d["text"]]).to(DEVICE)
        with torch.no_grad():
            te = model.encode_text(tok, normalize=True)[0]
        a = extract_chili_activations(d["image"], te)
        hm = get_hm(get_fm(a))
        for l in range(L):
            for h in range(H):
                iou_sums[l, h] += calculate_iou(hm[l, h], d["gt_mask"])
    means = iou_sums / len(calib_data)
    return 1.0 - np.exp(-ALPHA * means)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ds_dir = "partimagenet_1000_subset"
    ann_file = os.path.join(ds_dir, "subset_annotations.json")
    out_dir = "best_omp_visualizations"
    os.makedirs(out_dir, exist_ok=True)

    coco = COCO(ann_file)
    w_lh = calibrate(coco, ds_dir, k=50)

    print(f"\nScanning all images (threshold_baseline=0.5, threshold_omp={SPARSE_THRESHOLD}, "
          f"max_dict_cos_sim={MAX_DICT_COS_SIM}, max_atoms={MAX_ATOMS})...\n")

    img_ids = coco.getImgIds()
    per_image_records = []

    for img_id in tqdm(img_ids):
        info = coco.loadImgs([img_id])[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        if not anns:
            continue
        img_path = os.path.join(ds_dir, "images", info['file_name'])
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)

        categories = get_image_parts(coco, img_id)
        if len(categories) < 2:
            continue

        cat_names = [c['name'] for c in categories]
        prompts = [f"a photo of a {n}" for n in cat_names]
        tok = tokenizer(prompts).to(DEVICE)
        with torch.no_grad():
            text_embs = model.encode_text(tok, normalize=True)

        part_results = []

        for i, cat in enumerate(categories):
            target_emb = text_embs[i:i + 1]
            dict_embs = torch.cat([text_embs[:i], text_embs[i + 1:]], dim=0)

            if dict_embs.shape[0] > 0 and 0.0 < MAX_DICT_COS_SIM < 1.0:
                sim = (dict_embs @ target_emb.t()).squeeze(-1).abs()
                dict_embs = dict_embs[sim < MAX_DICT_COS_SIM]

            atoms = min(MAX_ATOMS, dict_embs.shape[0])
            sparse_emb = omp_sparse_residual(target_emb, dict_embs, max_atoms=atoms)

            gt_mask = get_binary_mask(coco, img_id, cat['id'], (IMAGE_SIZE, IMAGE_SIZE))
            if gt_mask.sum() == 0:
                continue

            a_baseline = extract_chili_activations(image, text_emb=target_emb[0])
            a_omp = extract_chili_activations(image, text_emb=sparse_emb[0])

            hm_baseline, pred_baseline = heatmap_to_pred(a_baseline, w_lh, gt_mask.shape, 0.5)
            hm_omp, pred_omp = heatmap_to_pred(a_omp, w_lh, gt_mask.shape, SPARSE_THRESHOLD)

            miou_b = compute_miou(pred_baseline, gt_mask)
            miou_o = compute_miou(pred_omp, gt_mask)

            part_results.append({
                'cat_name': cat['name'],
                'cat_id': cat['id'],
                'miou_baseline': miou_b,
                'miou_omp': miou_o,
                'hm_baseline': hm_baseline,
                'hm_omp': hm_omp,
                'pred_baseline': pred_baseline,
                'pred_omp': pred_omp,
                'gt_mask': gt_mask,
                'atoms_used': atoms,
            })

        if not part_results:
            continue

        avg_b = np.mean([p['miou_baseline'] for p in part_results])
        avg_o = np.mean([p['miou_omp'] for p in part_results])

        per_image_records.append({
            'img_id': img_id,
            'file_name': info['file_name'],
            'image': image,
            'avg_miou_baseline': avg_b,
            'avg_miou_omp': avg_o,
            'delta': avg_o - avg_b,
            'parts': part_results,
        })

    per_image_records.sort(key=lambda r: r['delta'], reverse=True)

    print(f"\n{'='*70}")
    print(f"  Top {TOP_K} images where OMP helps the most")
    print(f"{'='*70}")
    for rank, rec in enumerate(per_image_records[:TOP_K]):
        print(f"  #{rank+1:2d}  {rec['file_name']:40s}  "
              f"baseline={rec['avg_miou_baseline']:5.1f}  omp={rec['avg_miou_omp']:5.1f}  "
              f"delta=+{rec['delta']:5.1f}  parts={len(rec['parts'])}")

    # ── Visualization ─────────────────────────────────────────────────────────
    for rank, rec in enumerate(per_image_records[:TOP_K]):
        parts = rec['parts']
        n_parts = len(parts)

        fig, axes = plt.subplots(n_parts, 4, figsize=(16, 4 * n_parts))
        if n_parts == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            f"#{rank+1}  {rec['file_name']}\n"
            f"Avg mIoU: baseline={rec['avg_miou_baseline']:.1f}  |  "
            f"OMP={rec['avg_miou_omp']:.1f}  |  "
            f"delta=+{rec['delta']:.1f}",
            fontsize=14, fontweight='bold', y=1.02,
        )

        for row, part in enumerate(parts):
            img_np = np.array(rec['image'])

            # Col 0: original + GT outline
            axes[row, 0].imshow(img_np)
            gt_contours = cv2.findContours(
                part['gt_mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]
            overlay_gt = img_np.copy()
            cv2.drawContours(overlay_gt, gt_contours, -1, (0, 255, 0), 2)
            axes[row, 0].imshow(overlay_gt)
            axes[row, 0].set_title(f"GT: {part['cat_name']}", fontsize=11)
            axes[row, 0].axis('off')

            # Col 1: GT mask
            axes[row, 1].imshow(part['gt_mask'], cmap='gray', vmin=0, vmax=1)
            axes[row, 1].set_title("Ground Truth", fontsize=11)
            axes[row, 1].axis('off')

            # Col 2: baseline heatmap + prediction outline
            axes[row, 2].imshow(img_np)
            axes[row, 2].imshow(part['hm_baseline'], alpha=0.5, cmap='jet', vmin=0, vmax=1)
            b_contours = cv2.findContours(
                part['pred_baseline'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]
            overlay_b = img_np.copy()
            cv2.drawContours(overlay_b, b_contours, -1, (255, 0, 0), 2)
            axes[row, 2].imshow(overlay_b, alpha=0.3)
            axes[row, 2].set_title(f"Baseline  mIoU={part['miou_baseline']:.1f}", fontsize=11)
            axes[row, 2].axis('off')

            # Col 3: OMP heatmap + prediction outline
            axes[row, 3].imshow(img_np)
            axes[row, 3].imshow(part['hm_omp'], alpha=0.5, cmap='jet', vmin=0, vmax=1)
            o_contours = cv2.findContours(
                part['pred_omp'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]
            overlay_o = img_np.copy()
            cv2.drawContours(overlay_o, o_contours, -1, (0, 0, 255), 2)
            axes[row, 3].imshow(overlay_o, alpha=0.3)
            axes[row, 3].set_title(
                f"OMP (atoms={part['atoms_used']})  mIoU={part['miou_omp']:.1f}",
                fontsize=11,
            )
            axes[row, 3].axis('off')

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"top{rank+1:02d}_{rec['img_id']}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

    summary = {
        'hyperparams': {
            'sparse_threshold': SPARSE_THRESHOLD,
            'max_dict_cos_sim': MAX_DICT_COS_SIM,
            'max_atoms': MAX_ATOMS,
        },
        'top_images': [
            {
                'rank': i + 1,
                'img_id': r['img_id'],
                'file_name': r['file_name'],
                'avg_miou_baseline': round(r['avg_miou_baseline'], 2),
                'avg_miou_omp': round(r['avg_miou_omp'], 2),
                'delta': round(r['delta'], 2),
                'n_parts': len(r['parts']),
                'per_part': [
                    {
                        'name': p['cat_name'],
                        'miou_baseline': round(p['miou_baseline'], 2),
                        'miou_omp': round(p['miou_omp'], 2),
                    }
                    for p in r['parts']
                ],
            }
            for i, r in enumerate(per_image_records[:TOP_K])
        ],
    }
    summary_path = os.path.join(out_dir, "top10_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()

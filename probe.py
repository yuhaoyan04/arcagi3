"""
probe.py – LeWM 模型表征区分度验证

测试训练好的模型在不同动作输入（单动作/动作插值）下预测下一状态的差异，
验证模型是否能区分不同动作并在隐空间产生有意义的预测变化。

用法：
  python probe.py                                          # 随机权重 + 合成数据（验证代码）
  python probe.py --ckpt path/to/lewm_object.ckpt         # 加载 checkpoint
  python probe.py --ckpt ... --dataset pusht_expert_train  # 真实数据
"""

import argparse
import torch
import torch.nn.functional as F

import stable_pretraining as spt
import stable_worldmodel as swm

from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from utils import get_img_preprocessor, get_column_normalizer


# ── 模型构建：镜像 train.py::run() 中的构建方式，使用相同的类和参数 ──────────

def build_model(encoder_scale="tiny", patch_size=14, img_size=224,
                embed_dim=192, history_size=3, effective_act_dim=10,
                predictor_cfg=None):
    """构建 LeWM 模型，参数与 train.py::run() 中的构建逻辑完全一致。"""
    predictor_cfg = predictor_cfg or dict(
        depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1
    )
    encoder = spt.backbone.utils.vit_hf(
        encoder_scale, patch_size=patch_size, image_size=img_size,
        pretrained=False, use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size

    predictor = ARPredictor(
        num_frames=history_size, input_dim=embed_dim,
        hidden_dim=hidden_dim, output_dim=hidden_dim,
        **predictor_cfg,
    )
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=Embedder(input_dim=effective_act_dim, emb_dim=embed_dim),
        projector=MLP(input_dim=hidden_dim, hidden_dim=2048, output_dim=embed_dim,
                      norm_fn=torch.nn.BatchNorm1d),
        pred_proj=MLP(input_dim=hidden_dim, hidden_dim=2048, output_dim=embed_dim,
                      norm_fn=torch.nn.BatchNorm1d),
    )


# ── 模型加载：对应 utils.ModelObjectCallBack._dump_model 的 torch.save(model, path) ──

def load_model(ckpt_path, device="cpu"):
    """加载 object checkpoint（由 utils.ModelObjectCallBack._dump_model 保存）。"""
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    if hasattr(model, "model"):   # 兼容意外保存了 spt.Module wrapper 的情况
        model = model.model
    return model.to(device).eval().requires_grad_(False)


# ── 数据加载：复用 train.py 的 HDF5Dataset + get_img_preprocessor + get_column_normalizer ──

def load_dataset_batch(dataset_name, n_samples=4, history_size=3,
                       frameskip=5, img_size=224, device="cpu"):
    """从 HDF5 数据集加载一批样本，预处理与训练时完全一致。"""
    ds = swm.data.HDF5Dataset(
        name=dataset_name, num_steps=history_size + 1, frameskip=frameskip,
        keys_to_load=["pixels", "action"], transform=None,
    )
    ds.transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", img_size),
        get_column_normalizer(ds, "action", "action"),
    )
    idx = torch.randperm(len(ds))[:n_samples].tolist()
    samples = [ds[i] for i in idx]
    pixels = torch.stack([s["pixels"] for s in samples]).to(device)
    action = torch.nan_to_num(torch.stack([s["action"] for s in samples]), 0.0).to(device)
    return pixels, action


def synthetic_batch(batch, timesteps, effective_act_dim, img_size=224, device="cpu"):
    """合成数据，用于无 checkpoint/数据集时的代码验证。"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    pixels = torch.randn(batch, timesteps, 3, img_size, img_size) * std + mean
    action = torch.randn(batch, timesteps, effective_act_dim)
    return pixels.to(device), action.to(device)


# ── 核心推理：复用 train.py::lejepa_forward 中的 encode → predict 模式 ────────

@torch.no_grad()
def encode_and_predict(model, pixels, action):
    """
    复用 lejepa_forward 的推理模式：nan_to_num → encode → predict。

    pixels : (B, T, C, H, W)
    action : (B, T, effective_act_dim)
    返回   : emb (B,T,D),  pred_emb (B,T,D)
    """
    action = torch.nan_to_num(action, 0.0)
    info = model.encode({"pixels": pixels, "action": action})
    emb      = info["emb"]
    act_emb  = info["act_emb"]
    pred_emb = model.predict(emb, act_emb)
    return emb, pred_emb


# ── 分析函数 ──────────────────────────────────────────────────────────────────

def analyze_action_sensitivity(model, pixels, actions, label="action_sensitivity"):
    """固定历史帧，对 N 个不同动作各自预测，统计预测 embedding 的区分度。"""
    info = model.encode({"pixels": pixels})
    history_emb = info["emb"]

    preds = []
    for act in actions:
        act_emb = model.action_encoder(torch.nan_to_num(act, 0.0))
        pred = model.predict(history_emb, act_emb)[:, -1].mean(0)  # (D,)
        preds.append(pred)

    preds = torch.stack(preds)                                       # (N, D)
    l2    = torch.cdist(preds.unsqueeze(0), preds.unsqueeze(0))[0]  # (N, N)
    cos   = F.normalize(preds, dim=-1) @ F.normalize(preds, dim=-1).T
    mask  = ~torch.eye(len(actions), dtype=torch.bool, device=preds.device)

    return {
        "label": label,
        "n_actions": len(actions),
        "pred_l2_mean": l2[mask].mean().item(),
        "pred_l2_max":  l2.max().item(),
        "pred_cosine_sim_mean": cos[mask].mean().item(),
        "l2_matrix": l2,
    }


def analyze_action_interpolation(model, pixels, action_a, action_b,
                                 n_steps=9, label="action_interpolation"):
    """两个动作之间线性插值，观察预测 embedding 变化是否平滑单调。"""
    info = model.encode({"pixels": pixels})
    history_emb = info["emb"]

    alphas = torch.linspace(0, 1, n_steps)
    preds  = []
    for alpha in alphas:
        act     = (1 - alpha) * action_a + alpha * action_b
        act_emb = model.action_encoder(torch.nan_to_num(act, 0.0))
        pred    = model.predict(history_emb, act_emb)[:, -1].mean(0)
        preds.append(pred)

    preds = torch.stack(preds)                   # (n_steps, D)
    dist  = (preds - preds[0]).norm(dim=-1)      # 相对 α=0 的距离变化

    return {
        "label": label,
        "action_emb_dist_a_to_b": (
            model.action_encoder(action_b)[:, -1].mean(0) -
            model.action_encoder(action_a)[:, -1].mean(0)
        ).norm().item(),
        "pred_dist_a_to_b": dist[-1].item(),
        "pred_dist_at_alpha": {
            f"{a:.1f}": d.item() for a, d in zip(alphas.tolist(), dist.tolist())
        },
        "monotonicity_ratio": ((dist[1:] - dist[:-1]) >= 0).float().mean().item(),
    }


def analyze_embedding_stats(model, pixels_list, label="embedding_stats"):
    """统计 embedding 分布：范数、各维方差（各向同性指标）、样本间余弦相似度。"""
    embs = []
    for pixels in pixels_list:
        info = model.encode({"pixels": pixels})
        embs.append(info["emb"].reshape(-1, info["emb"].shape[-1]).cpu())
    embs = torch.cat(embs, 0)

    norms   = embs.norm(dim=-1)
    cos     = F.normalize(embs, dim=-1) @ F.normalize(embs, dim=-1).T
    mask    = ~torch.eye(len(embs), dtype=torch.bool)
    dim_std = embs.std(0)

    return {
        "label": label,
        "n_embeddings": len(embs),
        "norm_mean": norms.mean().item(),
        "norm_std":  norms.std().item(),
        "dim_std_mean": dim_std.mean().item(),
        "dim_std_min/max": f"{dim_std.min().item():.4f} / {dim_std.max().item():.4f}",
        "inter_sample_cosine_mean": cos[mask].mean().item(),
    }


# ── 输出 ──────────────────────────────────────────────────────────────────────

def print_result(res):
    print(f"\n{'─'*55}")
    print(f"  {res['label']}")
    print(f"{'─'*55}")
    for k, v in res.items():
        if k == "label":
            continue
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"      {kk}: {vv:.4f}" if isinstance(vv, float) else f"      {kk}: {vv}")
        elif isinstance(v, torch.Tensor):
            mat = v.cpu().numpy()
            header = "        " + "  ".join(f"A{i:<4}" for i in range(mat.shape[0]))
            print(f"  {k}:\n{header}")
            for i, row in enumerate(mat):
                print(f"  A{i}      " + "  ".join(f"{x:6.3f}" for x in row))
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         type=str, default=None)
    parser.add_argument("--dataset",      type=str, default=None,
                        help="HDF5 数据集名，如 pusht_expert_train")
    parser.add_argument("--n_samples",    type=int, default=4)
    parser.add_argument("--action_dim",   type=int, default=2)
    parser.add_argument("--frameskip",    type=int, default=5)
    parser.add_argument("--history_size", type=int, default=3)
    parser.add_argument("--img_size",     type=int, default=224)
    parser.add_argument("--n_actions",    type=int, default=5)
    parser.add_argument("--interp_steps", type=int, default=9)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 加载模型
    if args.ckpt:
        print(f"[probe] 加载 checkpoint: {args.ckpt}")
        model = load_model(args.ckpt, args.device)
    else:
        print("[probe] 使用随机初始化模型（无 checkpoint）")
        model = build_model(
            effective_act_dim=args.frameskip * args.action_dim,
            history_size=args.history_size,
            img_size=args.img_size,
        ).to(args.device).eval()

    eff_act_dim = model.action_encoder.patch_embed.in_channels
    print(f"[probe] device={args.device}  effective_act_dim={eff_act_dim}")

    # 加载数据
    B, T = args.n_samples, args.history_size
    if args.dataset:
        print(f"[probe] 加载数据集: {args.dataset}")
        pixels, action = load_dataset_batch(
            args.dataset, n_samples=B, history_size=T,
            frameskip=args.frameskip, img_size=args.img_size, device=args.device,
        )
    else:
        print("[probe] 使用合成数据（传入 --dataset 可使用真实数据）")
        pixels, action = synthetic_batch(B, T, eff_act_dim, args.img_size, args.device)

    # 分析 1：动作敏感度
    actions_list = [torch.randn_like(action) for _ in range(args.n_actions)]
    print_result(analyze_action_sensitivity(model, pixels, actions_list))

    # 分析 2：动作插值（大幅 & 小幅）
    act_a = torch.randn_like(action)
    act_b = torch.randn_like(action)
    print_result(analyze_action_interpolation(
        model, pixels, act_a, act_b, n_steps=args.interp_steps))
    print_result(analyze_action_interpolation(
        model, pixels, act_a, act_a * 0.01, n_steps=args.interp_steps,
        label="action_interpolation (large→small)"))

    # 分析 3：embedding 分布统计
    pixel_batches = [
        synthetic_batch(B, T, eff_act_dim, args.img_size, args.device)[0]
        for _ in range(3)
    ]
    print_result(analyze_embedding_stats(model, pixel_batches))

    print(f"\n{'─'*55}\n[probe] 完成。")


if __name__ == "__main__":
    main()

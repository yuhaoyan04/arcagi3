import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
except ImportError:
    SwanLabLogger = None
from omegaconf import OmegaConf, open_dict
from torch import nn

from jepa import SphericalJEPA
from module import (
    ARPredictor,
    Embedder,
    MLP,
    infonce_loss,
    spread_loss,
    uniformity_loss,
)
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def resolve_norm_fn(norm_name: str):
    norm_name = norm_name.lower()
    if norm_name in {"none", "identity"}:
        return None
    if norm_name in {"ln", "layernorm"}:
        return nn.LayerNorm
    if norm_name in {"bn", "batchnorm", "batchnorm1d"}:
        return nn.BatchNorm1d
    raise ValueError(f"Unsupported projection_head.norm_fn: {norm_name}")


def build_projection_head(input_dim: int, output_dim: int, cfg) -> nn.Module:
    head_type = cfg.type.lower()
    norm_name = cfg.get("norm_fn", "none")

    # Backward-compatible aliases for older configs.
    if head_type == "bn":
        head_type = "linear"
        norm_name = "batchnorm1d"
    elif head_type == "ln":
        head_type = "linear"
        norm_name = "layernorm"

    norm_fn = resolve_norm_fn(norm_name)

    if head_type == "linear":
        layers = [nn.Linear(input_dim, output_dim)]
        if norm_fn is not None:
            layers.append(norm_fn(output_dim))
        return nn.Sequential(*layers) if len(layers) > 1 else layers[0]

    if head_type == "mlp":
        return MLP(
            input_dim=input_dim,
            hidden_dim=cfg.get("hidden_dim", 2048),
            output_dim=output_dim,
            norm_fn=norm_fn,
        )

    raise ValueError(f"Unsupported projection_head.type: {head_type}")


def get_loss_space_tensors(output, *, pred_raw, pred_norm, n_preds: int, space: str):
    space = space.lower()
    if space == "raw":
        return pred_raw, output["emb_raw"][:, n_preds:]
    if space in {"normalized", "sphere"}:
        return pred_norm, output["emb"][:, n_preds:]
    raise ValueError(f"Unsupported loss space: {space}")


def get_regularizer_tensor(output, *, space: str):
    space = space.lower()
    if space == "raw":
        return output["emb_raw"]
    if space in {"normalized", "sphere"}:
        return output["emb"]
    raise ValueError(f"Unsupported loss.regularizer.space: {space}")


def get_context_tensor(output, *, space: str):
    space = space.lower()
    if space == "raw":
        return output["emb_raw"]
    if space in {"normalized", "sphere"}:
        return output["emb"]
    raise ValueError(f"Unsupported loss.pred.context_space: {space}")


def compute_pred_loss(pred: torch.Tensor, target: torch.Tensor, cfg) -> torch.Tensor:
    pred_type = cfg.loss.pred.get("type", "cosine").lower()
    if pred_type == "cosine":
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    if pred_type == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unsupported loss.pred.type: {pred_type}")


def swm_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute spherical losses.

    Losses:
      pred_loss   — configurable prediction loss in raw or normalized space
      reg_loss    — configurable anti-collapse loss in raw or normalized space
      loss        — pred_loss + λ * reg_loss
    """
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    reg_type = cfg.loss.regularizer.type
    lambd = cfg.loss.regularizer.weight
    reg_space = cfg.loss.regularizer.get("space", "normalized")
    pred_space = cfg.loss.pred.get("space", "normalized")
    context_space = cfg.loss.pred.get("context_space", pred_space)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    act_emb = output["act_emb"]

    # Make training-time autoregressive context explicit so raw-consistent
    # experiments can feed predictor history from the same space used for the
    # prediction target and planning rollout.
    ctx_emb = get_context_tensor(output, space=context_space)[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    pred_raw = self.model.predict_raw(ctx_emb, ctx_act)
    pred_emb = self.model.normalize_embeddings(pred_raw)

    pred_source, tgt_source = get_loss_space_tensors(
        output,
        pred_raw=pred_raw,
        pred_norm=pred_emb,
        n_preds=n_preds,
        space=pred_space,
    )
    output["pred_loss"] = compute_pred_loss(pred_source, tgt_source, cfg)

    reg_emb = get_regularizer_tensor(output, space=reg_space)
    if reg_type == "spread":
        output["spread_loss"] = spread_loss(reg_emb, cfg.loss.spread.margin)
        output["reg_loss"] = output["spread_loss"]
    elif reg_type == "uniformity":
        output["uniformity_loss"] = uniformity_loss(
            reg_emb,
            cfg.loss.uniformity.t,
            mode=cfg.loss.uniformity.get("mode", "all_pairs"),
            temporal_exclusion=cfg.loss.uniformity.get("temporal_exclusion", 0),
        )
        output["reg_loss"] = output["uniformity_loss"]
    elif reg_type == "infonce":
        output["infonce_loss"] = infonce_loss(
            pred_emb, tgt_emb, cfg.loss.infonce.temperature
        )
        output["reg_loss"] = output["infonce_loss"]
    else:
        raise ValueError(f"Unsupported loss.regularizer.type: {reg_type}")

    output["loss"] = output["pred_loss"] + lambd * output["reg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="swm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    head_cfg = cfg.encoder.projection_head
    projector = build_projection_head(hidden_dim, embed_dim, head_cfg)
    pred_proj = build_projection_head(hidden_dim, embed_dim, head_cfg)
    inference_cfg = cfg.wm.get("inference", {})
    pred_cfg = cfg.loss.get("pred", {})
    training_context_space = pred_cfg.get("context_space", pred_cfg.get("space", "normalized"))

    world_model = SphericalJEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        inference_rollout_state_space=inference_cfg.get("rollout_state_space", "normalized"),
        inference_cost_space=inference_cfg.get("cost_space", "normalized"),
        inference_cost_type=inference_cfg.get("cost_type", "cosine"),
        analysis_prediction_space=cfg.loss.pred.get("space", "normalized"),
        training_context_space=training_context_space,
    )

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        forward=partial(swm_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    backend = cfg.get("logger_backend", "swanlab")
    if backend == "swanlab" and cfg.swanlab.enabled:
        if SwanLabLogger is None:
            raise ImportError("swanlab is not installed. Run: pip install swanlab")
        logger = SwanLabLogger(**cfg.swanlab.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    elif backend == "wandb" and cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()

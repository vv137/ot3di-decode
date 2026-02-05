"""Distributed training script for OT3Di with DDP support."""

from __future__ import annotations

import os
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .data import ProstT5Dataset
from .data.dataset import collate_fn
from .losses import OT3DiLoss
from .metrics import compute_ot_metrics
from .model import OT3DiModel


class TokenizerWrapper:
    """Wrapper for tokenizer to make it picklable for multiprocessing."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def train(
    config_path: str | Path,
    data_path: str | Path | None = None,
    output_dir: str | Path = "./output",
) -> None:
    """Train OT3Di model with DDP support."""
    rank, local_rank, world_size = setup_distributed()
    is_main = is_main_process(rank)

    config = OmegaConf.load(config_path)
    output_dir = Path(output_dir)

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"World size: {world_size}")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        print(f"Using device: {device}")

        # Initialize WandB
        if config.get("wandb", {}).get("enabled", False):
            wandb.init(
                project=config.wandb.get("project", "ot3di"),
                entity=config.wandb.get("entity"),
                name=config.wandb.get("name"),
                config=OmegaConf.to_container(config, resolve=True),
            )

    # Model
    model = OT3DiModel(
        esm_model=config.model.esm_model,
        embed_dim=config.model.get("embed_dim"),  # None uses ESM hidden size
        num_tokens=config.model.num_tokens,
        ot_epsilon=config.ot.epsilon,
        ot_max_iters=config.ot.max_iters,
        ot_backend=config.ot.get("backend", "triton"),
        freeze_esm=config.model.freeze_esm,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    base_model = model.module if hasattr(model, "module") else model

    # Data
    if is_main:
        print("Loading ProstT5Dataset from HuggingFace cache...")

    dataset = ProstT5Dataset(
        split=config.data.get("split", "train"),
        max_length=config.data.max_length,
    )

    if is_main:
        print(f"Dataset size: {len(dataset)}")

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None

    # Use tokenizer directly to avoid pickling/sharing the entire model with workers
    # Use tokenizer directly to avoid pickling/sharing the entire model with workers
    tokenizer_wrapper = TokenizerWrapper(base_model.encoder.tokenizer)
    collate = partial(collate_fn, esm_batch_converter=tokenizer_wrapper)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.data.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    criterion = OT3DiLoss(alpha=config.training.alpha, use_soft_ce=True)

    save_every_steps = config.training.get("save_every_steps", 1000)
    grad_acc_steps = config.training.get("gradient_accumulation_steps", 1)

    if is_main:
        print(f"Training for {config.training.epochs} epochs")
        print(f"Gradient accumulation steps: {grad_acc_steps}")
        print(f"Effective batch size: {config.training.batch_size * grad_acc_steps * world_size}")
        print(f"Saving every {save_every_steps} steps + end of each epoch")

    # Training loop
    global_step = 0
    for epoch in range(config.training.epochs):
        model.train()

        if sampler is not None:
            sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", disable=not is_main)
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["esm_tokens"].to(device)
            attention_mask = batch.get("esm_attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            target_tokens = batch["threedi_tokens"].to(device)
            mask = batch["mask"].to(device)

            # Gradient accumulation context
            # Use no_sync() if we are accumulating gradients (not the last step of the accumulation)
            # This prevents DDP from synchronizing gradients at every step
            do_sync = (batch_idx + 1) % grad_acc_steps == 0
            sync_context = model.no_sync() if (world_size > 1 and not do_sync) else nullcontext()

            with sync_context:
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    target_tokens=target_tokens,
                )
                losses = criterion(outputs, target_tokens, base_model.num_tokens, mask)

                # Scale loss
                loss = losses["loss"] / grad_acc_steps
                loss.backward()

            # Update weights only after accumulating enough gradients
            if do_sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Calculate extra OT metrics
                ot_metrics = {"ot_cost": 0.0, "entropy": 0.0}
                if "P" in outputs and "cost" in outputs:
                    ot_metrics = compute_ot_metrics(outputs["P"].detach(), outputs["cost"].detach(), mask.detach())

                pbar.set_postfix(
                    step=global_step,
                    loss=losses["loss"].item(),
                    ce=losses["loss_ce"].item(),
                    ot=losses["loss_ot"].item(),
                    cost=f"{ot_metrics['ot_cost']:.2f}",
                )

                if is_main and config.get("wandb", {}).get("enabled", False):
                    log_interval = config.wandb.get("log_interval", 50)
                    if global_step % log_interval == 0:
                        wandb.log(
                            {
                                "train/loss": losses["loss"].item(),
                                "train/ce_loss": losses["loss_ce"].item(),
                                "train/ot_loss": losses["loss_ot"].item(),
                                "train/ot_cost_avg": ot_metrics["ot_cost"],
                                "train/entropy_avg": ot_metrics["entropy"],
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "train/epoch": epoch + 1,
                                "train/step": global_step,
                            },
                            step=global_step,
                        )

                # Save every N steps
                if is_main and global_step % save_every_steps == 0:
                    ckpt_path = output_dir / f"checkpoint_step_{global_step}.pt"
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "model_state_dict": base_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": OmegaConf.to_container(config),
                        },
                        ckpt_path,
                    )
                    print(f"\nSaved checkpoint: {ckpt_path}")

            total_loss += losses["loss"].item()
            num_batches += 1

        scheduler.step()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        if is_main:
            print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}, steps = {global_step}")

            # Save at end of each epoch
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": OmegaConf.to_container(config),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    if is_main and config.get("wandb", {}).get("enabled", False):
        wandb.finish()

    cleanup_distributed()

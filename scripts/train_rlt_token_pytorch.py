import dataclasses
import logging
import pathlib
import time

import jax
import safetensors.torch
import torch
import tqdm
import tyro

import openpi.models.pi0_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.rlt.config import RLTTokenConfig
from openpi.rlt.networks import RLTokenModule
from openpi.rlt.policy import resolve_local_checkpoint
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_libero"
    checkpoint_dir: str = "checkpoints/pi05_libero_pytorch"
    output_dir: str = "checkpoints/rlt/pi05_libero/rl_token"
    task_id: int | None = None
    device: str = "cuda"
    batch_size: int = 128
    grad_accum_steps: int = 1
    num_train_steps: int = 10_000
    lr: float = 1e-4
    alpha_vla: float = 0.0
    log_interval: int = 50
    save_interval: int = 1000
    seed: int = 42
    num_workers: int = 2
    wandb_enabled: bool = False
    token: RLTTokenConfig = dataclasses.field(default_factory=RLTTokenConfig)


def _load_vla(train_config: _config.TrainConfig, checkpoint_dir: pathlib.Path, device: torch.device) -> PI0Pytorch:
    model_cfg = train_config.model
    if not isinstance(model_cfg, openpi.models.pi0_config.Pi0Config):
        raise TypeError("RLT token training currently supports Pi0Config / pi0.5 only.")
    object.__setattr__(model_cfg, "dtype", train_config.pytorch_training_precision)
    model = PI0Pytorch(model_cfg).to(device)
    safetensors.torch.load_model(model, checkpoint_dir / "model.safetensors", device=str(device))
    return model


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint_dir = resolve_local_checkpoint(args.checkpoint_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = _config.get_config(args.config_name)
    train_config = dataclasses.replace(
        train_config,
        exp_name="rlt_token",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        wandb_enabled=args.wandb_enabled,
    )
    if args.task_id is not None:
        logging.info("Training task-specific RL token for task_id=%s", args.task_id)
    vla = _load_vla(train_config, checkpoint_dir, device)
    vla.train(args.alpha_vla > 0)
    if args.alpha_vla <= 0:
        for param in vla.parameters():
            param.requires_grad_(False)

    rl_token = RLTokenModule(args.token).to(device)
    params = list(rl_token.parameters()) + ([p for p in vla.parameters() if p.requires_grad] if args.alpha_vla > 0 else [])
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    loader = _data_loader.create_data_loader(train_config, framework="pytorch", shuffle=True, task_id=args.task_id)
    data_iter = iter(loader)

    start_time = time.time()
    pbar = tqdm.tqdm(range(args.num_train_steps), desc="RLT token")
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        token_loss_sum = 0.0
        vla_loss_sum = 0.0
        vla_loss_count = 0
        for _ in range(args.grad_accum_steps):
            try:
                observation, actions = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                observation, actions = next(data_iter)
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(device=device, dtype=torch.float32)

            with torch.set_grad_enabled(args.alpha_vla > 0):
                embeddings, padding_mask = vla.encode_prefix_tokens(observation, train=args.alpha_vla > 0)
            token_loss, _ = rl_token.reconstruction_loss(embeddings.detach(), padding_mask=~padding_mask)
            loss = token_loss
            vla_loss = None
            if args.alpha_vla > 0:
                vla_loss = vla(observation, actions).mean()
                loss = loss + args.alpha_vla * vla_loss
                vla_loss_sum += float(vla_loss.detach().cpu())
                vla_loss_count += 1
            (loss / args.grad_accum_steps).backward()
            loss_sum += float(loss.detach().cpu())
            token_loss_sum += float(token_loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        optimizer.step()
        loss_value = loss_sum / args.grad_accum_steps
        token_loss_value = token_loss_sum / args.grad_accum_steps
        vla_loss_value = vla_loss_sum / vla_loss_count if vla_loss_count else None

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            msg = (
                f"step={step} token_loss={token_loss_value:.5f} loss={loss_value:.5f} "
                f"effective_batch_size={args.batch_size * args.grad_accum_steps} time={elapsed:.1f}s"
            )
            if vla_loss_value is not None:
                msg += f" vla_loss={vla_loss_value:.5f}"
            logging.info(msg)
            pbar.set_postfix({"loss": f"{loss_value:.4f}", "token": f"{token_loss_value:.4f}"})
            start_time = time.time()

        if (step + 1) % args.save_interval == 0 or step + 1 == args.num_train_steps:
            payload = {
                "rl_token": rl_token.state_dict(),
                "token_config": dataclasses.asdict(args.token),
                "step": step + 1,
                "config_name": args.config_name,
                "checkpoint_dir": str(checkpoint_dir),
                "task_id": args.task_id,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "effective_batch_size": args.batch_size * args.grad_accum_steps,
            }
            torch.save(payload, output_dir / f"rl_token_{step + 1}.pt")
            torch.save(payload, output_dir / "latest.pt")
            logging.info("Saved RL token checkpoint to %s", output_dir / "latest.pt")


if __name__ == "__main__":
    tyro.cli(main)

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from typing import Any

from flax import serialization
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro

import openpi.models.model as _model
from openpi.rlt.jax_token import RLTokenAutoencoder
from openpi.rlt.jax_token import RLTTokenJaxConfig
from openpi.rlt.jax_token import reconstruction_loss
import openpi.shared.nnx_utils as nnx_utils
from openpi import transforms as _transforms
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_piper_greenblock_5090_jax_delta_v1"
    checkpoint_dir: str = (
        "/mnt2/pi05/openpi_jax_checkpoints/pi05_piper_greenblock_5090_jax_delta_v1/"
        "piper_greenblock_5090_delta_sft_30k_20260707/20000"
    )
    output_dir: str = (
        "/mnt2/pi05/openpi_rlt/rl_tokens/"
        "pi05_piper_greenblock_5090_full20k_rlt_token_v1_20260709"
    )
    batch_size: int = 4
    num_train_steps: int = 10_000
    lr: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    log_interval: int = 20
    save_interval: int = 1_000
    sample_num_steps: int = 10
    cache_after: bool = True
    cache_batch_size: int = 8
    cache_max_batches: int | None = None
    jax_cache_dir: str = "/mnt2/pi05/jax_cache"
    token: RLTTokenJaxConfig = dataclasses.field(default_factory=RLTTokenJaxConfig)


def _load_model(train_config: _config.TrainConfig, checkpoint_dir: pathlib.Path):
    if not (checkpoint_dir / "params").is_dir():
        raise FileNotFoundError(f"Missing params directory: {checkpoint_dir / 'params'}")
    return train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))


def _save_checkpoint(
    output_dir: pathlib.Path,
    *,
    step: int,
    params: Any,
    token_config: RLTTokenJaxConfig,
    args: Args,
    metrics: dict[str, float],
) -> pathlib.Path:
    ckpt_dir = output_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "params.msgpack").write_bytes(serialization.to_bytes(params))
    metadata = {
        "step": step,
        "token_config": dataclasses.asdict(token_config),
        "args": dataclasses.asdict(args),
        "metrics": metrics,
        "format": "jax_flax_rl_token_autoencoder_v1",
    }
    (ckpt_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    latest = output_dir / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(ckpt_dir.name)
    return ckpt_dir


def _make_train_step(token_model: RLTokenAutoencoder, tx: optax.GradientTransformation):
    @jax.jit
    def train_step(params, opt_state, rng, embeddings, valid_mask):
        def loss_fn(p):
            z_rl, reconstruction = token_model.apply(
                {"params": p},
                embeddings,
                valid_mask,
                train=True,
                rngs={"dropout": rng},
            )
            loss = reconstruction_loss(reconstruction, embeddings, valid_mask)
            metrics = {
                "loss": loss,
                "z_norm": jnp.mean(jnp.linalg.norm(z_rl, axis=-1)),
                "valid_tokens": jnp.mean(jnp.sum(valid_mask, axis=-1)),
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics = {**metrics, "grad_norm": optax.global_norm(grads), "loss": loss}
        return params, opt_state, metrics

    return train_step


def _unnormalize_state(state: np.ndarray, norm_stats: dict[str, _transforms.NormStats] | None) -> np.ndarray:
    if not norm_stats or "state" not in norm_stats:
        return state[..., :7].astype(np.float32)
    stats = norm_stats["state"]
    mean = np.asarray(stats.mean, dtype=np.float32)
    std = np.asarray(stats.std, dtype=np.float32)
    return (state[..., : mean.shape[-1]] * (std + 1e-6) + mean)[..., :7].astype(np.float32)


def _absolute_actions_per_sample(output_transform, state_batch: np.ndarray, actions_batch: np.ndarray) -> np.ndarray:
    outputs = []
    for state, actions in zip(state_batch, actions_batch, strict=True):
        transformed = output_transform({"state": state, "actions": actions})
        outputs.append(np.asarray(transformed["actions"], dtype=np.float32))
    return np.stack(outputs, axis=0).astype(np.float32)


def _cache_rlt_inputs(
    *,
    args: Args,
    output_dir: pathlib.Path,
    train_config: _config.TrainConfig,
    token_model: RLTokenAutoencoder,
    token_params,
    extract_prefix_tokens,
    sample_actions,
) -> dict[str, Any]:
    cache_dir = output_dir / "cache_z_rl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_config = dataclasses.replace(
        train_config,
        batch_size=args.cache_batch_size,
        num_workers=0,
        seed=args.seed,
    )
    loader = _data_loader.create_data_loader(
        cache_config,
        shuffle=False,
        num_batches=args.cache_max_batches,
        framework="jax",
    )
    data_config = loader.data_config()
    output_transform = _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )
    apply_token = jax.jit(lambda embeddings, valid: token_model.apply({"params": token_params}, embeddings, valid, train=False)[0])
    rng = jax.random.key(args.seed + 17)
    total = 0
    shards = []
    start = time.time()
    for batch_idx, (observation, _) in enumerate(loader):
        embeddings, valid_mask = extract_prefix_tokens(observation)
        z_rl = np.asarray(apply_token(embeddings, valid_mask), dtype=np.float32)
        rng, sample_rng = jax.random.split(rng)
        a_ref_model = sample_actions(sample_rng, observation, num_steps=args.sample_num_steps)
        state_model = np.asarray(observation.state)
        a_ref = _absolute_actions_per_sample(output_transform, state_model, np.asarray(a_ref_model))
        state = _unnormalize_state(state_model, data_config.norm_stats)
        path = cache_dir / f"shard_{batch_idx:05d}.npz"
        np.savez_compressed(
            path,
            z_rl=z_rl,
            state=state,
            a_ref=a_ref,
            prefix_valid_tokens=np.asarray(jnp.sum(valid_mask, axis=-1)),
            source=np.asarray(["sft_full_20k"] * z_rl.shape[0]),
        )
        shards.append(str(path))
        total += z_rl.shape[0]
        logging.info("cached shard=%s batch=%d total=%d", path, batch_idx, total)
    manifest = {
        "cache_dir": str(cache_dir),
        "num_samples": total,
        "num_shards": len(shards),
        "shards": shards,
        "fields": {
            "z_rl": "float32 [B, token_dim]",
            "state": "float32 [B, 7] raw robot state",
            "a_ref": "float32 [B, 50, 7] absolute reference action chunk",
            "prefix_valid_tokens": "int [B]",
            "source": "sft_full_20k",
        },
        "elapsed_s": time.time() - start,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    jax.config.update("jax_compilation_cache_dir", args.jax_cache_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    train_config = _config.get_config(args.config_name)
    train_config = dataclasses.replace(
        train_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        wandb_enabled=False,
    )
    logging.info("loading frozen JAX policy config=%s checkpoint=%s", args.config_name, checkpoint_dir)
    frozen_model = _load_model(train_config, checkpoint_dir)
    extract_prefix_tokens = nnx_utils.module_jit(frozen_model.encode_prefix_tokens)
    sample_actions = nnx_utils.module_jit(frozen_model.sample_actions)

    loader = _data_loader.create_data_loader(train_config, shuffle=True, framework="jax")
    data_iter = iter(loader)
    first_observation, _ = next(data_iter)
    first_embeddings, first_valid_mask = extract_prefix_tokens(first_observation)
    embedding_dim = int(first_embeddings.shape[-1])
    seq_len = int(first_embeddings.shape[1])
    token_config = dataclasses.replace(
        args.token,
        embedding_dim=embedding_dim,
        max_seq_len=max(args.token.max_seq_len, seq_len),
    )
    logging.info("prefix embeddings shape=%s valid_tokens_mean=%.1f", first_embeddings.shape, float(jnp.mean(jnp.sum(first_valid_mask, axis=-1))))
    logging.info("token_config=%s", token_config)

    token_model = RLTokenAutoencoder(token_config)
    rng = jax.random.key(args.seed)
    rng, init_rng = jax.random.split(rng)
    variables = token_model.init({"params": init_rng, "dropout": init_rng}, first_embeddings, first_valid_mask, train=True)
    params = variables["params"]
    tx = optax.adamw(args.lr)
    opt_state = tx.init(params)
    train_step = _make_train_step(token_model, tx)

    metrics = {}
    pbar = tqdm.tqdm(range(args.num_train_steps), desc="JAX RLT token")
    start = time.time()
    pending_batch = (first_observation, None)
    for step in pbar:
        if step == 0:
            observation = pending_batch[0]
        else:
            try:
                observation, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                observation, _ = next(data_iter)
        embeddings, valid_mask = extract_prefix_tokens(observation)
        rng, step_rng = jax.random.split(rng)
        params, opt_state, metrics_jax = train_step(params, opt_state, step_rng, embeddings, valid_mask)
        metrics = {k: float(np.asarray(v)) for k, v in metrics_jax.items()}
        if step % args.log_interval == 0:
            elapsed = time.time() - start
            logging.info(
                "step=%d loss=%.6f grad_norm=%.4f z_norm=%.4f valid_tokens=%.1f %.2fs/step",
                step,
                metrics["loss"],
                metrics["grad_norm"],
                metrics["z_norm"],
                metrics["valid_tokens"],
                elapsed / max(args.log_interval, 1),
            )
            pbar.set_postfix({"loss": f"{metrics['loss']:.5f}", "z": f"{metrics['z_norm']:.2f}"})
            start = time.time()
        step_number = step + 1
        if step_number % args.save_interval == 0 or step_number == args.num_train_steps:
            ckpt_path = _save_checkpoint(
                output_dir,
                step=step_number,
                params=jax.device_get(params),
                token_config=token_config,
                args=args,
                metrics=metrics,
            )
            logging.info("saved RL token checkpoint: %s", ckpt_path)

    final_path = _save_checkpoint(
        output_dir,
        step=args.num_train_steps,
        params=jax.device_get(params),
        token_config=token_config,
        args=args,
        metrics=metrics,
    )
    logging.info("final RL token checkpoint: %s", final_path)
    if args.cache_after:
        manifest = _cache_rlt_inputs(
            args=args,
            output_dir=output_dir,
            train_config=train_config,
            token_model=token_model,
            token_params=params,
            extract_prefix_tokens=extract_prefix_tokens,
            sample_actions=sample_actions,
        )
        logging.info("cached z_rl manifest: %s", manifest)


if __name__ == "__main__":
    main(tyro.cli(Args))

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from openpi.models.pi0 import make_attn_mask
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _load_token_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("rlt_token_validation_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import token module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync(value):
    return jax.block_until_ready(value)


def _benchmark(fn, *args, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        _sync(fn(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _sync(fn(*args))
        times.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "repeats": int(repeats),
    }


def _encode_prefix_tokens(self, observation: _model.Observation):
    observation = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), _ = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )
    return prefix_out.astype(jnp.float32), prefix_mask


def _tree_stats(tree: Any) -> dict[str, int]:
    leaves = jax.tree.leaves(tree)
    arrays = [np.asarray(x) for x in leaves]
    return {
        "leaf_count": len(arrays),
        "parameter_count": int(sum(x.size for x in arrays)),
        "parameter_bytes": int(sum(x.nbytes for x in arrays)),
    }


def _token_statistics(z: np.ndarray) -> dict[str, float | int]:
    z = np.asarray(z, dtype=np.float64)
    feature_var = np.var(z, axis=0)
    centered = z - z.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(z, axis=1)
    centered_norms = np.linalg.norm(centered, axis=1)
    normalized = z / np.maximum(norms[:, None], 1e-12)
    cosine = normalized @ normalized.T
    off_diag = cosine[~np.eye(len(z), dtype=bool)] if len(z) > 1 else np.asarray([])
    pair_l2 = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    off_l2 = pair_l2[~np.eye(len(z), dtype=bool)] if len(z) > 1 else np.asarray([])
    return {
        "samples": int(z.shape[0]),
        "dim": int(z.shape[1]),
        "finite_fraction": float(np.isfinite(z).mean()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "centered_norm_mean": float(centered_norms.mean()),
        "feature_variance_mean": float(feature_var.mean()),
        "feature_variance_median": float(np.median(feature_var)),
        "feature_variance_max": float(feature_var.max()),
        "active_feature_fraction_var_gt_1e-6": float(np.mean(feature_var > 1e-6)),
        "rounded_unique_rows_1e-5": int(np.unique(np.round(z, 5), axis=0).shape[0]),
        "pairwise_cosine_mean": float(off_diag.mean()) if off_diag.size else 1.0,
        "pairwise_cosine_std": float(off_diag.std()) if off_diag.size else 0.0,
        "pairwise_l2_mean": float(off_l2.mean()) if off_l2.size else 0.0,
        "pairwise_l2_min": float(off_l2.min()) if off_l2.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--sft-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--token-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--token-module", type=pathlib.Path, required=True)
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--latency-repeats", type=int, default=10)
    args = parser.parse_args()

    started = time.time()
    metadata_path = args.token_checkpoint / "metadata.json"
    params_path = args.token_checkpoint / "params.msgpack"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    token_lib = _load_token_module(args.token_module)
    token_config = token_lib.RLTTokenJaxConfig(**metadata["token_config"])

    train_config = _config.get_config(args.config_name)
    asset_id = train_config.data.repo_id
    data_factory = dataclasses.replace(
        train_config.data,
        assets=_config.AssetsConfig(
            assets_dir=str(args.sft_checkpoint / "assets"),
            asset_id=asset_id,
        ),
    )
    train_config = dataclasses.replace(
        train_config,
        data=data_factory,
        batch_size=args.batch_size,
        num_workers=0,
        seed=20260710,
        wandb_enabled=False,
    )
    loader = _data_loader.create_data_loader(
        train_config,
        shuffle=False,
        num_batches=args.num_batches,
        framework="jax",
    )
    batches = list(loader)
    if not batches:
        raise RuntimeError("Evaluation loader returned no batches")

    params = _model.restore_params(args.sft_checkpoint / "params", dtype=jnp.bfloat16)
    frozen_model = train_config.model.load(params)
    if not hasattr(type(frozen_model), "encode_prefix_tokens"):
        type(frozen_model).encode_prefix_tokens = _encode_prefix_tokens
    extract_prefix = nnx_utils.module_jit(frozen_model.encode_prefix_tokens)
    sample_actions = nnx_utils.module_jit(frozen_model.sample_actions)

    first_observation, _ = batches[0]
    first_embeddings, first_valid = extract_prefix(first_observation)
    first_embeddings, first_valid = _sync((first_embeddings, first_valid))

    token_model = token_lib.RLTokenAutoencoder(token_config)
    init_rng = jax.random.key(0)
    template = token_model.init(
        {"params": init_rng, "dropout": init_rng},
        first_embeddings,
        first_valid,
        train=False,
    )["params"]
    token_params = serialization.from_bytes(template, params_path.read_bytes())

    @jax.jit
    def token_full(p, embeddings, valid):
        z, reconstruction = token_model.apply(
            {"params": p}, embeddings, valid, train=False
        )
        loss = token_lib.reconstruction_loss(reconstruction, embeddings, valid)
        zero_loss = jnp.sum(
            jnp.mean(jnp.square(embeddings), axis=-1) * valid.astype(jnp.float32)
        ) / jnp.maximum(jnp.sum(valid), 1.0)
        return z, loss, zero_loss

    @jax.jit
    def token_only(p, embeddings, valid):
        return token_model.apply({"params": p}, embeddings, valid, train=False)[0]

    z_parts = []
    reconstruction_losses = []
    zero_losses = []
    valid_counts = []
    eval_prefix_ms = []
    for observation, _ in batches:
        tick = time.perf_counter()
        embeddings, valid = _sync(extract_prefix(observation))
        eval_prefix_ms.append((time.perf_counter() - tick) * 1000.0)
        z, loss, zero_loss = _sync(token_full(token_params, embeddings, valid))
        z_parts.append(np.asarray(z, dtype=np.float32))
        reconstruction_losses.append(float(np.asarray(loss)))
        zero_losses.append(float(np.asarray(zero_loss)))
        valid_counts.extend(np.asarray(jnp.sum(valid, axis=-1), dtype=np.int32).tolist())

    z_all = np.concatenate(z_parts, axis=0)
    z_a = np.asarray(_sync(token_only(token_params, first_embeddings, first_valid)))
    z_b = np.asarray(_sync(token_only(token_params, first_embeddings, first_valid)))

    fixed_rng = jax.random.key(987654)
    action_before = np.asarray(_sync(sample_actions(fixed_rng, first_observation, num_steps=10)))
    _sync(token_only(token_params, first_embeddings, first_valid))
    action_after = np.asarray(_sync(sample_actions(fixed_rng, first_observation, num_steps=10)))

    prefix_latency = _benchmark(
        extract_prefix,
        first_observation,
        warmup=1,
        repeats=args.latency_repeats,
    )
    token_latency = _benchmark(
        token_only,
        token_params,
        first_embeddings,
        first_valid,
        warmup=2,
        repeats=args.latency_repeats,
    )
    action_latency = _benchmark(
        lambda observation: sample_actions(fixed_rng, observation, num_steps=10),
        first_observation,
        warmup=1,
        repeats=max(3, min(args.latency_repeats, 5)),
    )

    metadata_checkpoint = pathlib.PurePosixPath(metadata["args"]["checkpoint_dir"])
    local_checkpoint = args.sft_checkpoint
    checkpoint_suffix_matches = tuple(metadata_checkpoint.parts[-3:]) == tuple(local_checkpoint.parts[-3:])
    latest = args.token_checkpoint.parent / "latest"
    report = {
        "status": "evaluated",
        "timestamp_unix": time.time(),
        "elapsed_s": time.time() - started,
        "device": str(jax.devices()[0]),
        "checkpoint": {
            "path": str(args.token_checkpoint),
            "format": metadata.get("format"),
            "step": metadata.get("step"),
            "params_size_bytes": params_path.stat().st_size,
            "params_sha256": _sha256(params_path),
            "latest_resolves_to_checkpoint": latest.resolve() == args.token_checkpoint.resolve(),
            "tree": _tree_stats(token_params),
        },
        "alignment": {
            "requested_config_name": args.config_name,
            "metadata_config_name": metadata["args"]["config_name"],
            "config_name_matches": metadata["args"]["config_name"] == args.config_name,
            "metadata_sft_checkpoint": str(metadata_checkpoint),
            "local_sft_checkpoint": str(local_checkpoint),
            "sft_checkpoint_suffix_matches": checkpoint_suffix_matches,
            "data_repo_id": train_config.data.repo_id,
            "prompt": train_config.policy_metadata.get("prompt") if train_config.policy_metadata else None,
            "action_coordinate": train_config.policy_metadata.get("action_coordinate") if train_config.policy_metadata else None,
        },
        "evaluation_subset": {
            "kind": "deterministic shuffle=False audit subset; not a strict training-held-out split",
            "num_batches": len(batches),
            "num_samples": int(z_all.shape[0]),
            "batch_size": args.batch_size,
            "valid_tokens_mean": float(np.mean(valid_counts)),
            "valid_tokens_min": int(np.min(valid_counts)),
            "valid_tokens_max": int(np.max(valid_counts)),
        },
        "reconstruction": {
            "audit_loss_mean": float(np.mean(reconstruction_losses)),
            "audit_loss_std": float(np.std(reconstruction_losses)),
            "zero_prediction_loss_mean": float(np.mean(zero_losses)),
            "loss_over_zero_baseline": float(np.mean(reconstruction_losses) / np.mean(zero_losses)),
            "final_training_loss": float(metadata.get("metrics", {}).get("loss", float("nan"))),
        },
        "z_rl": _token_statistics(z_all),
        "determinism": {
            "exact_equal": bool(np.array_equal(z_a, z_b)),
            "max_abs_difference": float(np.max(np.abs(z_a - z_b))),
        },
        "sft_action_invariance": {
            "exact_equal": bool(np.array_equal(action_before, action_after)),
            "max_abs_difference": float(np.max(np.abs(action_before - action_after))),
            "shape": list(action_before.shape),
            "fixed_rng": 987654,
        },
        "latency": {
            "batch_size": args.batch_size,
            "prefix_encoder": prefix_latency,
            "token_only_from_prefix": token_latency,
            "sequential_prefix_plus_token_mean_ms": prefix_latency["mean_ms"] + token_latency["mean_ms"],
            "pi05_sample_actions_10_steps": action_latency,
            "initial_eval_prefix_ms": eval_prefix_ms,
        },
        "training_metadata_metrics": metadata.get("metrics", {}),
    }
    report["checks"] = {
        "load_ok": True,
        "finite_z": report["z_rl"]["finite_fraction"] == 1.0,
        "noncollapsed_z": report["z_rl"]["rounded_unique_rows_1e-5"] > 1
        and report["z_rl"]["feature_variance_mean"] > 1e-6,
        "deterministic": report["determinism"]["exact_equal"],
        "sft_action_unchanged": report["sft_action_invariance"]["exact_equal"],
        "config_aligned": report["alignment"]["config_name_matches"]
        and report["alignment"]["sft_checkpoint_suffix_matches"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

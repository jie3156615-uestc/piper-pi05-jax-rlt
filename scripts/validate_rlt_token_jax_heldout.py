from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from flax import linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import openpi.models.model as _model
from openpi.policies import policy_config
from openpi.rlt.jax_token import _DecoderBlock
from openpi.rlt.jax_token import _EncoderBlock
from openpi.rlt.jax_token import reconstruction_loss
from openpi.rlt.jax_token import RLTokenAutoencoder
from openpi.rlt.jax_token import RLTTokenJaxConfig
from openpi.training import config as _config


@dataclasses.dataclass(frozen=True)
class HeldoutSample:
    episode_jsonl: str
    t: int
    global_image: str
    wrist_image: str
    state: np.ndarray


class RLTokenEncoderOnly(nn.Module):
    config: RLTTokenJaxConfig

    @nn.compact
    def __call__(self, embeddings: jax.Array, valid_mask: jax.Array) -> jax.Array:
        cfg = self.config
        batch_size, seq_len, dim = embeddings.shape
        if dim != cfg.embedding_dim:
            raise ValueError(f"Expected embedding_dim={cfg.embedding_dim}, got {dim}")
        rl_token = self.param("rl_token", nn.initializers.normal(0.02), (1, 1, dim))
        encoder_pos = self.param("encoder_pos", nn.initializers.normal(0.02), (1, cfg.max_seq_len + 1, dim))
        token = jnp.broadcast_to(rl_token, (batch_size, 1, dim))
        x = jnp.concatenate([embeddings, token], axis=1)
        x = x + encoder_pos[:, : seq_len + 1]
        encoder_valid = jnp.concatenate([valid_mask, jnp.ones((batch_size, 1), dtype=bool)], axis=1)
        for layer in range(cfg.num_encoder_layers):
            x = _EncoderBlock(cfg, name=f"encoder_{layer}")(x, encoder_valid, train=False)
        return nn.LayerNorm(name="z_norm")(x[:, -1])


class RLTokenDecoderOnly(nn.Module):
    config: RLTTokenJaxConfig

    @nn.compact
    def __call__(
        self,
        embeddings: jax.Array,
        valid_mask: jax.Array,
        z_rl: jax.Array,
    ) -> jax.Array:
        cfg = self.config
        batch_size, seq_len, dim = embeddings.shape
        decoder_pos = self.param("decoder_pos", nn.initializers.normal(0.02), (1, cfg.max_seq_len, dim))
        start = jnp.zeros((batch_size, 1, dim), dtype=embeddings.dtype)
        decoder_in = jnp.concatenate([start, jax.lax.stop_gradient(embeddings[:, :-1])], axis=1)
        y = decoder_in + decoder_pos[:, :seq_len]
        for layer in range(cfg.num_decoder_layers):
            y = _DecoderBlock(cfg, name=f"decoder_{layer}")(y, z_rl, valid_mask, train=False)
        return nn.Dense(dim, name="output")(nn.LayerNorm(name="decoder_out_norm")(y))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def collect_heldout_samples(root: Path, *, max_samples: int) -> list[HeldoutSample]:
    samples: list[HeldoutSample] = []
    episode_paths = sorted(root.rglob("episode.jsonl"), key=lambda path: path.as_posix())
    fractions = (0.2, 0.5, 0.8)
    for episode_index, episode_path in enumerate(episode_paths):
        rows = []
        for row in _read_rows(episode_path):
            global_rel = row.get("global_image")
            wrist_rel = row.get("wrist_image")
            state = np.asarray(row.get("state", []), dtype=np.float32)
            if not global_rel or not wrist_rel or state.shape != (7,) or not np.all(np.isfinite(state)):
                continue
            global_path = episode_path.parent / str(global_rel)
            wrist_path = episode_path.parent / str(wrist_rel)
            if global_path.is_file() and wrist_path.is_file():
                rows.append((row, global_path, wrist_path, state))
        if not rows:
            continue
        fraction = fractions[episode_index % len(fractions)]
        selected = rows[min(int(round((len(rows) - 1) * fraction)), len(rows) - 1)]
        row, global_path, wrist_path, state = selected
        samples.append(
            HeldoutSample(
                episode_jsonl=str(episode_path),
                t=int(row.get("t", 0)),
                global_image=str(global_path),
                wrist_image=str(wrist_path),
                state=state,
            )
        )
        if len(samples) >= max_samples:
            break
    if len(samples) < 2:
        raise RuntimeError(f"Need at least two held-out samples, found {len(samples)} under {root}")
    return samples


def _raw_observation(sample: HeldoutSample, prompt: str) -> dict[str, Any]:
    with Image.open(sample.global_image) as image:
        global_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(sample.wrist_image) as image:
        wrist_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return {
        "observation/image": global_image,
        "observation/wrist_image": wrist_image,
        "observation/state": sample.state.astype(np.float32, copy=True),
        "prompt": prompt,
    }


def _preprocess(policy: Any, raw_observation: dict[str, Any]) -> _model.Observation:
    inputs = policy._input_transform(jax.tree.map(lambda value: value, raw_observation))
    inputs = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], inputs)
    return _model.Observation.from_dict(inputs)


def _tree_bytes(tree: Any) -> int:
    return int(sum(math.prod(leaf.shape) * np.dtype(leaf.dtype).itemsize for leaf in jax.tree_util.tree_leaves(tree)))


def _cosine_distance_summary(z_values: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(z_values, axis=-1, keepdims=True)
    normalized = z_values / np.maximum(norms, 1e-12)
    similarity = normalized @ normalized.T
    distances = 1.0 - similarity
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    return {
        "min": float(np.min(upper)),
        "mean": float(np.mean(upper)),
        "max": float(np.max(upper)),
    }


def _effective_rank(z_values: np.ndarray) -> float:
    centered = z_values - np.mean(z_values, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular_values)
    if float(np.sum(energy)) <= 0.0:
        return 0.0
    probabilities = energy / np.sum(energy)
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _block_until_ready(value: Any) -> Any:
    return jax.tree.map(lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, value)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    token_dir = args.token_checkpoint.resolve()
    metadata_path = token_dir / "metadata.json"
    params_path = token_dir / "params.msgpack"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    token_config = RLTTokenJaxConfig(**metadata["token_config"])
    if metadata.get("format") != "jax_flax_rl_token_autoencoder_v1":
        raise RuntimeError(f"Unexpected token checkpoint format: {metadata.get('format')}")

    token_model = RLTokenAutoencoder(token_config)
    init_embeddings = jnp.zeros((1, 2, token_config.embedding_dim), dtype=jnp.float32)
    init_mask = jnp.ones((1, 2), dtype=bool)
    template = token_model.init(jax.random.key(0), init_embeddings, init_mask, train=False)["params"]
    params = serialization.from_bytes(template, params_path.read_bytes())

    encoder_names = {"rl_token", "encoder_pos", "z_norm"} | {
        f"encoder_{index}" for index in range(token_config.num_encoder_layers)
    }
    decoder_names = {"decoder_pos", "decoder_out_norm", "output"} | {
        f"decoder_{index}" for index in range(token_config.num_decoder_layers)
    }
    encoder_params = {name: params[name] for name in encoder_names}
    decoder_params = {name: params[name] for name in decoder_names}
    encoder_model = RLTokenEncoderOnly(token_config)
    decoder_model = RLTokenDecoderOnly(token_config)
    encode_token = jax.jit(lambda embeddings, valid: encoder_model.apply({"params": encoder_params}, embeddings, valid))
    decode_token = jax.jit(
        lambda embeddings, valid, z_rl: decoder_model.apply(
            {"params": decoder_params}, embeddings, valid, z_rl
        )
    )

    train_config = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        train_config,
        args.policy_checkpoint,
        default_prompt=args.prompt,
    )
    extract_prefix = jax.jit(policy._model.encode_prefix_tokens)
    sample_actions = jax.jit(policy._model.sample_actions, static_argnames=("num_steps",))
    samples = collect_heldout_samples(args.episodes_root, max_samples=args.max_samples)

    z_values: list[np.ndarray] = []
    reconstruction_losses: list[float] = []
    zero_z_losses: list[float] = []
    zero_output_losses: list[float] = []
    valid_token_counts: list[int] = []
    encoded: list[tuple[jax.Array, jax.Array, jax.Array]] = []
    observations: list[_model.Observation] = []
    prefix_times_ms: list[float] = []
    token_times_ms: list[float] = []

    for sample in samples:
        observation = _preprocess(policy, _raw_observation(sample, args.prompt))
        observations.append(observation)
        started = time.perf_counter()
        embeddings, valid_mask = extract_prefix(observation)
        _block_until_ready((embeddings, valid_mask))
        prefix_times_ms.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        z_rl = encode_token(embeddings, valid_mask)
        _block_until_ready(z_rl)
        token_times_ms.append((time.perf_counter() - started) * 1000.0)
        reconstruction = decode_token(embeddings, valid_mask, z_rl)
        zero_reconstruction = decode_token(embeddings, valid_mask, jnp.zeros_like(z_rl))
        loss = reconstruction_loss(reconstruction, embeddings, valid_mask)
        zero_z_loss = reconstruction_loss(zero_reconstruction, embeddings, valid_mask)
        zero_output_loss = reconstruction_loss(jnp.zeros_like(embeddings), embeddings, valid_mask)
        _block_until_ready((loss, zero_z_loss, zero_output_loss))
        z_values.append(np.asarray(z_rl[0], dtype=np.float32))
        reconstruction_losses.append(float(np.asarray(loss)))
        zero_z_losses.append(float(np.asarray(zero_z_loss)))
        zero_output_losses.append(float(np.asarray(zero_output_loss)))
        valid_token_counts.append(int(np.asarray(jnp.sum(valid_mask))))
        encoded.append((embeddings, valid_mask, z_rl))

    shuffled_z_losses = []
    for index, (embeddings, valid_mask, _z_rl) in enumerate(encoded):
        shuffled_z = encoded[(index + 1) % len(encoded)][2]
        shuffled_reconstruction = decode_token(embeddings, valid_mask, shuffled_z)
        shuffled_loss = reconstruction_loss(shuffled_reconstruction, embeddings, valid_mask)
        _block_until_ready(shuffled_loss)
        shuffled_z_losses.append(float(np.asarray(shuffled_loss)))

    first_embeddings, first_mask, first_z = encoded[0]
    repeat_z = encode_token(first_embeddings, first_mask)
    _block_until_ready(repeat_z)
    deterministic_max_abs = float(np.max(np.abs(np.asarray(first_z) - np.asarray(repeat_z))))

    for _ in range(2):
        _block_until_ready(encode_token(first_embeddings, first_mask))
    cached_token_times = []
    for _ in range(args.benchmark_iters):
        started = time.perf_counter()
        value = encode_token(first_embeddings, first_mask)
        _block_until_ready(value)
        cached_token_times.append((time.perf_counter() - started) * 1000.0)

    action_key = jax.random.key(args.action_seed)
    before_actions = sample_actions(action_key, observations[0], num_steps=args.action_steps)
    _block_until_ready(before_actions)
    _block_until_ready(encode_token(first_embeddings, first_mask))
    after_actions = sample_actions(action_key, observations[0], num_steps=args.action_steps)
    _block_until_ready(after_actions)
    action_max_abs_diff = float(np.max(np.abs(np.asarray(before_actions) - np.asarray(after_actions))))

    z_array = np.stack(z_values)
    reconstruction_mean = float(np.mean(reconstruction_losses))
    zero_z_mean = float(np.mean(zero_z_losses))
    shuffled_z_mean = float(np.mean(shuffled_z_losses))
    training_loss = float(metadata.get("metrics", {}).get("loss", float("nan")))
    heldout_loss_limit = max(0.25, training_loss * 2.0) if np.isfinite(training_loss) else 0.25
    checks = {
        "artifact_complete": metadata_path.is_file() and params_path.is_file(),
        "metadata_base_matches_requested": metadata["args"]["config_name"] == args.config_name
        and Path(metadata["args"]["checkpoint_dir"]).name == args.policy_checkpoint.name,
        "all_outputs_finite": bool(
            np.all(np.isfinite(z_array))
            and np.all(np.isfinite(reconstruction_losses))
            and np.all(np.isfinite(shuffled_z_losses))
        ),
        "deterministic": deterministic_max_abs <= 1e-6,
        "not_constant": float(np.mean(np.std(z_array, axis=0))) > 1e-5,
        "heldout_reconstruction_reasonable": reconstruction_mean <= heldout_loss_limit,
        "decoder_uses_z_vs_zero": reconstruction_mean < zero_z_mean * 0.99,
        "decoder_uses_sample_specific_z": reconstruction_mean < shuffled_z_mean * 0.99,
        "base_action_unchanged": action_max_abs_diff <= 1e-6,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "artifact": {
            "token_checkpoint": str(token_dir),
            "params_bytes": params_path.stat().st_size,
            "params_sha256": _sha256(params_path),
            "metadata_sha256": _sha256(metadata_path),
            "metadata": metadata,
            "parameter_bytes_total": _tree_bytes(params),
            "encoder_parameter_bytes": _tree_bytes(encoder_params),
            "decoder_parameter_bytes": _tree_bytes(decoder_params),
        },
        "base_policy": {
            "config_name": args.config_name,
            "checkpoint": str(args.policy_checkpoint),
            "prompt": args.prompt,
            "action_steps": args.action_steps,
            "action_max_abs_diff_after_token_eval": action_max_abs_diff,
        },
        "heldout": {
            "episodes_root": str(args.episodes_root),
            "num_samples": len(samples),
            "samples": [
                {
                    "episode_jsonl": sample.episode_jsonl,
                    "t": sample.t,
                    "global_image": sample.global_image,
                    "wrist_image": sample.wrist_image,
                }
                for sample in samples
            ],
            "valid_tokens": valid_token_counts,
            "training_final_loss": training_loss,
            "heuristic_heldout_loss_limit": heldout_loss_limit,
            "reconstruction_loss": {
                "mean": reconstruction_mean,
                "min": float(np.min(reconstruction_losses)),
                "max": float(np.max(reconstruction_losses)),
            },
            "zero_z_loss_mean": zero_z_mean,
            "shuffled_z_loss_mean": shuffled_z_mean,
            "zero_output_loss_mean": float(np.mean(zero_output_losses)),
            "actual_vs_zero_z_improvement_fraction": float((zero_z_mean - reconstruction_mean) / zero_z_mean),
            "actual_vs_shuffled_z_improvement_fraction": float(
                (shuffled_z_mean - reconstruction_mean) / shuffled_z_mean
            ),
        },
        "z_rl": {
            "shape": list(z_array.shape),
            "norm_mean": float(np.mean(np.linalg.norm(z_array, axis=-1))),
            "norm_min": float(np.min(np.linalg.norm(z_array, axis=-1))),
            "norm_max": float(np.max(np.linalg.norm(z_array, axis=-1))),
            "mean_dimension_std": float(np.mean(np.std(z_array, axis=0))),
            "max_dimension_std": float(np.max(np.std(z_array, axis=0))),
            "effective_rank": _effective_rank(z_array),
            "pairwise_cosine_distance": _cosine_distance_summary(z_array),
            "deterministic_max_abs_diff": deterministic_max_abs,
        },
        "latency_ms": {
            "prefix_first_or_cached_per_sample": prefix_times_ms,
            "token_first_or_cached_per_sample": token_times_ms,
            "token_cached_mean": float(np.mean(cached_token_times)),
            "token_cached_p95": float(np.percentile(cached_token_times, 95)),
            "token_cached_max": float(np.max(cached_token_times)),
            "benchmark_iters": args.benchmark_iters,
        },
        "environment": {
            "jax_version": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a JAX Piper RL-token checkpoint on held-out real episodes.")
    parser.add_argument("--token-checkpoint", type=Path, required=True)
    parser.add_argument("--config-name", default="pi05_piper_greenblock_5090_jax_delta_v1")
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--prompt", default="Put the green block into the box.")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--benchmark-iters", type=int, default=10)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--action-seed", type=int, default=123)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

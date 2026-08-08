from __future__ import annotations

import os
import pathlib
from typing import Any

import jax
import numpy as np
import safetensors.torch
import torch

from openpi import transforms
from openpi.models import model as _model
import openpi.models.pi0_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


class RLTPolicyInputs:
    """Owns openpi transforms and exposes tensors needed by RLT."""

    def __init__(
        self,
        train_config: _config.TrainConfig,
        checkpoint_dir: str | pathlib.Path,
        *,
        device: torch.device,
        default_prompt: str | None = None,
        sample_num_steps: int = 10,
    ):
        self.train_config = train_config
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.device = device
        self.sample_num_steps = sample_num_steps

        model_cfg = train_config.model
        if not isinstance(model_cfg, openpi.models.pi0_config.Pi0Config):
            raise TypeError("RLT currently supports Pi0Config / pi0.5 PyTorch models only.")
        object.__setattr__(model_cfg, "dtype", train_config.pytorch_training_precision)
        object.__setattr__(model_cfg, "pytorch_compile_mode", None)
        self.model = PI0Pytorch(model_cfg).to(device)
        weight_path = self.checkpoint_dir / "model.safetensors"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Expected a PyTorch checkpoint at {weight_path}. Convert the JAX checkpoint first with "
                "`examples/convert_jax_model_to_pytorch.py`, or pass a PyTorch fine-tuned checkpoint."
            )
        safetensors.torch.load_model(self.model, weight_path, device=str(device))
        self.model.eval()

        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        if data_config.asset_id is None:
            raise ValueError("Data config must define asset_id so norm stats can be loaded.")
        norm_stats = _checkpoints.load_norm_stats(self.checkpoint_dir / "assets", data_config.asset_id)
        self.input_transform = transforms.compose(
            [
                transforms.InjectDefaultPrompt(default_prompt),
                *data_config.data_transforms.inputs,
                transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        self.output_transform = transforms.compose(
            [
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ]
        )

    def transform_observation(self, element: dict[str, Any]) -> tuple[_model.Observation, np.ndarray]:
        raw_state = np.asarray(element["observation/state"], dtype=np.float32)
        inputs = self.input_transform(jax.tree.map(lambda x: x, element))
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self.device)[None, ...], inputs)
        return _model.Observation.from_dict(inputs), raw_state

    def transform_observations(self, elements: list[dict[str, Any]]) -> tuple[_model.Observation, np.ndarray]:
        if not elements:
            raise ValueError("Cannot transform an empty observation batch.")
        raw_states = np.stack([np.asarray(element["observation/state"], dtype=np.float32) for element in elements])
        transformed = [self.input_transform(jax.tree.map(lambda x: x, element)) for element in elements]
        batched = jax.tree.map(lambda *xs: np.stack([np.array(x) for x in xs], axis=0), *transformed)
        batched = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self.device), batched)
        return _model.Observation.from_dict(batched), raw_states

    @torch.no_grad()
    def sample_reference_action(self, observation: _model.Observation) -> np.ndarray:
        return self.sample_reference_actions(observation)[0]

    @torch.no_grad()
    def sample_reference_actions(self, observation: _model.Observation) -> np.ndarray:
        action = self.model.sample_actions(self.device, observation, num_steps=self.sample_num_steps)
        states = np.asarray(observation.state.detach().cpu())
        actions = np.asarray(action.detach().cpu())
        outputs = []
        for state, action_chunk in zip(states, actions, strict=True):
            transformed = self.output_transform({"state": state, "actions": action_chunk})
            outputs.append(np.asarray(transformed["actions"], dtype=np.float32))
        return np.stack(outputs).astype(np.float32)

    @torch.no_grad()
    def extract_rl_token(self, observation: _model.Observation, rl_token_module: torch.nn.Module) -> np.ndarray:
        return self.extract_rl_tokens(observation, rl_token_module)[0]

    @torch.no_grad()
    def extract_rl_tokens(self, observation: _model.Observation, rl_token_module: torch.nn.Module) -> np.ndarray:
        embeddings, padding_mask = self.model.encode_prefix_tokens(observation, train=False)
        rl_token = rl_token_module.encode(embeddings, padding_mask=~padding_mask)
        return np.asarray(rl_token.detach().cpu(), dtype=np.float32)

    def env_action_to_model_output(self, action_chunk: np.ndarray) -> dict[str, np.ndarray]:
        return {"actions": np.asarray(action_chunk, dtype=np.float32)}

    @torch.no_grad()
    def _sample_reference_action_legacy(self, observation: _model.Observation) -> np.ndarray:
        action = self.model.sample_actions(self.device, observation, num_steps=self.sample_num_steps)
        outputs = {
            "state": observation.state,
            "actions": action,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        return np.asarray(self.output_transform(outputs)["actions"], dtype=np.float32)


def resolve_local_checkpoint(path: str) -> pathlib.Path:
    if path.startswith("gs://"):
        raise ValueError(
            "RLT PyTorch training needs a local PyTorch checkpoint directory. Download/convert the checkpoint first, "
            f"then pass its local path instead of {path!r}."
        )
    local_path = pathlib.Path(os.path.expanduser(path)).resolve()
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    return local_path

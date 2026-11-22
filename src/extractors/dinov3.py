"""DINOv3 spatial patch-token extractor producing (B, C, H_p, W_p) tensors."""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
import torch

from src.config import DatasetConfig, DINOv3ExtractorConfig

logger = logging.getLogger(__name__)

_model_cache: dict[str, tuple[Any, Any]] = {}
_cache_lock = threading.Lock()


def compute_aligned_dinov3_grid(
    orig_h: int,
    orig_w: int,
    patch_size: int,
    stride: int,
    stride_factor: int = 1,
    target_short_edge: int | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute aligned (target_h, target_w) pixels and (h_patches, w_patches) grid."""
    if target_short_edge is not None and target_short_edge > 0:
        scale = float(target_short_edge) / float(min(orig_h, orig_w))
        scaled_h = round(orig_h * scale)
        scaled_w = round(orig_w * scale)
    else:
        scaled_h, scaled_w = orig_h, orig_w

    raw_nh = max(1, (scaled_h - patch_size) // stride + 1)
    raw_nw = max(1, (scaled_w - patch_size) // stride + 1)
    n_h = max(stride_factor, (raw_nh // stride_factor) * stride_factor)
    n_w = max(stride_factor, (raw_nw // stride_factor) * stride_factor)

    target_h = patch_size + (n_h - 1) * stride
    target_w = patch_size + (n_w - 1) * stride
    return (target_h, target_w), (n_h, n_w)


class DINOv3Extractor:
    """Extract L2-normalized DINOv3 spatial patch-token maps (B, C, H_p, W_p)."""

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        extractor_cfg: DINOv3ExtractorConfig,
        device: torch.device,
        stride_factor: int = 1,
    ) -> None:
        self.dataset_cfg = dataset_cfg
        self.extractor_cfg = extractor_cfg
        self.device = device
        self.stride_factor = max(1, stride_factor)

        self.patch_size = extractor_cfg.patch_size
        self.stride = extractor_cfg.stride
        self.prune_registers = extractor_cfg.prune_registers
        self.mean_centering_alpha = extractor_cfg.mean_centering_alpha
        self.target_short_edge = extractor_cfg.target_short_edge
        self.layers: tuple[int, ...] | None = (
            tuple(sorted(set(extractor_cfg.layers))) if extractor_cfg.layers else None
        )

        self._components_cache: tuple[Any, Any] | None = None
        self._prefix_tokens: int | None = None
        self._logged_dims = False

        (self.target_h, self.target_w), (self.h_patches, self.w_patches) = (
            compute_aligned_dinov3_grid(
                dataset_cfg.image_size[0],
                dataset_cfg.image_size[1],
                patch_size=self.patch_size,
                stride=self.stride,
                stride_factor=self.stride_factor,
                target_short_edge=self.target_short_edge,
            )
        )

        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(
            1, 3, 1, 1
        )
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(
            1, 3, 1, 1
        )
        self.pooler: torch.nn.AvgPool2d | None = (
            None
            if extractor_cfg.plain_last_hidden_state
            else torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
        )

    @property
    def _components(self) -> tuple[Any, Any]:
        if self._components_cache is not None:
            return self._components_cache

        cache_key = (
            f"dinov3_{self.extractor_cfg.model_name}_s{self.stride}_"
            f"pr{int(self.prune_registers)}_{self.device}"
        )
        if cache_key not in _model_cache:
            with _cache_lock:
                if cache_key not in _model_cache:
                    _model_cache[cache_key] = self._load_model()

        model, processor = _model_cache[cache_key]
        self._apply_patch_stride(model)
        self._configure_register_pruning(model)
        if (
            processor is not None
            and hasattr(processor, "image_mean")
            and hasattr(processor, "image_std")
        ):
            self._mean = torch.tensor(
                processor.image_mean, device=self.device, dtype=torch.float32
            ).view(1, 3, 1, 1)
            self._std = torch.tensor(
                processor.image_std, device=self.device, dtype=torch.float32
            ).view(1, 3, 1, 1)

        num_registers = int(
            getattr(
                model,
                "_native_num_registers",
                getattr(
                    model.config,
                    "num_register_tokens",
                    getattr(model.config, "num_storage_tokens", 0),
                ),
            )
        )
        active_registers = 0 if self.prune_registers else num_registers
        self._prefix_tokens = 1 + active_registers
        self._components_cache = (model, processor)
        return self._components_cache

    def _load_model(self) -> tuple[Any, Any]:
        from transformers import AutoModel  # pyright: ignore[reportMissingImports]

        model_name = self.extractor_cfg.model_name
        try:
            from transformers import (  # pyright: ignore[reportMissingImports]
                AutoImageProcessor,
            )

            processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        except (ImportError, RuntimeError, OSError, ValueError, AttributeError) as err:
            logger.warning(f"Failed to load processor for {model_name} ({err}); using ImageNet.")
            processor = None

        model = AutoModel.from_pretrained(model_name)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        model.to(self.device)

        num_registers = int(
            getattr(
                model.config,
                "num_register_tokens",
                getattr(model.config, "num_storage_tokens", 0),
            )
        )
        model._native_num_registers = num_registers
        self._apply_patch_stride(model)
        self._configure_register_pruning(model)
        return model, processor

    def _apply_patch_stride(self, model: Any) -> None:
        target_stride = (self.stride, self.stride)
        for mod in model.modules():
            if (
                isinstance(mod, torch.nn.Conv2d)
                and mod.kernel_size == (self.patch_size, self.patch_size)
            ):
                if mod.stride != target_stride:
                    mod.stride = target_stride
                break

        rope_mod = getattr(
            model,
            "rope_embeddings",
            getattr(getattr(model, "embeddings", None), "rope_embeddings", None),
        )
        if rope_mod is not None and not getattr(rope_mod, "_stride_patched", False):
            orig_rope_forward = rope_mod.forward

            def _patched_rope_forward(pixel_values: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
                if self.stride != self.patch_size and pixel_values.ndim == 4:
                    h_in, w_in = pixel_values.shape[2], pixel_values.shape[3]
                    h_p = (h_in - self.patch_size) // self.stride + 1
                    w_p = (w_in - self.patch_size) // self.stride + 1
                    dummy_pv = pixel_values[:, :, :1, :1].expand(
                        pixel_values.shape[0],
                        pixel_values.shape[1],
                        h_p * self.patch_size,
                        w_p * self.patch_size,
                    )
                    return orig_rope_forward(dummy_pv, *args, **kwargs)
                return orig_rope_forward(pixel_values, *args, **kwargs)

            rope_mod.forward = _patched_rope_forward
            rope_mod._stride_patched = True

    def _configure_register_pruning(self, model: Any) -> None:
        n_reg = int(
            getattr(
                model,
                "_native_num_registers",
                getattr(
                    model.config,
                    "num_register_tokens",
                    getattr(model.config, "num_storage_tokens", 0),
                ),
            )
        )
        model._native_num_registers = n_reg
        model._prune_registers_active = bool(self.prune_registers and n_reg > 0)

        embeddings_mod = getattr(model, "embeddings", None)
        if embeddings_mod is None:
            return

        for mod in model.modules():
            if mod is embeddings_mod:
                continue
            if hasattr(mod, "num_register_tokens"):
                mod.num_register_tokens = 0 if model._prune_registers_active else n_reg

        if not hasattr(embeddings_mod, "_register_prune_hook_handle"):

            def _prune_registers_forward_hook(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
            ) -> Any:
                if not getattr(model, "_prune_registers_active", False):
                    return output
                nr = int(getattr(model, "_native_num_registers", 0))
                if nr <= 0:
                    return output

                if (
                    isinstance(output, torch.Tensor)
                    and output.ndim == 3
                    and output.shape[1] > 1 + nr
                ):
                    return torch.cat([output[:, :1, :], output[:, 1 + nr :, :]], dim=1)

                if (
                    isinstance(output, tuple)
                    and len(output) > 0
                    and isinstance(output[0], torch.Tensor)
                    and output[0].ndim == 3
                    and output[0].shape[1] > 1 + nr
                ):
                    pruned = torch.cat([output[0][:, :1, :], output[0][:, 1 + nr :, :]], dim=1)
                    return (pruned, *output[1:])
                return output

            embeddings_mod._register_prune_hook_handle = embeddings_mod.register_forward_hook(
                _prune_registers_forward_hook
            )

    def _apply_mean_centering_and_norm(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.mean_centering_alpha > 0.0:
            spatial_mean = tokens.mean(dim=1, keepdim=True)
            tokens = tokens - (self.mean_centering_alpha * spatial_mean)
        return torch.nn.functional.normalize(tokens, p=2.0, dim=-1)

    def _select_patch_tokens(self, model: Any, outputs: Any) -> torch.Tensor:
        num_spatial = self.h_patches * self.w_patches
        seq_len = outputs.last_hidden_state.shape[1]
        if num_spatial > 0 and seq_len >= num_spatial:
            slice_start = seq_len - num_spatial
            self._prefix_tokens = slice_start
        else:
            slice_start = self._prefix_tokens if self._prefix_tokens is not None else 1

        if self.layers is None:
            raw_tokens = outputs.last_hidden_state[:, slice_start:, :].float()
            return self._apply_mean_centering_and_norm(raw_tokens)

        hidden_states = outputs.hidden_states
        max_idx = len(hidden_states) - 1
        layernorm = getattr(model, "layernorm", getattr(model, "norm", None))

        selected_features: list[torch.Tensor] = []
        for idx in self.layers:
            if idx > max_idx:
                raise IndexError(f"Invalid layer index {idx}. Model has {max_idx} layers.")
            layer_feats = hidden_states[idx][:, slice_start:, :].float()
            if layernorm is not None:
                layer_feats = layernorm(layer_feats)
            else:
                layer_feats = torch.nn.functional.layer_norm(layer_feats, (layer_feats.shape[-1],))
            selected_features.append(self._apply_mean_centering_and_norm(layer_feats))

        return torch.cat(selected_features, dim=-1)

    @property
    def out_channels(self) -> int:
        model, _ = self._components
        num_layers = len(self.layers) if self.layers else 1
        return int(model.config.hidden_size) * num_layers

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return (self.h_patches, self.w_patches)

    @property
    def is_unit_range(self) -> bool:
        """L2-normalized patch embeddings span [-1, 1], not [0, 1]."""
        return False

    @property
    def cache_features_for_training(self) -> bool:
        """Frozen DINOv3 features are deterministic and should be cached once per fit()."""
        return True

    def normalize_uint8_tensor(self, tensor_nchw_u8: torch.Tensor) -> torch.Tensor:
        """Extract (B, C, H_p, W_p) float32 DINOv3 features from (B, 3, H, W) uint8 RGB."""
        model, _ = self._components
        x = tensor_nchw_u8.to(device=self.device, dtype=torch.float32).div_(255.0)
        batch_size, _, orig_h, orig_w = x.shape

        if (orig_h, orig_w) != (self.target_h, self.target_w):
            x = torch.nn.functional.interpolate(
                x,
                size=(self.target_h, self.target_w),
                mode="bicubic",
                align_corners=False,
            )

        normalized = (x - self._mean) / self._std

        if not self._logged_dims:
            logger.info(
                f"DINOv3 grid: ({self.target_h}x{self.target_w}) -> "
                f"({self.h_patches}x{self.w_patches}) patches "
                f"(patch={self.patch_size}, stride={self.stride}, "
                f"prune_registers={self.prune_registers})"
            )
            self._logged_dims = True

        with torch.no_grad():
            outputs = model(
                pixel_values=normalized,
                output_hidden_states=(self.layers is not None),
            )
            patch_tokens = self._select_patch_tokens(model, outputs)
            channels = patch_tokens.shape[-1]
            feature_map = (
                patch_tokens.transpose(1, 2)
                .contiguous()
                .view(batch_size, channels, self.h_patches, self.w_patches)
            )
            if self.pooler is not None:
                feature_map = self.pooler(feature_map)
                feature_map = torch.nn.functional.normalize(feature_map, p=2.0, dim=1)
            elif self.layers is not None:
                feature_map = torch.nn.functional.normalize(feature_map, p=2.0, dim=1)

        return feature_map.contiguous()

    def extract_batch(self, batch_images: np.ndarray) -> torch.Tensor:
        """Convert (B, H, W, 3) uint8 RGB numpy batch to (B, C, H_p, W_p) DINOv3 tensor."""
        if batch_images.ndim != 4 or batch_images.shape[-1] != 3:
            raise ValueError(
                f"Expected (B, H, W, 3) RGB batch for DINOv3, got shape {batch_images.shape}"
            )
        contiguous = np.ascontiguousarray(batch_images)
        tensor_u8 = torch.from_numpy(contiguous).to(device=self.device, non_blocking=True)
        return self.normalize_uint8_tensor(tensor_u8.permute(0, 3, 1, 2))

    def to_uint8_images(self, tensor: torch.Tensor) -> np.ndarray:
        """Project (B, C, H_p, W_p) feature tensor to 3-group RGB uint8 visualization."""
        feat = tensor.detach().float()
        groups = torch.tensor_split(feat, 3, dim=1)
        rgb = torch.cat([g.mean(dim=1, keepdim=True) for g in groups], dim=1)

        b = rgb.shape[0]
        flat = rgb.view(b, -1)
        mins = flat.min(dim=1).values.view(b, 1, 1, 1)
        maxs = flat.max(dim=1).values.view(b, 1, 1, 1)
        rgb_norm = (rgb - mins) / (maxs - mins).clamp(min=1e-6)

        target_hw = self.dataset_cfg.image_size
        if rgb_norm.shape[-2:] != target_hw:
            rgb_norm = torch.nn.functional.interpolate(
                rgb_norm,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        vis = rgb_norm.clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8)
        return np.ascontiguousarray(vis.permute(0, 2, 3, 1).cpu().numpy())

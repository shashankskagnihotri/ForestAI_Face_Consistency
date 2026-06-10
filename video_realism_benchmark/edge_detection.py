"""DexiNed edge probability inference for sampled frames."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import warnings

import cv2
import numpy as np

from . import config
from .frame_sampling import SampledFrame, load_sampled_frame_rgb


@dataclass(frozen=True)
class EdgeMapResult:
    """Saved edge probability outputs for one sampled frame."""

    frame_index: int
    timestamp_sec: float
    edge_png_path: Path
    edge_npy_path: Path
    probability_map: np.ndarray


def _import_torch_and_dexined() -> tuple[object, type]:
    """Import the real DexiNed implementation and fail clearly if unavailable."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for DexiNed edge detection. Install the project "
            "requirements inside the conda environment."
        ) from exc

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"kornia\..*")
            from kornia.filters.dexined import DexiNed
    except ImportError as exc:
        raise ImportError(
            "Kornia's DexiNed implementation is required. Install `kornia` from "
            "requirements.txt; no Canny fallback is used."
        ) from exc

    return torch, DexiNed


class DexiNedEdgeDetector:
    """Load DexiNed once and run deterministic inference on sampled frames."""

    def __init__(self, weights_path: Path | None = None) -> None:
        self._torch, dexined_cls = _import_torch_and_dexined()
        self.device = self._resolve_device()
        self.weights_path = self._resolve_weights_path(weights_path)

        # Strictly load the requested pretrained DexiNed weights.
        self.model = dexined_cls(pretrained=False)
        try:
            state = self._torch.load(
                self.weights_path,
                map_location=self._torch.device("cpu"),
                weights_only=True,
            )
        except TypeError:
            state = self._torch.load(self.weights_path, map_location=self._torch.device("cpu"))
        except Exception as exc:
            raise RuntimeError(f"Failed to load DexiNed weights: {self.weights_path}") from exc

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"DexiNed weights are not a state_dict: {self.weights_path}")

        normalized_state = {
            key.replace("module.", "", 1): value
            for key, value in state.items()
            if hasattr(value, "shape")
        }
        try:
            self.model.load_state_dict(normalized_state, strict=True)
        except Exception as exc:
            raise RuntimeError(
                "DexiNed weights are incompatible with Kornia's DexiNed model. "
                f"Expected BIPED weights at {self.weights_path}."
            ) from exc

        # Deterministic inference settings. CPU is the default for reproducibility.
        self._torch.manual_seed(config.RANDOM_SEED)
        self._torch.use_deterministic_algorithms(True, warn_only=False)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self) -> object:
        """Resolve the configured torch device."""

        requested = config.EDGE_DETECTOR_DEVICE
        if requested == "cpu":
            return self._torch.device("cpu")
        if requested == "cuda":
            if not self._torch.cuda.is_available():
                raise RuntimeError("EDGE_DETECTOR_DEVICE is cuda but CUDA is not available.")
            self._torch.backends.cudnn.deterministic = True
            self._torch.backends.cudnn.benchmark = False
            return self._torch.device("cuda")
        raise RuntimeError(f"Unsupported EDGE_DETECTOR_DEVICE: {requested}")

    def _resolve_weights_path(self, weights_path: Path | None) -> Path:
        """Resolve the DexiNed weights, downloading the default checkpoint if needed."""

        configured_path = weights_path
        if configured_path is None:
            configured_path = Path(
                os.environ.get(
                    config.DEXINED_WEIGHTS_ENV_VAR,
                    str(config.DEFAULT_DEXINED_WEIGHTS_PATH),
                )
            )
        resolved = configured_path.expanduser().resolve()
        if not resolved.exists() or not resolved.is_file():
            self._download_weights(resolved)
        return resolved

    def _download_weights(self, target_path: Path) -> None:
        """Download the configured DexiNed checkpoint from Hugging Face Hub."""

        repo_id = os.environ.get(config.DEXINED_HF_REPO_ENV_VAR, config.DEFAULT_DEXINED_HF_REPO_ID)
        filename = os.environ.get(config.DEXINED_HF_FILENAME_ENV_VAR, config.DEFAULT_DEXINED_HF_FILENAME)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "DexiNed weights are missing and `huggingface_hub` is not installed. "
                "Install requirements.txt, then rerun so the checkpoint can be downloaded."
            ) from exc

        try:
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(target_path.parent),
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to download DexiNed weights from Hugging Face Hub. "
                f"Repo: {repo_id!r}; file: {filename!r}; target: {target_path}. "
                f"You can also place the checkpoint manually or set "
                f"{config.DEXINED_WEIGHTS_ENV_VAR}=/absolute/path/{filename}."
            ) from exc

        if downloaded_path.resolve() != target_path:
            shutil.copy2(downloaded_path, target_path)
        if not target_path.exists() or not target_path.is_file():
            raise RuntimeError(f"DexiNed checkpoint download did not create {target_path}.")

    def _preprocess(self, rgb_frame: np.ndarray) -> tuple[object, tuple[int, int]]:
        """Resize if needed, convert RGB uint8 to a torch BCHW tensor in [0, 255]."""

        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise RuntimeError(f"Expected an RGB frame, got shape {rgb_frame.shape}")

        original_height, original_width = rgb_frame.shape[:2]
        long_side = max(original_height, original_width)
        if long_side > config.EDGE_INPUT_MAX_LONG_SIDE:
            scale = config.EDGE_INPUT_MAX_LONG_SIDE / float(long_side)
            inference_width = max(1, int(round(original_width * scale)))
            inference_height = max(1, int(round(original_height * scale)))
            resized = cv2.resize(
                rgb_frame,
                (inference_width, inference_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = rgb_frame

        tensor = self._torch.from_numpy(resized).to(dtype=self._torch.float32)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor, (original_height, original_width)

    def _postprocess(self, model_output: object, original_shape: tuple[int, int]) -> np.ndarray:
        """Convert DexiNed logits into an original-size edge probability map."""

        if not isinstance(model_output, list) or not model_output:
            raise RuntimeError("DexiNed did not return the expected list of edge logits.")
        logits = model_output[-1]
        probabilities = self._torch.sigmoid(logits).detach().cpu().squeeze().numpy()
        if probabilities.ndim != 2:
            raise RuntimeError(f"DexiNed produced an invalid probability map: {probabilities.shape}")

        original_height, original_width = original_shape
        if probabilities.shape != (original_height, original_width):
            probabilities = cv2.resize(
                probabilities,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            )
        probabilities = np.clip(probabilities.astype(np.float32), 0.0, 1.0)
        return self._robust_normalize(probabilities)

    def _robust_normalize(self, probabilities: np.ndarray) -> np.ndarray:
        """Stretch low-contrast DexiNed outputs while preserving edge ordering."""

        lower = float(np.percentile(probabilities, config.EDGE_NORMALIZATION_LOWER_PERCENTILE))
        upper = float(np.percentile(probabilities, config.EDGE_NORMALIZATION_UPPER_PERCENTILE))
        if upper <= lower + config.EDGE_NORMALIZATION_EPS:
            upper = float(np.max(probabilities))
        if upper <= lower + config.EDGE_NORMALIZATION_EPS:
            return np.zeros_like(probabilities, dtype=np.float32)
        normalized = (probabilities - lower) / (upper - lower)
        return np.clip(normalized.astype(np.float32), 0.0, 1.0)

    def run_on_frame(self, sampled_frame: SampledFrame, output_dir: Path) -> EdgeMapResult:
        """Run DexiNed on one sampled frame and save PNG plus exact NPY maps."""

        rgb_frame = load_sampled_frame_rgb(sampled_frame)
        tensor, original_shape = self._preprocess(rgb_frame)
        with self._torch.inference_mode():
            model_output = self.model(tensor)
        probability_map = self._postprocess(model_output, original_shape)

        edge_png_path = output_dir / f"frame_{sampled_frame.frame_index:06d}_edge.png"
        edge_npy_path = output_dir / f"frame_{sampled_frame.frame_index:06d}_edge_probability.npy"
        edge_uint8 = np.round(probability_map * 255.0).astype(np.uint8)
        if not cv2.imwrite(str(edge_png_path), edge_uint8):
            raise RuntimeError(f"OpenCV failed to write edge map: {edge_png_path}")
        np.save(edge_npy_path, probability_map)

        return EdgeMapResult(
            frame_index=sampled_frame.frame_index,
            timestamp_sec=sampled_frame.timestamp_sec,
            edge_png_path=edge_png_path,
            edge_npy_path=edge_npy_path,
            probability_map=probability_map,
        )

    def run(self, sampled_frames: list[SampledFrame], output_dir: Path) -> list[EdgeMapResult]:
        """Run deterministic edge inference on every sampled frame."""

        if not sampled_frames:
            raise RuntimeError("No sampled frames were provided to the edge detector.")
        output_dir.mkdir(parents=True, exist_ok=True)
        return [self.run_on_frame(frame, output_dir) for frame in sampled_frames]

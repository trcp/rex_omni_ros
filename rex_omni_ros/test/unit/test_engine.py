"""Tests for the GPU-less parts of the engine (auto VRAM sizing, sleep/wake)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from rex_omni_ros.core import tasks
from rex_omni_ros.core.engine import (
    GIB,
    IMAGE_PAD_TOKEN,
    EngineConfig,
    InferenceRequest,
    RexOmniEngine,
    _checkpoint_weight_nbytes,
    auto_gpu_memory_utilization,
)

WEIGHT_NBYTES = int(3.5 * GIB)  # ≈ the AWQ checkpoint
TOTAL_VRAM = 24 * GIB


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    # Sparse files: st_size is what the estimator reads, no disk is used.
    with (tmp_path / "model.safetensors").open("wb") as file:
        file.truncate(WEIGHT_NBYTES)
    return tmp_path


def config_for(checkpoint: Path, **overrides: object) -> EngineConfig:
    return EngineConfig(
        model_path=str(checkpoint),
        gpu_memory_utilization=0.0,
        **overrides,  # type: ignore[arg-type]
    )


def test_weight_nbytes_sums_weight_files(checkpoint: Path) -> None:
    with (checkpoint / "extra.bin").open("wb") as file:
        file.truncate(GIB)
    (checkpoint / "config.json").write_text("{}")  # not a weight file

    assert _checkpoint_weight_nbytes(str(checkpoint)) == WEIGHT_NBYTES + GIB


def test_weight_nbytes_rejects_checkpoint_without_weights(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no weight files"):
        _checkpoint_weight_nbytes(str(tmp_path))


def test_auto_utilization_is_a_sane_fraction(checkpoint: Path) -> None:
    utilization = auto_gpu_memory_utilization(config_for(checkpoint), TOTAL_VRAM)

    # Weights alone are ~3.5 GiB; everything together stays well under 50%
    # of a 24 GiB GPU (measured minimum for the default config is ~6 GiB).
    assert WEIGHT_NBYTES / TOTAL_VRAM < utilization < 0.5


def test_auto_utilization_grows_with_context_and_image_budget(
    checkpoint: Path,
) -> None:
    base = auto_gpu_memory_utilization(config_for(checkpoint), TOTAL_VRAM)
    more_context = auto_gpu_memory_utilization(
        config_for(checkpoint, max_model_len=8192), TOTAL_VRAM
    )
    more_pixels = auto_gpu_memory_utilization(
        config_for(checkpoint, max_pixels=4 * 2007040), TOTAL_VRAM
    )

    assert more_context > base
    assert more_pixels > base


def test_auto_utilization_drops_cuda_graphs_when_eager(checkpoint: Path) -> None:
    default = auto_gpu_memory_utilization(config_for(checkpoint), TOTAL_VRAM)
    eager = auto_gpu_memory_utilization(
        config_for(checkpoint, enforce_eager=True), TOTAL_VRAM
    )

    assert eager < default


def test_auto_utilization_rejects_too_small_gpu(checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="GiB VRAM"):
        auto_gpu_memory_utilization(config_for(checkpoint), 4 * GIB)


class FakeLLM:
    """Records sleep/wake/generate calls; returns one empty completion."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def sleep(self, level: int = 1) -> None:
        self.calls.append(("sleep", level))

    def wake_up(self) -> None:
        self.calls.append(("wake_up",))

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        self.calls.append(("generate",))
        completion = SimpleNamespace(text="", token_ids=[], logprobs=None)
        return [SimpleNamespace(outputs=[completion])]


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return IMAGE_PAD_TOKEN

    def convert_ids_to_tokens(self, token_id):
        return ""


def started_engine(**overrides: object) -> RexOmniEngine:
    """An engine in the post-start() state, backed by fakes instead of vLLM."""
    engine = RexOmniEngine(EngineConfig(**overrides))  # type: ignore[arg-type]
    engine._llm = FakeLLM()
    engine._tokenizer = FakeTokenizer()
    return engine


def test_sleep_offloads_at_level_1_and_is_idempotent() -> None:
    engine = started_engine()

    engine.sleep()
    engine.sleep()

    assert engine.sleeping
    assert engine._llm.calls == [("sleep", 1)]


def test_wake_up_restores_and_is_idempotent() -> None:
    engine = started_engine()

    engine.wake_up()  # no-op while awake
    engine.sleep()
    engine.wake_up()
    engine.wake_up()

    assert not engine.sleeping
    assert engine._llm.calls == [("sleep", 1), ("wake_up",)]


def test_sleep_requires_started_engine() -> None:
    engine = RexOmniEngine(EngineConfig())

    with pytest.raises(RuntimeError, match="not started"):
        engine.sleep()
    with pytest.raises(RuntimeError, match="not started"):
        engine.wake_up()


def test_sleep_rejected_when_disabled() -> None:
    engine = started_engine(enable_sleep_mode=False)

    with pytest.raises(RuntimeError, match="sleep mode is disabled"):
        engine.sleep()


def test_infer_wakes_sleeping_engine() -> None:
    engine = started_engine()
    engine.sleep()

    engine.infer(
        InferenceRequest(
            image=Image.new("RGB", (64, 64)),
            task=tasks.TaskType.DETECTION,
            categories=["object"],
        )
    )

    assert not engine.sleeping
    assert engine._llm.calls == [("sleep", 1), ("wake_up",), ("generate",)]

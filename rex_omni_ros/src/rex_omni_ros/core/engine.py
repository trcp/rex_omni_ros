"""In-process vLLM inference engine for Rex-Omni.

vLLM is imported lazily in :meth:`RexOmniEngine.start` so that this module
(and everything that type-checks against :class:`Engine`) can be imported
without vLLM installed — unit and integration tests use a fake engine.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from PIL import Image

from rex_omni_ros.core import parser, preprocess, tasks
from rex_omni_ros.core.types import Annotation, Box, KeypointInstance

logger = logging.getLogger(__name__)

GIB = 1 << 30

KV_CACHE_DTYPE_BYTES = 2  # vLLM keeps the KV cache in fp16/bf16
CUDA_GRAPH_BYTES = int(0.55 * GIB)
BASE_OVERHEAD_BYTES = int(1.5 * GIB)
VIT_ACTIVATION_BYTES_PER_PIXEL = 36

QWEN_CHAT_TEMPLATE_FALLBACK = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "{prompt}<|im_end|>\n<|im_start|>assistant\n"
)
IMAGE_PAD_TOKEN = "<|image_pad|>"


@dataclass
class EngineConfig:
    """Model and sampling configuration (defaults follow upstream Rex-Omni)."""

    model_path: str = "IDEA-Research/Rex-Omni"
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 4096
    min_pixels: int = preprocess.DEFAULT_MIN_PIXELS
    max_pixels: int = preprocess.DEFAULT_MAX_PIXELS
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 0.8
    top_k: int = 1
    repetition_penalty: float = 1.05
    system_prompt: str = "You are a helpful assistant"
    enable_confidence: bool = True
    quantization: str = ""
    dtype: str = "auto"
    enforce_eager: bool = False
    warmup: bool = True
    compile_vit: bool = False
    enable_sleep_mode: bool = True
    # Passed through to vllm.LLM verbatim (e.g. speculative_config).
    extra_llm_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceRequest:
    """One perception request, independent of the transport (ROS) layer."""

    image: Image.Image
    task: tasks.TaskType
    categories: list[str] = field(default_factory=list)
    keypoint_type: str = ""
    visual_prompt_boxes: list[Box] = field(default_factory=list)


@dataclass
class InferenceResult:
    annotations: list[Annotation] = field(default_factory=list)
    keypoint_instances: list[KeypointInstance] = field(default_factory=list)
    raw_output: str = ""
    inference_time: float = 0.0


class Engine(Protocol):
    """Interface the ROS layer depends on; satisfied by fakes in tests."""

    def start(self) -> None: ...

    def infer(self, request: InferenceRequest) -> InferenceResult: ...

    def sleep(self) -> None: ...

    def wake_up(self) -> None: ...


def _checkpoint_weight_nbytes(model_path: str) -> int:
    """On-disk size of the model weights, which matches their GPU footprint.

    Downloads the weights for Hub ids; vLLM reuses the same cache right
    after, so nothing is fetched twice.
    """
    path = Path(model_path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(model_path, allow_patterns=["*.safetensors", "*.bin"])
        )
    nbytes = sum(
        file.stat().st_size
        for pattern in ("*.safetensors", "*.bin")
        for file in path.glob(pattern)
    )
    if nbytes == 0:
        raise ValueError(f"no weight files (*.safetensors, *.bin) under {path}")
    return nbytes


def _model_text_config(model_path: str) -> dict[str, Any]:
    """The text-model section of the model's config.json.

    Qwen2.5-VL-style configs keep the text keys at the top level; other
    multimodal configs nest them under ``text_config``. Downloads only
    config.json for Hub ids.
    """
    path = Path(model_path)
    if path.is_dir():
        config_file = path / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        config_file = Path(hf_hub_download(model_path, "config.json"))
    with config_file.open() as file:
        config = json.load(file)
    return cast("dict[str, Any]", config.get("text_config", config))


def kv_cache_bytes_per_token(model_path: str) -> int:
    """Per-token KV cache size in bytes, derived from the model's config.json.

    Raises:
        ValueError: If config.json lacks the required attention geometry.
    """
    text_config = _model_text_config(model_path)
    try:
        num_layers = int(text_config["num_hidden_layers"])
        num_kv_heads = int(text_config["num_key_value_heads"])
        head_dim = int(
            text_config.get("head_dim")
            or text_config["hidden_size"] // text_config["num_attention_heads"]
        )
    except KeyError as error:
        raise ValueError(
            f"cannot size the KV cache: config.json of {model_path} has no "
            f"{error} key; set gpu_memory_utilization explicitly"
        ) from None
    return 2 * num_layers * num_kv_heads * head_dim * KV_CACHE_DTYPE_BYTES  # K and V


def auto_gpu_memory_utilization(config: EngineConfig, total_vram_bytes: int) -> float:
    """The gpu_memory_utilization granting the minimum VRAM `config` needs.

    Sized as weights + fixed runtime overheads + a KV cache that fits exactly
    one max_model_len request — sufficient because the node serializes
    requests. The rest of the GPU stays free for other processes.
    """
    required = (
        _checkpoint_weight_nbytes(config.model_path)
        + BASE_OVERHEAD_BYTES
        + config.max_pixels * VIT_ACTIVATION_BYTES_PER_PIXEL
        + config.max_model_len * kv_cache_bytes_per_token(config.model_path)
    )
    if not config.enforce_eager:
        required += CUDA_GRAPH_BYTES
    if required > total_vram_bytes:
        raise ValueError(
            f"this configuration needs ~{required / GIB:.2f} GiB VRAM but the "
            f"GPU only has {total_vram_bytes / GIB:.2f} GiB; reduce "
            "max_model_len or max_pixels, or set enforce_eager"
        )
    utilization = required / total_vram_bytes
    logger.info(
        "auto gpu_memory_utilization: %.3f (%.2f GiB of %.2f GiB)",
        utilization,
        required / GIB,
        total_vram_bytes / GIB,
    )
    return utilization


def _patch_marlin_lm_head_support() -> None:
    """Let awq_marlin accept a quantized lm_head (vLLM 0.22.1 workaround).

    With ``lm_head: true`` in the quantization config (as produced by
    tools/quantize_lm_head.py), ``check_marlin_supports_layer`` is called on
    a :class:`ParallelLMHead`, but reads LinearBase-only attributes and
    raises ``AttributeError: 'ParallelLMHead' object has no attribute
    'output_size'``. Feed embedding-style layers through the same shape
    check using their own dimensions. Remove once vLLM fixes this upstream.
    """
    from vllm.model_executor.layers.quantization import awq_marlin
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    if getattr(awq_marlin.check_marlin_supports_layer, "_rex_omni_patch", False):
        return
    original = awq_marlin.check_marlin_supports_layer

    def patched(layer: Any, group_size: int) -> bool:
        if not hasattr(layer, "input_size") and hasattr(layer, "embedding_dim"):
            # Called from VocabParallelEmbedding.__init__ before the
            # per-partition sizes are assigned; derive them the same way.
            return marlin_utils.check_marlin_supports_shape(
                output_size_per_partition=layer.num_embeddings_padded // layer.tp_size,
                input_size_per_partition=layer.embedding_dim,
                input_size=layer.embedding_dim,
                group_size=group_size,
            )[0]
        return original(layer, group_size)

    patched._rex_omni_patch = True  # type: ignore[attr-defined]
    awq_marlin.check_marlin_supports_layer = patched


def _vram_usage_suffix() -> str:
    """Current GPU memory usage as a log suffix, or '' when unavailable.

    Reads torch from sys.modules instead of importing it so this stays free
    in unit tests; after start() torch is always loaded.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return ""
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - logging only, never fail the call
        return ""
    return f" (GPU: {(total - free) / GIB:.2f}/{total / GIB:.2f} GiB used)"


class RexOmniEngine:
    """vLLM-backed engine. Not thread-safe; serialize calls to :meth:`infer`."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._llm: Any = None
        self._tokenizer: Any = None
        self._sampling_params: Any = None
        self._sleeping = False

    def start(self) -> None:
        """Load the model. Takes tens of seconds; call once at node startup."""
        # Run the V1 EngineCore in this process instead of a forked child:
        # the ROS node process is multi-threaded (and may have touched CUDA),
        # and forking from such a process deadlocks or crashes CUDA init.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM, SamplingParams

        _patch_marlin_lm_head_support()

        config = self._config
        logger.info("loading Rex-Omni model from %s", config.model_path)
        gpu_memory_utilization = config.gpu_memory_utilization
        if gpu_memory_utilization == 0:
            import torch

            total_vram_bytes = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory
            gpu_memory_utilization = auto_gpu_memory_utilization(
                config, total_vram_bytes
            )
        llm_kwargs = dict(config.extra_llm_kwargs)
        if config.compile_vit:
            # Merge rather than overwrite so extra_llm_kwargs can still
            # carry its own compilation_config entries.
            compilation = dict(llm_kwargs.get("compilation_config") or {})
            compilation.setdefault("compile_mm_encoder", True)
            llm_kwargs["compilation_config"] = compilation
        self._llm = LLM(
            model=config.model_path,
            # The fast tokenizer (tokenizer.json) shipped with Rex-Omni maps
            # coordinate/special tokens beyond the embedding size; only the
            # slow tokenizer carries the id layout the model was trained with.
            tokenizer_mode="slow",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=config.max_model_len,
            quantization=cast(Any, config.quantization or None),
            dtype=cast(Any, config.dtype),
            enforce_eager=config.enforce_eager,
            enable_sleep_mode=config.enable_sleep_mode,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "min_pixels": config.min_pixels,
                "max_pixels": config.max_pixels,
            },
            **llm_kwargs,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            max_tokens=config.max_tokens,
            stop=["<|im_end|>"],
            skip_special_tokens=False,
            logprobs=0 if config.enable_confidence else None,
        )
        if config.warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Exercise the full request path once so the first real request
        does not pay one-time costs (processor init, kernel autotuning)."""
        start_time = time.monotonic()
        self.infer(
            InferenceRequest(
                image=Image.new("RGB", (224, 224)),
                task=tasks.TaskType.DETECTION,
                categories=["object"],
            )
        )
        logger.info("warmup request took %.2fs", time.monotonic() - start_time)

    @property
    def started(self) -> bool:
        return self._llm is not None

    @property
    def sleeping(self) -> bool:
        return self._sleeping

    def sleep(self) -> None:
        """Offload the weights to host RAM and drop the KV cache (vLLM sleep
        level 1), releasing the model's VRAM for other processes. The CUDA
        context (and CUDA graphs, unless enforce_eager) stays resident.
        Idempotent; needs free host RAM about the size of the weights."""
        if not self.started:
            raise RuntimeError("engine not started; call start() first")
        if not self._config.enable_sleep_mode:
            raise RuntimeError(
                "sleep mode is disabled; start the node with enable_sleep_mode"
            )
        if self._sleeping:
            return
        start_time = time.monotonic()
        self._llm.sleep(level=1)
        self._sleeping = True
        logger.info(
            "model offloaded to host RAM in %.2fs%s",
            time.monotonic() - start_time,
            _vram_usage_suffix(),
        )

    def wake_up(self) -> None:
        """Restore the weights to VRAM and reallocate the KV cache. Idempotent."""
        if not self.started:
            raise RuntimeError("engine not started; call start() first")
        if not self._sleeping:
            return
        start_time = time.monotonic()
        self._llm.wake_up()
        self._sleeping = False
        logger.info(
            "model restored to VRAM in %.2fs%s",
            time.monotonic() - start_time,
            _vram_usage_suffix(),
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        if not self.started:
            raise RuntimeError("engine not started; call start() first")
        if self._sleeping:
            logger.warning("inference requested while asleep; waking model up")
            self.wake_up()

        start_time = time.monotonic()
        image = request.image.convert("RGB")
        width, height = image.size

        prompt_text = tasks.build_prompt(
            task=request.task,
            categories=request.categories or None,
            keypoint_type=request.keypoint_type or None,
            visual_prompt_boxes=request.visual_prompt_boxes or None,
            image_width=width,
            image_height=height,
        )
        prompt = self._render_chat(prompt_text)

        outputs = self._llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}],
            sampling_params=self._sampling_params,
            use_tqdm=False,
        )
        completion = outputs[0].outputs[0]
        raw_output = completion.text

        result = InferenceResult(raw_output=raw_output)
        if request.task is tasks.TaskType.KEYPOINT:
            result.keypoint_instances = parser.parse_keypoint_instances(
                raw_output, width, height
            )
            items: list[Any] = list(result.keypoint_instances)
        else:
            result.annotations = parser.parse_annotations(raw_output, width, height)
            items = list(result.annotations)

        if self._config.enable_confidence and items:
            parser.assign_confidences(items, self._coord_token_probs(completion))

        result.inference_time = time.monotonic() - start_time
        return result

    def _render_chat(self, prompt_text: str) -> str:
        """Apply the model's chat template, with a Qwen-format fallback."""
        messages = [
            {"role": "system", "content": self._config.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]
        try:
            rendered = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if IMAGE_PAD_TOKEN in rendered:
                return rendered
            logger.warning(
                "chat template produced no %s token; using fallback template",
                IMAGE_PAD_TOKEN,
            )
        except Exception as error:  # noqa: BLE001 - template content is external
            logger.warning("apply_chat_template failed (%s); using fallback", error)
        return QWEN_CHAT_TEMPLATE_FALLBACK.format(
            system=self._config.system_prompt, prompt=prompt_text
        )

    def _coord_token_probs(self, completion: Any) -> list[float]:
        """Probabilities of generated coordinate tokens, in emission order."""
        if not completion.logprobs:
            return []
        probs: list[float] = []
        for token_id, logprob_dict in zip(completion.token_ids, completion.logprobs):
            token = self._tokenizer.convert_ids_to_tokens(int(token_id))
            if isinstance(token, str) and parser.COORD_TOKEN_PATTERN.fullmatch(token):
                entry = logprob_dict.get(token_id)
                probs.append(math.exp(entry.logprob) if entry is not None else 0.0)
        return probs

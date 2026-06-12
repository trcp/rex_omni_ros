"""In-process vLLM inference engine for Rex-Omni.

vLLM is imported lazily in :meth:`RexOmniEngine.start` so that this module
(and everything that type-checks against :class:`Engine`) can be imported
without vLLM installed — unit and integration tests use a fake engine.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from PIL import Image

from rex_omni_ros.core import parser, preprocess, tasks
from rex_omni_ros.core.types import Annotation, Box, KeypointInstance

logger = logging.getLogger(__name__)

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
    quantization: str = ""  # e.g. "awq"; empty selects no explicit quantization
    dtype: str = "auto"
    enforce_eager: bool = False
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


class RexOmniEngine:
    """vLLM-backed engine. Not thread-safe; serialize calls to :meth:`infer`."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._llm: Any = None
        self._tokenizer: Any = None
        self._sampling_params: Any = None

    def start(self) -> None:
        """Load the model. Takes tens of seconds; call once at node startup."""
        # Run the V1 EngineCore in this process instead of a forked child:
        # the ROS node process is multi-threaded (and may have touched CUDA),
        # and forking from such a process deadlocks or crashes CUDA init.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM, SamplingParams

        config = self._config
        logger.info("loading Rex-Omni model from %s", config.model_path)
        self._llm = LLM(
            model=config.model_path,
            # The fast tokenizer (tokenizer.json) shipped with Rex-Omni maps
            # coordinate/special tokens beyond the embedding size; only the
            # slow tokenizer carries the id layout the model was trained with.
            tokenizer_mode="slow",
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            quantization=cast(Any, config.quantization or None),
            dtype=cast(Any, config.dtype),
            enforce_eager=config.enforce_eager,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "min_pixels": config.min_pixels,
                "max_pixels": config.max_pixels,
            },
            **config.extra_llm_kwargs,
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

    @property
    def started(self) -> bool:
        return self._llm is not None

    def infer(self, request: InferenceRequest) -> InferenceResult:
        if not self.started:
            raise RuntimeError("engine not started; call start() first")

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
        model_image = preprocess.resize_for_model(
            image,
            min_pixels=self._config.min_pixels,
            max_pixels=self._config.max_pixels,
        )

        outputs = self._llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": model_image}}],
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

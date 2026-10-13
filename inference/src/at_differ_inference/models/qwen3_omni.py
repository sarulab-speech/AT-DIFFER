"""Qwen3-Omni captioner backed by vLLM offline inference.

The engine is built once and accepts a list of (audio, text) pairs per call,
letting vLLM continuous-batch them internally.

The heavy dependencies (vllm / transformers / qwen_omni_utils / static_ffmpeg)
are imported at module load, so this module must only be imported on a node
where they are installed (i.e. via the registry's lazy import).
"""

import os

import static_ffmpeg

static_ffmpeg.add_paths()

from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor
from vllm import LLM, SamplingParams

from at_differ_inference.models.base import AudioCaptioner, AudioTextPair

# Local directory of the base model (see README for how to download it).
_DEFAULT_MODEL_PATH = "../outputs/models/Qwen3-Omni-30B-A3B-Thinking"


class Qwen3OmniCaptioner(AudioCaptioner):
    """Qwen3-Omni-30B-A3B (Thinking) via vLLM."""

    name = "qwen3-omni-thinking"

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 32768,
        max_num_seqs: int = 16,
        dtype: str = "bfloat16",
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 4096,
    ) -> None:
        self.model_path = os.path.expanduser(model_path)

        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"video": 1, "audio": 1, "image": 1},
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            dtype=dtype,
            enforce_eager=True,
            mm_encoder_attn_backend="TORCH_SDPA",
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)

    def generate(self, pairs: list[AudioTextPair]) -> list[str]:
        if not pairs:
            return []

        inputs = [self._build_input(pair) for pair in pairs]
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]

    def _build_input(self, pair: AudioTextPair) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": pair.audio_path},
                    {"type": "text", "text": pair.text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

        multi_modal_data: dict = {}
        if images is not None:
            multi_modal_data["image"] = images
        if videos is not None:
            multi_modal_data["video"] = videos
        if audios is not None:
            multi_modal_data["audio"] = audios

        return {"prompt": prompt, "multi_modal_data": multi_modal_data}


# Empty reasoning block appended after the generation prompt to suppress thinking.
# Must match at_differ_finetune.models.qwen3_omni.EMPTY_THINK_BLOCK exactly: the fine-tuned
# adapter was trained to continue from this prefix, so any difference here changes
# what the model is being asked to continue.
_EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"


class Qwen3OmniNoThinkCaptioner(Qwen3OmniCaptioner):
    """Qwen3-Omni without the reasoning block, for LoRA fine-tuned checkpoints.

    The base Thinking checkpoint emits a long ``<think>`` block before the three
    answer lines, and generation can hit ``max_tokens`` before ``</think>`` closes.
    Fine-tuning teaches the model to answer directly, so here we close the reasoning
    block up front and let it continue with the three lines.

    Pass ``--model-path`` pointing at a merged checkpoint produced by
    ``at-differ-finetune merge``; the base weights alone were never trained
    for this format and will not honour it.
    """

    name = "qwen3-omni-nothink"

    def _build_input(self, pair: AudioTextPair) -> dict:
        inputs = super()._build_input(pair)
        inputs["prompt"] += _EMPTY_THINK_BLOCK
        return inputs

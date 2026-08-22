"""Policy wrappers turning chat history into raw assistant generations."""

from __future__ import annotations

from typing import Any


class HFChatPolicy:
    """HuggingFace generate-based policy matching the SFT inference contract.

    The chat template pre-emits ``<|im_start|>assistant\n<think>\n`` when
    thinking is enabled, so decoding starts inside the think block and the raw
    completion contains only its closing tag - parse_model_output handles both
    shapes.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_new_tokens: int = 1024,
        enable_thinking: bool = True,
        bf16: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.bfloat16 if bf16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

    def build_prompt(self, history: list[dict[str, Any]]) -> str:
        """Renders history through the official chat template with a generation header."""
        rendered = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        return str(rendered)

    def generate(self, history: list[dict[str, Any]]) -> str:
        import torch

        prompt_text = self.build_prompt(history)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-4),
                top_p=self.top_p,
                top_k=self.top_k,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs.input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        # Trim everything after the turn-ending marker.
        return str(text).split("<|im_end|>")[0]


class VLLMChatPolicy:
    """vLLM-backed policy for fast final evaluations (requires vllm extra)."""

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_new_tokens: int = 1024,
        enable_thinking: bool = True,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 16384,
    ) -> None:
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
        )
        self.enable_thinking = enable_thinking

    def build_prompt(self, history: list[dict[str, Any]]) -> str:
        rendered = self.llm.get_tokenizer().apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        return str(rendered)

    def generate(self, history: list[dict[str, Any]]) -> str:
        prompt_text = self.build_prompt(history)
        output = self.llm.generate([prompt_text], self.sampling)[0]
        return str(output.outputs[0].text)

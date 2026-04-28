import signal
import sys
import threading
from contextlib import contextmanager

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEEPSEEK_PROVER_V2_PROMPT = """Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""


class LLMGenerationTimeoutError(TimeoutError):
    pass


@contextmanager
def _generation_timeout(timeout_sec: int | None):
    if timeout_sec is None or timeout_sec <= 0:
        yield
        return
    # signal.alarm only works reliably on Unix main thread.
    if sys.platform.startswith("win") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handle_timeout(signum, frame):
        del signum, frame
        raise LLMGenerationTimeoutError(f"LLM generation timed out after {timeout_sec}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


class ProverGenerator:
    def __init__(self, model_cfg: dict):
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"], use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name_or_path"],
            device_map=model_cfg.get("device_map", "auto"),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        adapter_path = model_cfg.get("adapter_path")
        if adapter_path:
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model
        self.model.eval()
        self.max_new_tokens = model_cfg.get("max_new_tokens", 256)
        self.temperature = model_cfg.get("temperature", 0.2)
        self.top_p = model_cfg.get("top_p", 0.9)
        self.do_sample = model_cfg.get("do_sample", False)
        self.num_return_sequences = model_cfg.get("num_return_sequences", 1)
        self.inference_timeout_sec = model_cfg.get("inference_timeout_sec")

    def build_prompt(self, statement: str) -> str:
        return (
            "You are a Lean theorem prover assistant.\n"
            "Given the theorem declaration, generate a valid Lean proof only.\n\n"
            f"{statement}\n\n"
            "-- proof:"
        )

    def _decode_generations(self, prompt: str, out: torch.Tensor) -> list[str]:
        texts: list[str] = []
        for seq in out:
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            texts.append(text[len(prompt) :].strip() if text.startswith(prompt) else text.strip())
        return texts

    def _decode_generations_from_input_len(self, out: torch.Tensor, input_len: int) -> list[str]:
        texts: list[str] = []
        for seq in out:
            continuation = seq[input_len:]
            texts.append(self.tokenizer.decode(continuation, skip_special_tokens=True).strip())
        return texts

    def generate_proofs(self, statement: str, num_samples: int = 1) -> list[str]:
        prompt = self.build_prompt(statement)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        want_samples = max(1, int(num_samples))
        use_sampling = self.do_sample or want_samples > 1
        with _generation_timeout(self.inference_timeout_sec):
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=use_sampling,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=want_samples,
                )
        return self._decode_generations(prompt, out)

    def generate_proof(self, statement: str) -> str:
        return self.generate_proofs(statement, num_samples=1)[0]


class DeepSeekProverV2Generator(ProverGenerator):
    def __init__(self, model_cfg: dict):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name_or_path"],
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name_or_path"],
            device_map=model_cfg.get("device_map", "auto"),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        adapter_path = model_cfg.get("adapter_path")
        if adapter_path:
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model
        self.model.eval()
        self.max_new_tokens = model_cfg.get("max_new_tokens", 256)
        self.temperature = model_cfg.get("temperature", 0.2)
        self.top_p = model_cfg.get("top_p", 0.9)
        self.do_sample = model_cfg.get("do_sample", False)
        self.num_return_sequences = model_cfg.get("num_return_sequences", 1)
        self.inference_timeout_sec = model_cfg.get("inference_timeout_sec")

    def build_prompt(self, statement: str) -> str:
        return DEEPSEEK_PROVER_V2_PROMPT.format(statement)

    def generate_proofs(self, statement: str, num_samples: int = 1) -> list[str]:
        prompt = self.build_prompt(statement)
        chat = [{"role": "user", "content": prompt}]
        chat_inputs = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if hasattr(chat_inputs, "to"):
            chat_inputs = chat_inputs.to(self.model.device)

        if isinstance(chat_inputs, torch.Tensor):
            model_inputs = {"input_ids": chat_inputs}
            input_len = int(chat_inputs.shape[-1])
        else:
            model_inputs = dict(chat_inputs)
            input_ids = model_inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Chat template output does not contain input_ids.")
            input_len = int(input_ids.shape[-1])

        want_samples = max(1, int(num_samples))
        use_sampling = self.do_sample or want_samples > 1
        with _generation_timeout(self.inference_timeout_sec):
            with torch.no_grad():
                out = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=use_sampling,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=want_samples,
                )
        return self._decode_generations_from_input_len(out, input_len)


def build_prover_generator(model_cfg: dict) -> ProverGenerator:
    model_name = str(model_cfg.get("name_or_path", "")).lower()
    model_type = str(model_cfg.get("model_type", "auto")).lower()
    if model_type == "auto":
        if "deepseek-ai/deepseek-prover-v2" in model_name:
            model_type = "deepseek_prover_v2"
        else:
            model_type = "generic"

    if model_type == "deepseek_prover_v2":
        return DeepSeekProverV2Generator(model_cfg)
    if model_type == "generic":
        return ProverGenerator(model_cfg)
    raise ValueError(f"Unsupported model_type: {model_type}")


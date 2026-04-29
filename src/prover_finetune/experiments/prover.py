import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

DEEPSEEK_PROVER_V2_PROMPT = """Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""


class LLMGenerationTimeoutError(TimeoutError):
    pass


class _WallClockStoppingCriteria(StoppingCriteria):
    def __init__(self, timeout_sec: int):
        self.timeout_sec = int(timeout_sec)
        self.start_time = time.perf_counter()
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        del input_ids, scores, kwargs
        self.triggered = (time.perf_counter() - self.start_time) >= self.timeout_sec
        return self.triggered


class ProverGenerator:
    @staticmethod
    def _load_model(model_cfg: dict, device_map: str, trust_remote_code: bool = False):
        base_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if trust_remote_code:
            base_kwargs["trust_remote_code"] = True

        attn_impl = model_cfg.get("attn_implementation")
        if attn_impl:
            try:
                return AutoModelForCausalLM.from_pretrained(
                    model_cfg["name_or_path"],
                    attn_implementation=attn_impl,
                    **base_kwargs,
                )
            except Exception:
                # Fallback to default attention implementation for compatibility.
                pass

        return AutoModelForCausalLM.from_pretrained(
            model_cfg["name_or_path"],
            **base_kwargs,
        )

    def __init__(self, model_cfg: dict):
        require_gpu = bool(model_cfg.get("require_gpu", True))
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "GPU is required for model loading (require_gpu=True), "
                "but torch.cuda.is_available() is False."
            )
        gpu_device = model_cfg.get("gpu_device")
        if gpu_device is not None:
            gpu_device_str = str(gpu_device)
            default_device_map = (
                gpu_device_str if gpu_device_str.startswith("cuda:") else f"cuda:{gpu_device_str}"
            )
        else:
            default_device_map = "cuda:0" if torch.cuda.is_available() else "auto"
        device_map = model_cfg.get("device_map", default_device_map)

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"], use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = self._load_model(model_cfg, device_map, trust_remote_code=False)
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
        decoded = self.tokenizer.batch_decode(out)
        texts: list[str] = []
        for text in decoded:
            texts.append(text[len(prompt) :].strip() if text.startswith(prompt) else text.strip())
        return texts

    def _build_stopping_criteria(self):
        timeout_sec = self.inference_timeout_sec
        if timeout_sec is None or int(timeout_sec) <= 0:
            return None, None
        criterion = _WallClockStoppingCriteria(int(timeout_sec))
        return StoppingCriteriaList([criterion]), criterion

    def _build_max_time(self) -> float | None:
        timeout_sec = self.inference_timeout_sec
        if timeout_sec is None:
            return None
        timeout = float(timeout_sec)
        if timeout <= 0:
            return None
        return timeout

    def generate_proofs(self, statement: str, num_samples: int = 1) -> list[str]:
        prompt = self.build_prompt(statement)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)
        want_samples = max(1, int(num_samples))
        use_sampling = self.do_sample or want_samples > 1
        input_len = int(inputs["input_ids"].shape[-1])
        stopping_criteria, timeout_criterion = self._build_stopping_criteria()
        max_time = self._build_max_time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=use_sampling,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=want_samples,
                stopping_criteria=stopping_criteria,
                max_time=max_time,
            )
        if timeout_criterion is not None and timeout_criterion.triggered:
            raise LLMGenerationTimeoutError(
                f"LLM generation timed out after {int(self.inference_timeout_sec)}s"
            )
        return self._decode_generations(prompt, out)

    def generate_proof(self, statement: str) -> str:
        return self.generate_proofs(statement, num_samples=1)[0]


class DeepSeekProverV2Generator(ProverGenerator):
    def __init__(self, model_cfg: dict):
        require_gpu = bool(model_cfg.get("require_gpu", True))
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "GPU is required for model loading (require_gpu=True), "
                "but torch.cuda.is_available() is False."
            )
        gpu_device = model_cfg.get("gpu_device")
        if gpu_device is not None:
            gpu_device_str = str(gpu_device)
            default_device_map = (
                gpu_device_str if gpu_device_str.startswith("cuda:") else f"cuda:{gpu_device_str}"
            )
        else:
            default_device_map = "cuda:0" if torch.cuda.is_available() else "auto"
        device_map = model_cfg.get("device_map", default_device_map)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name_or_path"],
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = self._load_model(model_cfg, device_map, trust_remote_code=True)
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
        inputs = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        if isinstance(inputs, torch.Tensor):
            model_inputs = {"input_ids": inputs}
        else:
            model_inputs = dict(inputs)
        if "input_ids" not in model_inputs:
            raise ValueError("Chat template output does not contain input_ids.")
        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(
                model_inputs["input_ids"], dtype=torch.long
            )
        input_len = int(model_inputs["input_ids"].shape[-1])

        want_samples = max(1, int(num_samples))
        use_sampling = self.do_sample or want_samples > 1
        stopping_criteria, timeout_criterion = self._build_stopping_criteria()
        max_time = self._build_max_time()
        with torch.no_grad():
            out = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=use_sampling,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=want_samples,
                stopping_criteria=stopping_criteria,
                max_time=max_time,
            )
        if timeout_criterion is not None and timeout_criterion.triggered:
            raise LLMGenerationTimeoutError(
                f"LLM generation timed out after {int(self.inference_timeout_sec)}s"
            )
        continuations = out[:, input_len:]
        return [text.strip() for text in self.tokenizer.batch_decode(continuations)]


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


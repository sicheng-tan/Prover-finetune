import signal
import sys
import threading
from contextlib import contextmanager

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEEPSEEK_PROVER_V2_PROMPT = """Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""


class LLMGenerationTimeoutError(TimeoutError):
    pass


def _resolve_tensor_parallel_size(model_cfg: dict) -> int:
    value = model_cfg.get("tensor_parallel_size", 1)
    if value is None:
        return 1
    return max(1, int(value))


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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name_or_path"],
            use_fast=True,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LLM(
            model_cfg["name_or_path"],
            tokenizer=model_cfg["name_or_path"],
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            tensor_parallel_size=_resolve_tensor_parallel_size(model_cfg),
            gpu_memory_utilization=float(model_cfg.get("gpu_memory_utilization", 0.9)),
            max_model_len=model_cfg.get("max_model_len"),
        )
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

    def generate_proofs(self, statement: str, num_samples: int = 1) -> list[str]:
        prompt = self.build_prompt(statement)
        want_samples = max(1, int(num_samples))
        use_sampling = self.do_sample or want_samples > 1
        temperature = self.temperature if use_sampling else 0.0
        sampling_params = SamplingParams(
            n=want_samples,
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=self.top_p,
        )
        with _generation_timeout(self.inference_timeout_sec):
            outputs = self.model.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        if not outputs:
            return [""]
        return [item.text.strip() for item in outputs[0].outputs]

    def generate_proofs_batch(self, statements: list[str], num_samples: int = 1) -> list[list[str]]:
        if not statements:
            return []
        prompts = [self.build_prompt(s) for s in statements]
        want_samples = max(1, int(num_samples))
        use_sampling = self.do_sample or want_samples > 1
        temperature = self.temperature if use_sampling else 0.0
        sampling_params = SamplingParams(
            n=want_samples,
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=self.top_p,
        )
        with _generation_timeout(self.inference_timeout_sec):
            outputs = self.model.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        results: list[list[str]] = []
        for item in outputs:
            results.append([o.text.strip() for o in item.outputs] or [""])
        return results

    def generate_proof(self, statement: str) -> str:
        return self.generate_proofs(statement, num_samples=1)[0]


class DeepSeekProverV2Generator(ProverGenerator):
    def __init__(self, model_cfg: dict):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name_or_path"],
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LLM(
            model_cfg["name_or_path"],
            tokenizer=model_cfg["name_or_path"],
            trust_remote_code=True,
            tensor_parallel_size=_resolve_tensor_parallel_size(model_cfg),
            gpu_memory_utilization=float(model_cfg.get("gpu_memory_utilization", 0.9)),
            max_model_len=model_cfg.get("max_model_len"),
        )
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
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        want_samples = max(1, int(num_samples))
        use_sampling = self.do_sample or want_samples > 1
        temperature = self.temperature if use_sampling else 0.0
        sampling_params = SamplingParams(
            n=want_samples,
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=self.top_p,
        )
        with _generation_timeout(self.inference_timeout_sec):
            outputs = self.model.generate(
                [prompt_with_chat_template],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        if not outputs:
            return [""]
        return [item.text.strip() for item in outputs[0].outputs]

    def generate_proofs_batch(self, statements: list[str], num_samples: int = 1) -> list[list[str]]:
        if not statements:
            return []
        prompts = []
        for statement in statements:
            prompt = self.build_prompt(statement)
            chat = [{"role": "user", "content": prompt}]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        want_samples = max(1, int(num_samples))
        use_sampling = self.do_sample or want_samples > 1
        temperature = self.temperature if use_sampling else 0.0
        sampling_params = SamplingParams(
            n=want_samples,
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=self.top_p,
        )
        with _generation_timeout(self.inference_timeout_sec):
            outputs = self.model.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

        results: list[list[str]] = []
        for item in outputs:
            results.append([o.text.strip() for o in item.outputs] or [""])
        return results


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


import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEEPSEEK_PROVER_V2_PROMPT = """Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""


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

    def generate_proofs(self, statement: str, num_samples: int = 1) -> list[str]:
        prompt = self.build_prompt(statement)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        want_samples = max(1, int(num_samples))
        use_sampling = self.do_sample or want_samples > 1
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
    def build_prompt(self, statement: str) -> str:
        return DEEPSEEK_PROVER_V2_PROMPT.format(statement)


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


import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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

    def build_prompt(self, statement: str) -> str:
        return (
            "You are a Lean theorem prover assistant.\n"
            "Given the theorem declaration, generate a valid Lean proof only.\n\n"
            f"{statement}\n\n"
            "-- proof:"
        )

    def generate_proof(self, statement: str) -> str:
        prompt = self.build_prompt(statement)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()


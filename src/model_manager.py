from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import List, Dict, Optional
import os

class HFModelManager:
    """Manages multiple Hugging Face models with efficient loading/unloading"""
    
    _instances = {}  # Cache loaded models
    
    def __init__(
        self, 
        model_name: str,
        use_quantization: bool = True,
        device: str = "auto"
    ):
        if model_name in self._instances:
            self.model = self._instances[model_name]["model"]
            self.tokenizer = self._instances[model_name]["tokenizer"]
        else:
            self.model_name = model_name
            self.use_quantization = use_quantization and torch.cuda.is_available()
            self._load_model()
            self._instances[model_name] = {
                "model": self.model,
                "tokenizer": self.tokenizer
            }
    
    def _load_model(self):
        print(f" Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        print(f" Model loaded successfully")
    
    def chat_completion(
        self, 
        messages: List[Dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        prompt = self._format_messages(messages)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages based on model type"""
        
        # Detect model family
        model_lower = self.model_name.lower()
        
        if "llama-3" in model_lower or "llama3" in model_lower:
            return self._format_llama3(messages)
        elif "hermes" in model_lower:
            return self._format_hermes(messages)
        elif "mixtral" in model_lower or "mistral" in model_lower:
            return self._format_mistral(messages)
        else:
            return self._format_generic(messages)
    
    def _format_llama3(self, messages: List[Dict]) -> str:
        formatted = "<|begin_of_text|>"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>"
            elif role == "user":
                formatted += f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"
            elif role == "assistant":
                formatted += f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>"
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n"
        return formatted
    
    def _format_hermes(self, messages: List[Dict]) -> str:
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted
    
    def _format_mistral(self, messages: List[Dict]) -> str:
        formatted = ""
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                if i == 0:
                    formatted = f"<s>[INST] {content}\n\n"
                else:
                    formatted += f"{content}\n\n"
            elif role == "user":
                if formatted and not formatted.endswith("[INST] "):
                    formatted += f"<s>[INST] {content} [/INST]"
                else:
                    formatted += f"{content} [/INST]"
            elif role == "assistant":
                formatted += f" {content}</s>\n"
        return formatted
    
    def _format_generic(self, messages: List[Dict]) -> str:
        formatted = ""
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted += f"{role}: {content}\n\n"
        formatted += "Assistant:"
        return formatted


# Model factory for each agent role
def get_planner_model():
    return HFModelManager("mistralai/Mistral-7B-Instruct-v0.2")

def get_research_model():
    return HFModelManager("NousResearch/Hermes-2-Pro-Llama-3-8B")

def get_writer_model():
    return HFModelManager("mistralai/Mixtral-8x7B-Instruct-v0.1")

def get_editor_model():
    return HFModelManager("meta-llama/Meta-Llama-3-8B-Instruct")

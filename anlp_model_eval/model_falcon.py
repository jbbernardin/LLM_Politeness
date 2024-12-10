# model_falcon.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class FalconModel:
    def __init__(self, device='cuda'):
        model_name = "tiiuae/falcon-7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.device = device

    def generate_answer(self, context, question):
        prompt = f"{context}\n\nQ: {question}\nA:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=2000, max_new_tokens=128, do_sample=False)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        answer_part = answer.split("A:")[-1].strip()
        return answer_part
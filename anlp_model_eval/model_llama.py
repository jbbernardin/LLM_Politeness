# model_llama.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaModel:
    def __init__(self, device='cuda'):
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.device = device

    def generate_answer(self, context, question):
        prompt = f"<s>[INST] <<SYS>>You are a helpful assistant.<</SYS>>\n{context}\n\nUser: {question}\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=2000, max_new_tokens=128, do_sample=False)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract the assistant's answer (this might vary based on model prompting format)
        # We'll assume the assistant's answer follows "Assistant:"
        if "Assistant:" in answer:
            answer_part = answer.split("Assistant:")[-1].strip()
        else:
            answer_part = answer
        return answer_part

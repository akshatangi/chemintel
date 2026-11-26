"""Pluggable text generators for the demo and for API-based evaluation.

hf     : local Hugging Face model (optionally with a LoRA adapter). CPU is slow but works for small models.
openai : any OpenAI-compatible chat endpoint (OpenAI, Ollama at http://localhost:11434/v1, LM Studio, vLLM, ...).
Nothing here ships a key or a model; the user supplies them.
"""
import json, os, urllib.request


class HFGenerator:
    def __init__(self, model, adapter=None):
        from .modeling import load_model
        self.tok, self.model = load_model(model, adapter=adapter)

    def generate(self, prompt, max_new_tokens=300, temperature=0.2):
        import torch
        from .modeling import device_of
        if getattr(self.tok, "chat_template", None):
            prompt = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(device_of(self.model))
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                      temperature=max(temperature, 1e-5) if temperature > 0 else None,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class OpenAICompatGenerator:
    def __init__(self, model, base_url="https://api.openai.com/v1", api_key=None):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "none")

    def generate(self, prompt, max_new_tokens=300, temperature=0.2):
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": max_new_tokens, "temperature": temperature}).encode()
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()


RAG_QA_TEMPLATE = (
    "You are a chemistry assistant. Use the retrieved passages when they help. "
    "Cite passages you use as [1], [2]. If the passages do not contain the answer, say so clearly "
    "and answer from general knowledge, marking it as such.\n\n{ctx}\nQuestion: {q}\nAnswer:")


def build_rag_prompt(question, passages, max_chars=900):
    ctx = "Retrieved passages (ChemPile text, may be irrelevant):\n" + "\n".join(
        f"[{i}] {p['text'][:max_chars]}" for i, p in enumerate(passages, 1)) if passages else ""
    return RAG_QA_TEMPLATE.format(ctx=ctx, q=question)

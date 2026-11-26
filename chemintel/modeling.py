import os


def load_model(model_name, adapter=None, four_bit=None, trainable=False):
    """Load tokenizer + causal LM. 4-bit (QLoRA) is used when CUDA and bitsandbytes are available.
    On CPU (smoke tests only) the model is loaded in fp32 without quantisation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cuda = torch.cuda.is_available()
    mps = (not cuda) and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if four_bit is None:
        four_bit = cuda
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if (cuda and torch.cuda.is_bf16_supported()) else torch.float16)
        kwargs["device_map"] = {"": 0}
    elif cuda:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if mps and not four_bit:
        model = model.to("mps")   # Apple Silicon; default dtype, as in the user's own script. Not tested here (no Mac).
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    if not trainable:
        model.eval()
    return tok, model


def device_of(model):
    return next(model.parameters()).device

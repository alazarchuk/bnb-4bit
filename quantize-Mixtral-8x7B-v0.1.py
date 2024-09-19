import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float32

model_name = "mistralai/Mixtral-8x7B-v0.1"
quant_path = "Mixtral-8x7B-v0.1-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

model.save_pretrained("../Models/" + quant_path, safetensors=True)
tokenizer.save_pretrained("../Models/" + quant_path)

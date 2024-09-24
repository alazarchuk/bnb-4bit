import os

## Saves about 2 Gb of memory, just enough to fit the model in a 24 Gb GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# Check if the current CUDA device supports bfloat16 precision
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float32

# Define the model name and the path where the quantized model will be saved
model_name = "microsoft/Phi-3.5-MoE-instruct"
quant_path = "Phi-3.5-MoE-instruct-bnb-4bit"

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure the quantization settings for loading the model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Specify the quantization type
    bnb_4bit_compute_dtype=compute_dtype,  # Set the compute dtype based on CUDA support
    bnb_4bit_use_double_quant=True,  # Enable double quantization for better accuracy
)

# Load the model with the specified quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, trust_remote_code=True
)

# Save the quantized model and tokenizer to the specified path
model.save_pretrained("../Models/" + quant_path, safetensors=True)
tokenizer.save_pretrained("../Models/" + quant_path)

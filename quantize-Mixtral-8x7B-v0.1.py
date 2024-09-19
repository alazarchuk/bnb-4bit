import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Check if the current CUDA device supports bfloat16 (bf16) precision
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16  # Use bfloat16 for computation if supported
else:
    compute_dtype = torch.float32  # Otherwise, fall back to float32

# Define the model name and the path where the quantized model will be saved
model_name = "mistralai/Mixtral-8x7B-v0.1"
quant_path = "Mixtral-8x7B-v0.1-bnb-4bit"

# Load the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure the quantization settings for loading the model in 4-bit precision
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Specify the quantization type as nf4
    bnb_4bit_compute_dtype=compute_dtype,  # Set the computation data type
    bnb_4bit_use_double_quant=True,  # Enable double quantization
)

# Load the model with the specified quantization configuration
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

# Save the quantized model and tokenizer to the specified directory
model.save_pretrained("../Models/" + quant_path, safetensors=True)
tokenizer.save_pretrained("../Models/" + quant_path)

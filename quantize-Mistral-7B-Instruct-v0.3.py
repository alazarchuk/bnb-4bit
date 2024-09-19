import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Check if the current CUDA device supports bfloat16 precision
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float32

# Define the model name and the path where the quantized model will be saved
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
quant_path = "Mistral-7B-Instruct-v0.3-bnb-4bit"

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
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

# Save the quantized model and tokenizer to the specified path
model.save_pretrained("../Models/" + quant_path, safetensors=True)
tokenizer.save_pretrained("../Models/" + quant_path)

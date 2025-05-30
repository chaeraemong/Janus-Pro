import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from models import MultiModalityCausalLM, VLChatProcessor

# Load the base model
base_model_name = "deepseek-ai/Janus-Pro-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load the processor (e.g., tokenizer + vision processor)
processor: VLChatProcessor = VLChatProcessor.from_pretrained(base_model_name)

# Load the saved LoRA checkpoint
lora_checkpoint_dir = "./janus_lora_act/checkpoint-9933"    # directory saved by Trainer
peft_model = PeftModel.from_pretrained(base_model, lora_checkpoint_dir, torch_dtype=torch.bfloat16)

# Merge LoRA parameters into the base model and remove PEFT layers
merged_model = peft_model.merge_and_unload()                # now merged_model has LoRA weights integrated into the base model

# Save the merged model and processor
output_dir = "./janus_merged_checkpoint-9933"
merged_model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print(f"Merged model saved to {output_dir}")

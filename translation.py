import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model_name = "facebook/m2m100_418M"
model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)

def translate_text(text, src_lang, tgt_lang):
		# Set the tokenizer's source language
		tokenizer.src_lang = src_lang

		# Tokenizer optimization: Adjusting the maximum length
		max_length = 128
		encoded_text = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True).to(device)

		# Specify the forced_bos_token_id in the generate() method instead of modifying the model config
		forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

		# Generate translation using the model
		generated_tokens = model.generate(**encoded_text, forced_bos_token_id=forced_bos_token_id)

		# Decode the generated tokens to get the translated text
		return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

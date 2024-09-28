from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch

tokenizer = AutoTokenizer.from_pretrained("/home/zikaixiao/zikaixiao/LongLoRA-main/models/llama-3-8B-262k")
model = AutoModelForCausalLM.from_pretrained("/home/zikaixiao/zikaixiao/LongLoRA-main/models/llama-3-8B-262k", torch_dtype=torch.float16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
set_seed(42)

text = "On what date was the Declaration of Independence officially signed?"
inputs = tokenizer(text, return_tensors="pt").to(device)

vanilla_output = model.generate(**inputs, do_sample=False, max_new_tokens=50)
print(tokenizer.batch_decode(vanilla_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))

dola_high_output = model.generate(**inputs, do_sample=False, max_new_tokens=50, dola_layers='high')
print(tokenizer.batch_decode(dola_high_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))

dola_custom_output = model.generate(**inputs, do_sample=False, max_new_tokens=50, dola_layers=[28,30], repetition_penalty=1.2)
print(tokenizer.batch_decode(dola_custom_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
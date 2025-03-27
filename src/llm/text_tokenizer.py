from transformers import GPT2Tokenizer

def gpt2_tokenizer(pretrained_model_name_or_path, additional_special_tokens):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                              truncation_side='right', 
                                              additional_special_tokens=list(additional_special_tokens))
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import GPT2Model
import torch


class GPT2ForCausalLMFlashAttn(AutoModelForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, vocab_size=None, debug=False, *model_args, **kwargs):
        gpt2_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        if debug:
            gpt2_config.num_hidden_layers = 2
            return cls(gpt2_config)
        
        model = super().from_pretrained(    
            pretrained_model_name_or_path, *model_args, config=gpt2_config, **kwargs
        )

        if vocab_size is not None:
            print(f'Length of tokenizer and resize embedding: {vocab_size}')
            model.resize_token_embeddings(vocab_size)

        return model
    
    @classmethod
    def from_config(cls, pretrained_model_name_or_path, vocab_size=None, *model_args, **kwargs):
        gpt2_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = super().from_config(gpt2_config, *model_args, **kwargs)

        if vocab_size is not None:
            print(f'Length of tokenizer and resize embedding: {vocab_size}')
            model.resize_token_embeddings(vocab_size)

        return model


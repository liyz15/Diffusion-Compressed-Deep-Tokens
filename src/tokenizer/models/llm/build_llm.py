import sys
import hydra
from omegaconf import OmegaConf

IDENTITY_CONFIG = {
    'target': 'torch.nn.Identity',
}

def build_model(
    agent_config_path: str,
    llm_config_path: str,
    tokenizer_config_path: str,
):
    llm_model_cfg = OmegaConf.load(llm_config_path)
    llm_model = hydra.utils.instantiate(llm_model_cfg)

    agent_model_cfg = OmegaConf.load(agent_config_path)
    agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm_model)

    tokenizer_cfg = OmegaConf.load(tokenizer_config_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)

    return agent_model, tokenizer

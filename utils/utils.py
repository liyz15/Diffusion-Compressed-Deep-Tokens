import importlib
import numpy as np
import cv2
import torch
import torch.distributed as dist
import os
import functools
from collections import OrderedDict


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """ 
    name: full name of source para
    para_list: partial name of target para 
    """
    istarget=False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [np.load(os.path.join(data_dir, data_name))['arr_0'] for data_name in os.listdir(data_dir)]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)['arr_0'] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data   


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )


@functools.lru_cache(maxsize=1)
def load_watermark_template(temp_path):
    if not os.path.exists(temp_path):
        raise FileNotFoundError("Template file not found in {}".format(temp_path))
    with np.load(temp_path) as temp:
        return temp["alpha"], temp["W"], temp["temp_lap"]

def remove_watermark(images, temp_path="./shutterstock_temp.npz"):
    """
    images: list of ndarray
    """

    alpha, W, temp_lap = load_watermark_template(temp_path)

    h, w = images[0].shape[:2]
    temp_h, temp_w = alpha.shape

    # Compute the location offset based on the first image
    # We assume the watermark is always at the same location for the same video
    src_image = images[0]
    image_gray = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
    image_lap = cv2.Laplacian(image_gray, cv2.CV_64F)
    image_lap = (
        (image_lap - image_lap.min()) / (image_lap.max() - image_lap.min()) * 255.0
    )
    image_lap = image_lap.astype(np.uint8)

    res = cv2.matchTemplate(image_lap, temp_lap, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    src_alpha, src_W = np.zeros((h, w)), np.zeros((h, w))

    src_alpha[max_loc[1] : max_loc[1] + temp_h, max_loc[0] : max_loc[0] + temp_w] = (
        alpha
    )
    src_W[max_loc[1] : max_loc[1] + temp_h, max_loc[0] : max_loc[0] + temp_w] = W
    src_alphaW = src_alpha * src_W

    images = [image.astype(np.float32) for image in images]
    images = [
        (image - src_alphaW[:, :, None]) / (1 - src_alpha[:, :, None])
        for image in images
    ]
    images = [np.clip(image, 0, 255).astype(np.uint8) for image in images]
    return images


def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None


def load_checkpoints(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s"%pretrained_ckpt

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        # try:
        if 'state_dict' in pl_sd.keys():
            state_dict = pl_sd['state_dict']
            new_state_dict = OrderedDict()
            
            # Change naming from 256 ckpt
            for k in list(state_dict.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_state_dict[new_key] = state_dict[k]
                else:
                    new_state_dict[k] = state_dict[k]
            
            # Modify in_channels
            in_channels = model_cfg.params.unet_config.params.in_channels
            input_key = 'model.diffusion_model.input_blocks.0.0.weight'
            in_channels_ckpt = new_state_dict[input_key].shape[1]

            if in_channels != in_channels_ckpt:
                if in_channels < in_channels_ckpt:
                    new_state_dict[input_key] = new_state_dict[input_key][:, :in_channels]
                else:
                    new_state_dict[input_key] = torch.cat([
                        new_state_dict[input_key], 
                        torch.zeros_like(new_state_dict[input_key])[:, :in_channels-in_channels_ckpt]], 
                        dim=1)    
            
            # Modify query number
            num_queries = model_cfg.params.image_proj_stage_config.params.num_queries
            if 'video_length' in model_cfg.params.image_proj_stage_config.params:
                raise NotImplementedError("video_length is not None, this will result in num_queries * video_length queries")
                video_length = model_cfg.params.image_proj_stage_config.params.video_length
                num_queries = num_queries * video_length
            
            num_queries_ckpt = new_state_dict['image_proj_model.latents'].shape[1]

            if num_queries != num_queries_ckpt:
                if num_queries < num_queries_ckpt:
                    new_state_dict['image_proj_model.latents'] = new_state_dict['image_proj_model.latents'][:, :num_queries]
                else:
                    new_state_dict['image_proj_model.latents'] = torch.cat([
                        new_state_dict['image_proj_model.latents'], 
                        torch.zeros_like(new_state_dict['image_proj_model.latents'])[:, :num_queries-num_queries_ckpt]], 
                        dim=1)
            
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            keys_to_tolerate = [".alpha", ".to_k_ip", "to_v_ip", "image_proj_model.norm_out"]
            unexpected = [k for k in unexpected if not any([p in k for p in keys_to_tolerate])]
            print("Unexpected keys in pretrained checkpoint: ", unexpected)
            assert len(unexpected) == 0, f"Unexpected keys found in pretrained checkpoint: {unexpected}"
            prefix_to_tolerate = ["image_cat_proj_model", "image_proj_model.final_proj_out", "vae_mean_proj", 
                                  "vae_logvar_proj", "vae_decoder", "image_proj_model.self_attn_layers", 
                                  "llm_agent"]
            print("Missing keys in pretrained checkpoint: ", missing)
            missing = [k for k in missing if not any([k.startswith(p) for p in prefix_to_tolerate])]
            assert len(missing) == 0, f"Missing keys in pretrained checkpoint: {missing}"

        elif 'module' in pl_sd.keys() or '_forward_module.betas' in pl_sd.keys():
            if 'module' in pl_sd.keys():
                pl_sd = pl_sd['module']
            # deepspeed
            new_pl_sd = OrderedDict()
            for key in pl_sd.keys():
                new_pl_sd[key[16:]]=pl_sd[key]
            missing, unexpected = model.load_state_dict(new_pl_sd, strict=False)
            prefix_to_tolerate = ["image_cat_proj_model", "image_proj_model.final_proj_out", "vae_mean_proj", 
                                  "vae_logvar_proj", "vae_decoder", "image_proj_model.self_attn_layers", 
                                  "llm_agent"]
            missing = [k for k in missing if not any([k.startswith(p) for p in prefix_to_tolerate])]
            assert len(missing) == 0, f"Missing keys in pretrained checkpoint: {missing}"
            assert len(unexpected) == 0, f"Unexpected keys found in pretrained checkpoint: {unexpected}"
        else:
            model.load_state_dict(pl_sd)
        # except:
            #     model.load_state_dict(pl_sd)
    else:
        print(">>> Start training from scratch")

    return model
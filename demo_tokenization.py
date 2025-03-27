import os
import sys
import torch
import torchvision
from torchvision import transforms
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from collections import OrderedDict
import time
import argparse

# Add the parent directory to the path to import from lvdm
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tokenizer.models.samplers.ddim import DDIMSampler
from utils.utils import instantiate_from_config
from src.tokenizer.models.ddpm3d import vae_sample_latent

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k, v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        if "module" in state_dict:
            state_dict = state_dict["module"]
        for key in state_dict.keys():
            new_pl_sd[key[16:]] = state_dict[key]
        model.load_state_dict(new_pl_sd)
    print(">>> model checkpoint loaded.")
    return model

def image_guided_synthesis(
    model,
    videos,
    noise_shape,
    ddim_steps=50,
    ddim_eta=1.0,
    unconditional_guidance_scale=7.5,
    cfg_img=None,
    fs=6,
    timestep_spacing="uniform",
    guidance_rescale=0.7,
    video_length=16,
    rm_tkn=-1,
):
    ddim_sampler = DDIMSampler(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    prompts = [""] * batch_size

    assert model.image_proj_type == "headrear"
    assert videos.shape[2] == 2

    img = rearrange(videos, "b c t h w -> (b t) c h w")

    img_emb_unarranged = model.embedder(img)  ## blc
    img_emb_unarranged = model.image_proj_model(img_emb_unarranged)

    if model.use_vae:
        means = model.vae_mean_proj(img_emb_unarranged)
        logvar = model.vae_logvar_proj(img_emb_unarranged)
        img_emb = vae_sample_latent(means, logvar, do_sample=False)
    else:
        img_emb = img_emb_unarranged

    if rm_tkn >= 0:
        # Zero out the token
        img_emb[:, :(rm_tkn + 1)] = 0

    img_emb = rearrange(img_emb, "(b t) l c -> b t l c", t=2)
    assert video_length % 2 == 0
    img_emb = img_emb.repeat_interleave(video_length // 2, dim=1)
    img_emb = rearrange(img_emb, "b t l c -> b (t l) c")

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}

    if model.model.conditioning_key == 'hybrid_concat':
        img_cat_cond = model.image_cat_proj_model(img_emb_unarranged)  # (b t) c h w
        t = 2
        img_cat_cond = rearrange(img_cat_cond, "(b t) c h w -> b c t h w", t=t)
        img_cat_cond = img_cat_cond.repeat_interleave(noise_shape[2] // t, dim=2)
        
        cond["c_concat"] = [img_cat_cond] # b c t h w

    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(videos[:, :, 0]))  ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        if model.use_vae:
            means = model.vae_mean_proj(uc_img_emb)
            logvar = model.vae_logvar_proj(uc_img_emb)
            uc_img_emb = vae_sample_latent(means, logvar, do_sample=False)

        uc_img_emb = repeat(uc_img_emb, "b l c -> b t l c", t=noise_shape[2])
        uc_img_emb = rearrange(uc_img_emb, "b t l c -> b (t l) c")

        uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
        if "c_concat" in cond:
            uc["c_concat"] = cond["c_concat"]
    else:
        uc = None

    kwargs = {"unconditional_conditioning_img_nonetext": None}

    samples, _ = ddim_sampler.sample(
        S=ddim_steps,
        conditioning=cond,
        batch_size=batch_size,
        shape=noise_shape[1:],
        verbose=False,
        unconditional_guidance_scale=unconditional_guidance_scale,
        unconditional_conditioning=uc,
        eta=ddim_eta,
        cfg_img=cfg_img,
        mask=None,
        x0=None,
        fs=fs,
        timestep_spacing=timestep_spacing,
        guidance_rescale=guidance_rescale,
        **kwargs,
    )

    ## reconstruct from latent to pixel space
    batch_images = model.decode_first_stage(samples)  # [b, c, t, h, w]

    return batch_images

def save_video(samples, output_path, fps=8):
    samples = samples.detach().cpu()
    samples = torch.clamp(samples.float(), -1.0, 1.0)
    samples = samples[0, ...]
    samples = (samples + 1.0) / 2.0
    samples = (samples * 255).to(torch.uint8).permute(1, 2, 3, 0)  # thwc
    print(samples.shape)
    torchvision.io.write_video(
        output_path, samples, fps=fps, video_codec="h264", options={"crf": "10"}
    )

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Video tokenization demo")
    parser.add_argument("--ckpt_path", type=str, default='models/pytorch_model.bin',
                        help="Path to the model checkpoint")
    parser.add_argument("--config_path", type=str, default='configs/tokenizer/config.yaml',
                        help="Path to the model config file")
    parser.add_argument("--input_video", type=str, default="assets/videos/main_tokenization/7172413_ori.mp4",
                        help="Path to the input video file")
    parser.add_argument("--output_video", type=str, default="results/tokenizer_output/reconstructed.mp4",
                        help="Path to save the output video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--height", type=int, default=256, help="Video height")
    parser.add_argument("--width", type=int, default=256, help="Video width")
    parser.add_argument("--frame_stride", type=int, default=6, 
                        help="Frame stride (larger value for larger motion)")
    parser.add_argument("--video_length", type=int, default=16, help="Output video length")
    
    args = parser.parse_args()
    
    # Set the random seed
    seed_everything(args.seed)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_video)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model configuration
    config = OmegaConf.load(args.config_path)
    model_config = config.pop("model", OmegaConf.create())
    model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
    
    # Load the model
    model = instantiate_from_config(model_config)
    model.perframe_ae = False  # Default from inference_motion_multiple.py
    model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu"), strict=False)

    model = model.cuda()
    model.eval()
    
    # Read input video and extract frames
    print(f"Reading video from {args.input_video}")
    video, _, _ = torchvision.io.read_video(args.input_video, pts_unit='sec')
    # Convert to [0,1] range and then to [-1,1]
    video = video.float() / 255.0 * 2.0 - 1.0
    
    # Prepare transform for resizing
    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.CenterCrop((args.height, args.width))
    ])
    
    # Extract first and last frame for the model
    num_frames = video.shape[0]
    if num_frames < 2:
        raise ValueError("Video must have at least 2 frames")
    
    first_frame = video[0].permute(2, 0, 1)  # THW -> CHW
    last_frame = video[-1].permute(2, 0, 1)
    
    # Apply transforms
    first_frame = transform(first_frame.unsqueeze(0)).squeeze(0)
    last_frame = transform(last_frame.unsqueeze(0)).squeeze(0)
    
    # Stack frames for model input
    input_frames = torch.stack([first_frame, last_frame], dim=1).unsqueeze(0).cuda()  # [1, C, 2, H, W]
    
    # Prepare noise shape for the model
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    noise_shape = [1, channels, args.video_length, h, w]
    
    # Generate the reconstructed video
    print("Generating reconstructed video...")
    start = time.time()
    
    with torch.no_grad():
        reconstructed_video = image_guided_synthesis(
            model=model,
            videos=input_frames,
            noise_shape=noise_shape,
            ddim_steps=50,
            ddim_eta=1.0,
            unconditional_guidance_scale=7.5,
            fs=args.frame_stride,
            timestep_spacing="uniform",
            guidance_rescale=0.7,
            video_length=args.video_length
        )
    
    # Save the reconstructed video
    save_video(reconstructed_video, args.output_video)
    
    print(f"Reconstructed video saved to {args.output_video}")
    print(f"Time used: {(time.time() - start):.2f} seconds")

if __name__ == "__main__":
    main() 
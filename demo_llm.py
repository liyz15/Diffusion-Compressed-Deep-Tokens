import os, torch
from omegaconf import OmegaConf
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
import torchvision

from src.tokenizer.models.samplers.ddim import DDIMSampler
from src.tokenizer.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config

def load_model():
    """Load the model with default settings"""
    # Default settings from the bash script
    ckpt = "models/gpt2_large.pt"
    config_path = "configs/llm/tokenizer.yaml"
    
    # Load model config
    config = OmegaConf.load(config_path)
    model_config = config.pop("model", OmegaConf.create())
    
    # Set model config parameters
    model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
    model_config.params.llm_config.generate_kwargs.var_scale = 0.01
    model_config.params.llm_config.generate_kwargs.do_sample = True
    model_config.params.llm_config.generate_kwargs.num_frames_to_gen = 2
    
    # Initialize and load model
    model = instantiate_from_config(model_config)
    model = model.cuda()
    model.perframe_ae = False

    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    
    model.eval()
    return model


def encode(model, text_prompt):
    """Encode the text prompt into visual features using the LLM"""
    with torch.no_grad():
        img_emb = model.llm_generate([text_prompt])["pred"][0]
    return img_emb.cpu()


@torch.no_grad()
def image_guided_synthesis(
    model,
    img_emb,
    noise_shape,
    ddim_steps=50,
    ddim_eta=1.0,
    unconditional_guidance_scale=7.5,
    cfg_img=None,
    fs=6,
    multiple_cond_cfg=False,
    timestep_spacing="uniform",
    guidance_rescale=0.7,
    **kwargs,
):
    assert (
        model.image_proj_type == "headrear"
    ), "This evaluation is based on predicted features, only headrear is supported"
    ddim_sampler = (
        DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    )
    batch_size = noise_shape[0]
    assert batch_size == 1, "Error: batch size should be 1 for inference"
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    prompts = [""] * batch_size

    if model.image_proj_type == "headrear":
        # img_emb = rearrange(img_emb, "(b t) l c -> b t l c", t=2)
        assert noise_shape[2] % 2 == 0
        img_emb = img_emb.repeat_interleave(noise_shape[2] // 2, dim=1)
        img_emb = rearrange(img_emb, "b t l c -> b (t l) c")
    else:
        raise NotImplementedError

    cond_emb = model.get_learned_conditioning([""] * batch_size)
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}

    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)

        b, c, t, h, w = noise_shape
        emtpy_img = torch.zeros((b, 3, h * 8, w * 8), device=model.device)
        uc_img_emb = model.embedder(torch.zeros_like(emtpy_img))  ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)  # b l c
        uc_img_emb = repeat(uc_img_emb, "b l c -> b t l c", t=t)
        uc_img_emb = rearrange(uc_img_emb, "b t l c -> b (t l) c")
        uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb, img_emb], dim=1)]}
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

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
        mask=cond_mask,
        x0=z0,
        fs=fs,
        timestep_spacing=timestep_spacing,
        guidance_rescale=guidance_rescale,
        **kwargs,
    )

    ## reconstruct from latent to pixel space
    batch_images = model.decode_first_stage(samples)

    return batch_images


def decode(model, img_emb):
    """Decode the visual features into a video"""
    # Set parameters for video generation
    height, width = 256, 256
    h, w = height // 8, width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = 16
    noise_shape = [1, channels, n_frames, h, w]
    
    samples_list = []
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        # Process each pair of consecutive frames
        for i in range(len(img_emb) - 1):
            current_img_emb = img_emb[[i, i + 1]].unsqueeze(0).to("cuda")
            
            # Use the original image_guided_synthesis function
            batch_samples = image_guided_synthesis(
                model,
                current_img_emb,
                noise_shape,
                ddim_steps=50,
                ddim_eta=1.0,
                unconditional_guidance_scale=7.5,
                cfg_img=None,
                fs=6,  # frame stride
                multiple_cond_cfg=False,
                timestep_spacing="uniform",
                guidance_rescale=0.7,
            )
            samples_list.append(batch_samples.cpu())
        
        # Concatenate all samples
        all_samples = torch.cat(samples_list, dim=2)
        
    return all_samples


def save_video(samples, output_path, fps=8):
    """Save the generated video to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    samples = samples.detach().cpu()
    samples = torch.clamp(samples.float(), -1.0, 1.0)
    samples = samples[0, ...]
    samples = (samples + 1.0) / 2.0
    samples = (samples * 255).to(torch.uint8).permute(1, 2, 3, 0)  # thwc
    
    torchvision.io.write_video(
        output_path, samples, fps=fps, video_codec="h264", options={"crf": "10"}
    )
    print(f"Video saved to {output_path}")


def main():
    """Main function to generate video from text prompt"""
    # Set seed for reproducibility
    seed_everything(42)
    
    prompt = "A time-lapse of clouds moving across a blue sky."
    
    # Generate safe filename from prompt
    safe_filename = "_".join(prompt.split())
    output_path = f"results/llm_output/{safe_filename}.mp4"
    
    print(f"Generating video for prompt: '{prompt}'")
    
    # Load model
    model = load_model()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Encode text to visual features
        print("Encoding text to visual features...")
        img_emb = encode(model, prompt)
        print(f"Generated features shape: {img_emb.shape}")
        
        # Decode features to video
        print("Decoding features to video...")
        video = decode(model, img_emb)
        
    # Save video
    save_video(video, output_path)

if __name__ == "__main__":
    main() 
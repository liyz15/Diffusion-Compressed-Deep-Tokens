import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import LogitsProcessor, LogitsProcessorList
from collections.abc import Iterable
from tqdm import tqdm
import itertools

BOI_TOKEN = "<img>"
EOI_TOKEN = "</img>"
IMG_TOKEN = "<img_{:05d}>"


def gmm_split_params(gmm_params, k, d, var_scale=1.0):
    # Note that returned weights are logits instead of probabilities
    batch_size, num_tokens, _ = gmm_params.shape

    means = gmm_params[..., : k * d].reshape(batch_size, num_tokens, k, d)

    variances = gmm_params[..., k * d : 2 * k * d].reshape(batch_size, num_tokens, k, d)
    variances = torch.clamp(F.softplus(variances), min=1e-5)
    variances = variances * var_scale

    weights = gmm_params[..., 2 * k * d :]

    return means, variances, weights


def compute_gmm_loss(means, variances, weights, targets, loss_balance=None, batch_size=None):
    # loss_balance (sum(num_valid_frames), num_encoder_tokens)
    # Note that the input weights are logits
    assert means.shape[-1] == targets.shape[-1]

    # Create the Gaussian Mixture Model
    mixture = D.Categorical(logits=weights)
    components = D.Independent(D.Normal(means, torch.sqrt(variances)), 1)
    gmm = D.MixtureSameFamily(mixture, components)

    # Compute the negative log-likelihood and scale it by the dimensionality d
    log_probs = gmm.log_prob(targets)
    if loss_balance is not None:
        assert batch_size is not None, "batch_size should be provided for loss balance"
        num_tokens = targets.shape[-2]
        log_probs = log_probs * loss_balance
        nll = -log_probs.sum() / batch_size / num_tokens / targets.shape[-1]
    else:
        nll = -log_probs.mean() / targets.shape[-1]  # Scale NLL by dimension d

    return nll


def gmm_predict(means, weights):
    # Note that the input weights are logits
    weighted_means = torch.einsum("bnkd,bnk->bnd", means, weights.softmax(-1))
    return weighted_means


def gmm_sample_weighted(means, variances, weights):
    # Note that the input weights are logits

    # Reshape means and variances
    std = torch.sqrt(variances)
    normal_dist = D.Normal(means, std)
    samples = normal_dist.sample()  # Shape: (batch_size, num_tokens, k, d)

    probs = weights.softmax(-1)
    samples = (samples * probs.unsqueeze(-1)).sum(-2)

    return samples


def gmm_sample(means, variances, weights):
    # Note that the input weights are logits

    batch_size, num_tokens, k, d = means.shape

    # Reshape weights and sample component indices
    weights_flat = weights.view(-1, k)  # Flatten to (batch_size * num_tokens, k)
    mixture_dist = D.Categorical(logits=weights_flat)
    component_indices = mixture_dist.sample()  # Shape: (batch_size * num_tokens,)

    # Reshape means and variances and select based on sampled indices
    means_flat = means.view(-1, k, d)  # Flatten to (batch_size * num_tokens, k, d)
    variances_flat = variances.view(-1, k, d)  # Same as means

    # Use advanced indexing to select the means and variances
    means_selected = means_flat[
        torch.arange(means_flat.size(0)), component_indices
    ]  # Shape: (batch_size * num_tokens, d)
    variances_selected = variances_flat[
        torch.arange(variances_flat.size(0)), component_indices
    ]

    # Compute standard deviations and sample from the normal distributions
    std_selected = torch.sqrt(variances_selected)
    normal_dist = D.Normal(means_selected, std_selected)
    samples_flat = normal_dist.sample()  # Shape: (batch_size * num_tokens, d)

    # Reshape samples back to (batch_size, num_tokens, d)
    samples = samples_flat.view(batch_size, num_tokens, d)

    return samples


def gmm_params_to_predictions(params, k, do_sample=False, var_scale=1.0, weighted_sample=False):
    # The output of GMM loss is GMM params, we need to sample from it
    d = (params.shape[-1] - k) // k // 2
    assert 2 * k * d + k == params.shape[-1], \
        "Invalid GMM params, k = {}, 2 * k * d + k = {}".format(k, params.shape[-1])
    means, variances, weights = gmm_split_params(params, k, d, var_scale=var_scale)
    if do_sample:
        if weighted_sample:
            predictions = gmm_sample_weighted(means, variances, weights)
        else:
            predictions = gmm_sample(means, variances, weights)
    else:
        predictions = gmm_predict(means, weights)
    
    return predictions


def vae_split_params(params):
    # We assume the params are in the form of (batch_size, num_tokens, 2 * d)
    d = params.shape[-1] // 2
    assert 2 * d == params.shape[-1], "Invalid VAE params, 2 * d = {}".format(params.shape[-1])

    means = params[..., :d]
    logvar = params[..., d:]

    return means, logvar


def vae_kl_loss(means, logvar):
    variance = torch.exp(logvar)
    kl_loss = -0.5 * (1 + logvar - means.pow(2) - variance).sum(-1)
    return kl_loss.mean()


def vae_sample_latent(means, logvar, do_sample=True):
    std = torch.exp(0.5 * logvar)
    if do_sample:
        eps = torch.randn_like(logvar)
        latent = means + eps * std
    else:
        latent = means
    return latent


def vae_params_to_predictions(params, vae_decoder, do_sample=True):
    means, logvar = vae_split_params(params)

    latent = vae_sample_latent(means, logvar, do_sample=do_sample)
    latent = vae_sample_latent(means, logvar, do_sample=do_sample)
    predictions = vae_decoder(latent)

    return predictions


def vec_loss(
    rec,
    target,
    loss_type="cosine",
    reduction="mean",

    # GMM loss params
    num_gmm_kernel=None,
    gmm_with_mse_and_l1=False,

    # VAE loss params
    vae_decoder=None,
    kl_loss_scale=0.001,
    loss_balance=None,
    batch_size=None,
):
    if loss_type == "cosine":
        assert reduction == "mean", "cosine loss only supports mean reduction"
        assert loss_balance is None, "loss balance is only supported for gmm loss"
        return 1 - F.cosine_similarity(rec, target, dim=-1).mean()
    elif loss_type == "mse":
        assert reduction in [
            "mean",
            "batchmean",
        ], "mse loss only supports mean or batchmean reduction, got {}".format(
            reduction
        )
        assert loss_balance is None, "loss balance is only supported for gmm loss"
        loss = F.mse_loss(rec, target, reduction="none")
        if (
            reduction == "batchmean" and loss.dim() > 1
        ):  # Sum over all dimensions except batch
            loss = loss.sum(dim=-1)
        return loss.mean()
    elif loss_type == "l1":
        assert reduction == "mean"
        assert loss_balance is None, "loss balance is only supported for gmm loss"
        loss = F.l1_loss(rec, target)
        return loss
    elif loss_type == "gmm":
        assert reduction == "mean"
        assert num_gmm_kernel is not None, "num_gmm_kernel should be provided"
        means, variances, weights = gmm_split_params(
            rec, num_gmm_kernel, target.shape[-1]
        )
        gmm_loss = compute_gmm_loss(
            means, variances, weights, target, loss_balance, batch_size
        )

        if gmm_with_mse_and_l1:
            assert loss_balance is None
            gmm_preds = gmm_predict(means, weights)
            mse_loss = vec_loss(gmm_preds, target, loss_type="mse", reduction="mean")
            l1_loss = vec_loss(gmm_preds, target, loss_type="l1", reduction="mean")
            gmm_loss = gmm_loss + mse_loss + l1_loss
        return gmm_loss
    elif loss_type == "vae":
        assert reduction == "mean"
        assert loss_balance is None, "loss balance is only supported for gmm loss"
        means, logvar = vae_split_params(rec)
        kl_loss = vae_kl_loss(means, logvar)
        latent = vae_sample_latent(means, logvar)
        sample = vae_decoder(latent)
        mse_loss = vec_loss(sample, target, loss_type="mse", reduction="mean")
        return mse_loss + kl_loss * kl_loss_scale
    elif loss_type == "l1l2":
        assert reduction == "mean"
        assert loss_balance is None, "loss balance is only supported for gmm loss"
        l1_loss = vec_loss(rec, target, loss_type="l1", reduction=reduction)
        l2_loss = vec_loss(rec, target, loss_type="mse", reduction=reduction)
        return l1_loss + l2_loss
    elif loss_type == "l1l2cos":
        assert reduction == "mean"
        assert loss_balance is None, "loss balance is only supported for gmm loss"
        l1_loss = vec_loss(rec, target, loss_type="l1", reduction=reduction)
        l2_loss = vec_loss(rec, target, loss_type="mse", reduction=reduction)
        cos_loss = vec_loss(rec, target, loss_type="cosine", reduction=reduction)
        return l1_loss + l2_loss + cos_loss
    else:
        return ValueError(f"loss_type {loss_type} not supported")


@torch.no_grad()
def per_frame_loss(rec, target, valid_mask, loss_type="cosine"):
    if loss_type == "cosine":
        loss = 1 - F.cosine_similarity(rec, target, dim=-1)
    elif loss_type == "mse":
        loss = F.mse_loss(rec, target, reduction="none").mean(-1)
    else:
        raise ValueError(f"loss_type {loss_type} not supported")

    # Calculate per frame loss
    num_valid_frames = valid_mask.sum(-1)
    frame_idx = torch.cat([torch.arange(n) for n in num_valid_frames]).to(loss.device)
    max_num_valid_frames = num_valid_frames.max().item()

    loss = loss.reshape(len(frame_idx), -1)
    per_frame_loss = {
        "per_frame_loss_{}".format(i): torch.tensor(-1.0, device=loss.device)
        for i in range(16)
    }
    for i in range(max_num_valid_frames):
        key = "per_frame_loss_{}".format(i)
        per_frame_loss[key] = loss[frame_idx == i].mean()

    return per_frame_loss


@torch.no_grad()
def inter_frame_distance(target, valid_mask):
    # target: (sum(num_valid_frames), num_img_tokens, dim)
    # valid_mask: (bz, num_frames)
    num_valid_frames = valid_mask.sum(-1)
    frame_idx = torch.cat([torch.arange(n) for n in num_valid_frames]).to(target.device)
    max_num_valid_frames = num_valid_frames.max().item()

    frame_distance = (
        F.mse_loss(target[:-1], target[1:], reduction="none").mean(-1).mean(-1)
    )  # (sum(num_valid_frames) - 1,)
    valid_distance_mask = frame_idx[:-1] < frame_idx[1:]

    frame_distance = frame_distance[
        valid_distance_mask
    ]  # (sum(num_valid_frames) - bz,)
    frame_idx = frame_idx[1:][valid_distance_mask]

    inter_frame_distance = {
        "inter_frame_distance_{}".format(i): torch.tensor(-1.0, device=target.device)
        for i in range(16)
    }
    for i in range(1, max_num_valid_frames):
        key = "inter_frame_distance_{}".format(i)
        inter_frame_distance[key] = frame_distance[frame_idx == i].mean()

    return inter_frame_distance

@torch.no_grad()
def text_vision_norm(hidden_states, input_ids, video_ids_mask):
    """
    hidden_states: n_layers x (bz, seq_len, dim)
    inputs_embeds: (bz, seq_len, dim)
    video_ids_mask: (bz, seq_len), True for valid video ids
    """
    final_norm = {}
    for l, hidden_state in enumerate(hidden_states):
        norm = torch.norm(hidden_state, dim=-1)
        norm = norm.float()

        vision_norm = norm[video_ids_mask]
        final_norm[f"vision_norm_layer_{l}"] = vision_norm.mean()

        boi_mask = (input_ids == 32001)
        boi_mask = torch.logical_or(boi_mask, input_ids == 2)
        boi_mask = boi_mask.long()
        first_true_locs = torch.where(boi_mask.any(dim=1), boi_mask.argmax(dim=1), -1)
        text_ids_mask = torch.zeros_like(video_ids_mask)
        for i in range(len(first_true_locs)):
            text_ids_mask[i, : first_true_locs[i]] = True
        text_norm = norm[text_ids_mask]
        final_norm[f"text_norm_layer_{l}"] = text_norm.mean()
    
    final_norm["text_norm"] = torch.mean(torch.tensor([final_norm[key] for key in final_norm.keys() if "text_norm" in key]))
    final_norm["vision_norm"] = torch.mean(torch.tensor([final_norm[key] for key in final_norm.keys() if "vision_norm" in key]))
            
    return final_norm


class VideoPosToken(nn.Module):
    def __init__(self, num_img_tokens=64, max_length=512, dim=5120):
        super().__init__()
        self.num_img_tokens = num_img_tokens
        self.max_length = max_length
        self.dim = dim

        max_num_frames = max_length // num_img_tokens
        self.frame_token = nn.Parameter(torch.randn(max_num_frames, 1, dim))
        self.pos_token = nn.Parameter(torch.randn(1, num_img_tokens, dim))

        self._init_weight()

    def _init_weight(self):
        nn.init.trunc_normal_(self.frame_token, std=0.02)
        nn.init.trunc_normal_(self.pos_token, std=0.02)

    def forward(self, video_embeds, video_frame_mask):
        """
        video_embeds: (sum(num_valid_frames), num_img_tokens, dim)
        video_frame_mask: (bz, num_frames), True for valid video frames
        """
        num_valid_frames = video_frame_mask.sum(-1)
        assert (num_valid_frames > 0).all()

        frame_idx = torch.cat([torch.arange(n) for n in num_valid_frames]).to(
            video_embeds.device
        )

        return video_embeds + self.frame_token[frame_idx] + self.pos_token
        # try:
        # except:
        #     assert False, (
        #         video_embeds.shape,
        #         self.frame_token.shape,
        #         self.frame_token[frame_idx].shape,
        #         self.pos_token.shape,
        #     )

    def add_single(self, video_embeds, frame_idx, img_idx):
        return (
            video_embeds + self.frame_token[[frame_idx]] + self.pos_token[:, [img_idx]]
        )


class ContinuousLVLMSimple(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        input_resampler: nn.Module,
        output_resampler: nn.Module,
        pre_input_resampler=None,
        num_img_tokens=64,
        max_length=512,
        use_cosine_loss=True,
        llm_loss_weight=1.0,
        rec_loss_weight=1.0,
        llm_rec_loss_weight=0.0,
        pre_normalizer=None,
        loss_reduction="mean",
        loss_func="mse",
        num_gmm_kernel=16,
        gmm_with_mse_and_l1=False,
        vae_decoder: nn.Module = None,
        vae_kl_loss_scale=0.001,
        log_norms=False,
        balance_loss=False,
    ) -> None:
        super().__init__()
        self.llm = llm

        # 1. Pre input resampler
        if pre_input_resampler is not None:
            self.pre_input_resampler = pre_input_resampler
        else:
            self.pre_input_resampler = nn.Identity()

        # 2. Normalize the input
        if pre_normalizer is not None:
            # This normalizer is used to scale the output of pre_input_resampler
            # The embeds outside this model is always not normalized
            pre_normalizer = torch.tensor(pre_normalizer).view(1, -1, 1)
            self.register_buffer("pre_normalizer", pre_normalizer)
        else:
            self.pre_normalizer = None

        self.input_resampler = input_resampler
        self.output_resampler = output_resampler
        if use_cosine_loss:
            print(
                "Warning: use_cosine_loss is a deprecated argument and should be set to False, use loss_func instead"
            )
            assert (
                loss_func == "cosine"
            ), "loss_func should be cosine if use_cosine_loss is True, got {}".format(
                loss_func
            )
            self.loss_func = "cosine"
        else:
            self.loss_func = loss_func
        
        if self.loss_func == "vae":
            assert vae_decoder is not None, "vae_decoder should be provided for VAE loss"
        self.vae_decoder = vae_decoder
        self.vae_kl_loss_scale = vae_kl_loss_scale

        self.loss_reduction = loss_reduction

        if llm is not None:
            dim = llm.config.hidden_size
            self.video_pos_token = VideoPosToken(
                num_img_tokens=num_img_tokens,
                max_length=max_length,
                dim=dim,
            )

        self.rec_loss_weight = rec_loss_weight
        self.llm_loss_weight = llm_loss_weight
        self.llm_rec_loss_weight = llm_rec_loss_weight

        self.num_gmm_kernel = num_gmm_kernel
        self.gmm_with_mse_and_l1 = gmm_with_mse_and_l1

        self.log_norms = log_norms

        self.balance_loss = balance_loss

        self.set_trainable()
        self.init_weights()

    def set_trainable(self):
        # The pre_input_resampler is pretrained
        self.pre_input_resampler.requires_grad_(False)

        self.input_resampler.requires_grad_(True)
        self.output_resampler.requires_grad_(True)

        if self.llm is not None:
            self.llm.requires_grad_(True)
            self.video_pos_token.requires_grad_(True)
        
        if self.vae_decoder is not None:
            self.vae_decoder.requires_grad_(True)

    def params_to_opt(self):
        opt_params = itertools.chain(
            self.input_resampler.parameters(),
            self.output_resampler.parameters(),
        )
        if self.llm is not None:
            opt_params = itertools.chain(
                opt_params,
                self.llm.parameters(),
                self.video_pos_token.parameters(),
            )
        if self.vae_decoder is not None:
            opt_params = itertools.chain(
                opt_params,
                self.vae_decoder.parameters(),
            )

        return opt_params
    
    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        
        self.input_resampler.apply(_init_weights)
        self.output_resampler.apply(_init_weights)
        if self.vae_decoder is not None:
            self.vae_decoder.apply(_init_weights)

    def forward_resampler(
        self,
        input_ids,
        video_ids_mask,
        video_frame_mask,
        video_embeds,
        output_pred=False,
    ):
        video_ids_mask = video_ids_mask.bool()
        video_frame_mask = video_frame_mask.bool()

        video_embeds = video_embeds[video_frame_mask.view(-1)]
        video_embeds = self.pre_input_resampler(video_embeds).detach()
        if self.pre_normalizer is not None:
            video_embeds = video_embeds / self.pre_normalizer

        video_embeds_in = self.input_resampler(video_embeds)
        if self.loss_func == "cosine":
            video_embeds_in = F.normalize(video_embeds_in, dim=-1)

        video_embeds_out = self.output_resampler(video_embeds_in)

        rec_loss = vec_loss(
            video_embeds_out,
            video_embeds,
            loss_type=self.loss_func,
            reduction=self.loss_reduction,
        )

        output = {
            "total_loss": rec_loss,
        }

        if output_pred:
            output["pred"] = video_embeds_out

        return output

    def forward(
        self,
        input_ids,
        video_ids_mask,
        video_frame_mask,
        video_embeds,
        query_video_ids_mask=None,
        gt_video_frame_mask=None,
        attention_mask=None,
        position_ids=None,
        return_pred=False,
    ):
        """
        input_ids: (bz, seq_len)
        video_ids_mask: (bz, seq_len), True for valid video ids
        video_frame_mask: (bz, num_frames), True for valid video frames
        video_embeds: (bz * num_frames, num_encoder_tokens, dim)
        """
        if self.llm is None:  # Only train resampler
            if query_video_ids_mask is not None:
                raise NotImplementedError
            return self.forward_resampler(
                input_ids,
                video_ids_mask,
                video_frame_mask,
                video_embeds,
                output_pred=return_pred,
            )

        # Make sure all masks are boolean
        video_ids_mask = video_ids_mask.bool()
        video_frame_mask = video_frame_mask.bool()

        # 1. PREPARE AND RESAMPLE INPUT
        # Extract valid video frames, (sum(num_valid_frames), num_encoder_tokens, dim)
        # This is the final target
        video_embeds = video_embeds[video_frame_mask.view(-1)]
        video_embeds = self.pre_input_resampler(video_embeds).detach()
        if self.pre_normalizer is not None:
            # This needs to be scaled back at inference
            video_embeds = video_embeds / self.pre_normalizer

        # Resample video embeds, (sum(num_valid_frames), num_img_token, dim)
        video_embeds_in = self.input_resampler(video_embeds)
        num_img_tokens = video_embeds_in.shape[1]
        if self.loss_func == "cosine":
            video_embeds_in = F.normalize(video_embeds_in, dim=-1)

        if self.balance_loss:
            # Prepare loss weight here, we want every sample to be equal regardless of the number of frames
            loss_balance = torch.zeros(
                tuple(video_frame_mask.shape) + (num_img_tokens,), device=video_embeds.device
            )  # (bz, num_frames, num_img_tokens)
            loss_balance[video_frame_mask] = 1.0
            loss_balance = loss_balance / (
                loss_balance.sum(dim=(1, 2), keepdim=True) + 1e-6
            ) * num_img_tokens
            loss_balance = loss_balance[video_frame_mask]
            batch_size = video_frame_mask.shape[0]
        else:
            loss_balance = None
            batch_size = None

        # 2. DETACHED RECONSTRUCTION LOSS
        # Resample back (sum(num_valid_frames), num_encoder_tokens, dim)
        video_embeds_out = self.output_resampler(video_embeds_in)
        if self.loss_func not in ["gmm", "vae"]:
            rec_loss = vec_loss(
                video_embeds_out,
                video_embeds,
                loss_type=self.loss_func,
                reduction=self.loss_reduction,
            )
        else:
            # Skip this when using GMM loss as the dim differs
            rec_loss = torch.tensor(0.0, device=video_embeds.device)

        # Detach here to avoid collapse, no gradient flow to resampler from here
        # If we train end2end with llm, we should not detach here
        if self.rec_loss_weight > 0:
            assert self.llm_rec_loss_weight == 0, "llm_rec_loss_weight should be 0"
            video_embeds_in = video_embeds_in.detach()
        else:
            assert self.llm_rec_loss_weight > 0

        # 3. PREPARE LLM INPUT
        # Embed text input tokens, (bz, seq_len, dim)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        # Add video pos token, (sum(num_valid_frames), dim)
        video_embeds_in_with_pos = self.video_pos_token(
            video_embeds_in, video_frame_mask
        )
        if gt_video_frame_mask is not None:
            gt_video_frame_mask = gt_video_frame_mask.bool()
            # When use_bidirectional_query, video_frame_mask contains num_frames frames
            # But we only need (num_frames - 1) as GT embeds in LLM
            # Here we construct a mask of shape (sum(num_valid_frames),)
            gt_mask = gt_video_frame_mask[video_frame_mask]
            video_embeds_in_with_pos = video_embeds_in_with_pos[gt_mask]

        # Reshape video embeds, (sum(num_valid_frames) * num_img_token, dim)
        video_embeds_in_with_pos = video_embeds_in_with_pos.reshape(
            -1, video_embeds_in_with_pos.shape[-1]
        )
        # Put video frames into the input sequence
        inputs_embeds[video_ids_mask] = video_embeds_in_with_pos

        # 4. LLM LOSS
        # Inference wiht LLM. We will compute the loss later.
        output = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        if self.log_norms:
            tv_norm = text_vision_norm(output.hidden_states, input_ids, video_ids_mask)
        last_hidden_state = output.hidden_states[-1]  # (bz, seq_len, dim)

        if gt_video_frame_mask is None:
            # Auto regressive mode
            # n-th prediction is the (n+1)-th token
            # get the prediction for (0, 1, ..., seq_len-2) tokens
            # # (sum(num_valid_frames) * num_img_tokens, dim)
            shift_mask = video_ids_mask[..., 1:]
            llm_pred = last_hidden_state[..., :-1, :]
            llm_pred = llm_pred[shift_mask]
        else:
            # Each query predict one token
            llm_pred = last_hidden_state[query_video_ids_mask]

        if self.loss_func not in ["gmm", "vae"]:
            llm_loss = vec_loss(
                llm_pred,
                # No matter whether video_embeds_in is detached, we need to detach it here
                video_embeds_in.reshape(-1, video_embeds_in.shape[-1]).detach(),
                loss_type=self.loss_func,
                reduction=self.loss_reduction,
            )
        else:
            # Skip this when using GMM loss as the dim differs
            llm_loss = torch.tensor(0.0, device=video_embeds.device)

        # 4. END TO END RECONSTRUCTION LOSS
        llm_pred_unflatten = llm_pred.reshape(-1, num_img_tokens, llm_pred.shape[-1])
        video_embeds_out = self.output_resampler(llm_pred_unflatten)
        llm_rec_loss = vec_loss(
            video_embeds_out,
            video_embeds,
            loss_type=self.loss_func,
            reduction=self.loss_reduction,
            # GMM params
            num_gmm_kernel=self.num_gmm_kernel,
            gmm_with_mse_and_l1=self.gmm_with_mse_and_l1,
            # VAE params
            vae_decoder=self.vae_decoder,
            kl_loss_scale=self.vae_kl_loss_scale,
            loss_balance=loss_balance,
            batch_size=batch_size,
        )

        total_loss = (
            llm_loss * self.llm_loss_weight
            + rec_loss * self.rec_loss_weight
            + llm_rec_loss * self.llm_rec_loss_weight
        )

        output = {}

        if self.log_norms:
            output.update(tv_norm)
        
        # For log purpose
        if self.loss_func == "vae":
            with torch.no_grad():
                means, log_vars = vae_split_params(video_embeds_out)
                kl_loss = vae_kl_loss(means, log_vars)
            
            output.update({"vae_kl_loss": kl_loss})

        output.update(
            {
                "total_loss": total_loss,
                "llm_loss": llm_loss,
                "rec_loss": rec_loss,
                "llm_rec_loss": llm_rec_loss,
            }
        )

        if return_pred:
            output["pred"] = self.output_resampler(
                llm_pred.reshape(-1, num_img_tokens, llm_pred.shape[-1])
            )

        return output

    @classmethod
    def from_pretrained(
        cls,
        llm,
        input_resampler,
        output_resampler=None,
        pretrained_model_path=None,
        skip_pre_input_resampler=False,
        **kwargs,
    ):
        debug = kwargs.pop("debug", False)
        model = cls(
            llm=llm,
            input_resampler=input_resampler,
            output_resampler=output_resampler,
            **kwargs,
        )
        if debug:
            print("Debugging mode, not loading pretrained model")
            return model
        if pretrained_model_path is not None:
            print(
                "Loading pretrained agent model from {}".format(pretrained_model_path)
            )
            ckpt = torch.load(pretrained_model_path, map_location="cpu")
            
            # Check if the checkpoint is from deepspeed stage 1
            if "input_resampler.pos_embed" not in ckpt and "module" in ckpt:
                print("Loading deepspeed stage 1 checkpoint")
                ckpt = ckpt["module"]

            # Check if the checkpoint is from qwen resampler
            if "input_resampler.pos_embed" in ckpt:
                src_in_token = ckpt["input_resampler.pos_embed"].shape[0]
                tgt_in_token = input_resampler.pos_embed.shape[0]
                if src_in_token != tgt_in_token:
                    print(
                        "Resampling position embeddings from {} to {}".format(
                            src_in_token, tgt_in_token
                        )
                    )
                    ckpt["input_resampler.pos_embed"] = ckpt[
                        "input_resampler.pos_embed"
                    ][:tgt_in_token]
                    ckpt["input_resampler.query"] = ckpt["input_resampler.query"][
                        :tgt_in_token
                    ]

            # Check if the last layer need to be ignored
            if (
                model.state_dict()["output_resampler.2.weight"].shape
                != ckpt["output_resampler.2.weight"].shape
            ):
                print("Output resampler weight shape mismatch, ignoring the last layer")
                ckpt.pop("output_resampler.2.weight")
                ckpt.pop("output_resampler.2.bias")
            
            if (
                # A single-modal norm checkpoint
                "llm.base_model.model.model.layers.0.input_layernorm.original_module.weight" in ckpt and
                # A multi-modal norm model
                "llm.base_model.model.model.layers.0.input_layernorm.original_module.norms.0.weight" in model.state_dict()
            ):
                ori_llm_norm_keys = [
                    k for k in ckpt.keys() if k.startswith("llm") and 
                    any([norm_key in k for norm_key in ["input_layernorm", "post_attention_layernorm", "norm"]])
                ]
                new_llm_norm_keys = [
                    k for k in model.state_dict().keys() if k.startswith("llm") and
                    any([norm_key in k for norm_key in ["input_layernorm", "post_attention_layernorm", "norm"]])
                ]
                for k in new_llm_norm_keys:
                    # TODO: This only handles two modalities
                    ori_k = k.replace("norms.0.", "").replace("norms.1.", "")
                    ckpt[k] = ckpt[ori_k].detach().clone()
                
                for k in ori_llm_norm_keys:
                    del ckpt[k]
                
            if skip_pre_input_resampler:
                ckpt = {k: v for k, v in ckpt.items() if "pre_input_resampler" not in k}

            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(
                "agent model, missing keys: ",
                missing,
                "\n",
                "unexpected keys:",
                unexpected,
            )
        return model

    @torch.inference_mode()
    def generate(
        self,
        text_prompt=None,
        tokenizer=None,
        text_embeds=None,
        video_embeds=None,
        num_img_tokens=64,
        num_frames_to_gen=None,
        add_sep=False,
        use_bidirectional_query=False,
        do_sample=False,
        var_scale=1.0,
        weighted_sample=False,
        text_length=None,
        disable_tqdm=False,
        cfg_scale=1.0,
        **kwargs,
    ):
        device = next(self.parameters()).device
        assert not (add_sep and use_bidirectional_query), "add_sep and use_bidirectional_query cannot be both True"
        # 1. Prepare text input
        if text_prompt is None and text_embeds is None:
            print(
                "WARNING: no text input provided, replacing text_prompt with empty text"
            )
            text_prompt = ""

        if text_embeds is None and text_prompt is not None:
            if tokenizer is None:
                raise ValueError("tokenizer should be provided if text_prompt is used")
            
            if isinstance(text_prompt, str):
                text_prompt = [text_prompt]
            
            text_prompt = [s.strip() for s in text_prompt]
            num_text_prompt = len(text_prompt)
            if cfg_scale > 1.0:  # Add an empty text prompt for classifer free guidance
                assert text_length is not None, "text_length should be provided for classifer free guidance"
                text_prompt = text_prompt + [""] * num_text_prompt

            if text_length is None and len(text_prompt) > 1:
                raise ValueError(
                    "text_length should be provided for padding if multiple text prompts are used"
                )
            
            if not hasattr(self, "boi_token_input_id"):  # Only need to compute once
                self.boi_token_input_id = tokenizer(
                    BOI_TOKEN, return_tensors="pt", add_special_tokens=False
                ).input_ids
                self.boi_token_embed = self.llm.get_input_embeddings()(
                    self.boi_token_input_id.to(device=device)
                )
            
            if text_length is not None:
                input_ids = tokenizer(
                    text_prompt, max_length=text_length, padding='max_length',
                    truncation=True, add_special_tokens=False, return_tensors='pt',
                ).input_ids
            else:
                input_ids = tokenizer(
                    text_prompt, add_special_tokens=False, return_tensors="pt"
                ).input_ids
            
            input_ids = torch.cat(
                [input_ids, self.boi_token_input_id.expand(len(input_ids), 1)], dim=1
            )

            text_embeds = self.llm.get_input_embeddings()(
                input_ids.to(device=device)
            )

        # 2. Prepare video input
        if video_embeds is not None:
            if self.pre_normalizer is not None:
                video_embeds = video_embeds / self.pre_normalizer
            
            video_embeds = self.input_resampler(video_embeds)
            if self.loss_func == "cosine":
                video_embeds = F.normalize(video_embeds, dim=-1)

            bz, existing_num_frames, _, _ = video_embeds.shape
            video_embeds = video_embeds.reshape(bz * existing_num_frames, num_img_tokens, -1)
            video_embeds = self.video_pos_token(
                video_embeds,
                torch.ones(bz, existing_num_frames).bool(),
            )

            video_embeds = video_embeds.reshape(bz, existing_num_frames, num_img_tokens, -1)
            video_embeds = torch.cat(
                [video_embeds, self.boi_token_embed.expand(bz, existing_num_frames, 1, -1)], dim=2
            )
            video_embeds = video_embeds.reshape(bz, -1, video_embeds.shape[-1])
            if text_embeds is not None:
                inputs_embeds = torch.cat([text_embeds, video_embeds], dim=1)
            else:
                inputs_embeds = video_embeds
        else:
            assert (
                text_embeds is not None
            ), "text_embeds should be provided if video_embeds is None"
            inputs_embeds = text_embeds

        # 3. Compute target number of tokens
        max_length = kwargs.pop("max_length", 512)
        if not add_sep:
            max_num_frames_to_gen = (
                max_length - inputs_embeds.shape[1]
            ) // num_img_tokens
        else:
            # Plus one here for the first boi token
            max_num_frames_to_gen = (max_length - inputs_embeds.shape[1] + 1) // (
                num_img_tokens + 1
            )

        if num_frames_to_gen is None or num_frames_to_gen > max_num_frames_to_gen:
            num_frames_to_gen = max_num_frames_to_gen

        num_visual_tokens_to_gen = num_frames_to_gen * num_img_tokens

        # 4. Generate
        start = 0 if video_embeds is None else video_embeds.shape[1]
        pred = []
        if not use_bidirectional_query:
            num_tokens_to_gen_at_once = 1
            past_key_values = None
        else:  
            # Generate num_img_tokens tokens at once.
            # Pass the inputs_embeds before the loop starts such that the input_embeds in the loop
            # is always query_embeds, this makes it easier to deal with attention mask
            num_tokens_to_gen_at_once = num_img_tokens

            output = self.llm(
                inputs_embeds=inputs_embeds,
                return_dict=True,
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values = output.past_key_values

            # Pre-compute query tokens
            query_tokens = "".join([IMG_TOKEN.format(i) for i in range(num_img_tokens)])
            query_ids = tokenizer(
                query_tokens, return_tensors="pt", add_special_tokens=False
            )
            query_embeds = self.llm.get_input_embeddings()(
                query_ids.input_ids.to(device=device)
            )
            inputs_embeds = query_embeds
        
        iterater = range(start, start + num_visual_tokens_to_gen, num_tokens_to_gen_at_once)
        if not disable_tqdm:
            iterater = tqdm(iterater)

        for i in iterater:
            if use_bidirectional_query:
                attention_mask = torch.ones(  # Sees all previous tokens and itself
                    (1, query_embeds.shape[1], query_embeds.shape[1] + past_key_values[-1][0].shape[2]),
                    device=query_embeds.device, dtype=torch.bool
                )
            else:  # Causal mask
                attention_mask = None

            output = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )

            # With past_key_vales, we can optionally pass only one token as input
            inputs_embeds = output.hidden_states[-1][:, -num_tokens_to_gen_at_once:, :]
            if self.loss_func == "cosine":
                inputs_embeds = F.normalize(inputs_embeds, dim=-1)
            
            # Prepare for next iter
            if (
                self.rec_loss_weight == 0.0
                and self.llm_rec_loss_weight == 1.0
                and self.llm_loss_weight == 0.0
            ):  # Make predictions based on the hidden_state
                inputs_embeds = self.output_resampler(inputs_embeds)
                if self.loss_func == "gmm":
                    # The outputs of GMM loss is GMM params, sample from it
                    inputs_embeds = gmm_params_to_predictions(inputs_embeds, self.num_gmm_kernel, do_sample, var_scale=var_scale, weighted_sample=weighted_sample)
                elif self.loss_func == "vae":
                    inputs_embeds = vae_params_to_predictions(inputs_embeds, self.vae_decoder, do_sample)


            if cfg_scale > 1.0:
                # The last prediction is the classifer free guidance
                inputs_embeds[:num_text_prompt] = inputs_embeds[num_text_prompt:] + (
                    inputs_embeds[:num_text_prompt] - inputs_embeds[num_text_prompt:]
                ) * cfg_scale
                inputs_embeds[num_text_prompt:] = inputs_embeds[:num_text_prompt]

            # Save the predictions for output video embeds
            pred.append(inputs_embeds)

            if (
                self.rec_loss_weight == 0.0
                and self.llm_rec_loss_weight == 1.0
                and self.llm_loss_weight == 0.0
            ):
                # In this case, we need to pass the input to input resampler
                # Because the predediction is original video embeds
                inputs_embeds = self.input_resampler(inputs_embeds)
            
            # TODO: This is hard to understand, refactor
            inputs_embeds = torch.cat(
                [
                    self.video_pos_token.add_single(inputs_embeds[:, [j - i], :], j // num_img_tokens, j % num_img_tokens)
                    for j in range(i, i + num_tokens_to_gen_at_once)
                ],
                dim=1,
            )

            # If add seperator, the next token of every num_img_tokens is a seperator
            # Append here
            if add_sep and (i + 1) % num_img_tokens == 0:
                inputs_embeds = torch.cat([inputs_embeds, self.boi_token_embed.expand(
                    len(inputs_embeds), 1, -1
                )], dim=1)
            
            # We need to pass the real video embeds to the next iteration
            if use_bidirectional_query:
                output = self.llm(
                    inputs_embeds=inputs_embeds,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                inputs_embeds = query_embeds  # Always use query_embeds as input for next iteration

            past_key_values = output.past_key_values

        pred = torch.cat(pred, dim=1)
        pred = pred.reshape(pred.shape[0], -1, num_img_tokens, pred.shape[-1])

        if self.pre_normalizer is not None:  # Scale back for output
            pred = pred * self.pre_normalizer

        return {
            "pred": pred,
            "text_embeds": text_embeds,
        }

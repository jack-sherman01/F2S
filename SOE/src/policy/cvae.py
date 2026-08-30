import torch
import torch.nn as nn

from policy.diffusion import DiffusionUNetPolicy
from policy.img_encoder.multi_image_obs_encoder import MultiImageObsEncoder
from policy.vqvae_modules.vqvae import EncoderMLP
import einops

class CVAE(nn.Module):
    def __init__(
        self, 
        num_action = 20,
        obs_shape_meta = dict(
            color_image = dict(
                shape = [3, 64, 64],
                type = 'rgb'                    
            )
        ),
        action_dim = 10,
        use_group_norm = True,
        resnet_out_features = 64,
        readout_dim = None,
        hidden_dim = None,
        
        style_dim = 64, 
        kl_target_mu_is_zero = True,
        kl_weight = 1.0,
        
        # parameters passed to step
        **kwargs
    ):
        super().__init__()
        self.action_encoder_mu = EncoderMLP(
            input_dim=num_action * action_dim, output_dim=style_dim
        )
        self.action_encoder_logvar = EncoderMLP(
            input_dim=num_action * action_dim, output_dim=style_dim
        )

        num_obs = 1
        if not obs_shape_meta:
            print("Empty obs_shape_meta detected, assuming unconditional policy.")
            self.img_encoder = None
            obs_feature_dim = 0
        else:
            self.img_encoder = MultiImageObsEncoder(
                shape_meta = dict(
                    action = dict(shape = [action_dim]),
                    obs = obs_shape_meta,
                ),
                use_group_norm = use_group_norm,
                share_rgb_model = False,
                imagenet_norm = True,
                resnet_out_features = resnet_out_features
            )
            obs_feature_dim = self.img_encoder.output_shape()[0]
        print("obs_feature_dim:", obs_feature_dim)
        if readout_dim is not None:
            assert hidden_dim is not None
            self.bottleneck = EncoderMLP(
                input_dim=obs_feature_dim, 
                output_dim=readout_dim,
                hidden_dim=hidden_dim,
            )
        else:
            self.bottleneck = None
            readout_dim = obs_feature_dim
        self.style_dim = style_dim
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, readout_dim+style_dim, **kwargs)
        self.action_dim = action_dim
        self.kl_target_mu_is_zero = kl_target_mu_is_zero
        self.kl_weight = kl_weight

    def forward(
            self, obs_dict, actions = None,
            return_intermediate = False,
            style = None, # unlike ACT, the default style is sampled from randn, instead of zeros
        ):
        if style is None:
            if actions is not None:
                flattened_action = einops.rearrange(actions, "B T Da -> B (T Da)")
                mu = self.action_encoder_mu(flattened_action)
                logvar = self.action_encoder_logvar(flattened_action)
                std = torch.exp(logvar / 2)
                eps = torch.randn_like(std)
                style = mu + eps * std
            else:
                style = None

        if self.img_encoder is not None:
            readout = self.img_encoder(obs_dict)
            if self.bottleneck is not None:
                readout = self.bottleneck(readout)
        else:
            readout = None
        
        if style is None:
            if readout is not None:
                style = torch.randn((readout.shape[0], self.style_dim), device=readout.device)
            else:
                # assume batch size of 1
                style = torch.randn((1, self.style_dim), device=next(self.action_decoder.model.parameters()).device)

        readout = torch.cat([readout, style], dim=-1) if readout is not None else style

        if return_intermediate:
            assert actions is None, "currently only support inference-time returning intermediate actions"
            assert self.img_encoder is not None, "unconditional policy does not support returning intermediate actions currently"
            with torch.no_grad():
                action_pred, action_list = self.action_decoder.predict_action_and_return_intermediate(readout)
            return action_pred, action_list

        if actions is not None:
            raw_loss = self.action_decoder.compute_weighted_loss(readout, actions)
            pred_loss = raw_loss.mean()
            if self.kl_target_mu_is_zero:
                kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
            else:
                kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - logvar.exp(), dim=-1))
            loss = self.kl_weight * kl_loss + pred_loss
            return {"kl_loss": kl_loss, "pred_loss": pred_loss, "loss": loss}
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred

    
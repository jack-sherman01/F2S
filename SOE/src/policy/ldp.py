import torch
import torch.nn as nn
import einops

from policy.diffusion import DiffusionUNetPolicy
from policy.img_encoder.multi_image_obs_encoder import MultiImageObsEncoder
from policy.vqvae_modules.vqvae import EncoderMLP

class LDP(nn.Module):
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
        n_latent_dims = 64,
        use_group_norm = True,
        resnet_out_features = 64,
        readout_dim = None,
        hidden_dim = None,

        # parameters passed to step
        **kwargs
    ):
        super().__init__()
        
        # AE
        self.action_encoder = EncoderMLP(
            input_dim=num_action * action_dim, output_dim=n_latent_dims
        )
        self.action_decoder = EncoderMLP(
            input_dim=n_latent_dims, output_dim=num_action * action_dim
        )
        self.action_dim = action_dim

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
        self.latent_diffusion = DiffusionUNetPolicy(n_latent_dims, 1, num_obs, readout_dim, kernel_size=1, **kwargs)

    def forward(
            self, obs_dict, actions = None
        ):
        if self.img_encoder is not None:
            readout = self.img_encoder(obs_dict)
            if self.bottleneck is not None:
                readout = self.bottleneck(readout)
        else:
            readout = None

        if actions is not None:
            latent_actions = einops.rearrange(actions, "B T Da -> B (T Da)")
            latent_actions = self.action_encoder(latent_actions)
            latent_actions = einops.rearrange(latent_actions, "B (T Da) -> B T Da", T=1)
            raw_loss = self.latent_diffusion.compute_weighted_loss(readout, latent_actions.detach())
            loss = raw_loss.mean()
            return loss
        else:
            with torch.no_grad():
                latent_action_pred = self.latent_diffusion.predict_action(readout)
                latent_action_pred = einops.rearrange(latent_action_pred, "B T Da -> B (T Da)")
                action_pred = self.action_decoder(latent_action_pred)
                action_pred = einops.rearrange(action_pred, "B (T Da) -> B T Da", Da=self.action_dim)
            return action_pred

    def get_latent_action(
            self, obs_dict, actions = None
        ):
        readout = self.img_encoder(obs_dict)
        if self.bottleneck is not None:
            readout = self.bottleneck(readout)
        latent_action_pred = self.latent_diffusion.predict_action(readout)
        return latent_action_pred
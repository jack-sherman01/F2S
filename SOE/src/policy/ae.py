import torch
import torch.nn as nn

from policy.diffusion import DiffusionUNetPolicy
from policy.vqvae_modules.vqvae import EncoderMLP
import einops

class AE(nn.Module):
    def __init__(
        self, 
        num_action = 20,
        action_dim = 10,
        obs_shape_meta = None,
        n_latent_dims = 64, 
        decoder_type = "diffusion",
        
        # parameters passed to step
        **kwargs
    ):
        super().__init__()
        self.action_encoder = EncoderMLP(
            input_dim=num_action * action_dim, output_dim=n_latent_dims
        )

        num_obs = 1
        if decoder_type == "diffusion":
            self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, n_latent_dims, **kwargs)
        elif decoder_type == "mlp":
            self.action_decoder = EncoderMLP(
                input_dim=n_latent_dims, output_dim=num_action * action_dim
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        self.decoder_type = decoder_type

    def forward(
            self, obs_dict, actions = None
        ):
        readout = einops.rearrange(actions, "B T Da -> B (T Da)")
        readout = self.action_encoder(readout)
        if self.decoder_type == "diffusion":
            loss = self.action_decoder.compute_loss(readout, actions)
        elif self.decoder_type == "mlp":
            pred_actions = self.action_decoder(readout)
            loss = nn.functional.mse_loss(
                einops.rearrange(pred_actions, "B (T Da) -> B T Da", T=actions.shape[1]),
                actions
            )
        return loss
    
    def get_latent_action(
            self, obs_dict, actions = None
        ):
        readout = einops.rearrange(actions, "B T Da -> B (T Da)")
        readout = self.action_encoder(readout)
        return readout
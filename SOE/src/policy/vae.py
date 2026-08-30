import torch
import torch.nn as nn

from policy.diffusion import DiffusionUNetPolicy
from policy.vqvae_modules.vqvae import EncoderMLP
import einops

# modified, not vanilla VAE
class VAE(nn.Module):
    def __init__(
        self, 
        num_action = 20,
        action_dim = 10,
        obs_shape_meta = None,
        n_latent_dims = 64, 
        kl_target_mu_is_zero = False,
        
        # parameters passed to step
        **kwargs
    ):
        super().__init__()
        self.action_encoder_mu = EncoderMLP(
            input_dim=num_action * action_dim, output_dim=n_latent_dims
        )
        self.action_encoder_logvar = EncoderMLP(
            input_dim=num_action * action_dim, output_dim=n_latent_dims
        )

        num_obs = 1
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, n_latent_dims, **kwargs)
        self.kl_target_mu_is_zero = kl_target_mu_is_zero

    def forward(
            self, obs_dict, actions = None
        ):
        readout = einops.rearrange(actions, "B T Da -> B (T Da)")
        # readout = self.action_encoder(readout)
        mu = self.action_encoder_mu(readout)
        logvar = self.action_encoder_logvar(readout)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        readout = mu + eps * std

        if self.kl_target_mu_is_zero:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            kl_loss = -0.5 * torch.sum(1 + logvar - logvar.exp())
        pred_loss = self.action_decoder.compute_loss(readout, actions)
        loss = kl_loss + pred_loss

        return {"kl_loss": kl_loss, "pred_loss": pred_loss, "loss": loss}
    
    def get_latent_action(
            self, obs_dict, actions = None
        ):
        readout = einops.rearrange(actions, "B T Da -> B (T Da)")
        mu = self.action_encoder_mu(readout)
        logvar = self.action_encoder_logvar(readout)
        return mu, logvar
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.diffusion import DiffusionUNetPolicy
from policy.img_encoder.multi_image_obs_encoder import MultiImageObsEncoder
from policy.vqvae_modules.vqvae import EncoderMLP

class MIDP(nn.Module):
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
        weight_type = None,
        use_group_norm = True,
        resnet_out_features = 64,
        readout_dim = None,
        hidden_dim = None,
        style_dim = 4,
        dist_factor = None,
        dist_factor_scale = 2.0,
        constraint_type = "readout", # 'readout' or 'action' or 'loss'
        fixed_lagrange_multiplier = None,

        # parameters passed to step
        **kwargs
    ):
        super().__init__()
        num_obs = 1
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
        
        assert readout_dim is not None
        assert hidden_dim is not None
        self.style_dim = style_dim
        self.readout_dim = readout_dim
        self.bottleneck = EncoderMLP(
            input_dim=obs_feature_dim + self.style_dim,
            output_dim=readout_dim,
            hidden_dim=hidden_dim,
        )

        self.style_discriminator = EncoderMLP(
            input_dim=readout_dim*2,
            output_dim=self.style_dim,
            hidden_dim=hidden_dim,
        )
        self.log_lagrange_multiplier = nn.Parameter(torch.zeros(1))
        if dist_factor is None:
            self.dist_factor = dist_factor_scale * math.sqrt(self.style_dim) / math.sqrt(readout_dim)
        else:
            self.dist_factor = dist_factor
        self.constraint_type = constraint_type
        self.fixed_lagrange_multiplier = fixed_lagrange_multiplier

        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, readout_dim, **kwargs)
        self.action_dim = action_dim
        self.weight_type = weight_type

    def compute_action_dist(self, action_pred, action_pred_origin):
        return torch.norm(action_pred - action_pred_origin, dim=-1).mean()

    def forward(
            self, obs_dict, actions = None, 
            return_intermediate = False, 
            weights = None,
            style = None,
        ):
        obs_embedding = self.img_encoder(obs_dict)
        if style is None:
            # sample style
            style = torch.randn(obs_embedding.shape[0], self.style_dim, device=obs_embedding.device)
        readout = self.bottleneck(torch.cat([obs_embedding, style], dim=-1))

        if return_intermediate:
            assert actions is None, "currently only support inference-time returning intermediate actions"
            assert self.img_encoder is not None, "unconditional policy does not support returning intermediate actions currently"
            with torch.no_grad():
                action_pred, action_list = self.action_decoder.predict_action_and_return_intermediate(readout)
            return action_pred, action_list

        if actions is not None:
            # loss = self.action_decoder.compute_loss(readout, actions)
            raw_bc_loss = self.action_decoder.compute_weighted_loss(readout, actions, weights=weights, weight_type=self.weight_type)
            bc_loss = raw_bc_loss.mean()

            style_origin = torch.zeros_like(style, device=style.device)
            readout_origin = self.bottleneck(torch.cat([obs_embedding, style_origin], dim=-1))
            style_pred = self.style_discriminator(torch.cat([readout, readout_origin], dim=-1))
            # style_loss = -torch.sum(style_pred * style, dim=-1).mean()
            style_loss = F.mse_loss(style_pred, style, reduction='mean')
            style_loss = style_loss.mean()

            style_dist_to_origin = torch.norm(style_pred - style_origin, dim=-1)
            style_dist_to_origin_mean = torch.mean(style_dist_to_origin)
            readout_dist_to_origin = torch.norm(readout - readout_origin, dim=-1)
            readout_dist_to_origin_mean = torch.mean(readout_dist_to_origin)    
            if self.constraint_type == "action":
                with torch.no_grad():
                    action_pred = self.action_decoder.predict_action(readout)
                    action_pred_origin = self.action_decoder.predict_action(readout_origin)
                    action_dist_to_origin = self.compute_action_dist(action_pred, action_pred_origin)
                lagrange_penalty = torch.mean(torch.clamp(
                    style_dist_to_origin - action_dist_to_origin.detach() * self.dist_factor, min=-1e-6))
            elif self.constraint_type == "readout":
                # lagrange_penalty = torch.mean(style_dist_to_origin - readout_dist_to_origin * self.dist_factor)
                lagrange_penalty = torch.mean(torch.clamp(
                    style_dist_to_origin - readout_dist_to_origin * self.dist_factor, min=-1e-6))
                # lagrange_penalty_non_neg = torch.clamp(lagrange_penalty, min=-1e-6)
            elif self.constraint_type == "loss":
                # print("raw_bc_loss shape:", raw_bc_loss.shape)
                # print("style_dist_to_origin shape:", style_dist_to_origin.shape)
                lagrange_penalty = torch.mean(torch.clamp(
                    style_dist_to_origin - torch.mean(raw_bc_loss, dim=-1).detach() * self.dist_factor, min=-1e-6))
            else:
                raise ValueError("Unknown constraint type: {}".format(self.constraint_type))

            if self.fixed_lagrange_multiplier is None:
                lagrange_multiplier = torch.exp(self.log_lagrange_multiplier) # ensure positive
                style_loss = style_loss + lagrange_multiplier.detach() * lagrange_penalty - lagrange_multiplier * lagrange_penalty.detach()
            else:
                style_loss = style_loss + lagrange_penalty * self.fixed_lagrange_multiplier

            loss = bc_loss + style_loss
            ret_dict = {
                "loss": loss,
                "bc_loss": bc_loss,
                "style_loss": style_loss,
                "lagrange_penalty": lagrange_penalty,
                # "lagrange_multiplier": lagrange_multiplier,
                "style_dist_to_origin_mean": style_dist_to_origin_mean,
                "readout_dist_to_origin_mean": readout_dist_to_origin_mean,
            }
            if self.fixed_lagrange_multiplier is None:
                ret_dict["lagrange_multiplier"] = torch.exp(self.log_lagrange_multiplier)
            if self.constraint_type == "action":
                ret_dict["action_dist_to_origin"] = action_dist_to_origin
            return ret_dict
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred

    def get_latent_action(
            self, obs_dict, actions = None, style = None,
        ):
        obs_embedding = self.img_encoder(obs_dict)
        if style is None:
            # sample style
            style = torch.randn(obs_embedding.shape[0], self.style_dim, device=obs_embedding.device)
        readout = self.bottleneck(torch.cat([obs_embedding, style], dim=-1))
        return readout
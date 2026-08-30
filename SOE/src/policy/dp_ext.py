import torch
import torch.nn as nn
import numpy as np

from policy.diffusion import DiffusionUNetPolicy
from policy.img_encoder.multi_image_obs_encoder import MultiImageObsEncoder
from policy.vqvae_modules.vqvae import EncoderMLP

class DPExt(nn.Module):
    def __init__(
        self, 
        num_action = 20,
        obs_shape_meta = dict(
            color_image = dict(
                shape = [3, 64, 64],
                type = 'rgb'                    
            )
        ),
        resize_shape = None,
        crop_shape = None,
        random_crop = True,
        action_dim = 10, 
        weight_type = None,
        use_group_norm = True,
        resnet_out_features = 64,
        readout_dim = None,
        hidden_dim = None,
        style_dim = 4,

        predict_gaussian = False,
        kl_weight = 1.0,
        recon_loss_weight = 0.0,
        ext_loss_weight = 1.0,
        use_mu_in_recon = False,

        # parameters passed to step
        **kwargs
    ):
        super().__init__()
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
                resize_shape = resize_shape,
                crop_shape = crop_shape,
                random_crop = random_crop,
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

        self.extension_down_module = EncoderMLP(
            input_dim=readout_dim, 
            output_dim=style_dim if not predict_gaussian else style_dim * 2,
            hidden_dim=hidden_dim,
        )
        self.extension_up_module = EncoderMLP(
            input_dim=style_dim,
            output_dim=readout_dim,
            hidden_dim=hidden_dim,
        )
        self.style_dim = style_dim
        self.enable_exploration_extension = False
        self.noise_scale = 0.5

        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, readout_dim, **kwargs)
        self.action_dim = action_dim
        self.weight_type = weight_type

        self.predict_gaussian = predict_gaussian
        self.kl_weight = kl_weight
        self.recon_loss_weight = recon_loss_weight
        self.ext_loss_weight = ext_loss_weight
        self.use_mu_in_recon = use_mu_in_recon

    def fuse_readout(self, readout, readout_from_ext, w = 1.0):
        # use readout_from_ext only by default
        return readout + w * (readout_from_ext - readout)

    def forward(
            self, obs_dict, actions = None, 
            return_intermediate = False, 
            weights = None,
            debug = False,
            std_mask = None,
            uniform_exploration = False
        ):
        training = actions is not None
        
        if self.img_encoder is not None:
            readout = self.img_encoder(obs_dict)
            if self.bottleneck is not None:
                readout = self.bottleneck(readout)
        else:
            readout = None

        # print("readout:", readout)
        if training or self.enable_exploration_extension:
            readout_from_ext = self.extension_down_module(readout.detach())
            if self.predict_gaussian:
                mu, logvar = readout_from_ext.chunk(2, dim=-1)
                std = torch.exp(0.5 * logvar)
                if debug:
                    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
                    print("mu:", mu[0].cpu().detach().numpy())
                    print("std:", std[0].cpu().detach().numpy())
                if training:
                    readout_from_ext = mu + std * torch.randn_like(std)
                    if self.use_mu_in_recon:
                        readout_from_ext_mu = self.extension_up_module(mu)
                elif self.enable_exploration_extension:
                    if not uniform_exploration:
                        std_noise = torch.randn_like(std)
                    else:
                        std_noise = torch.tensor(
                            np.linspace(-2, 2, num=std.shape[0], dtype=np.float32).reshape(-1, 1).repeat(std.shape[1], axis=1),
                            device=std.device
                        )
                    if std_mask is not None:
                        readout_from_ext = mu + std * std_noise * torch.tensor(std_mask, device=readout.device) * self.noise_scale
                    else:
                        # print("readout_from_ext:", readout_from_ext) 
                        readout_from_ext = mu + std * std_noise * self.noise_scale
                if debug:
                    print("readout_from_ext:", readout_from_ext.cpu().detach().numpy())
            else:
                if self.enable_exploration_extension:
                    readout_from_ext = readout_from_ext + torch.randn_like(readout_from_ext) * self.noise_scale
            readout_from_ext = self.extension_up_module(readout_from_ext)
            # print("readout_from_ext:", readout_from_ext)

        if self.enable_exploration_extension:
            readout = self.fuse_readout(readout, readout_from_ext)

        if return_intermediate:
            assert training, "currently only support inference-time returning intermediate actions"
            assert self.img_encoder is not None, "unconditional policy does not support returning intermediate actions currently"
            with torch.no_grad():
                action_pred, action_list = self.action_decoder.predict_action_and_return_intermediate(readout)
            return action_pred, action_list

        if training:
            # loss = self.action_decoder.compute_loss(readout, actions)
            pred_loss = self.action_decoder.compute_weighted_loss(
                readout, actions, weights=weights, weight_type=self.weight_type
            ).mean()

            ext_loss = self.action_decoder.compute_weighted_loss(
                readout_from_ext, actions, weights=weights, weight_type=self.weight_type
            ).mean()

            if self.use_mu_in_recon:
                recon_loss = torch.mean(
                    (readout.detach() - readout_from_ext_mu) ** 2
                )
            else:
                recon_loss = torch.mean(
                    (readout.detach() - readout_from_ext) ** 2
                )

            if self.predict_gaussian:
                kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
            else:
                kl_loss = torch.tensor(0.0, device=readout.device)

            loss = pred_loss + ext_loss * self.ext_loss_weight + kl_loss * self.kl_weight + recon_loss * self.recon_loss_weight
            return {
                "loss": loss,
                "pred_loss": pred_loss,
                "ext_loss": ext_loss,
                "kl_loss": kl_loss,
                "recon_loss": recon_loss,
            }
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred

    def backward(self, loss):
        loss["pred_loss"].backward(retain_graph=True)
        # save initial param requires_grad state
        requires_grad_state = {}
        for name, param in self.action_decoder.named_parameters():
            requires_grad_state[name] = param.requires_grad
            param.requires_grad = False
        (loss["ext_loss"] * self.ext_loss_weight).backward(retain_graph=True)
        if self.predict_gaussian:
            (loss["kl_loss"] * self.kl_weight).backward(retain_graph=True)
        (loss["recon_loss"] * self.recon_loss_weight).backward()
        # restore initial param requires_grad state
        for name, param in self.action_decoder.named_parameters():
            param.requires_grad = requires_grad_state[name]

    def get_latent_action(
            self, obs_dict, actions = None
        ):
        readout = self.img_encoder(obs_dict)
        if self.bottleneck is not None:
            readout = self.bottleneck(readout)
        readout_from_ext = self.extension_down_module(readout)
        assert self.predict_gaussian
        mu, logvar = readout_from_ext.chunk(2, dim=-1)
        return mu, logvar
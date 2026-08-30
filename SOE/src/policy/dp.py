import torch
import torch.nn as nn

from policy.diffusion import DiffusionUNetPolicy
from policy.img_encoder.multi_image_obs_encoder import MultiImageObsEncoder
from policy.vqvae_modules.vqvae import EncoderMLP

class DP(nn.Module):
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
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, readout_dim, **kwargs)
        self.action_dim = action_dim
        self.weight_type = weight_type

    def forward(
            self, obs_dict, actions = None, 
            return_intermediate = False, 
            weights = None,
            readout_offset = None,
        ):
        if self.img_encoder is not None:
            readout = self.img_encoder(obs_dict)
            if self.bottleneck is not None:
                readout = self.bottleneck(readout)
        else:
            readout = None
        
        if readout_offset is not None:
            readout = readout + readout_offset
            # print("readout:", readout)

        if return_intermediate:
            assert actions is None, "currently only support inference-time returning intermediate actions"
            assert self.img_encoder is not None, "unconditional policy does not support returning intermediate actions currently"
            with torch.no_grad():
                action_pred, action_list = self.action_decoder.predict_action_and_return_intermediate(readout)
            return action_pred, action_list

        if actions is not None:
            # loss = self.action_decoder.compute_loss(readout, actions)
            raw_loss = self.action_decoder.compute_weighted_loss(readout, actions, weights=weights, weight_type=self.weight_type)
            loss = raw_loss.mean()
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred

    def get_latent_action(
            self, obs_dict, actions = None
        ):
        readout = self.img_encoder(obs_dict)
        if self.bottleneck is not None:
            readout = self.bottleneck(readout)
        return readout
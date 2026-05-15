import torch.nn as nn
import torch.nn.functional as F

from classification.models.vmamba import Permute, VSSBlock
from changedetection.models.dysample import DySample
from changedetection.models.moes.dense_MoE import InteractionAdaptiveMoE


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class SCDMoEUnifiedDecoder(nn.Module):
    """Multi-scale decoder used by SCD-MoE."""

    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super().__init__()
        self.st_block_41 = InteractionAdaptiveMoE(dim=encoder_dims[-1], unified_dim=256, outdim=128)
        self.st_block_31 = InteractionAdaptiveMoE(dim=encoder_dims[-2], unified_dim=256, outdim=128)
        self.st_block_21 = InteractionAdaptiveMoE(dim=encoder_dims[-3], unified_dim=256, outdim=128)
        self.st_block_11 = InteractionAdaptiveMoE(dim=encoder_dims[-4], unified_dim=256, outdim=128)

        self.smooth_layer_3 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_2 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_1 = ResBlock(in_channels=128, out_channels=128, stride=1)

        self.ssm3 = self._make_ssm_block(channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs)
        self.ssm2 = self._make_ssm_block(channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs)
        self.ssm1 = self._make_ssm_block(channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs)

        self.dysample3 = DySample(128)
        self.dysample2 = DySample(128)
        self.dysample1 = DySample(128)

    @staticmethod
    def _make_ssm_block(channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        return nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(
                hidden_dim=128,
                drop_path=0.1,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=kwargs["ssm_d_state"],
                ssm_ratio=kwargs["ssm_ratio"],
                ssm_dt_rank=kwargs["ssm_dt_rank"],
                ssm_act_layer=ssm_act_layer,
                ssm_conv=kwargs["ssm_conv"],
                ssm_conv_bias=kwargs["ssm_conv_bias"],
                ssm_drop_rate=kwargs["ssm_drop_rate"],
                ssm_init=kwargs["ssm_init"],
                forward_type=kwargs["forward_type"],
                mlp_ratio=kwargs["mlp_ratio"],
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=kwargs["mlp_drop_rate"],
                gmlp=kwargs["gmlp"],
                use_checkpoint=kwargs["use_checkpoint"],
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

    def _dyupsample_add_3(self, x, y):
        return self.dysample3(x) + y

    def _dyupsample_add_2(self, x, y):
        return self.dysample2(x) + y

    def _dyupsample_add_1(self, x, y):
        return self.dysample1(x) + y

    def forward(self, pre_features, post_features, istrain=True):
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features
        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        p4 = self.st_block_41(pre_feat_4, post_feat_4, istrain)

        p3 = self.st_block_31(pre_feat_3, post_feat_3, istrain)
        p3 = self.ssm3(p3)
        p3 = self._dyupsample_add_3(p4, p3)
        p3 = self.smooth_layer_3(p3)

        p2 = self.st_block_21(pre_feat_2, post_feat_2, istrain)
        p2 = self.ssm2(p2)
        p2 = self._dyupsample_add_2(p3, p2)
        p2 = self.smooth_layer_2(p2)

        p1 = self.st_block_11(pre_feat_1, post_feat_1, istrain)
        p1 = self.ssm1(p1)
        p1 = self._dyupsample_add_1(p2, p1)
        p1 = self.smooth_layer_1(p1)
        return p1

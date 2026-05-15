import torch.nn as nn
import torch.nn.functional as F

from classification.models.vmamba import LayerNorm2d
from changedetection.models.decoder import SCDMoEUnifiedDecoder
from changedetection.models.Mamba_backbone import Backbone_VSSM
from changedetection.models.moes.dense_MoE import TaskDecouplingMoEHead


class SCDMoE(nn.Module):
    """SCD-MoE model used for all released checkpoints."""

    def __init__(self, output_cd, output_clf, pretrained, **kwargs):
        super().__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)

        norm_layers = {
            "ln": nn.LayerNorm,
            "ln2d": LayerNorm2d,
            "bn": nn.BatchNorm2d,
        }
        act_layers = {
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
        }

        norm_layer = norm_layers.get(kwargs["norm_layer"].lower(), None)
        ssm_act_layer = act_layers.get(kwargs["ssm_act_layer"].lower(), None)
        mlp_act_layer = act_layers.get(kwargs["mlp_act_layer"].lower(), None)
        clean_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in ["norm_layer", "ssm_act_layer", "mlp_act_layer"]
        }

        self.decoder = SCDMoEUnifiedDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )
        self.Task_MoE_Head = TaskDecouplingMoEHead(in_dim=128)

        self.main_clf_cd = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)
        self.aux_clf2 = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)

    def forward(self, pre_data, post_data, istrain=True):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        output_feature = self.decoder(pre_features, post_features, istrain)
        t1_feature, t2_feature, cd_feature = self.Task_MoE_Head(output_feature, istrain)

        output_bcd = self.main_clf_cd(cd_feature)
        output_t1 = self.aux_clf(t1_feature)
        output_t2 = self.aux_clf2(t2_feature)

        output_bcd = F.interpolate(output_bcd, size=pre_data.size()[-2:], mode="bilinear")
        output_t1 = F.interpolate(output_t1, size=pre_data.size()[-2:], mode="bilinear")
        output_t2 = F.interpolate(output_t2, size=post_data.size()[-2:], mode="bilinear")
        return output_bcd, output_t1, output_t2, 0

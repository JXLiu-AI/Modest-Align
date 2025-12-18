import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.models.fusion_models.fusion_outputs import FusionOutputFeatures, FusionOutput
from lavis.models.fusion_models.utils import compute_sim_matrix, slerp


class Block(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(dim, int(expansion_factor * dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(expansion_factor * dim), dim),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.fn(self.ln(x))


# Small network for predicting noise intensity, return value range can be Sigmoid / Softplus, ensuring non-negative
class NoisePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()  # Or nn.Softplus(), depending on the desired range
        )

    def forward(self, x):
        # x: [B, input_dim]
        return self.mlp(x)  # [B, out_dim]


@registry.register_model("mlp_contrastive_fusion")
class MLPContrastiveFusion(BaseModel):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/fusion_models/mlp_contrastive_fusion_base.yaml",
    }

    def __init__(
        self,
        vis_embed_dim,
        text_embed_dim,
        proj_embed_dim,
        proj_bias=True,
        num_layers_vis=1,
        num_layers_text=1,
        expansion_factor=4,
        dropout=0.,
        unimodal_loss_coeff=1.0
    ):
        super().__init__()
        self.vis_embed_dim = vis_embed_dim
        self.text_embed_dim = text_embed_dim
        self.proj_embed_dim = proj_embed_dim
        self.unimodal_loss_coeff = unimodal_loss_coeff
        self.mixup_alpha = -1

        # ---- Original Projection Layers ----
        self.vis_proj = nn.Sequential(
            *[Block(vis_embed_dim, expansion_factor, dropout) for _ in range(num_layers_vis)],
            nn.LayerNorm(vis_embed_dim),
            nn.Linear(vis_embed_dim, proj_embed_dim, bias=proj_bias),
        )

        self.text_proj = nn.Sequential(
            *[Block(text_embed_dim, expansion_factor, dropout) for _ in range(num_layers_text)],
            nn.LayerNorm(text_embed_dim),
            nn.Linear(text_embed_dim, proj_embed_dim, bias=proj_bias),
        )

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.prompt_length = 10
        self.label_smoothing = 0.1

        # =======================================
        # B.1) Small MLP for predicting noise intensity (adaptive)
        # =======================================
        self.vis_noise_predictor = NoisePredictor(input_dim=vis_embed_dim, hidden_dim=256, out_dim=vis_embed_dim)
        self.text_noise_predictor = NoisePredictor(input_dim=text_embed_dim, hidden_dim=256, out_dim=text_embed_dim)

    def proj_vis(self, vis_embed):
        return self.vis_proj(vis_embed)

    def proj_text(self, text_embed):
        return self.text_proj(text_embed)

    def forward(self, samples):
        vis_embed = samples["vis_embed"]   # [B, D_v]
        text_embed = samples["text_embed"] # [B, D_t]
        batch_size = vis_embed.size(0)

        # ---- Adaptive Noise Injection ----
        if self.training:
            # 1) Dynamically predict noise scaling factor
            vis_noise_scale = self.vis_noise_predictor(vis_embed)   # [B, D_v]
            text_noise_scale = self.text_noise_predictor(text_embed) # [B, D_t]

            # 2) Generate random noise and inject
            vis_noise = torch.randn_like(vis_embed)
            text_noise = torch.randn_like(text_embed)

            vis_embed = vis_embed + vis_noise_scale * vis_noise
            text_embed = text_embed + text_noise_scale * text_noise

        # ---- Mixup (Optional) ----
        if self.mixup_alpha > 0 and self.training:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = torch.randperm(batch_size).to(vis_embed.device)
            vis_embed = lam * vis_embed + (1 - lam) * vis_embed[index]
            text_embed = lam * text_embed + (1 - lam) * text_embed[index]

        # ---- Contrastive Learning Subsequent Process ----
        vis_embed_proj = self.proj_vis(vis_embed)
        text_embed_proj = self.proj_text(text_embed)
        vis_embed_proj_norm = F.normalize(vis_embed_proj, dim=-1)
        text_embed_proj_norm = F.normalize(text_embed_proj, dim=-1)

        sim_v2t = vis_embed_proj_norm @ text_embed_proj_norm.T / self.temp
        sim_t2v = text_embed_proj_norm @ vis_embed_proj_norm.T / self.temp

        bs = vis_embed_proj.size(0)
        targets = torch.arange(bs, device=vis_embed.device)

        loss_v2t = F.cross_entropy(sim_v2t, targets)
        loss_t2v = F.cross_entropy(sim_t2v, targets)
        loss = (loss_v2t + loss_t2v) / 2

        return FusionOutput(
            intermediate_output=FusionOutputFeatures(
                vis_embeds=vis_embed,
                vis_embeds_proj=vis_embed_proj,
                text_embeds=text_embed,
                text_embeds_proj=text_embed_proj,
            ),
            loss=loss,
        )

    def compute_sim_matrix(self, data_loader, **kwargs):
        return compute_sim_matrix(model=self, data_loader=data_loader)

    def predict(self, samples):
        vis_embed = samples["vis_embed"]
        target = samples["label"]

        vis_embed_proj_norm = F.normalize(self.proj_vis(vis_embed), dim=-1)
        logits = 100.0 * vis_embed_proj_norm @ self.classifier
        return {"predictions": logits, "targets": target}

    @classmethod
    def from_config(cls, cfg):
        vis_embed_dim = cfg.vis_embed_dim
        text_embed_dim = cfg.text_embed_dim
        proj_embed_dim = cfg.proj_embed_dim
        proj_bias = cfg.get("proj_bias", True)
        num_layers_vis = cfg.get("num_layers_vis", 1)
        num_layers_text = cfg.get("num_layers_text", 1)
        expansion_factor = cfg.get("expansion_factor", 4)
        dropout = cfg.get("dropout", 0.)
        unimodal_loss_coeff = cfg.get("unimodal_loss_coeff", 1.0)

        model = cls(
            vis_embed_dim=vis_embed_dim,
            text_embed_dim=text_embed_dim,
            proj_embed_dim=proj_embed_dim,
            proj_bias=proj_bias,
            num_layers_vis=num_layers_vis,
            num_layers_text=num_layers_text,
            expansion_factor=expansion_factor,
            dropout=dropout,
            unimodal_loss_coeff=unimodal_loss_coeff,
        )

        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            msg = model.load_checkpoint(url_or_filename=pretrain_path)
        else:
            logging.info("No pretrained weights are loaded.")

        return model
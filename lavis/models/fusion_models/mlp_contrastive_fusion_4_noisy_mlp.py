import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.models.fusion_models.fusion_outputs import FusionOutputFeatures, FusionOutput
from lavis.models.fusion_models.utils import compute_sim_matrix, slerp


# -------------------------------------------------
# Basic Modules
# -------------------------------------------------
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


# -------------------------------------------------
# NEW (1) Unified Noise Mixer: Outputs mixing weights alpha and scaling sigma simultaneously
# -------------------------------------------------
class NoiseMixer(nn.Module):
    """
    Input a pooled cross-modal feature, output:
        alpha : [B, k]  -- k noise distribution weights (softmax)
        sigma_v: [B, 1]  -- Visual noise amplitude (softplus)
        sigma_t: [B, 1]  -- Text noise amplitude (softplus)
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_components: int = 4):
        super().__init__()
        self.k = num_components
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Output k + 2 dimensions: first k dimensions logits, last 2 dimensions raw sigma
        self.head = nn.Linear(hidden_dim, num_components + 2)

    def forward(self, feat):
        h = self.mlp(feat)                        # [B, h]
        out = self.head(h)                       # [B, k+2]
        alpha_logits, sigma_raw = out[:, :self.k], out[:, self.k:]
        alpha = F.softmax(alpha_logits, dim=-1)  # Ensure sum is 1
        sigma = F.softplus(sigma_raw)            # Ensure non-negative
        return alpha, sigma  # sigma[:,0]->Visual, sigma[:,1]->Text


# -------------------------------------------------
# Main Model
# -------------------------------------------------
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
        unimodal_loss_coeff=1.0,
        num_noise_components=4,           # NEW: Number of noise distributions (Gaussian/Uniform/Beta/Laplace)
    ):
        super().__init__()
        self.vis_embed_dim = vis_embed_dim
        self.text_embed_dim = text_embed_dim
        self.proj_embed_dim = proj_embed_dim
        self.unimodal_loss_coeff = unimodal_loss_coeff
        self.mixup_alpha = -1
        self.k = num_noise_components

        # -------------- Projection Layers --------------
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

        # Temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.07))

        # -------------- NEW (2): Noise Mixer --------------
        mixer_in_dim = vis_embed_dim + text_embed_dim
        self.noise_mixer = NoiseMixer(
            in_dim=mixer_in_dim,
            hidden_dim=256,
            num_components=self.k
        )

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, samples):
        vis_embed = samples["vis_embed"]      # [B, D_v]
        text_embed = samples["text_embed"]    # [B, D_t]
        device = vis_embed.device
        B = vis_embed.size(0)

        # ---------- 1) Training Phase: Dynamic Noise ----------
        if self.training:

            # (1) Get global pooled features and predict alpha, sigma
            ctx_vis = vis_embed.mean(dim=1) if vis_embed.ndim == 3 else vis_embed
            ctx_txt = text_embed.mean(dim=1) if text_embed.ndim == 3 else text_embed
            ctx = torch.cat([ctx_vis, ctx_txt], dim=-1)       # [B, D_v+D_t]
            alpha, sigma = self.noise_mixer(ctx)              # alpha:[B,k], sigma:[B,2]

            # (2) Sample k types of noise respectively (sample once, then weight by alpha)
            eps_stack_vis  = torch.zeros(B, self.k, *vis_embed.shape[1:], device=device)
            eps_stack_text = torch.zeros(B, self.k, *text_embed.shape[1:], device=device)

            # Gaussian
            eps_stack_vis[:, 0]  = torch.randn_like(vis_embed)
            eps_stack_text[:, 0] = torch.randn_like(text_embed)
            # Uniform
            eps_stack_vis[:, 1]  = torch.rand_like(vis_embed) * 2 - 1
            eps_stack_text[:, 1] = torch.rand_like(text_embed) * 2 - 1
            # Beta(2,2)
            beta = torch.distributions.Beta(2, 2)
            eps_stack_vis[:, 2]  = 2 * beta.sample(vis_embed.shape).to(device) - 1
            eps_stack_text[:, 2] = 2 * beta.sample(text_embed.shape).to(device) - 1
            # Laplace
            lap = torch.distributions.Laplace(0, 1)
            eps_stack_vis[:, 3]  = lap.sample(vis_embed.shape).to(device)
            eps_stack_text[:, 3] = lap.sample(text_embed.shape).to(device)

            # (3) Use alpha to do weighted sum to get mixed noise
            alpha_vis  = alpha.unsqueeze(-1)            # [B,k,1]
            alpha_text = alpha.unsqueeze(-1)
            noise_vis  = (alpha_vis  * eps_stack_vis ).sum(dim=1)
            noise_text = (alpha_text * eps_stack_text).sum(dim=1)

            # (4) Inject after scaling
            vis_embed  = vis_embed  + sigma[:, 0:1] * noise_vis
            text_embed = text_embed + sigma[:, 1:2] * noise_text

        # ---------- 2) (Optional) Mixup ----------
        if self.mixup_alpha > 0 and self.training:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = torch.randperm(B, device=device)
            vis_embed  = lam * vis_embed  + (1 - lam) * vis_embed[index]
            text_embed = lam * text_embed + (1 - lam) * text_embed[index]

        # ---------- 3) Projection + Normalization ----------
        vis_proj  = F.normalize(self.vis_proj(vis_embed),  dim=-1)
        text_proj = F.normalize(self.text_proj(text_embed), dim=-1)

        sim_v2t = vis_proj @ text_proj.T / self.temp
        sim_t2v = text_proj @ vis_proj.T / self.temp

        targets = torch.arange(B, device=device)
        loss = 0.5 * (F.cross_entropy(sim_v2t, targets) +
                      F.cross_entropy(sim_t2v, targets))

        return FusionOutput(
            intermediate_output=FusionOutputFeatures(
                vis_embeds=vis_embed,
                vis_embeds_proj=vis_proj,
                text_embeds=text_embed,
                text_embeds_proj=text_proj,
            ),
            loss=loss,
        )

    # -------------------------------------------------
    # Other auxiliary interfaces and from_config remain unchanged
    # -------------------------------------------------
    def proj_vis(self, vis_embed):
        return self.vis_proj(vis_embed)

    def proj_text(self, text_embed):
        return self.text_proj(text_embed)

    def compute_sim_matrix(self, data_loader, **kwargs):
        return compute_sim_matrix(model=self, data_loader=data_loader)

    def predict(self, samples):
        vis_embed = samples["vis_embed"]
        target = samples["label"]
        vis_proj = F.normalize(self.proj_vis(vis_embed), dim=-1)
        logits = 100.0 * vis_proj @ self.classifier
        return {"predictions": logits, "targets": target}

    @classmethod
    def from_config(cls, cfg):
        model = cls(
            vis_embed_dim      = cfg.vis_embed_dim,
            text_embed_dim     = cfg.text_embed_dim,
            proj_embed_dim     = cfg.proj_embed_dim,
            proj_bias          = cfg.get("proj_bias", True),
            num_layers_vis     = cfg.get("num_layers_vis", 1),
            num_layers_text    = cfg.get("num_layers_text", 1),
            expansion_factor   = cfg.get("expansion_factor", 4),
            dropout            = cfg.get("dropout", 0.),
            unimodal_loss_coeff= cfg.get("unimodal_loss_coeff", 1.0),
            num_noise_components=4,
        )
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            model.load_checkpoint(url_or_filename=pretrain_path)
        else:
            logging.info("No pretrained weights are loaded.")
        return model
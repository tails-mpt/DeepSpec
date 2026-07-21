import os

from deepspec.trainer import Qwen3Eagle3Trainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR


# NVIDIA-Nemotron-3-Nano-30B-A3B (nemotron_h hybrid Mamba-2/MoE) -- v2 recipe.
# Fixes vs v1 (config#1/eagle3_nemotron3_nano): (1) DEPTH-triad aux layers that
# include Mamba positions [2,13,26,39,48] instead of attention-only
# [5,12,26,33,42] -- the only working Eagle3 head for this arch (chankhavu
# Nemotron-Cascade2) captures a Mamba+Attn+Mamba depth triad, NOT attention-only.
# (2) trained on Nemotron self-distill data (target-generated), not raw ShareGPT.
# (3) full 10 epochs (v1 was undertrained at 3, LR hit 0 while acc still climbing).
#
# EAGLE-3.1 "FC-norm" ABLATION: identical to eagle3_nemotron3_nano_v2 except
# fc_norm=True -- RMSNorm the fused aux hidden states before the fusion linear
# (self.fc). Single-variable ablation against the v2 baseline; distinct exp_name
# so it lands in its own checkpoint dir.
NEMOTRON3_NANO = "/mnt/persistent/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

project_name = "deepspec"
exp_name = "eagle3_fcnorm_nemotron3_nano_v2"
seed = 0

model = dict(
    target_model_name_or_path=NEMOTRON3_NANO,
    # DEPTH-based early/mid/late triad (52 layers): ~4/25/50/75/92% depth.
    # 26 is attention (mid); 2,13,39,48 are Mamba/MoE positions -> the draft
    # sees both state-space (long-range) and attention (short-range) features.
    # Strictly increasing, none final (51).
    target_layer_ids=[2, 13, 26, 39, 48],
    ttt_length=7,
    step_loss_decay=0.8,
    draft_num_hidden_layers=1,
    draft_intermediate_size=8064,
    fc_norm=True,
)

train = dict(
    trainer_cls=Qwen3Eagle3Trainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=200,
)

data = dict(
    target_cache_path=None,
    chat_template="nemotron",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg

    return cfg

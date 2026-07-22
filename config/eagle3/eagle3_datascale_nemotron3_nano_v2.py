import os

from deepspec.trainer import Qwen3Eagle3Trainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR


# NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (nemotron_h hybrid Mamba-2/MoE)
# DATA-SCALE variant of the v2 recipe. IDENTICAL to
# config/eagle3/eagle3_nemotron3_nano_v2.py in every hyperparameter (depth-triad
# aux [2,13,26,39,48], ttt_length=7, draft_num_hidden_layers=1, LR, batch, ...);
# the ONLY differences are (a) exp_name (separate ckpt/tb dir) and (b) it is
# trained from a LARGER self-distill corpus cache (the 45k _all + a slice of the
# 350k _extra2 completions). Cross-framework counterpart to the H200 SpecForge
# `v3scale` bet: does data scale lift the DeepSpec eagle3 head too?
# Corpus size + epochs are chosen at launch (train-data-path controls the cache;
# EPOCHS env overrides num_train_epochs) so this file stays a drop-in of v2.
NEMOTRON3_NANO = "/mnt/persistent/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

project_name = "deepspec"
exp_name = "eagle3_datascale_nemotron3_nano_v2"
seed = 0

model = dict(
    target_model_name_or_path=NEMOTRON3_NANO,
    # DEPTH-based early/mid/late triad (52 layers): ~4/25/50/75/92% depth.
    # 26 is attention (mid); 2,13,39,48 are Mamba/MoE positions -> the draft
    # sees both state-space (long-range) and attention (short-range) features.
    # Strictly increasing, none final (51). MUST match the cache's aux layers.
    target_layer_ids=[2, 13, 26, 39, 48],
    ttt_length=7,
    step_loss_decay=0.8,
    draft_num_hidden_layers=1,
    draft_intermediate_size=8064,
)

train = dict(
    trainer_cls=Qwen3Eagle3Trainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    # Data-scale regime uses fewer epochs (the H200 v3scale bet is 3ep on the
    # large corpus, NOT 10ep). Override at launch with EPOCHS=3; the default here
    # stays 10 so the file is a byte-for-byte-hyperparameter clone of v2.
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

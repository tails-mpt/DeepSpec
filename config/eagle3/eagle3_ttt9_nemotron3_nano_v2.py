import os

from deepspec.trainer import Qwen3Eagle3Trainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR


# NVIDIA-Nemotron-3-Nano-30B-A3B (nemotron_h hybrid Mamba-2/MoE) -- v2 recipe,
# SINGLE-VARIABLE LEVER: ttt_length 7 -> 9 (everything else identical to
# eagle3_nemotron3_nano_v2.py). Motivation: the v2 heads' per-position accept
# decays fast (dspark v2 target_mean pos0 0.82 -> pos6 0.15); training the draft
# to predict further ahead (longer test-time-training horizon) directly targets
# the late-position acceptance that caps chat-tau (~2.2). Complements the H200
# data-scale bet on a DIFFERENT axis (training horizon vs data volume). [2026-07-21]
NEMOTRON3_NANO = "/mnt/persistent/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

project_name = "deepspec"
exp_name = "eagle3_ttt9_nemotron3_nano_v2"
seed = 0

model = dict(
    target_model_name_or_path=NEMOTRON3_NANO,
    target_layer_ids=[2, 13, 26, 39, 48],
    ttt_length=9,
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

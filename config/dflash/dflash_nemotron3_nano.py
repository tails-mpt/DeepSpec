import os

from deepspec.trainer import Qwen3DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR


# NVIDIA-Nemotron-3-Nano-30B-A3B (model_type=nemotron_h, hybrid Mamba-2/MoE).
# DFlash shares the DSpark modeling/config builder; the dense Qwen3-style draft
# is synthesized from the target dims by
# deepspec.modeling.dspark.qwen3.config.build_qwen3_draft_base_config.
NEMOTRON3_NANO = "/mnt/persistent/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

project_name = "deepspec"
exp_name = "dflash_block7_nemotron3_nano"
seed = 42

model = dict(
    target_model_name_or_path=NEMOTRON3_NANO,
    block_size=7,
    num_draft_layers=5,
    # 5 of the 6 nemotron_h attention layers (see eagle3 config for rationale).
    target_layer_ids=[5, 12, 26, 33, 42],
    # Reserved, never-emitted special token `<SPECIAL_999>` (id 999) in the
    # Nemotron vocab (0..131071), used as the DFlash noise/mask token.
    mask_token_id=999,
    num_anchors=512,
    # Dense SwiGLU width for the 5 draft layers (Nemotron's 1856 is narrow for
    # a 2688-hidden draft; ~3x Qwen3-style). Tune if the draft underfits.
    draft_intermediate_size=8064,

    # Disable markov head.
    markov_rank=0,

    # Disable confidence head.
    confidence_head_alpha=0.0,

    # CE-only loss.
    loss_decay_gamma=4.0,
    ce_loss_alpha=1.0,
    l1_loss_alpha=0.0,
)

train = dict(
    trainer_cls=Qwen3DSparkTrainer,
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
    torch_compile=True,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    chat_template="nemotron",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name=str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg

    return cfg

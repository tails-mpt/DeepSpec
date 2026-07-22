import copy

from transformers import Qwen3Config

from deepspec.modeling.eagle3.common import validate_eagle3_target_layer_ids


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


def build_qwen3_draft_base_config(target_config, model_args):
    """Build a Qwen3Config carrying the target's dims for the dense draft head.

    The Qwen3* draft modules import Qwen3 internals (Qwen3MLP needs
    ``hidden_act``, Qwen3RMSNorm needs ``rms_norm_eps``, Qwen3RotaryEmbedding
    needs ``rope_parameters``). A NemotronHConfig has none of those under those
    names, so a raw ``copy.deepcopy(target_config)`` cannot drive the Qwen3
    draft. For nemotron_h we synthesize a Qwen3Config from the target's
    attention dims; for native Qwen3 targets we keep the deepcopy path.
    """
    if str(getattr(target_config, "model_type", "")) != "nemotron_h":
        return copy.deepcopy(target_config)

    hidden_size = int(target_config.hidden_size)
    num_attention_heads = int(target_config.num_attention_heads)
    default_intermediate = int(
        getattr(target_config, "intermediate_size", None) or 4 * hidden_size
    )
    intermediate_size = int(
        getattr(model_args, "draft_intermediate_size", default_intermediate)
    )
    rms_norm_eps = float(
        getattr(target_config, "norm_eps", None)
        or getattr(target_config, "layer_norm_epsilon", None)
        or 1e-5
    )
    return Qwen3Config(
        vocab_size=int(target_config.vocab_size),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=int(target_config.num_hidden_layers),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(target_config.num_key_value_heads),
        head_dim=int(
            getattr(target_config, "head_dim", hidden_size // num_attention_heads)
        ),
        hidden_act="silu",
        max_position_embeddings=int(
            getattr(target_config, "max_position_embeddings", 32768)
        ),
        rms_norm_eps=rms_norm_eps,
        rope_theta=float(getattr(target_config, "rope_theta", 10000.0)),
        attention_bias=bool(getattr(target_config, "attention_bias", False)),
        attention_dropout=float(getattr(target_config, "attention_dropout", 0.0)),
        tie_word_embeddings=False,
        pad_token_id=getattr(target_config, "pad_token_id", None),
        bos_token_id=getattr(target_config, "bos_token_id", None),
        eos_token_id=getattr(target_config, "eos_token_id", None),
    )


def build_draft_config(*, target_config, model_args):
    target_layer_ids = validate_eagle3_target_layer_ids(
        model_args.target_layer_ids,
        int(target_config.num_hidden_layers),
    )
    ttt_length = int(model_args.ttt_length)
    assert ttt_length >= 1, f"ttt_length must be >= 1, got {ttt_length}"
    step_loss_decay = float(model_args.step_loss_decay)
    assert step_loss_decay > 0.0, (
        "step_loss_decay must be > 0.0, "
        f"got {step_loss_decay}"
    )
    draft_num_hidden_layers = int(model_args.draft_num_hidden_layers)
    assert draft_num_hidden_layers >= 1, (
        "draft_num_hidden_layers must be >= 1, "
        f"got {draft_num_hidden_layers}"
    )

    draft_config = build_qwen3_draft_base_config(target_config, model_args)
    draft_config.architectures = ["Qwen3Eagle3Model"]
    draft_config.num_target_layers = int(target_config.num_hidden_layers)
    draft_config.num_hidden_layers = draft_num_hidden_layers
    draft_config.layer_types = ["full_attention"] * draft_num_hidden_layers
    draft_config.target_model_name_or_path = str(model_args.target_model_name_or_path)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.ttt_length = ttt_length
    draft_config.step_loss_decay = step_loss_decay
    draft_config.draft_num_hidden_layers = draft_num_hidden_layers
    # EAGLE-3.1 "FC-norm" lever: RMSNorm the fused aux hidden states before the
    # fusion projection (self.fc). Default False -> byte-for-byte identical to
    # the v2 recipe. See Qwen3Eagle3Model.__init__ / project_hidden_states.
    # fc_norm       = one RMSNorm over the full 5*hidden concat.
    # fc_norm_perslice = one RMSNorm per hidden-size source slice (register
    #                    alignment). Mutually exclusive; both default False.
    draft_config.fc_norm = bool(getattr(model_args, "fc_norm", False))
    draft_config.fc_norm_perslice = bool(
        getattr(model_args, "fc_norm_perslice", False)
    )
    draft_config.tie_word_embeddings = False
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    return draft_config


__all__ = [
    "build_draft_config",
]

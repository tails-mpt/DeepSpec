import copy

from transformers import Qwen3Config

from deepspec.modeling.dspark.common import validate_target_layer_ids


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


def build_draft_config(
    target_config,
    model_args,
):
    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    layer_types = ["full_attention"] * num_draft_layers
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    draft_config = build_qwen3_draft_base_config(target_config, model_args)
    draft_config.architectures = ["Qwen3DSparkModel"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


__all__ = [
    "build_draft_config",
]

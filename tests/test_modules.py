"""Unit tests for the model components in src/sjepa/modules.

Each test checks one component. We check shapes, data types, masking, gradient
flow, and the moving-average update. The tests use small tensors so they run
fast on a CPU.
"""

import torch

from sjepa.config import SJEPAConfig
from sjepa.modules import (
    BlockMaskGenerator,
    ClusterHead,
    ConvFeatureExtractor,
    ConvPositionalEncoding,
    EmaEncoder,
    Fp32LayerNorm,
    JEPAObjective,
    JEPAPredictor,
    KLDivergenceLoss,
    MultiHeadSelfAttention,
    SpeechEncoder,
    SwitchedEmaScheduler,
    TransformerEncoderLayer,
    build_padding_mask,
    scale_gradient,
)


# ---------------------------------------------------------------------------
# Normalization and gradient scaling
# ---------------------------------------------------------------------------

def test_fp32_layer_norm_keeps_shape_and_dtype():
    norm = Fp32LayerNorm(16)
    x = torch.randn(2, 5, 16, dtype=torch.float32)
    out = norm(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_scale_gradient_scales_backward_only():
    x = torch.ones(4, requires_grad=True)
    y = scale_gradient(x, 0.1).sum()
    y.backward()
    # The value passes through, but the gradient is scaled by 0.1.
    assert torch.allclose(x.grad, torch.full((4,), 0.1))


# ---------------------------------------------------------------------------
# CNN feature extractor and positional encoding
# ---------------------------------------------------------------------------

def test_feature_extractor_output_frames():
    extractor = ConvFeatureExtractor(conv_dim=32)
    wav = torch.randn(2, 1, 8000)
    out = extractor(wav)
    # The total stride is 320, so 8000 samples give about 24 frames.
    assert out.shape[0] == 2
    assert out.shape[1] == 32
    assert abs(out.shape[2] - 8000 // 320) <= 2


def test_positional_encoding_preserves_shape():
    pos = ConvPositionalEncoding(embed_dim=24, kernel_size=8, groups=4)
    x = torch.randn(2, 15, 24)
    out = pos(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Attention and transformer layer
# ---------------------------------------------------------------------------

def test_attention_output_shape():
    attn = MultiHeadSelfAttention(embed_dim=24, num_heads=4)
    x = torch.randn(2, 7, 24)
    out = attn(x)
    assert out.shape == x.shape


def test_attention_respects_padding_mask():
    attn = MultiHeadSelfAttention(embed_dim=16, num_heads=4)
    attn.eval()
    x = torch.randn(1, 6, 16)
    full = torch.ones(1, 6, dtype=torch.bool)
    part = full.clone()
    part[0, 4:] = False
    # When we hide the last two keys, the output must change.
    out_full = attn(x, key_padding_mask=full)
    out_part = attn(x, key_padding_mask=part)
    assert not torch.allclose(out_full, out_part)


def test_transformer_layer_shape():
    layer = TransformerEncoderLayer(embed_dim=24, num_heads=4, ffn_dim=48)
    x = torch.randn(2, 9, 24)
    out = layer(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def test_padding_mask_marks_real_frames():
    mask = build_padding_mask(2, 10, [10, 6], torch.device("cpu"))
    assert mask[0].all()
    assert mask[1, :6].all()
    assert not mask[1, 6:].any()


def test_block_mask_ratio_matches_target():
    generator = BlockMaskGenerator(mask_ratio=0.65, mask_length=10)
    # Average over many long sequences so the random spans even out. The mean
    # masked fraction must sit close to the paper target of 0.65.
    ratios = [
        generator.generate(1, 600, [600], torch.device("cpu")).float().mean()
        for _ in range(100)
    ]
    mean_ratio = float(torch.stack(ratios).mean())
    assert 0.60 < mean_ratio < 0.72


def test_block_mask_only_masks_real_frames():
    generator = BlockMaskGenerator(mask_ratio=0.65, mask_length=5)
    mask = generator.generate(1, 50, [20], torch.device("cpu"))
    # Frames after the real length must never be masked.
    assert not mask[0, 20:].any()


# ---------------------------------------------------------------------------
# Encoder, predictor, cluster head
# ---------------------------------------------------------------------------

def _tiny_config():
    return SJEPAConfig(conv_dim=32, hidden_dim=48, num_layers=2, num_heads=4,
                       ffn_dim=64, predictor_layers=1, predictor_heads=4,
                       predictor_ffn_dim=64, num_clusters=10, max_frames=128)


def test_encoder_forward_and_layers():
    config = _tiny_config()
    encoder = SpeechEncoder(config)
    encoder.eval()
    wav = torch.randn(2, 1, 8000)
    out = encoder(wav)
    layers = encoder(wav, return_layers=True)
    assert out.shape[0] == 2 and out.shape[2] == config.hidden_dim
    assert len(layers) == config.num_layers
    assert torch.allclose(layers[-1], out)


def test_encoder_extract_layer_index_check():
    config = _tiny_config()
    encoder = SpeechEncoder(config)
    encoder.eval()
    wav = torch.randn(1, 1, 4000)
    feat = encoder.extract_layer(wav, 0)
    assert feat.shape[-1] == config.hidden_dim
    try:
        encoder.extract_layer(wav, 99)
        raised = False
    except IndexError:
        raised = True
    assert raised


def test_predictor_injects_mask_token():
    config = _tiny_config()
    predictor = JEPAPredictor(config)
    predictor.eval()
    context = torch.randn(2, 12, config.hidden_dim)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[:, 3:6] = True
    out = predictor(context, mask)
    assert out.shape == context.shape


def test_cluster_head_output_shape():
    head = ClusterHead(hidden_dim=48, num_clusters=10)
    x = torch.randn(2, 7, 48)
    out = head(x)
    assert out.shape == (2, 7, 10)


# ---------------------------------------------------------------------------
# EMA encoder
# ---------------------------------------------------------------------------

def test_switched_ema_scheduler_flips_rate():
    scheduler = SwitchedEmaScheduler(alpha_fast=0.9, alpha_slow=0.99,
                                     switch_every=100)
    assert scheduler.decay(0) == 0.9
    assert scheduler.decay(150) == 0.99
    assert scheduler.decay(250) == 0.9


def test_ema_update_moves_weights_toward_online():
    config = _tiny_config()
    online = SpeechEncoder(config)
    ema = EmaEncoder(online, SwitchedEmaScheduler(switch_every=10))
    # Make the online weights different from the EMA copy.
    with torch.no_grad():
        for param in online.parameters():
            param.add_(1.0)
    before = next(iter(ema.encoder.parameters())).clone()
    ema.update(online, step=0)
    after = next(iter(ema.encoder.parameters()))
    # The EMA weight must move a little toward the online weight, not jump.
    assert not torch.allclose(before, after)
    assert (after - before).abs().mean() < 1.0


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def test_kl_loss_is_zero_for_empty_selection():
    loss_fn = KLDivergenceLoss()
    logits = torch.randn(2, 5, 4, requires_grad=True)
    targets = torch.softmax(torch.randn(2, 5, 4), dim=-1)
    selection = torch.zeros(2, 5, dtype=torch.bool)
    loss = loss_fn(logits, targets, selection)
    assert float(loss.detach()) == 0.0


def test_objective_masked_only_versus_with_visible():
    targets = torch.softmax(torch.randn(2, 6, 4), dim=-1)
    logits_m = torch.randn(2, 6, 4)
    logits_v = torch.randn(2, 6, 4)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 2:4] = True
    padding = torch.ones(2, 6, dtype=torch.bool)
    masked_only = JEPAObjective(use_visible_loss=False)
    with_visible = JEPAObjective(use_visible_loss=True)
    res_m = masked_only(logits_m, logits_v, targets, mask, padding)
    res_v = with_visible(logits_m, logits_v, targets, mask, padding)
    # The masked-only loss equals its own masked term.
    assert torch.allclose(res_m["loss"], res_m["loss_masked"])
    # Adding the visible term gives a different (larger or equal) total.
    assert not torch.allclose(res_m["loss"], res_v["loss"])

"""Named configurations of the ENVC network."""
from .envc import ENVC
from .efp.predictor import DEFAULT_DECODER_WIDTHS, COMPACT_DECODER_WIDTHS

MODEL_CONFIGS = {
    # Full model.
    'envc': {'frontend_widths': DEFAULT_DECODER_WIDTHS, 'refinement_scale': 1.0},
    # Lightweight model: narrower frame-prediction decoder and context refinement at half resolution.
    'envc-l': {'frontend_widths': COMPACT_DECODER_WIDTHS, 'refinement_scale': 0.5},
}


def create_model(version='envc', **kwargs):
    if version not in MODEL_CONFIGS:
        raise ValueError(f'Unknown model version: {version} (choose from {sorted(MODEL_CONFIGS)})')
    model = ENVC(**MODEL_CONFIGS[version], **kwargs)
    model.model_version = version
    return model

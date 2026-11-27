"""Совместимость со старым импортом сводок; реализация находится в quantlab.ui."""

from quantlab.ui.presentation import (
    MODE_LABELS, dataset_summary, empty_state, inference_summary, model_summary,
    quant_summary, workspace_header,
)

__all__ = [
    'MODE_LABELS', 'dataset_summary', 'empty_state', 'inference_summary',
    'model_summary', 'quant_summary', 'workspace_header',
]

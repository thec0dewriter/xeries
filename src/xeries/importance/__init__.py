"""Feature importance calculation methods."""

from xeries.importance.permutation import ConditionalPermutationImportance
from xeries.importance.shap import BatchSHAP, ConditionalSHAP

# from xeries.importance.causal import CausalFeatureImportance
# from xeries.importance.dropping import ConditionalDropImportance
# from xeries.importance.shapiq import ConditionalSHAPIQ, SHAPIQResult

__all__ = [
    "BatchSHAP",
    # "CausalFeatureImportance",
    # "ConditionalDropImportance",
    "ConditionalPermutationImportance",
    "ConditionalSHAP",
    # "ConditionalSHAPIQ",
    # "SHAPIQResult",
]

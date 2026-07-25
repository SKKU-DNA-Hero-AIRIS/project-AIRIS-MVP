"""AIRIS context-aware decision engine."""

from .control import (
    ActuationPlan,
    ControlDecision,
    analyze_hybrid,
    baseline_control_level,
    decision_report,
    plan_actuation,
)
from .engine import DecisionResult, analyze_context
from .features import FEATURE_NAMES, extract_features, feature_vector
from .ml_policy import ControlPolicy, MLPrediction, load_policy
from .models import (
    EnvironmentContext,
    FacilityContext,
    HealthContext,
    LEVELS,
    MultimodalContext,
    VisionContext,
)
from .rules import ALL_RULES, DAMPING_RULES, SAFETY_RULES, evaluate_rules
from .scenarios import build_context, load_context_samples

__all__ = [
    "ALL_RULES",
    "ActuationPlan",
    "ControlDecision",
    "ControlPolicy",
    "DAMPING_RULES",
    "DecisionResult",
    "EnvironmentContext",
    "FEATURE_NAMES",
    "FacilityContext",
    "HealthContext",
    "LEVELS",
    "MLPrediction",
    "MultimodalContext",
    "SAFETY_RULES",
    "VisionContext",
    "analyze_context",
    "analyze_hybrid",
    "baseline_control_level",
    "build_context",
    "decision_report",
    "evaluate_rules",
    "extract_features",
    "feature_vector",
    "load_context_samples",
    "load_policy",
    "plan_actuation",
]

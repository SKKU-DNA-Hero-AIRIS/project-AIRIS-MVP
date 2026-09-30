"""AIRIS context-aware decision engine."""

from .body_profile import (
    BodyProfile,
    UserProfile,
    UserTraits,
    body_from_measurements,
    estimate_body_profile,
    load_user_profiles,
    traits_from_gait,
)
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
from .gate_physics import GateGeometry, GateSetpoints, comfort_limits
from .ml_policy import ControlPolicy, MLPrediction, load_policy
from .models import (
    EnvironmentContext,
    FacilityContext,
    HealthContext,
    LEVELS,
    MultimodalContext,
    VisionContext,
)
from .personalization import PersonalizedPlan, plan_personalized
from .rules import ALL_RULES, DAMPING_RULES, SAFETY_RULES, evaluate_rules
from .scenarios import build_context, load_context_samples

__all__ = [
    "ALL_RULES",
    "ActuationPlan",
    "BodyProfile",
    "ControlDecision",
    "ControlPolicy",
    "DAMPING_RULES",
    "DecisionResult",
    "EnvironmentContext",
    "FEATURE_NAMES",
    "FacilityContext",
    "GateGeometry",
    "GateSetpoints",
    "HealthContext",
    "LEVELS",
    "MLPrediction",
    "MultimodalContext",
    "PersonalizedPlan",
    "SAFETY_RULES",
    "UserProfile",
    "UserTraits",
    "VisionContext",
    "analyze_context",
    "analyze_hybrid",
    "baseline_control_level",
    "body_from_measurements",
    "build_context",
    "comfort_limits",
    "decision_report",
    "estimate_body_profile",
    "evaluate_rules",
    "extract_features",
    "feature_vector",
    "load_context_samples",
    "load_policy",
    "load_user_profiles",
    "plan_actuation",
    "plan_personalized",
    "traits_from_gait",
]

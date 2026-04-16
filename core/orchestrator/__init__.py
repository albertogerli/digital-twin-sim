"""Dynamic Contextual Activation — the Orchestrator layer.

Sits between the briefing and the simulation engine to:
1. Semantically retrieve the right stakeholders for any crisis
2. Escalate agent activation dynamically across rounds
3. Score contagion risk and predict escalation thresholds
4. Translate crisis metrics into financial market impact
"""

from core.orchestrator.retriever import SemanticRetriever
from core.orchestrator.escalation import EscalationEngine
from core.orchestrator.contagion import ContagionScorer
from core.orchestrator.financial_impact import FinancialImpactScorer

__all__ = [
    "SemanticRetriever",
    "EscalationEngine",
    "ContagionScorer",
    "FinancialImpactScorer",
]

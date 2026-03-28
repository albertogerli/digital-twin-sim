"""Token usage and cost tracking across LLM providers."""

from dataclasses import dataclass, field


@dataclass
class UsageStats:
    """Tracks cumulative token usage and costs."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    call_count: int = 0
    errors: int = 0
    calls_by_component: dict = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int,
               cost_per_1m_input: float, cost_per_1m_output: float,
               component: str = "unknown") -> float:
        """Record a single LLM call's usage and return the cost."""
        cost_in = (input_tokens / 1_000_000) * cost_per_1m_input
        cost_out = (output_tokens / 1_000_000) * cost_per_1m_output
        cost = cost_in + cost_out

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.call_count += 1

        if component not in self.calls_by_component:
            self.calls_by_component[component] = {"calls": 0, "cost": 0.0}
        self.calls_by_component[component]["calls"] += 1
        self.calls_by_component[component]["cost"] += cost

        return cost

    def summary(self) -> str:
        lines = [
            f"  Total calls: {self.call_count} | Errors: {self.errors}",
            f"  Tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out",
            f"  Cost: ${self.total_cost:.4f}",
        ]
        if self.calls_by_component:
            lines.append("  By component:")
            for comp, data in sorted(self.calls_by_component.items()):
                lines.append(f"    {comp}: {data['calls']} calls, ${data['cost']:.4f}")
        return "\n".join(lines)

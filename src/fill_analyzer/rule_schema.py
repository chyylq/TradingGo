"""
Rule schema definitions for the fill analyzer module.

This module provides dataclasses that allow trading rules to be expressed in a
format that is both readable by humans and consumable by downstream systems,
such as the backtesting pipeline. The goal is to capture the conditions under
which trades should occur, the resulting actions, and metadata that helps with
interpretability and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class RuleMetadata:
    """
    Metadata describing the provenance and scope of a trade rule.
    """

    name: str
    description: str
    instrument: Optional[str] = None
    strategy: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_by: Optional[str] = None


@dataclass
class RuleCondition:
    """
    A primitive condition that must be satisfied for the rule to trigger.

    Conditions are defined by a type identifier and a parameter dictionary.
    Each condition also carries a short natural language description to aid
    human review.
    """

    condition_type: str
    parameters: Dict[str, Any]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the condition to a JSON-friendly dictionary.
        """
        payload = asdict(self)
        return payload


@dataclass
class ConditionGroup:
    """
    A logical grouping of one or more conditions.

    The operator defines how child nodes are combined (for example, AND, OR).
    Nested groups allow complex logical expressions without forcing the system
    to parse bespoke string expressions.
    """

    operator: str
    conditions: List[RuleCondition] = field(default_factory=list)
    groups: List["ConditionGroup"] = field(default_factory=list)
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the condition group to a JSON-friendly dictionary.
        """
        return {
            "operator": self.operator,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "groups": [group.to_dict() for group in self.groups],
            "summary": self.summary,
        }


@dataclass
class RuleAction:
    """
    An action that should be executed when the rule fires.
    """

    action_type: str
    parameters: Dict[str, Any]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the action to a JSON-friendly dictionary.
        """
        payload = asdict(self)
        return payload


@dataclass
class RuleMetrics:
    """
    Evaluation metrics that describe how well the rule matches observed fills.

    Recall measures the share of real fills covered by the rule. Precision
    measures how often the rule would have fired relative to actual fills.
    Additional metrics can be recorded in the extras dictionary.
    """

    recall: Optional[float] = None
    precision: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the metrics to a JSON-friendly dictionary.
        """
        return {
            "recall": self.recall,
            "precision": self.precision,
            "extras": self.extras,
        }


@dataclass
class TradeRule:
    """
    Aggregate structure representing a trading rule inferred from fills.

    The rule combines metadata, a logical condition tree, actions to take, and
    optional evaluation metrics or confidence scores. The `notes` field captures
    diagnostic details useful for human reviewers.
    """

    metadata: RuleMetadata
    condition_tree: ConditionGroup
    actions: List[RuleAction]
    confidence: Optional[float] = None
    metrics: Optional[RuleMetrics] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the rule to a JSON-friendly dictionary.
        """
        return {
            "metadata": asdict(self.metadata),
            "condition_tree": self.condition_tree.to_dict(),
            "actions": [action.to_dict() for action in self.actions],
            "confidence": self.confidence,
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
            "notes": self.notes,
        }

    def to_text(self) -> str:
        """
        Create a human-readable summary of the rule and its evaluation metrics.
        """
        lines: List[str] = []
        lines.append(f"Rule: {self.metadata.name}")
        lines.append(f"Description: {self.metadata.description}")
        if self.metadata.instrument:
            lines.append(f"Instrument: {self.metadata.instrument}")
        if self.metadata.strategy:
            lines.append(f"Strategy: {self.metadata.strategy}")
        if self.metadata.tags:
            lines.append(f"Tags: {', '.join(self.metadata.tags)}")
        if self.confidence is not None:
            lines.append(f"Confidence: {self.confidence:.2f}")
        if self.metrics:
            recall_display = (
                f"{self.metrics.recall:.3f}"
                if self.metrics.recall is not None
                else "n/a"
            )
            precision_display = (
                f"{self.metrics.precision:.3f}"
                if self.metrics.precision is not None
                else "n/a"
            )
            lines.append(
                f"Metrics - Recall: {recall_display}, Precision: {precision_display}"
            )
            if self.metrics.extras:
                extras_parts = [
                    f"{key}: {value}"
                    for key, value in sorted(self.metrics.extras.items())
                ]
                lines.append(f"Additional Metrics: {', '.join(extras_parts)}")
        lines.append("Conditions:")
        lines.extend(_render_group(self.condition_tree, indent_level=1))
        lines.append("Actions:")
        for action in self.actions:
            lines.append(f"  - {action.summary} ({action.action_type})")
        if self.notes:
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)


def _render_group(group: ConditionGroup, indent_level: int) -> List[str]:
    """
    Render a condition group and its children for human-readable output.
    """
    indent = "  " * indent_level
    lines: List[str] = []
    header = f"{indent}- {group.operator}"
    if group.summary:
        header += f": {group.summary}"
    lines.append(header)
    for condition in group.conditions:
        lines.append(
            f"{indent}  * {condition.summary} ({condition.condition_type})"
        )
    for child_group in group.groups:
        lines.extend(_render_group(child_group, indent_level=indent_level + 1))
    return lines


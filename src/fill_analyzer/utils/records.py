"""
Dataclass definitions and shared types for the fill analyzer utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class FillRecord:
    """
    Normalized representation of an execution fill.
    """

    timestamp: str
    price: Optional[float] = None
    quantity: Optional[float] = None
    side: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FillSchema:
    """
    Schema definition describing how to parse a fill CSV file.

    The mapping fields refer to column names in the CSV. Optional metadata
    fields allow callers to capture additional columns without changing the
    core schema. Converters may be supplied to transform raw string values.
    """

    timestamp: str
    price: Optional[str] = None
    quantity: Optional[str] = None
    side: Optional[str] = None
    metadata_fields: Optional[List[str]] = None
    converters: Dict[str, Callable[[str], Any]] = field(default_factory=dict)
    dialect: Optional[str] = None
    delimiter: Optional[str] = None


@dataclass
class FillSourceConfig:
    """
    Configuration describing where and how to load fills.

    Attributes:
        source: Path to the underlying resource (file path for CSV, JSON, etc.).
        format: Loader key that selects the correct loader implementation.
        schema_path: Optional path to a serialized schema definition.
        schema: Optional explicit schema. Overrides `schema_path` if provided.
        options: Additional loader-specific keyword arguments (for example,
            encoding for CSV files). The dictionary values should be JSON
            compatible so that configuration can be stored externally.
    """

    source: Path
    format: str = "csv"
    schema_path: Optional[Path] = None
    schema: Optional[FillSchema] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuoteRecord:
    """
    Normalized representation of market data quotes.
    """

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuoteSchema:
    """
    Schema configuration for parsing quote data.
    """

    timestamp: str
    open: str
    high: str
    low: str
    close: str
    volume: Optional[str] = None
    open_interest: Optional[str] = None
    metadata_fields: Optional[List[str]] = None
    converters: Dict[str, Callable[[str], Any]] = field(default_factory=dict)
    dialect: Optional[str] = None
    delimiter: Optional[str] = None
    has_header: bool = True
    fieldnames: Optional[List[str]] = None


@dataclass
class QuoteSourceConfig:
    """
    Configuration describing where and how to load quote data.
    """

    source: Path
    format: str = "csv"
    schema_path: Optional[Path] = None
    schema: Optional[QuoteSchema] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuoteSlice:
    """
    Window of quote data aligned to a fill.
    """

    start_timestamp: str
    end_timestamp: str
    data: List[Dict[str, Any]]


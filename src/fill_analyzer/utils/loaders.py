"""
Loader implementations and helper functions for fill and quote data.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from pandas import DataFrame

from .records import (
    FillSchema,
    FillSourceConfig,
    QuoteSchema,
    QuoteSourceConfig,
)

logger = logging.getLogger(__name__)

FillLoader = Callable[[FillSourceConfig, FillSchema], DataFrame]
QuoteLoader = Callable[[QuoteSourceConfig, QuoteSchema], DataFrame]

FILL_LOADERS: Dict[str, FillLoader] = {}
QUOTE_LOADERS: Dict[str, QuoteLoader] = {}


def register_fill_loader(name: str, loader: FillLoader) -> None:
    """
    Register a fill loader implementation under a given format key.
    """

    FILL_LOADERS[name] = loader


def register_quote_loader(name: str, loader: QuoteLoader) -> None:
    """
    Register a quote loader implementation under a given format key.
    """

    QUOTE_LOADERS[name] = loader


def load_fills(config: FillSourceConfig) -> DataFrame:
    """
    Load fills using the loader specified in the configuration and return a DataFrame.

    A schema must be supplied either directly via `config.schema` or through
    a serialized schema file referenced by `config.schema_path`.
    """

    if config.schema is None and config.schema_path is None:
        raise ValueError("FillSourceConfig requires a schema or schema_path")

    schema = (
        config.schema
        if config.schema is not None
        else read_fill_schema(config.schema_path)
    )

    loader = FILL_LOADERS.get(config.format)
    if loader is None:
        raise ValueError(f"No fill loader registered for format '{config.format}'")

    return loader(config, schema)


def load_quotes(config: QuoteSourceConfig) -> DataFrame:
    """
    Load quotes using the configured loader and schema and return a DataFrame.
    """

    if config.schema is None and config.schema_path is None:
        raise ValueError("QuoteSourceConfig requires a schema or schema_path")

    schema = (
        config.schema
        if config.schema is not None
        else read_quote_schema(config.schema_path)
    )

    loader = QUOTE_LOADERS.get(config.format)
    if loader is None:
        raise ValueError(f"No quote loader registered for format '{config.format}'")

    return loader(config, schema)


def read_fill_schema(path: Optional[Path]) -> FillSchema:
    """
    Read a fill schema from a JSON file and construct a `FillSchema` instance.
    """

    payload = _read_schema_payload(path)

    converters_spec = payload.get("converters", {})
    converters: Dict[str, Callable[[str], Any]] = {}
    for column, converter_name in converters_spec.items():
        converters[column] = _resolve_converter(str(converter_name))

    metadata_fields = payload.get("metadata_fields")
    if metadata_fields is not None and not isinstance(metadata_fields, list):
        raise ValueError("metadata_fields in schema must be a list when provided")

    return FillSchema(
        timestamp=payload["timestamp"],
        price=payload.get("price"),
        quantity=payload.get("quantity"),
        side=payload.get("side"),
        metadata_fields=metadata_fields,
        converters=converters,
        dialect=payload.get("dialect"),
        delimiter=payload.get("delimiter"),
    )


def read_quote_schema(path: Optional[Path]) -> QuoteSchema:
    """
    Read a quote schema from JSON and construct a `QuoteSchema`.
    """

    payload = _read_schema_payload(path)

    converters_spec = payload.get("converters", {})
    converters: Dict[str, Callable[[str], Any]] = {}
    for column, converter_name in converters_spec.items():
        converters[column] = _resolve_converter(str(converter_name))

    metadata_fields = payload.get("metadata_fields")
    if metadata_fields is not None and not isinstance(metadata_fields, list):
        raise ValueError("metadata_fields in schema must be a list when provided")

    fieldnames = payload.get("fieldnames")
    if fieldnames is not None and not isinstance(fieldnames, list):
        raise ValueError("fieldnames in schema must be a list when provided")

    return QuoteSchema(
        timestamp=payload["timestamp"],
        open=payload["open"],
        high=payload["high"],
        low=payload["low"],
        close=payload["close"],
        volume=payload.get("volume"),
        open_interest=payload.get("open_interest"),
        metadata_fields=metadata_fields,
        converters=converters,
        dialect=payload.get("dialect"),
        delimiter=payload.get("delimiter"),
        has_header=payload.get("has_header", True),
        fieldnames=fieldnames,
    )


def _read_schema_payload(path: Optional[Path]) -> Dict[str, Any]:
    """
    Read JSON payload from the provided schema path.
    """

    if path is None:
        raise ValueError("Schema path must be provided")

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Schema file not found: {resolved_path}")

    with resolved_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    return payload


def _load_fills_from_csv(
    config: FillSourceConfig,
    schema: FillSchema,
) -> DataFrame:
    """
    Load raw fills from a CSV file and normalize into a DataFrame.
    """

    path = Path(config.source)
    if not path.exists():
        raise FileNotFoundError(f"Fill source not found: {path}")

    logger.info("Loading fills from %s", path)

    records: List[Dict[str, Any]] = []
    encoding = config.options.get("encoding", "utf-8")
    open_kwargs: Dict[str, Any] = {"encoding": encoding, "newline": ""}
    with path.open(**open_kwargs) as handle:
        if schema.dialect:
            reader = csv.DictReader(handle, dialect=schema.dialect)
        elif schema.delimiter:
            reader = csv.DictReader(handle, delimiter=schema.delimiter)
        else:
            reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"No header row detected in {path}")

        mapped_columns = {
            column
            for column in [
                schema.timestamp,
                schema.price,
                schema.quantity,
                schema.side,
            ]
            if column is not None
        }

        metadata_fields = (
            schema.metadata_fields
            if schema.metadata_fields is not None
            else [name for name in reader.fieldnames if name not in mapped_columns]
        )

        for row_index, row in enumerate(reader, start=1):
            timestamp_value = _apply_converter(
                row, schema.timestamp, schema.converters
            )
            if timestamp_value is None:
                raise ValueError(
                    f"Missing timestamp at row {row_index} in {path} "
                    f"(column: {schema.timestamp})"
                )

            price_value = _parse_optional_float(
                row, schema.price, schema.converters
            )
            quantity_value = _parse_optional_float(
                row, schema.quantity, schema.converters
            )
            side_value = _apply_converter(
                row, schema.side, schema.converters
            )

            metadata: Dict[str, Any] = {}
            for field_name in metadata_fields:
                if field_name in row:
                    metadata[field_name] = _apply_converter(
                        row, field_name, schema.converters
                    )

            row_data: Dict[str, Any] = {
                "timestamp": str(timestamp_value),
                "price": price_value,
                "quantity": quantity_value,
                "side": str(side_value) if side_value is not None else None,
                "metadata": metadata,
            }
            row_data.update(metadata)
            records.append(row_data)

    metadata_columns = metadata_fields if metadata_fields is not None else []
    base_columns = ["timestamp", "price", "quantity", "side", "metadata"]
    ordered_columns = list(dict.fromkeys(base_columns + metadata_columns))
    dataframe = pd.DataFrame(records, columns=ordered_columns)

    logger.info("Loaded %d fills from %s", len(dataframe), path)
    return dataframe


def _load_quotes_from_csv(
    config: QuoteSourceConfig,
    schema: QuoteSchema,
) -> DataFrame:
    """
    Load quotes from CSV-based data sources into a DataFrame.
    """

    path = Path(config.source)
    if not path.exists():
        raise FileNotFoundError(f"Quote source not found: {path}")

    logger.info("Loading quotes from %s", path)

    records: List[Dict[str, Any]] = []
    encoding = config.options.get("encoding", "utf-8")
    open_kwargs: Dict[str, Any] = {"encoding": encoding, "newline": ""}
    with path.open(**open_kwargs) as handle:
        if schema.dialect:
            reader = csv.DictReader(handle, dialect=schema.dialect)
        elif schema.delimiter:
            reader = csv.DictReader(handle, delimiter=schema.delimiter)
        else:
            reader = csv.DictReader(handle)

        if not schema.has_header:
            if schema.fieldnames is None:
                raise ValueError(
                    "Quote schema with has_header false requires fieldnames"
                )
            reader.fieldnames = schema.fieldnames

        if reader.fieldnames is None:
            raise ValueError(f"Unable to determine field names for {path}")

        mapped_columns = {
            column
            for column in [
                schema.timestamp,
                schema.open,
                schema.high,
                schema.low,
                schema.close,
                schema.volume,
                schema.open_interest,
            ]
            if column is not None
        }

        metadata_fields = (
            schema.metadata_fields
            if schema.metadata_fields is not None
            else [name for name in reader.fieldnames if name not in mapped_columns]
        )

        for row_index, row in enumerate(reader, start=1):
            timestamp_value = _apply_converter(
                row, schema.timestamp, schema.converters
            )
            if timestamp_value is None:
                raise ValueError(
                    f"Missing timestamp at row {row_index} in {path} "
                    f"(column: {schema.timestamp})"
                )

            open_value = _parse_required_float(
                row, schema.open, schema.converters, "open"
            )
            high_value = _parse_required_float(
                row, schema.high, schema.converters, "high"
            )
            low_value = _parse_required_float(
                row, schema.low, schema.converters, "low"
            )
            close_value = _parse_required_float(
                row, schema.close, schema.converters, "close"
            )
            volume_value = _parse_optional_float(
                row, schema.volume, schema.converters
            )
            open_interest_value = _parse_optional_float(
                row, schema.open_interest, schema.converters
            )

            metadata: Dict[str, Any] = {}
            for field_name in metadata_fields:
                if field_name in row:
                    metadata[field_name] = _apply_converter(
                        row, field_name, schema.converters
                    )

            row_data: Dict[str, Any] = {
                "timestamp": str(timestamp_value),
                "open": open_value,
                "high": high_value,
                "low": low_value,
                "close": close_value,
                "volume": volume_value,
                "open_interest": open_interest_value,
                "metadata": metadata,
            }
            row_data.update(metadata)
            records.append(row_data)

    metadata_columns = metadata_fields if metadata_fields is not None else []
    base_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
        "metadata",
    ]
    ordered_columns = list(dict.fromkeys(base_columns + metadata_columns))
    dataframe = pd.DataFrame(records, columns=ordered_columns)

    logger.info("Loaded %d quotes from %s", len(dataframe), path)
    return dataframe


# Register default loaders.
register_fill_loader("csv", _load_fills_from_csv)
register_quote_loader("csv", _load_quotes_from_csv)


def _apply_converter(
    row: Dict[str, Any],
    column: Optional[str],
    converters: Dict[str, Callable[[str], Any]],
) -> Any:
    """
    Apply an optional converter to a column value from the CSV row.
    """

    if column is None:
        return None
    if column not in row:
        raise ValueError(f"Column '{column}' not found in row: {row}")

    raw_value = row[column]
    converter = converters.get(column)
    if converter is not None:
        return converter(raw_value)
    return raw_value or None


def _parse_optional_float(
    row: Dict[str, Any],
    column: Optional[str],
    converters: Dict[str, Callable[[str], Any]],
) -> Optional[float]:
    """
    Parse a float value from the row using the provided converters when present.
    """

    value = _apply_converter(row, column, converters)
    if value is None or value == "":
        return None
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unable to parse float from column '{column}': {value}"
        ) from exc


def _parse_required_float(
    row: Dict[str, Any],
    column: Optional[str],
    converters: Dict[str, Callable[[str], Any]],
    label: str,
) -> float:
    """
    Parse a required float column.
    """

    value = _parse_optional_float(row, column, converters)
    if value is None:
        raise ValueError(f"Missing required {label} value in column '{column}'")
    return value


ConverterFunc = Callable[[Optional[str]], Optional[float]]


def _resolve_converter(name: str) -> Callable[[str], Any]:
    """
    Map a converter name from the schema file to a callable.
    """

    lookup: Dict[str, Callable[[str], Any]] = {
        "float": float,
        "int": int,
        "str": str,
        "lower": lambda value: value.lower() if value is not None else None,
        "upper": lambda value: value.upper() if value is not None else None,
        "strip": lambda value: value.strip() if value is not None else None,
        "percent": lambda value: _convert_percent(value),
    }

    if name not in lookup:
        raise ValueError(f"Unsupported converter '{name}' in fill schema")
    return lookup[name]


def _convert_percent(value: Optional[str]) -> Optional[float]:
    """
    Convert percentage strings like '0.5%' to float (0.005).
    """

    if value is None:
        return None
    trimmed = value.strip()
    if trimmed.endswith("%"):
        trimmed = trimmed[:-1]
    if trimmed == "":
        return None
    return float(trimmed) / 100.0


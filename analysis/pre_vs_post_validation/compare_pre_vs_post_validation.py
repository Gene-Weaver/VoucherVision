#!/usr/bin/env python3
"""Compare pre-validation LLM transcriptions against latest reviewed workbooks.

This script is designed for the Comparison_PreVsPostValidation project folders
and analyzes all projects in the home directory by default. It:

1. Loads `transcribed_prior_to_subsetting.xlsx`.
2. Finds the latest `transcribed__edited__*.xlsx`.
3. Aligns rows by `catalogNumber`.
4. Excludes rows where `country == "x"` on either side.
5. Scores exact agreement and character-level edit effort for reviewable fields.
6. Separates Gemini and non-Gemini projects into cohorts for comparison.

Outputs are written as CSV, JSON, and Markdown so they can be reused in other
analysis workflows.
"""

from __future__ import annotations

import argparse
import csv
import json
import posixpath
import re
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET

try:
    from Levenshtein import distance as fast_levenshtein_distance
except ImportError:
    fast_levenshtein_distance = None


DEFAULT_HOME = Path("/Users/willwe/Downloads/Comparison_PreVsPostValidation")
DEFAULT_GEMINI_PROJECTS = [
    "2023_10_09_I5_bamaral_AllAsia_Onagr",
    "2023_10_13_I4_kathia_AllAsia_Api",
    "2023_11_01_I4_mikedmac_AllAsia_Ole",
]
GEMINI_METHOD = "Gemini-1.5-Pro"
GEMINI_COHORT = "Gemini-1.5-Pro"
NON_GEMINI_COHORT = "Non-Gemini"
MIXED_COHORT = "Mixed"
UNKNOWN_COHORT = "Unknown"

SOURCE_WORKBOOK = "transcribed_prior_to_subsetting.xlsx"
REVIEWED_PATTERN = "transcribed__edited__*.xlsx"
DEFAULT_KEY_COLUMNS = ["catalogNumber"]
COUNTRY_COLUMN = "country"

# Default analysis excludes IDs plus machine/runtime provenance that was not
# meaningfully "human reviewed" during validation.
DEFAULT_EXCLUDED_COLUMNS = {
    "catalogNumber",
    "filename",
    "WFO_override_OCR",
    "WFO_exact_match",
    "WFO_exact_match_name",
    "WFO_best_match",
    "WFO_candidate_names",
    "WFO_placement",
    "GEO_override_OCR",
    "GEO_method",
    "GEO_formatted_full_string",
    "GEO_decimal_lat",
    "GEO_decimal_long",
    "GEO_city",
    "GEO_county",
    "GEO_state",
    "GEO_state_code",
    "GEO_country",
    "GEO_country_code",
    "GEO_continent",
    "current_time",
    "inference_time_s",
    "tool_time_s",
    "max_cpu",
    "max_ram_gb",
    "n_gpus",
    "max_gpu_load",
    "max_gpu_vram_gb",
    "total_gpu_vram_gb",
    "capability_score",
    "run_name",
    "prompt",
    "LLM",
    "tokens_in",
    "tokens_out",
    "LM2_collage",
    "OCR_method",
    "OCR_double",
    "OCR_trOCR",
    "path_to_crop",
    "path_to_original",
    "path_to_content",
    "path_to_helper",
}

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKGREL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {
    "main": MAIN_NS,
    "rel": REL_NS,
    "pkgrel": PKGREL_NS,
}
COL_RE = re.compile(r"([A-Z]+)")


@dataclass
class WorkbookRow:
    values: dict[str, str]
    excel_row_number: int
    row_index: int


@dataclass
class WorkbookData:
    path: Path
    sheet_name: str
    headers: list[str]
    rows: list[WorkbookRow]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--home",
        type=Path,
        default=DEFAULT_HOME,
        help="Directory containing one folder per project.",
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=None,
        help="Project folder names to analyze. Defaults to all projects in --home.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory where reports will be written.",
    )
    parser.add_argument(
        "--key-columns",
        nargs="+",
        default=DEFAULT_KEY_COLUMNS,
        help="Columns used to align records across source and reviewed sheets.",
    )
    parser.add_argument(
        "--country-column",
        default=COUNTRY_COLUMN,
        help='Rows with this column equal to "x" are excluded.',
    )
    parser.add_argument(
        "--include-all-columns",
        action="store_true",
        help="Analyze every shared column instead of only reviewable fields.",
    )
    parser.add_argument(
        "--exclude-columns",
        nargs="+",
        default=[],
        help="Additional columns to exclude from scoring.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many columns/rows to highlight in the Markdown summary.",
    )
    return parser.parse_args()


def normalize_cell_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def discover_projects(home: Path) -> list[str]:
    return sorted(
        project_dir.name
        for project_dir in home.iterdir()
        if project_dir.is_dir() and (project_dir / SOURCE_WORKBOOK).exists()
    )


def col_to_index(cell_ref: str) -> int:
    match = COL_RE.match(cell_ref or "")
    if not match:
        return -1
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - 64)
    return index - 1


def get_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values = []
    for si in root.findall("main:si", NS):
        text = "".join(node.text or "" for node in si.iterfind(".//main:t", NS))
        values.append(text)
    return values


def cell_xml_value(cell: ET.Element, shared_strings: list[str]) -> str | None:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iterfind(".//main:t", NS))
    value_node = cell.find("main:v", NS)
    if value_node is None:
        return None
    raw = value_node.text
    if raw is None:
        return None
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError):
            return raw
    return raw


def normalize_target(target: str) -> str:
    target = target.lstrip("/")
    if target.startswith("xl/"):
        return posixpath.normpath(target)
    return posixpath.normpath(posixpath.join("xl", target))


def first_sheet_target(zf: ZipFile) -> tuple[str, str]:
    workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
    rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.get("Id"): normalize_target(rel.get("Target", ""))
        for rel in rels_root.findall("pkgrel:Relationship", NS)
    }
    first_sheet = workbook_root.find("main:sheets/main:sheet", NS)
    if first_sheet is None:
        raise ValueError("Workbook does not contain any sheets.")
    rid = first_sheet.get(f"{{{REL_NS}}}id")
    if rid not in rel_map:
        raise ValueError("Could not resolve the first sheet relationship.")
    return first_sheet.get("name", "Sheet1"), rel_map[rid]


def load_workbook_rows(path: Path) -> WorkbookData:
    with ZipFile(path) as zf:
        shared_strings = get_shared_strings(zf)
        sheet_name, target = first_sheet_target(zf)
        if target not in zf.namelist():
            raise ValueError(f"Worksheet path not found inside archive: {target}")
        sheet_root = ET.fromstring(zf.read(target))
        xml_rows = sheet_root.findall("main:sheetData/main:row", NS)
        if not xml_rows:
            raise ValueError("Worksheet is empty.")

        header_map: dict[int, str] = {}
        first_row = xml_rows[0]
        for cell in first_row.findall("main:c", NS):
            col_idx = col_to_index(cell.get("r", ""))
            header_map[col_idx] = normalize_cell_value(cell_xml_value(cell, shared_strings))

        max_col = max(header_map) if header_map else -1
        headers = [header_map.get(index, f"column_{index + 1}") for index in range(max_col + 1)]

        rows: list[WorkbookRow] = []
        for row_index, xml_row in enumerate(xml_rows[1:], start=0):
            values_by_col: dict[int, str] = {}
            for cell in xml_row.findall("main:c", NS):
                col_idx = col_to_index(cell.get("r", ""))
                values_by_col[col_idx] = normalize_cell_value(cell_xml_value(cell, shared_strings))

            values = {
                header: values_by_col.get(index, "")
                for index, header in enumerate(headers)
            }
            rows.append(
                WorkbookRow(
                    values=values,
                    excel_row_number=int(xml_row.get("r", "0") or 0),
                    row_index=row_index,
                )
            )

    return WorkbookData(path=path, sheet_name=sheet_name, headers=headers, rows=rows)


def levenshtein_distance(left: str, right: str) -> int:
    if fast_levenshtein_distance is not None:
        return fast_levenshtein_distance(left, right)
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insertions = current[j - 1] + 1
            deletions = previous[j] + 1
            substitutions = previous[j - 1] + (left_char != right_char)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def build_key(row: WorkbookRow, key_columns: list[str]) -> tuple[str, ...]:
    return tuple(row.values.get(column, "") for column in key_columns)


def is_country_x(row: WorkbookRow | None, country_column: str) -> bool:
    if row is None:
        return False
    return row.values.get(country_column, "").strip().lower() == "x"


def safe_divide(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def percent(value: float) -> float:
    return round(value * 100.0, 4)


def choose_compare_columns(
    source_headers: list[str],
    target_headers: list[str],
    include_all_columns: bool,
    extra_excluded_columns: list[str],
) -> list[str]:
    shared = [header for header in source_headers if header in set(target_headers)]

    source_filename_idx = source_headers.index("filename") if "filename" in source_headers else None
    target_filename_idx = target_headers.index("filename") if "filename" in target_headers else None
    if (
        source_filename_idx is not None
        and target_filename_idx is not None
        and source_filename_idx > 0
        and target_filename_idx > 0
    ):
        source_left = set(source_headers[:source_filename_idx])
        target_left = set(target_headers[:target_filename_idx])
        shared = [header for header in shared if header in source_left and header in target_left]

    excluded = set(extra_excluded_columns)
    excluded.add("filename")
    if not include_all_columns:
        excluded |= DEFAULT_EXCLUDED_COLUMNS
    return [column for column in shared if column not in excluded]


def latest_reviewed_workbook(project_dir: Path) -> Path:
    matches = sorted(project_dir.glob(REVIEWED_PATTERN))
    if not matches:
        raise FileNotFoundError(
            f"No reviewed workbook matching {REVIEWED_PATTERN!r} found in {project_dir}"
        )
    return matches[-1]


def detect_ocr_methods(data: WorkbookData) -> list[str]:
    if "OCR_method" not in data.headers:
        return []
    return sorted(
        {
            row.values.get("OCR_method", "").strip()
            for row in data.rows
            if row.values.get("OCR_method", "").strip()
        }
    )


def classify_cohort(ocr_methods: list[str]) -> tuple[str, str]:
    if not ocr_methods:
        return "unknown", UNKNOWN_COHORT
    if all(method == GEMINI_METHOD for method in ocr_methods):
        return " | ".join(ocr_methods), GEMINI_COHORT
    if all(method != GEMINI_METHOD for method in ocr_methods):
        return " | ".join(ocr_methods), NON_GEMINI_COHORT
    return " | ".join(ocr_methods), MIXED_COHORT


def make_writer(path: Path, fieldnames: list[str]) -> tuple[Any, csv.DictWriter]:
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


def summarize_project(
    project_name: str,
    source_data: WorkbookData,
    target_data: WorkbookData,
    key_columns: list[str],
    country_column: str,
    include_all_columns: bool,
    extra_excluded_columns: list[str],
    ocr_method: str,
    cohort: str,
) -> dict[str, Any]:
    compare_columns = choose_compare_columns(
        source_data.headers,
        target_data.headers,
        include_all_columns=include_all_columns,
        extra_excluded_columns=extra_excluded_columns,
    )
    for key_column in key_columns:
        if key_column not in source_data.headers or key_column not in target_data.headers:
            raise KeyError(f"Missing key column {key_column!r} in project {project_name}")

    source_keys = [build_key(row, key_columns) for row in source_data.rows]
    target_keys = [build_key(row, key_columns) for row in target_data.rows]
    source_counts = Counter(source_keys)
    target_counts = Counter(target_keys)
    duplicate_keys = sorted(
        {
            "|".join(key)
            for key, count in source_counts.items()
            if count > 1
        }
        | {
            "|".join(key)
            for key, count in target_counts.items()
            if count > 1
        }
    )

    source_lookup = {build_key(row, key_columns): row for row in source_data.rows}
    target_lookup = {build_key(row, key_columns): row for row in target_data.rows}
    matched_keys = sorted(set(source_lookup) & set(target_lookup))
    unmatched_source_keys = sorted(set(source_lookup) - set(target_lookup))
    unmatched_target_keys = sorted(set(target_lookup) - set(source_lookup))

    column_metrics: dict[str, dict[str, Any]] = {}
    for column in compare_columns:
        column_metrics[column] = {
            "project": project_name,
            "cohort": cohort,
            "ocr_method": ocr_method,
            "column": column,
            "compared_cells": 0,
            "exact_cells": 0,
            "changed_cells": 0,
            "total_char_edit_distance": 0,
            "char_similarity_sum": 0.0,
            "source_non_empty_cells": 0,
            "target_non_empty_cells": 0,
            "filled_cells": 0,
            "cleared_cells": 0,
            "modified_non_empty_cells": 0,
            "changed_edit_distances": [],
        }

    row_metrics: list[dict[str, Any]] = []
    cell_diffs: list[dict[str, Any]] = []

    compared_rows = 0
    excluded_country_x_rows = 0
    exact_rows = 0
    compared_cells = 0
    exact_cells = 0
    changed_cells = 0
    total_char_edit_distance = 0
    char_similarity_sum = 0.0

    for key in matched_keys:
        source_row = source_lookup[key]
        target_row = target_lookup[key]
        if is_country_x(source_row, country_column) or is_country_x(target_row, country_column):
            excluded_country_x_rows += 1
            continue

        compared_rows += 1
        row_changed_cells = 0
        row_exact_cells = 0
        row_char_edit_distance = 0
        changed_columns: list[str] = []

        for column in compare_columns:
            source_value = source_row.values.get(column, "")
            target_value = target_row.values.get(column, "")
            edit_distance = levenshtein_distance(source_value, target_value)
            max_len = max(len(source_value), len(target_value), 1)
            similarity = 1.0 - (edit_distance / max_len)
            is_exact = source_value == target_value

            compared_cells += 1
            row_metric = column_metrics[column]
            row_metric["compared_cells"] += 1
            row_metric["total_char_edit_distance"] += edit_distance
            row_metric["char_similarity_sum"] += similarity
            char_similarity_sum += similarity
            total_char_edit_distance += edit_distance

            if source_value:
                row_metric["source_non_empty_cells"] += 1
            if target_value:
                row_metric["target_non_empty_cells"] += 1

            if is_exact:
                exact_cells += 1
                row_exact_cells += 1
                row_metric["exact_cells"] += 1
            else:
                changed_cells += 1
                row_changed_cells += 1
                row_char_edit_distance += edit_distance
                changed_columns.append(column)
                row_metric["changed_cells"] += 1
                row_metric["changed_edit_distances"].append(edit_distance)

                if not source_value and target_value:
                    change_type = "filled"
                    row_metric["filled_cells"] += 1
                elif source_value and not target_value:
                    change_type = "cleared"
                    row_metric["cleared_cells"] += 1
                else:
                    change_type = "modified"
                    row_metric["modified_non_empty_cells"] += 1

                cell_diffs.append(
                    {
                        "project": project_name,
                        "cohort": cohort,
                        "ocr_method": ocr_method,
                        "catalogNumber": target_row.values.get("catalogNumber", ""),
                        "column": column,
                        "source_value": source_value,
                        "reviewed_value": target_value,
                        "edit_distance": edit_distance,
                        "source_length": len(source_value),
                        "reviewed_length": len(target_value),
                        "change_type": change_type,
                        "source_excel_row": source_row.excel_row_number,
                        "reviewed_excel_row": target_row.excel_row_number,
                    }
                )

        if row_changed_cells == 0:
            exact_rows += 1

        row_metrics.append(
            {
                "project": project_name,
                "cohort": cohort,
                "ocr_method": ocr_method,
                "catalogNumber": target_row.values.get("catalogNumber", ""),
                "source_excel_row": source_row.excel_row_number,
                "reviewed_excel_row": target_row.excel_row_number,
                "changed_cells": row_changed_cells,
                "exact_cells": row_exact_cells,
                "compared_cells": len(compare_columns),
                "exact_cell_agreement_pct": percent(
                    safe_divide(row_exact_cells, len(compare_columns))
                ),
                "total_char_edit_distance": row_char_edit_distance,
                "changed_columns": "; ".join(changed_columns),
            }
        )

    column_rows: list[dict[str, Any]] = []
    for column, stats in column_metrics.items():
        changed_distances = stats.pop("changed_edit_distances")
        column_char_similarity_sum = stats.pop("char_similarity_sum")
        column_rows.append(
            {
                **stats,
                "exact_match_pct": percent(
                    safe_divide(stats["exact_cells"], stats["compared_cells"])
                ),
                "changed_cell_pct": percent(
                    safe_divide(stats["changed_cells"], stats["compared_cells"])
                ),
                "avg_char_edit_distance_per_cell": round(
                    safe_divide(
                        stats["total_char_edit_distance"], stats["compared_cells"]
                    ),
                    4,
                ),
                "avg_char_edit_distance_per_changed_cell": round(
                    safe_divide(
                        stats["total_char_edit_distance"], stats["changed_cells"]
                    ),
                    4,
                ),
                "median_char_edit_distance_changed_cells": round(
                    median(changed_distances), 4
                )
                if changed_distances
                else 0.0,
                "avg_char_similarity_pct": percent(
                    safe_divide(column_char_similarity_sum, stats["compared_cells"])
                ),
            }
        )

    summary = {
        "project": project_name,
        "cohort": cohort,
        "ocr_method": ocr_method,
        "source_workbook": source_data.path.name,
        "reviewed_workbook": target_data.path.name,
        "sheet_name": target_data.sheet_name,
        "source_rows": len(source_data.rows),
        "reviewed_rows": len(target_data.rows),
        "matched_rows": len(matched_keys),
        "unmatched_source_rows": len(unmatched_source_keys),
        "unmatched_reviewed_rows": len(unmatched_target_keys),
        "excluded_country_x_rows": excluded_country_x_rows,
        "compared_rows": compared_rows,
        "compare_columns": len(compare_columns),
        "compared_cells": compared_cells,
        "exact_cells": exact_cells,
        "changed_cells": changed_cells,
        "exact_rows": exact_rows,
        "exact_cell_agreement_pct": percent(safe_divide(exact_cells, compared_cells)),
        "exact_row_agreement_pct": percent(safe_divide(exact_rows, compared_rows)),
        "total_char_edit_distance": total_char_edit_distance,
        "avg_char_edit_distance_per_cell": round(
            safe_divide(total_char_edit_distance, compared_cells), 4
        ),
        "avg_char_edit_distance_per_changed_cell": round(
            safe_divide(total_char_edit_distance, changed_cells), 4
        ),
        "avg_char_similarity_pct": percent(
            safe_divide(char_similarity_sum, compared_cells)
        ),
        "duplicate_key_count": len(duplicate_keys),
        "duplicate_keys": duplicate_keys,
    }

    return {
        "summary": summary,
        "column_rows": column_rows,
        "row_rows": row_metrics,
        "cell_rows": cell_diffs,
        "compare_columns": compare_columns,
        "unmatched_source_keys": [" | ".join(key) for key in unmatched_source_keys],
        "unmatched_reviewed_keys": [" | ".join(key) for key in unmatched_target_keys],
    }


def format_pct(value: float) -> str:
    return f"{value:.2f}%"


def format_markdown_value(value: str, max_len: int = 80) -> str:
    text = value if value else "[blank]"
    text = text.replace("\n", "\\n").replace("|", "\\|")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return f"`{text}`"


def build_cohort_summaries(project_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in project_summaries:
        grouped[summary["cohort"]].append(summary)

    output: list[dict[str, Any]] = []
    for cohort in [GEMINI_COHORT, NON_GEMINI_COHORT, MIXED_COHORT, UNKNOWN_COHORT]:
        summaries = grouped.get(cohort, [])
        if not summaries:
            continue

        compared_rows = sum(item["compared_rows"] for item in summaries)
        compared_cells = sum(item["compared_cells"] for item in summaries)
        changed_cells = sum(item["changed_cells"] for item in summaries)
        total_char_edit_distance = sum(item["total_char_edit_distance"] for item in summaries)
        exact_cells = sum(item["exact_cells"] for item in summaries)
        exact_rows = sum(item["exact_rows"] for item in summaries)
        excluded_country_x_rows = sum(item["excluded_country_x_rows"] for item in summaries)

        output.append(
            {
                "cohort": cohort,
                "project_count": len(summaries),
                "ocr_methods": " ; ".join(sorted({item["ocr_method"] for item in summaries})),
                "compared_rows": compared_rows,
                "excluded_country_x_rows": excluded_country_x_rows,
                "compared_cells": compared_cells,
                "exact_cells": exact_cells,
                "changed_cells": changed_cells,
                "exact_rows": exact_rows,
                "exact_cell_agreement_pct": percent(safe_divide(exact_cells, compared_cells)),
                "exact_row_agreement_pct": percent(safe_divide(exact_rows, compared_rows)),
                "avg_project_exact_cell_agreement_pct": round(
                    mean(item["exact_cell_agreement_pct"] for item in summaries), 4
                ),
                "avg_project_exact_row_agreement_pct": round(
                    mean(item["exact_row_agreement_pct"] for item in summaries), 4
                ),
                "total_char_edit_distance": total_char_edit_distance,
                "avg_char_edit_distance_per_cell": round(
                    safe_divide(total_char_edit_distance, compared_cells), 4
                ),
                "avg_char_edit_distance_per_changed_cell": round(
                    safe_divide(total_char_edit_distance, changed_cells), 4
                ),
                "avg_char_similarity_pct": round(
                    mean(item["avg_char_similarity_pct"] for item in summaries), 4
                ),
            }
        )

    return output


def write_markdown_summary(
    path: Path,
    project_summaries: list[dict[str, Any]],
    cohort_summaries: list[dict[str, Any]],
    column_rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    top_n: int,
) -> None:
    lines: list[str] = []
    lines.append("# Pre vs Post Validation Comparison")
    lines.append("")
    lines.append("## Recommended Metrics")
    lines.append("")
    lines.append(
        "- Exact cell agreement: share of compared cells where the original LLM value already matched the reviewed value."
    )
    lines.append(
        "- Exact row agreement: share of included records where every reviewable field already matched."
    )
    lines.append(
        "- Character edit distance: minimum insertions, deletions, and substitutions needed to convert the LLM value into the reviewed value."
    )
    lines.append(
        "- Changed-cell rate by column: which fields most often needed intervention."
    )
    lines.append(
        "- Fill / clear / modify counts: whether review effort mostly added missing data, removed hallucinations, or rewrote existing text."
    )
    lines.append(
        "- Row-level edit burden: which specimens required the most cleanup overall."
    )
    lines.append("")
    lines.append("Rows with `country == \"x\"` on either side are excluded.")
    lines.append("")
    lines.append("## Final Gemini vs Non-Gemini Comparison")
    lines.append("")
    lines.append(
        "| Cohort | Projects | Compared Rows | Exact Cell Agreement | Exact Row Agreement | Avg Project Exact Cell Agreement | Total Char Edit Distance | Avg Char Edit per Changed Cell |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for summary in cohort_summaries:
        lines.append(
            f"| {summary['cohort']} | {summary['project_count']} | {summary['compared_rows']} | "
            f"{format_pct(summary['exact_cell_agreement_pct'])} | "
            f"{format_pct(summary['exact_row_agreement_pct'])} | "
            f"{format_pct(summary['avg_project_exact_cell_agreement_pct'])} | "
            f"{summary['total_char_edit_distance']} | "
            f"{summary['avg_char_edit_distance_per_changed_cell']:.2f} |"
        )

    grouped_columns: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in column_rows:
        grouped_columns[row["project"]].append(row)

    grouped_cell_diffs: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in cell_rows:
        grouped_cell_diffs[row["project"]][row["column"]].append(row)

    grouped_projects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in project_summaries:
        grouped_projects[summary["cohort"]].append(summary)

    for cohort in [GEMINI_COHORT, NON_GEMINI_COHORT, MIXED_COHORT, UNKNOWN_COHORT]:
        cohort_projects = sorted(
            grouped_projects.get(cohort, []),
            key=lambda row: row["project"],
        )
        if not cohort_projects:
            continue

        lines.append("")
        lines.append(f"## {cohort} Projects")
        lines.append("")
        lines.append("| Project | OCR Method | Compared Rows | Excluded `country=x` Rows | Exact Cell Agreement | Exact Row Agreement | Changed Cells | Total Char Edit Distance |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for summary in cohort_projects:
            lines.append(
                f"| {summary['project']} | {summary['ocr_method']} | {summary['compared_rows']} | "
                f"{summary['excluded_country_x_rows']} | "
                f"{format_pct(summary['exact_cell_agreement_pct'])} | "
                f"{format_pct(summary['exact_row_agreement_pct'])} | "
                f"{summary['changed_cells']} | "
                f"{summary['total_char_edit_distance']} |"
            )

        for summary in cohort_projects:
            project = summary["project"]
            lines.append("")
            lines.append(f"### {project}")
            lines.append("")
            lines.append("#### Most Accurate Columns")
            lines.append("")
            lines.append("| Column | Exact Match | Changed Cells | Avg Char Edit per Changed Cell |")
            lines.append("| --- | ---: | ---: | ---: |")
            top_columns = sorted(
                grouped_columns[project],
                key=lambda row: (-row["exact_match_pct"], row["changed_cells"], row["column"]),
            )[:top_n]
            for row in top_columns:
                lines.append(
                    f"| {row['column']} | {format_pct(row['exact_match_pct'])} | "
                    f"{row['changed_cells']} | {row['avg_char_edit_distance_per_changed_cell']:.2f} |"
                )

            lines.append("")
            lines.append("#### Least Accurate Columns")
            lines.append("")
            lines.append("| Column | Exact Match | Changed Cells | Total Char Edit Distance |")
            lines.append("| --- | ---: | ---: | ---: |")
            bottom_columns = sorted(
                grouped_columns[project],
                key=lambda row: (
                    row["exact_match_pct"],
                    -row["changed_cells"],
                    -row["total_char_edit_distance"],
                    row["column"],
                ),
            )[:top_n]
            for row in bottom_columns:
                lines.append(
                    f"| {row['column']} | {format_pct(row['exact_match_pct'])} | "
                    f"{row['changed_cells']} | {row['total_char_edit_distance']} |"
                )

            lines.append("")
            lines.append("#### Correction Examples By Least Accurate Column")
            lines.append("")
            for row in bottom_columns:
                column_name = row["column"]
                lines.append(f"##### {column_name}")
                lines.append("")
                lines.append(f"- Exact match: {format_pct(row['exact_match_pct'])}")
                lines.append(
                    f"- Changed cells: {row['changed_cells']} of {row['compared_cells']}"
                )
                lines.append(
                    f"- Change mix: filled={row['filled_cells']}, cleared={row['cleared_cells']}, modified={row['modified_non_empty_cells']}"
                )
                lines.append(
                    f"- Total char edit distance: {row['total_char_edit_distance']}"
                )
                lines.append("")
                lines.append("| catalogNumber | From | To | Change Type | Edit Distance |")
                lines.append("| --- | --- | --- | --- | ---: |")
                column_diffs = sorted(
                    grouped_cell_diffs[project][column_name],
                    key=lambda diff: (
                        -diff["edit_distance"],
                        diff["catalogNumber"],
                        diff["source_value"],
                        diff["reviewed_value"],
                    ),
                )[:top_n]
                for diff in column_diffs:
                    lines.append(
                        f"| {diff['catalogNumber']} | "
                        f"{format_markdown_value(diff['source_value'])} | "
                        f"{format_markdown_value(diff['reviewed_value'])} | "
                        f"{diff['change_type']} | "
                        f"{diff['edit_distance']} |"
                    )
                lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project_names = args.projects or discover_projects(args.home)

    project_summaries: list[dict[str, Any]] = []
    all_column_rows: list[dict[str, Any]] = []
    all_row_rows: list[dict[str, Any]] = []
    all_cell_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []

    for project_name in project_names:
        project_dir = args.home / project_name
        if not project_dir.exists():
            raise FileNotFoundError(f"Project directory does not exist: {project_dir}")

        source_path = project_dir / SOURCE_WORKBOOK
        reviewed_path = latest_reviewed_workbook(project_dir)

        source_data = load_workbook_rows(source_path)
        reviewed_data = load_workbook_rows(reviewed_path)
        ocr_method, cohort = classify_cohort(detect_ocr_methods(source_data))
        result = summarize_project(
            project_name=project_name,
            source_data=source_data,
            target_data=reviewed_data,
            key_columns=args.key_columns,
            country_column=args.country_column,
            include_all_columns=args.include_all_columns,
            extra_excluded_columns=args.exclude_columns,
            ocr_method=ocr_method,
            cohort=cohort,
        )

        project_summaries.append(result["summary"])
        all_column_rows.extend(result["column_rows"])
        all_row_rows.extend(result["row_rows"])
        all_cell_rows.extend(result["cell_rows"])

        for key in result["unmatched_source_keys"]:
            unmatched_rows.append(
                {
                    "project": project_name,
                    "cohort": cohort,
                    "ocr_method": ocr_method,
                    "side": "source_only",
                    "key": key,
                }
            )
        for key in result["unmatched_reviewed_keys"]:
            unmatched_rows.append(
                {
                    "project": project_name,
                    "cohort": cohort,
                    "ocr_method": ocr_method,
                    "side": "reviewed_only",
                    "key": key,
                }
            )

    project_summaries = sorted(
        project_summaries,
        key=lambda row: (row["cohort"], row["project"]),
    )
    cohort_summaries = build_cohort_summaries(project_summaries)

    summary_csv = output_dir / "project_summary.csv"
    cohort_csv = output_dir / "cohort_summary.csv"
    column_csv = output_dir / "column_summary.csv"
    row_csv = output_dir / "row_summary.csv"
    cell_csv = output_dir / "cell_diffs.csv"
    unmatched_csv = output_dir / "unmatched_rows.csv"
    summary_json = output_dir / "project_summary.json"
    summary_md = output_dir / "summary.md"
    summary_by_column_md = output_dir / "summary_by_column.md"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_ts_md = output_dir / f"summary_{timestamp}.md"
    summary_by_column_ts_md = output_dir / f"summary_by_column_{timestamp}.md"

    handle, writer = make_writer(summary_csv, list(project_summaries[0].keys()))
    try:
        writer.writerows(project_summaries)
    finally:
        handle.close()

    handle, writer = make_writer(cohort_csv, list(cohort_summaries[0].keys()))
    try:
        writer.writerows(cohort_summaries)
    finally:
        handle.close()

    handle, writer = make_writer(column_csv, list(all_column_rows[0].keys()))
    try:
        writer.writerows(all_column_rows)
    finally:
        handle.close()

    handle, writer = make_writer(row_csv, list(all_row_rows[0].keys()))
    try:
        writer.writerows(all_row_rows)
    finally:
        handle.close()

    if all_cell_rows:
        handle, writer = make_writer(cell_csv, list(all_cell_rows[0].keys()))
        try:
            writer.writerows(all_cell_rows)
        finally:
            handle.close()
    else:
        cell_csv.write_text("", encoding="utf-8")

    if unmatched_rows:
        handle, writer = make_writer(unmatched_csv, list(unmatched_rows[0].keys()))
        try:
            writer.writerows(unmatched_rows)
        finally:
            handle.close()
    else:
        unmatched_csv.write_text("", encoding="utf-8")

    summary_json.write_text(
        json.dumps(project_summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(
        path=summary_md,
        project_summaries=project_summaries,
        cohort_summaries=cohort_summaries,
        column_rows=all_column_rows,
        cell_rows=all_cell_rows,
        top_n=args.top_n,
    )
    write_markdown_summary(
        path=summary_by_column_md,
        project_summaries=project_summaries,
        cohort_summaries=cohort_summaries,
        column_rows=all_column_rows,
        cell_rows=all_cell_rows,
        top_n=args.top_n,
    )
    write_markdown_summary(
        path=summary_ts_md,
        project_summaries=project_summaries,
        cohort_summaries=cohort_summaries,
        column_rows=all_column_rows,
        cell_rows=all_cell_rows,
        top_n=args.top_n,
    )
    write_markdown_summary(
        path=summary_by_column_ts_md,
        project_summaries=project_summaries,
        cohort_summaries=cohort_summaries,
        column_rows=all_column_rows,
        cell_rows=all_cell_rows,
        top_n=args.top_n,
    )

    print(f"Wrote reports to {output_dir}")
    print(f"- {summary_csv.name}")
    print(f"- {cohort_csv.name}")
    print(f"- {column_csv.name}")
    print(f"- {row_csv.name}")
    print(f"- {cell_csv.name}")
    print(f"- {unmatched_csv.name}")
    print(f"- {summary_json.name}")
    print(f"- {summary_md.name}")
    print(f"- {summary_by_column_md.name}")
    print(f"- {summary_ts_md.name}")
    print(f"- {summary_by_column_ts_md.name}")


if __name__ == "__main__":
    main()

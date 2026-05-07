#!/usr/bin/env python3
"""Compare 4 new-model VV runs against historical original and reviewed workbooks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from compare_pre_vs_post_validation import (
    COUNTRY_COLUMN,
    DEFAULT_EXCLUDED_COLUMNS,
    DEFAULT_HOME,
    REVIEWED_PATTERN,
    SOURCE_WORKBOOK,
    WorkbookData,
    WorkbookRow,
    format_markdown_value,
    format_pct,
    is_country_x,
    latest_reviewed_workbook,
    levenshtein_distance,
    load_workbook_rows,
    make_writer,
    percent,
    safe_divide,
)


DEFAULT_NEW_RUN_HOME = Path("/Users/willwe/Desktop/run_vvgo_on_old_vv_OUT")
CATALOG_PREFIX_RE = re.compile(r"^(MICH-V-)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-home",
        type=Path,
        default=DEFAULT_HOME,
        help="Directory containing the historical Comparison_PreVsPostValidation projects.",
    )
    parser.add_argument(
        "--new-run-home",
        type=Path,
        default=DEFAULT_NEW_RUN_HOME,
        help="Directory containing OUT_<project> folders with new results.xlsx files.",
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=None,
        help="Optional subset of project names without the OUT_ prefix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output_new_runs_vs_historical",
        help="Directory where reports will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many columns/examples to show in the Markdown summary.",
    )
    parser.add_argument(
        "--exclude-columns",
        nargs="+",
        default=[],
        help="Additional columns to exclude from scoring.",
    )
    return parser.parse_args()


def discover_new_projects(new_run_home: Path) -> list[str]:
    return sorted(
        folder.name.removeprefix("OUT_")
        for folder in new_run_home.iterdir()
        if folder.is_dir() and folder.name.startswith("OUT_") and (folder / "results.xlsx").exists()
    )


def normalize_catalog_number(value: str) -> str:
    text = (value or "").strip().upper()
    text = CATALOG_PREFIX_RE.sub("", text)
    return text


def row_key(row: WorkbookRow) -> str:
    return normalize_catalog_number(row.values.get("catalogNumber", ""))


def collect_model_info(new_run_dir: Path) -> tuple[str, str]:
    ocr_models: Counter[str] = Counter()
    parser_models: Counter[str] = Counter()
    for json_path in sorted(new_run_dir.glob("*.json"))[:25]:
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        ocr_info = data.get("ocr_info", {})
        if isinstance(ocr_info, dict):
            for model_name in ocr_info:
                ocr_models[model_name] += 1
        parsing_info = data.get("parsing_info", {})
        if isinstance(parsing_info, dict):
            model_name = parsing_info.get("model")
            if model_name:
                parser_models[str(model_name)] += 1
    ocr_model = ", ".join(name for name, _ in ocr_models.most_common()) or "unknown"
    parser_model = ", ".join(name for name, _ in parser_models.most_common()) or "unknown"
    return ocr_model, parser_model


def choose_three_way_columns(
    old_headers: list[str],
    reviewed_headers: list[str],
    new_headers: list[str],
    extra_excluded_columns: list[str],
) -> list[str]:
    shared = [
        header
        for header in reviewed_headers
        if header in set(old_headers) and header in set(new_headers)
    ]

    old_filename_idx = old_headers.index("filename") if "filename" in old_headers else None
    reviewed_filename_idx = reviewed_headers.index("filename") if "filename" in reviewed_headers else None
    if (
        old_filename_idx is not None
        and reviewed_filename_idx is not None
        and old_filename_idx > 0
        and reviewed_filename_idx > 0
    ):
        old_left = set(old_headers[:old_filename_idx])
        reviewed_left = set(reviewed_headers[:reviewed_filename_idx])
        shared = [header for header in shared if header in old_left and header in reviewed_left]

    excluded = set(DEFAULT_EXCLUDED_COLUMNS)
    excluded.add("filename")
    excluded.update(extra_excluded_columns)
    return [header for header in shared if header not in excluded]


def summarize_triplet(
    project_name: str,
    old_data: WorkbookData,
    reviewed_data: WorkbookData,
    new_data: WorkbookData,
    new_ocr_model: str,
    new_parser_model: str,
    extra_excluded_columns: list[str],
) -> dict[str, Any]:
    compare_columns = choose_three_way_columns(
        old_headers=old_data.headers,
        reviewed_headers=reviewed_data.headers,
        new_headers=new_data.headers,
        extra_excluded_columns=extra_excluded_columns,
    )

    old_lookup = {row_key(row): row for row in old_data.rows if row_key(row)}
    reviewed_lookup = {row_key(row): row for row in reviewed_data.rows if row_key(row)}
    new_lookup = {row_key(row): row for row in new_data.rows if row_key(row)}
    matched_keys = sorted(set(old_lookup) & set(reviewed_lookup) & set(new_lookup))

    excluded_country_x_rows = 0
    compared_rows = 0

    old_exact_rows = 0
    new_exact_rows = 0
    old_exact_cells = 0
    new_exact_cells = 0
    compared_cells = 0

    old_total_edit = 0
    new_total_edit = 0
    new_better_cells = 0
    old_better_cells = 0
    tie_cells = 0
    new_matches_reviewed_only = 0
    old_matches_reviewed_only = 0

    column_rows_map: dict[str, dict[str, Any]] = {}
    for column in compare_columns:
        column_rows_map[column] = {
            "project": project_name,
            "column": column,
            "compared_cells": 0,
            "old_exact_cells": 0,
            "new_exact_cells": 0,
            "old_total_char_edit_distance": 0,
            "new_total_char_edit_distance": 0,
            "new_better_cells": 0,
            "old_better_cells": 0,
            "tie_cells": 0,
            "new_matches_reviewed_only": 0,
            "old_matches_reviewed_only": 0,
        }

    cell_rows: list[dict[str, Any]] = []

    for key in matched_keys:
        old_row = old_lookup[key]
        reviewed_row = reviewed_lookup[key]
        new_row = new_lookup[key]
        if (
            is_country_x(old_row, COUNTRY_COLUMN)
            or is_country_x(reviewed_row, COUNTRY_COLUMN)
            or is_country_x(new_row, COUNTRY_COLUMN)
        ):
            excluded_country_x_rows += 1
            continue

        compared_rows += 1
        old_row_exact = True
        new_row_exact = True

        for column in compare_columns:
            old_value = old_row.values.get(column, "")
            reviewed_value = reviewed_row.values.get(column, "")
            new_value = new_row.values.get(column, "")

            old_edit = levenshtein_distance(old_value, reviewed_value)
            new_edit = levenshtein_distance(new_value, reviewed_value)

            compared_cells += 1
            old_total_edit += old_edit
            new_total_edit += new_edit

            column_stats = column_rows_map[column]
            column_stats["compared_cells"] += 1
            column_stats["old_total_char_edit_distance"] += old_edit
            column_stats["new_total_char_edit_distance"] += new_edit

            if old_value == reviewed_value:
                old_exact_cells += 1
                column_stats["old_exact_cells"] += 1
            else:
                old_row_exact = False

            if new_value == reviewed_value:
                new_exact_cells += 1
                column_stats["new_exact_cells"] += 1
            else:
                new_row_exact = False

            if new_edit < old_edit:
                new_better_cells += 1
                column_stats["new_better_cells"] += 1
                winner = "new"
            elif old_edit < new_edit:
                old_better_cells += 1
                column_stats["old_better_cells"] += 1
                winner = "old"
            else:
                tie_cells += 1
                column_stats["tie_cells"] += 1
                winner = "tie"

            if new_value == reviewed_value and old_value != reviewed_value:
                new_matches_reviewed_only += 1
                column_stats["new_matches_reviewed_only"] += 1
            elif old_value == reviewed_value and new_value != reviewed_value:
                old_matches_reviewed_only += 1
                column_stats["old_matches_reviewed_only"] += 1

            if old_value != reviewed_value or new_value != reviewed_value:
                cell_rows.append(
                    {
                        "project": project_name,
                        "catalogNumber": reviewed_row.values.get("catalogNumber", ""),
                        "column": column,
                        "old_value": old_value,
                        "new_value": new_value,
                        "reviewed_value": reviewed_value,
                        "old_edit_distance": old_edit,
                        "new_edit_distance": new_edit,
                        "winner": winner,
                    }
                )

        if old_row_exact:
            old_exact_rows += 1
        if new_row_exact:
            new_exact_rows += 1

    column_rows: list[dict[str, Any]] = []
    for column, stats in column_rows_map.items():
        compared = stats["compared_cells"]
        old_exact_pct = percent(safe_divide(stats["old_exact_cells"], compared))
        new_exact_pct = percent(safe_divide(stats["new_exact_cells"], compared))
        column_rows.append(
            {
                **stats,
                "old_exact_match_pct": old_exact_pct,
                "new_exact_match_pct": new_exact_pct,
                "delta_exact_match_pct": round(new_exact_pct - old_exact_pct, 4),
                "old_avg_char_edit_distance_per_cell": round(
                    safe_divide(stats["old_total_char_edit_distance"], compared), 4
                ),
                "new_avg_char_edit_distance_per_cell": round(
                    safe_divide(stats["new_total_char_edit_distance"], compared), 4
                ),
                "delta_total_char_edit_distance": stats["new_total_char_edit_distance"]
                - stats["old_total_char_edit_distance"],
            }
        )

    summary = {
        "project": project_name,
        "new_ocr_model": new_ocr_model,
        "new_parser_model": new_parser_model,
        "historical_original_workbook": old_data.path.name,
        "historical_reviewed_workbook": reviewed_data.path.name,
        "new_results_workbook": new_data.path.name,
        "matched_rows_all_three": len(matched_keys),
        "excluded_country_x_rows": excluded_country_x_rows,
        "compared_rows": compared_rows,
        "compare_columns": len(compare_columns),
        "compared_cells": compared_cells,
        "old_exact_cells": old_exact_cells,
        "new_exact_cells": new_exact_cells,
        "old_exact_rows": old_exact_rows,
        "new_exact_rows": new_exact_rows,
        "old_exact_cell_agreement_pct": percent(safe_divide(old_exact_cells, compared_cells)),
        "new_exact_cell_agreement_pct": percent(safe_divide(new_exact_cells, compared_cells)),
        "delta_exact_cell_agreement_pct": round(
            percent(safe_divide(new_exact_cells, compared_cells))
            - percent(safe_divide(old_exact_cells, compared_cells)),
            4,
        ),
        "old_exact_row_agreement_pct": percent(safe_divide(old_exact_rows, compared_rows)),
        "new_exact_row_agreement_pct": percent(safe_divide(new_exact_rows, compared_rows)),
        "delta_exact_row_agreement_pct": round(
            percent(safe_divide(new_exact_rows, compared_rows))
            - percent(safe_divide(old_exact_rows, compared_rows)),
            4,
        ),
        "old_total_char_edit_distance": old_total_edit,
        "new_total_char_edit_distance": new_total_edit,
        "delta_total_char_edit_distance": new_total_edit - old_total_edit,
        "old_avg_char_edit_distance_per_cell": round(safe_divide(old_total_edit, compared_cells), 4),
        "new_avg_char_edit_distance_per_cell": round(safe_divide(new_total_edit, compared_cells), 4),
        "old_avg_char_edit_distance_per_changed_cell": round(
            safe_divide(old_total_edit, compared_cells - old_exact_cells), 4
        ),
        "new_avg_char_edit_distance_per_changed_cell": round(
            safe_divide(new_total_edit, compared_cells - new_exact_cells), 4
        ),
        "new_better_cells": new_better_cells,
        "old_better_cells": old_better_cells,
        "tie_cells": tie_cells,
        "new_matches_reviewed_only": new_matches_reviewed_only,
        "old_matches_reviewed_only": old_matches_reviewed_only,
    }

    return {
        "summary": summary,
        "column_rows": column_rows,
        "cell_rows": cell_rows,
    }


def write_markdown_summary(
    path: Path,
    summaries: list[dict[str, Any]],
    column_rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    top_n: int,
) -> None:
    grouped_columns: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in column_rows:
        grouped_columns[row["project"]].append(row)

    grouped_cells: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in cell_rows:
        grouped_cells[row["project"]][row["column"]].append(row)

    lines: list[str] = []
    lines.append("# New Runs vs Historical Comparison")
    lines.append("")
    lines.append(
        "This report compares each new-model run against the same historical project's original LLM output and reviewed final workbook."
    )
    lines.append("")
    lines.append("Rows are aligned by normalized `catalogNumber` and excluded if any side has `country == \"x\"`.")
    lines.append("")
    lines.append("## Overall Summary")
    lines.append("")
    lines.append(
        "| Project | New OCR Model | New Parser Model | Compared Rows | Old Exact Cell Agreement | New Exact Cell Agreement | Delta | Old Char Edit Distance | New Char Edit Distance | Delta |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in summaries:
        lines.append(
            f"| {summary['project']} | {summary['new_ocr_model']} | {summary['new_parser_model']} | "
            f"{summary['compared_rows']} | "
            f"{format_pct(summary['old_exact_cell_agreement_pct'])} | "
            f"{format_pct(summary['new_exact_cell_agreement_pct'])} | "
            f"{summary['delta_exact_cell_agreement_pct']:+.2f} | "
            f"{summary['old_total_char_edit_distance']} | "
            f"{summary['new_total_char_edit_distance']} | "
            f"{summary['delta_total_char_edit_distance']:+d} |"
        )

    lines.append("")
    lines.append("## Aggregate Across 4 Projects")
    lines.append("")
    total_compared_cells = sum(item["compared_cells"] for item in summaries)
    total_compared_rows = sum(item["compared_rows"] for item in summaries)
    old_exact_cells = sum(item["old_exact_cells"] for item in summaries)
    new_exact_cells = sum(item["new_exact_cells"] for item in summaries)
    old_total_edit = sum(item["old_total_char_edit_distance"] for item in summaries)
    new_total_edit = sum(item["new_total_char_edit_distance"] for item in summaries)
    lines.append(
        f"- Compared rows: {total_compared_rows}"
    )
    lines.append(
        f"- Old exact cell agreement: {format_pct(percent(safe_divide(old_exact_cells, total_compared_cells)))}"
    )
    lines.append(
        f"- New exact cell agreement: {format_pct(percent(safe_divide(new_exact_cells, total_compared_cells)))}"
    )
    lines.append(
        f"- Exact cell delta: {percent(safe_divide(new_exact_cells, total_compared_cells)) - percent(safe_divide(old_exact_cells, total_compared_cells)):+.2f} points"
    )
    lines.append(f"- Old total char edit distance to reviewed: {old_total_edit}")
    lines.append(f"- New total char edit distance to reviewed: {new_total_edit}")
    lines.append(f"- Edit distance delta: {new_total_edit - old_total_edit:+d}")

    for summary in summaries:
        project = summary["project"]
        lines.append("")
        lines.append(f"## {project}")
        lines.append("")
        lines.append(
            f"- New run models: OCR={summary['new_ocr_model']}; parser={summary['new_parser_model']}"
        )
        lines.append(
            f"- Compared rows: {summary['compared_rows']} (excluded `country=x`: {summary['excluded_country_x_rows']})"
        )
        lines.append(
            f"- Old exact cell agreement vs reviewed: {format_pct(summary['old_exact_cell_agreement_pct'])}"
        )
        lines.append(
            f"- New exact cell agreement vs reviewed: {format_pct(summary['new_exact_cell_agreement_pct'])}"
        )
        lines.append(
            f"- Exact cell delta: {summary['delta_exact_cell_agreement_pct']:+.2f} points"
        )
        lines.append(
            f"- Old total char edit distance vs reviewed: {summary['old_total_char_edit_distance']}"
        )
        lines.append(
            f"- New total char edit distance vs reviewed: {summary['new_total_char_edit_distance']}"
        )
        lines.append(
            f"- Edit distance delta: {summary['delta_total_char_edit_distance']:+d}"
        )
        lines.append(
            f"- Cell wins: new={summary['new_better_cells']}, old={summary['old_better_cells']}, tie={summary['tie_cells']}"
        )
        lines.append("")
        lines.append("### Most Improved Columns")
        lines.append("")
        lines.append("| Column | Old Exact | New Exact | Delta | Old Edit Distance | New Edit Distance |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        improved = sorted(
            grouped_columns[project],
            key=lambda row: (-row["delta_exact_match_pct"], row["delta_total_char_edit_distance"], row["column"]),
        )[:top_n]
        for row in improved:
            lines.append(
                f"| {row['column']} | {format_pct(row['old_exact_match_pct'])} | "
                f"{format_pct(row['new_exact_match_pct'])} | {row['delta_exact_match_pct']:+.2f} | "
                f"{row['old_total_char_edit_distance']} | {row['new_total_char_edit_distance']} |"
            )

        lines.append("")
        lines.append("### Most Regressed Columns")
        lines.append("")
        lines.append("| Column | Old Exact | New Exact | Delta | Old Edit Distance | New Edit Distance |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        regressed = sorted(
            grouped_columns[project],
            key=lambda row: (row["delta_exact_match_pct"], -row["delta_total_char_edit_distance"], row["column"]),
        )[:top_n]
        for row in regressed:
            lines.append(
                f"| {row['column']} | {format_pct(row['old_exact_match_pct'])} | "
                f"{format_pct(row['new_exact_match_pct'])} | {row['delta_exact_match_pct']:+.2f} | "
                f"{row['old_total_char_edit_distance']} | {row['new_total_char_edit_distance']} |"
            )

        lines.append("")
        lines.append("### Example New Wins")
        lines.append("")
        lines.append("| Column | catalogNumber | Old | New | Reviewed | Old Edit | New Edit |")
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: |")
        new_wins = sorted(
            [row for row in cell_rows if row["project"] == project and row["winner"] == "new"],
            key=lambda row: (row["new_edit_distance"] - row["old_edit_distance"], row["column"], row["catalogNumber"]),
        )[:top_n]
        for row in new_wins:
            lines.append(
                f"| {row['column']} | {row['catalogNumber']} | {format_markdown_value(row['old_value'])} | "
                f"{format_markdown_value(row['new_value'])} | {format_markdown_value(row['reviewed_value'])} | "
                f"{row['old_edit_distance']} | {row['new_edit_distance']} |"
            )

        lines.append("")
        lines.append("### Example Old Wins")
        lines.append("")
        lines.append("| Column | catalogNumber | Old | New | Reviewed | Old Edit | New Edit |")
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: |")
        old_wins = sorted(
            [row for row in cell_rows if row["project"] == project and row["winner"] == "old"],
            key=lambda row: (row["old_edit_distance"] - row["new_edit_distance"], row["column"], row["catalogNumber"]),
        )[:top_n]
        for row in old_wins:
            lines.append(
                f"| {row['column']} | {row['catalogNumber']} | {format_markdown_value(row['old_value'])} | "
                f"{format_markdown_value(row['new_value'])} | {format_markdown_value(row['reviewed_value'])} | "
                f"{row['old_edit_distance']} | {row['new_edit_distance']} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = args.projects or discover_new_projects(args.new_run_home)
    summaries: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []

    for project_name in projects:
        new_run_dir = args.new_run_home / f"OUT_{project_name}"
        historical_dir = args.historical_home / project_name
        if not new_run_dir.exists():
            raise FileNotFoundError(f"Missing new-run directory: {new_run_dir}")
        if not historical_dir.exists():
            raise FileNotFoundError(f"Missing historical project directory: {historical_dir}")

        old_data = load_workbook_rows(historical_dir / SOURCE_WORKBOOK)
        reviewed_data = load_workbook_rows(latest_reviewed_workbook(historical_dir))
        new_data = load_workbook_rows(new_run_dir / "results.xlsx")
        new_ocr_model, new_parser_model = collect_model_info(new_run_dir)

        result = summarize_triplet(
            project_name=project_name,
            old_data=old_data,
            reviewed_data=reviewed_data,
            new_data=new_data,
            new_ocr_model=new_ocr_model,
            new_parser_model=new_parser_model,
            extra_excluded_columns=args.exclude_columns,
        )
        summaries.append(result["summary"])
        column_rows.extend(result["column_rows"])
        cell_rows.extend(result["cell_rows"])

    summaries = sorted(summaries, key=lambda row: row["project"])
    summary_csv = output_dir / "new_vs_historical_project_summary.csv"
    column_csv = output_dir / "new_vs_historical_column_summary.csv"
    cell_csv = output_dir / "new_vs_historical_cell_comparison.csv"
    summary_md = output_dir / "new_vs_historical_summary.md"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_ts_md = output_dir / f"new_vs_historical_summary_{timestamp}.md"

    handle, writer = make_writer(summary_csv, list(summaries[0].keys()))
    try:
        writer.writerows(summaries)
    finally:
        handle.close()

    handle, writer = make_writer(column_csv, list(column_rows[0].keys()))
    try:
        writer.writerows(column_rows)
    finally:
        handle.close()

    handle, writer = make_writer(cell_csv, list(cell_rows[0].keys()))
    try:
        writer.writerows(cell_rows)
    finally:
        handle.close()

    write_markdown_summary(summary_md, summaries, column_rows, cell_rows, args.top_n)
    write_markdown_summary(summary_ts_md, summaries, column_rows, cell_rows, args.top_n)

    print(f"Wrote reports to {output_dir}")
    print(f"- {summary_csv.name}")
    print(f"- {column_csv.name}")
    print(f"- {cell_csv.name}")
    print(f"- {summary_md.name}")
    print(f"- {summary_ts_md.name}")


if __name__ == "__main__":
    main()

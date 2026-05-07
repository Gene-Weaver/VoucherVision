# Pre vs Post Validation Comparison

This folder contains a reusable comparison script for measuring how much human
editing was required to turn the original LLM transcription into the final
reviewed workbook.

By default, it scans all projects under
`/Users/willwe/Downloads/Comparison_PreVsPostValidation`, classifies them by
`OCR_method`, and keeps Gemini vs non-Gemini results separate in the reports.

## What it compares

For each project, the script:

1. Loads `transcribed_prior_to_subsetting.xlsx`.
2. Finds the latest `transcribed__edited__*.xlsx`.
3. Aligns records by `catalogNumber`.
4. Excludes rows where `country == "x"` on either side.
5. Compares the shared reviewable fields column by column.
6. Groups projects into cohorts based on `OCR_method`.
7. Produces a final Gemini vs non-Gemini accuracy comparison.

By default, the reported metrics focus on reviewable specimen fields and exclude
IDs plus machine/runtime provenance columns such as prompts, token counts,
`OCR_method`, file paths, and WFO/GEO helper outputs.

Use `--include-all-columns` if you want a full workbook-wide comparison.
Use `--exclude-columns` to drop extra fields without editing the script.

## Recommended metrics

These are the most useful metrics for describing agreement and human effort:

- Exact cell agreement: percent of compared cells where the LLM already matched the reviewed value.
- Exact row agreement: percent of included specimens where every compared field already matched.
- Changed-cell rate by column: which fields most frequently required review.
- Character edit distance: how many insertions, deletions, or substitutions were needed per changed cell.
- Total character edit distance by row: which specimens demanded the most cleanup.
- Fill / clear / modify counts: whether human effort mostly added missing text, removed bad text, or rewrote existing text.
- Average character similarity: a fine-grained score that still gives partial credit when the reviewed value is close to the original.

## Outputs

The script writes these files to `output/` by default:

- `project_summary.csv`: one row per project with high-level agreement and edit-burden metrics.
- `cohort_summary.csv`: aggregated metrics for Gemini and non-Gemini project cohorts.
- `column_summary.csv`: per-project, per-column accuracy and edit-distance metrics.
- `row_summary.csv`: per-record edit burden.
- `cell_diffs.csv`: one row per changed cell with old value, new value, and edit distance.
- `unmatched_rows.csv`: keys present on only one side.
- `project_summary.json`: machine-readable summary.
- `summary.md`: narrative report with cohort comparison plus per-project column-level error examples.
- `summary_by_column.md`: duplicate Markdown report on a fresh path for easier viewing in editors.

## Usage

From the repo root:

```bash
python3 analysis/pre_vs_post_validation/compare_pre_vs_post_validation.py
```

Custom project list:

```bash
python3 analysis/pre_vs_post_validation/compare_pre_vs_post_validation.py \
  --projects 2023_10_09_I5_bamaral_AllAsia_Onagr 2023_10_13_I4_kathia_AllAsia_Api
```

Compare every shared column instead of only reviewable fields:

```bash
python3 analysis/pre_vs_post_validation/compare_pre_vs_post_validation.py \
  --include-all-columns
```

Exclude a few noisy fields from the scoring pass:

```bash
python3 analysis/pre_vs_post_validation/compare_pre_vs_post_validation.py \
  --exclude-columns additionalText identifiedRemarks
```

Custom output directory:

```bash
python3 analysis/pre_vs_post_validation/compare_pre_vs_post_validation.py \
  --output-dir analysis/pre_vs_post_validation/output_run_01
```

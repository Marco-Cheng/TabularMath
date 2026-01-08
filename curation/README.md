# TabularMath Table Curation

This folder documents how the per-problem parquet tables were produced before they were copied into `tabularmath/artifacts/datasets/`.  Everything needed to rebuild the public tables (inputs, helper modules, and driver scripts) lives here so practitioners can audit or regenerate the data without touching the evaluation code.

## Quick start

```bash
cd tabularmath/curation
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Rebuild GSM8K + AIME tables (2048 rows/problem), normalize them,
# and verify that every parquet contains 2,048 unique rows.
make all

# Copy the outputs into the main repo once satisfied.
rsync -a tabular/gsm8k_problem_tables_2048 ../../artifacts/datasets/
rsync -a tabular/aime_problem_tables_2048 ../../artifacts/datasets/
rsync -a tabular/gsm8k_problem_tables_2048_norm ../../artifacts/datasets/
rsync -a tabular/aime_problem_tables_2048_norm ../../artifacts/datasets/
```

The `Makefile` spells out the parameters we used for the public release (row counts, seeds, and exact AIME problem IDs).  You can also run the Python scripts directly; the sections below explain the knobs in more detail.

## 0. Inputs

All of the intermediate artifacts used during curation are already checked into `data/`:

| File | Purpose |
| --- | --- |
| `data/aime24_seed.csv`, `data/aime25_seed.csv` | Original problem statements + gold answers for AIME 2024/2025 |
| `data/augmented_aime24.jsonl`, `data/augmented_aime25.jsonl` | Augmented variants emitted by `augmentation` (contains generators, verifiers, and sampled assignments) |
| `data/augmented_gsm8k.jsonl` | Augmented GSM8K variants (same structure as the AIME JSONLs) |

If you need to rebuild these JSONLs from scratch, follow the scripts inside `augmentation/` (see `augmentation/pipeline.py` for the multi-pass augmentation loop) to regenerate augmented variants from the seed CSVs.  The remainder of this guide assumes the `augmented_*.jsonl` files already exist.

## 1. Per-problem table generation

Use `make_problem_tables.py` to turn each augmented JSON entry into a CSV/Parquet pair containing numeric features and the target `y`.  The same command works for GSM8K and AIME; adjust `rows_per_problem`, `max_problems`, and `problem_ids` as needed.

All of the arguments we used for the release are codified in the `Makefile`, so you can simply run:

```bash
$ make tables-gsm8k
$ make tables-aime
```

Useful flags:

- `--drop_duplicates` removes duplicate feature rows when augmented variants collide.
- `--require_unique_rows` toggles strict mode: if uniqueness isn’t satisfied the script aborts and you can re-run with a larger `--max_sampling_factor` or `--time_limit_minutes`.
- `--problem_ids` lets you restrict the export to a subset (e.g., the 14 AIME problems that met the 2048-row requirement).

All tables are saved as both CSV and Parquet in the requested `out_dir` (and the `_2048_backup` folders keep previous snapshots).

GSM8K is processed in bulk, whereas the AIME export is restricted to the 14 problems that yield at least 2 048 unique rows:

```
2024-I-1, 2024-I-3, 2024-I-4, 2024-I-7, 2024-I-8, 2024-I-14,
2024-II-1, 2024-II-4, 2024-II-6, 2024-II-8, 2024-II-9,
2024-II-10, 2024-II-13, 2024-II-14
```

If you prefer to call the script directly, remember to point `PYTHONPATH` at this folder so `augmentation` can be imported:

```bash
PYTHONPATH=tabularmath/curation python3 tabularmath/curation/make_problem_tables.py \
  --aug_jsonl tabularmath/curation/data/augmented_gsm8k.jsonl \
  --out_dir tabular/gsm8k_problem_tables_2048 \
  --rows_per_problem 2048 \
  --drop_duplicates --require_unique_rows --max_sampling_factor 80 \
  --seed 2025
```

## 2. Normalized feature copies

Some experiments use feature-standardized versions of the tables.  Generate those with `normalize_tables.py` or the `normalize-*` Make targets:

```bash
$ make normalize-gsm8k
$ make normalize-aime
```

Every numeric feature column (everything except `y`) is z-scored; zero-variance columns are replaced with zeros.  Copy these `_norm` directories into `tabularmath/artifacts/datasets/` only if you need the normalized variants—they are not part of the default release bundle.

## 3. Verification

Run `make verify` (or call `check_tables.py` directly) before copying the data across.  The script ensures that every parquet in the output directories has the requested number of rows, that all rows are unique, and that the `y` column is present.

```bash
python3 check_tables.py --dir tabular/gsm8k_problem_tables_2048 --expected_rows 2048 --require_unique
```

## 4. Packaging for publication

Once the tables have been generated:

1. Copy the curated 2,048-row directories into `tabularmath/artifacts/datasets/`:

   ```bash
   rsync -a tabular/gsm8k_problem_tables_2048 tabularmath/artifacts/datasets/
   rsync -a tabular/aime_problem_tables_2048 tabularmath/artifacts/datasets/
   ```

   (Optional) Repeat the `rsync` commands for the `_norm` directories if you generated normalized copies.

2. Regenerate the manifests to point at these paths:

   ```bash
   perl -0pi -e 's#tabular/#tabularmath/artifacts/datasets/#g' ../evaluation/data_manifests/*.txt
   ```

3. Run the smoke tests described in the root README to ensure evaluation scripts can find the parquet files.

Following these steps reconstructs the exact feature tables used by the TabularMath benchmark and keeps the curation process transparent for new contributors.  If you improve the augmentation logic or add new sources (e.g., Beyond-AIME), document the additional JSONL files and commands here so the same clarity extends to future releases.

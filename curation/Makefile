PYTHON ?= python3
CUR_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

ROWS ?= 2048
SEED ?= 2025

TABULAR_DIR ?= $(CUR_DIR)/tabular
GSM8K_JSON := $(CUR_DIR)/data/augmented_gsm8k.jsonl
AIME_JSON := $(CUR_DIR)/data/augmented_aime25.jsonl

AIME_IDS := 2024-I-1,2024-I-3,2024-I-4,2024-I-7,2024-I-8,2024-I-14,2024-II-1,2024-II-4,2024-II-6,2024-II-8,2024-II-9,2024-II-10,2024-II-13,2024-II-14

GSM8K_DIR := $(TABULAR_DIR)/gsm8k_problem_tables_$(ROWS)
AIME_DIR := $(TABULAR_DIR)/aime_problem_tables_$(ROWS)
GSM8K_NORM_DIR := $(GSM8K_DIR)_norm
AIME_NORM_DIR := $(AIME_DIR)_norm

.PHONY: tables-gsm8k tables-aime normalize-gsm8k normalize-aime all verify clean-tabs

tables-gsm8k:
	@echo "Building GSM8K tables in $(GSM8K_DIR)"
	PYTHONPATH=$(CUR_DIR) $(PYTHON) $(CUR_DIR)/make_problem_tables.py \
		--aug_jsonl $(GSM8K_JSON) \
		--out_dir $(GSM8K_DIR) \
		--rows_per_problem $(ROWS) \
		--drop_duplicates \
		--require_unique_rows \
		--max_sampling_factor 80 \
		--seed $(SEED)

tables-aime:
	@echo "Building AIME tables in $(AIME_DIR)"
	PYTHONPATH=$(CUR_DIR) $(PYTHON) $(CUR_DIR)/make_problem_tables.py \
		--aug_jsonl $(AIME_JSON) \
		--out_dir $(AIME_DIR) \
		--rows_per_problem $(ROWS) \
		--problem_ids $(AIME_IDS) \
		--drop_duplicates \
		--require_unique_rows \
		--max_sampling_factor 80 \
		--seed $(SEED)

normalize-gsm8k: tables-gsm8k
	@echo "Normalizing GSM8K tables into $(GSM8K_NORM_DIR)"
	$(PYTHON) $(CUR_DIR)/normalize_tables.py \
		--input_dir $(GSM8K_DIR) \
		--output_dir $(GSM8K_NORM_DIR)

normalize-aime: tables-aime
	@echo "Normalizing AIME tables into $(AIME_NORM_DIR)"
	$(PYTHON) $(CUR_DIR)/normalize_tables.py \
		--input_dir $(AIME_DIR) \
		--output_dir $(AIME_NORM_DIR)

verify:
	$(PYTHON) $(CUR_DIR)/check_tables.py --dir $(GSM8K_DIR) --expected_rows $(ROWS) --require_unique
	$(PYTHON) $(CUR_DIR)/check_tables.py --dir $(AIME_DIR) --expected_rows $(ROWS) --require_unique

all: tables-gsm8k tables-aime normalize-gsm8k normalize-aime verify

clean-tabs:
	rm -rf $(TABULAR_DIR)

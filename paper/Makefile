# Paper build: data assets -> figures -> PDF.
# Run from the repository root as `make paper`, or here as `make -C paper paper`.

PY ?= python
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
TEX := green-sarc.tex

.PHONY: paper paper-data paper-figures paper-binding-budget paper-realtrace paper-real-arrival paper-adversarial paper-grid paper-sensitivity-grid paper-clean

# Binding-budget gate sweep (§9): full stack under a finite budget, 20 seeds.
paper-binding-budget:
	cd $(ROOT) && $(PY) paper/scripts/run_binding_budget.py \
		--out paper/data/binding_budget_sweep.json

# Real-trace coverage validation (§10): ShareGPT replay, token counts only.
# Caches paper/data/sharegpt_subset.parquet on first run (git-ignored).
paper-realtrace:
	cd $(ROOT) && $(PY) paper/scripts/run_realtrace_replay.py \
		--out paper/data/realtrace_calibration.json \
		--shift-out paper/data/realtrace_shift.json

# Real-arrival ablation (§11): BurstGPT replay, token counts only.
# Caches paper/data/burstgpt_subset.parquet on first run (git-ignored).
paper-real-arrival:
	cd $(ROOT) && $(PY) paper/scripts/run_real_arrival.py \
		--out paper/data/real_arrival.json

# Real grid intensity (§11.5): fetch + cache GB zones from the UK Carbon
# Intensity API (no account). The grid_sensitivity block itself is produced by
# paper-real-arrival, which computes carbon under each cached zone.
paper-grid:
	cd $(ROOT) && $(PY) paper/scripts/load_grid.py

# Joint sensitivity grid (§12.5): 80 cells (delta x cap x route) x 20 seeds.
paper-sensitivity-grid:
	cd $(ROOT) && $(PY) paper/scripts/run_sensitivity_grid.py \
		--out paper/data/sensitivity_grid.json --seeds 20

# Adversarial robustness study (§13): three attack classes vs the gate.
paper-adversarial:
	cd $(ROOT) && $(PY) paper/scripts/run_adversarial.py \
		--out paper/data/adversarial.json

# Regenerate every reproducible data asset.
paper-data: paper-binding-budget paper-realtrace paper-real-arrival paper-sensitivity-grid paper-adversarial
	cd $(ROOT) && $(PY) examples/learning_curve/run_demo.py \
		--emit-json paper/data/learning_curve.json --n 400 --seed 7
	cd $(ROOT) && $(PY) -m paper.scripts.gen_data --seeds 20 \
		--out paper/data/ibp_ablation.json

# Build the 7 figures (PDFs) from the committed data assets.
paper-figures:
	cd $(ROOT) && $(PY) -m paper.scripts.build_figures

# Compile the PDF (two passes resolve cross-references). Requires a LaTeX
# toolchain locally; CI (.github/workflows/paper.yml) renders it otherwise.
paper: paper-figures
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

paper-clean:
	rm -f green-sarc.aux green-sarc.log green-sarc.out green-sarc.pdf
	rm -f figures/*.pdf

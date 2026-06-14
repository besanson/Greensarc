# Paper build: data assets -> figures -> PDF.
# Run from the repository root as `make paper`, or here as `make -C paper paper`.

PY ?= python
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
TEX := green-sarc.tex

.PHONY: paper paper-data paper-figures paper-binding-budget paper-realtrace paper-real-arrival paper-adversarial paper-grid paper-fetch-grid paper-sensitivity-grid paper-multistep paper-clean arxiv

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
# real grid (§11.5): the grid_sensitivity block is produced by paper-real-arrival,
# which reads the committed CSVs at paper/data/grid/. paper-grid just validates
# the cached zones; refreshing the data needs the API key (paper-fetch-grid).
paper-grid:
	cd $(ROOT) && $(PY) paper/scripts/load_grid.py

# Optional: re-fetch real grid intensity from the ElectricityMaps API (§11.5).
# Needs ELECTRICITYMAPS_API_KEY; not a paper-data dependency, since the committed
# CSVs reproduce §11.5 without any API access.
paper-fetch-grid:
	cd $(ROOT) && $(PY) paper/scripts/fetch_grid.py --zones IT,US-CAL-CISO --year 2024

# Joint sensitivity grid (§12.5): 80 cells (delta x cap x route) x 20 seeds.
paper-sensitivity-grid:
	cd $(ROOT) && $(PY) paper/scripts/run_sensitivity_grid.py \
		--out paper/data/sensitivity_grid.json --seeds 20

# Multi-step real-trace ablation (§11.6): SWE-rebench OpenHands plans, token
# counts only. Caches paper/data/swe_rebench_subset.parquet (git-ignored).
paper-multistep:
	cd $(ROOT) && $(PY) paper/scripts/run_multistep_replay.py \
		--out paper/data/multistep_replay.json --max-trajectories 3000

# Adversarial robustness study (§13): three attack classes vs the gate.
paper-adversarial:
	cd $(ROOT) && $(PY) paper/scripts/run_adversarial.py \
		--out paper/data/adversarial.json

# Regenerate every reproducible data asset.
paper-data: paper-binding-budget paper-realtrace paper-real-arrival paper-multistep paper-sensitivity-grid paper-adversarial
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

# Assemble a self-contained arXiv upload: the .tex plus only the figures it
# references. The bibliography is inline (thebibliography), so no .bib/.bbl is
# needed; \pdfoutput=1 in the .tex forces arXiv to use pdflatex for the PDF
# figures. Produces paper/green-sarc-arxiv.tar.gz (figures must be built first).
arxiv:
	rm -rf arxiv-build && mkdir -p arxiv-build/figures
	cp $(TEX) arxiv-build/
	for f in $$(grep -oE '\{[a-zA-Z0-9_]+\.pdf\}' $(TEX) | tr -d '{}' | sort -u); do \
		cp figures/$$f arxiv-build/figures/ || { echo "missing figure: $$f (run make paper-figures)"; exit 1; }; \
	done
	cd arxiv-build && tar czf ../green-sarc-arxiv.tar.gz .
	@echo "wrote paper/green-sarc-arxiv.tar.gz"

paper-clean:
	rm -f green-sarc.aux green-sarc.log green-sarc.out green-sarc.pdf
	rm -f figures/*.pdf
	rm -rf arxiv-build green-sarc-arxiv.tar.gz green-sarc-arxiv.zip

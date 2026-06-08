# Paper build: data assets -> figures -> PDF.
# Run from the repository root as `make paper`, or here as `make -C paper paper`.

PY ?= python
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
TEX := green-sarc.tex

.PHONY: paper paper-data paper-figures paper-clean

# Regenerate the reproducible data assets from the benchmark (20 seeds).
paper-data:
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

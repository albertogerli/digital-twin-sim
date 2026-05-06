# Submission build — regenerate all PDFs from markdown sources + figures.
#
# Prereqs (one-time):
#   pandoc  — already installed via Homebrew
#   xelatex — TinyTeX (~150MB, no sudo): curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh
#   python  — numpy, matplotlib (already in repo's requirements.txt)
#
# Usage:
#   make            → build everything
#   make figures    → regenerate figures only
#   make pdfs       → regenerate paper PDFs (assumes figures + cover letters exist)
#   make clean      → remove all generated artefacts (PDFs + LaTeX intermediates)
#
# The Makefile is deliberately stateless: each `make` rebuilds from source so
# there is no risk of stale figures being shipped in a submission package.

PANDOC := pandoc
XELATEX := xelatex
TLBIN := $(HOME)/Library/TinyTeX/bin/universal-darwin
PYTHON := python3

PAPERS := paper1_oprisk paper2_llm_abm
SOURCES := $(foreach p,$(PAPERS),$(p)/paper.md)
PDF_PAPERS := $(foreach p,$(PAPERS),$(p)/paper.pdf)
PDF_COVERS := $(foreach p,$(PAPERS),$(p)/cover_letter.pdf)
TEX_PAPERS := $(foreach p,$(PAPERS),$(p)/paper.tex)

FIG_SCRIPT := figures/generate_figures.py
FIG_OUT := figures/fig01_cost_vs_shock_loglog.pdf

.PHONY: all figures pdfs clean check

all: figures pdfs

figures: $(FIG_OUT)

$(FIG_OUT): $(FIG_SCRIPT)
	cd .. && $(PYTHON) submission/figures/generate_figures.py

pdfs: $(PDF_PAPERS) $(PDF_COVERS)

%/paper.tex: %/paper.md
	$(PANDOC) $< -o $@ --standalone --pdf-engine=xelatex

%/paper.pdf: %/paper.tex
	cd $(dir $@) && PATH=$(TLBIN):$$PATH $(XELATEX) -interaction=nonstopmode paper.tex >/dev/null
	cd $(dir $@) && PATH=$(TLBIN):$$PATH $(XELATEX) -interaction=nonstopmode paper.tex >/dev/null
	cd $(dir $@) && rm -f paper.aux paper.log paper.out paper.toc

%/cover_letter.pdf: %/cover_letter.md
	cd $(dir $@) && PATH=$(TLBIN):$$PATH $(PANDOC) $(notdir $<) -o $(notdir $@) --pdf-engine=xelatex

check:
	@echo "Files in each submission package:"
	@for p in $(PAPERS); do \
		echo "  $$p:"; \
		ls -la $$p/ 2>/dev/null | tail -n +2; \
		echo; \
	done

clean:
	rm -f $(PDF_PAPERS) $(PDF_COVERS) $(TEX_PAPERS)
	rm -f $(foreach p,$(PAPERS),$(p)/paper.aux $(p)/paper.log $(p)/paper.out $(p)/paper.toc)
	rm -f figures/*.pdf figures/*.png

# Convenience target: rebuild a single paper end-to-end
%-only:
	$(MAKE) $*/paper.pdf $*/cover_letter.pdf

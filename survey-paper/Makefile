# Alaya Taxonomy Paper - Build Pipeline
# Author: Albert K.T. Hui <albert@securityronin.com>
#
# Usage:
#   make          - Build the paper (PDF)
#   make clean    - Remove build artifacts
#   make watch    - Rebuild on file changes (requires fswatch)
#   make wc       - Word count
#   make lint     - Check for common LaTeX issues

PAPER = main
LATEX = pdflatex
BIBTEX = bibtex
SHELL = /bin/bash

# Build directory for intermediate files
BUILD = build

.PHONY: all clean watch wc lint

all: $(PAPER).pdf

$(BUILD):
	mkdir -p $(BUILD)

$(PAPER).pdf: $(PAPER).tex references.bib $(wildcard sections/*.tex) | $(BUILD)
	$(LATEX) -output-directory=$(BUILD) $(PAPER).tex
	cp $(BUILD)/$(PAPER).aux . 2>/dev/null; true
	$(BIBTEX) $(PAPER)
	cp $(PAPER).bbl $(BUILD)/ 2>/dev/null; true
	$(LATEX) -output-directory=$(BUILD) $(PAPER).tex
	$(LATEX) -output-directory=$(BUILD) $(PAPER).tex
	cp $(BUILD)/$(PAPER).pdf .

clean:
	rm -rf $(BUILD)
	rm -f $(PAPER).pdf $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out

watch:
	@echo "Watching for changes..."
	fswatch -o $(PAPER).tex sections/*.tex references.bib | while read; do make; done

wc:
	@texcount -inc -total $(PAPER).tex 2>/dev/null || \
		(echo "texcount not installed. Approximate:" && \
		 cat $(PAPER).tex sections/*.tex | grep -v '^%' | wc -w)

lint:
	@echo "Checking for common issues..."
	@grep -n '\\\\cite{.*,.*,.*,.*}' sections/*.tex && echo "WARNING: >3 citations in single \\cite{}" || true
	@grep -n '[^~]\\\\cite{' sections/*.tex && echo "WARNING: Missing ~ before \\cite{}" || true
	@grep -rn 'TODO\|FIXME\|XXX' sections/*.tex $(PAPER).tex && echo "WARNING: TODOs found" || true

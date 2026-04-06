# Makefile for PNA Paper Compilation

PAPER = pna_paper
TEX = $(PAPER).tex
PDF = $(PAPER).pdf
BIB = $(PAPER).bib

.PHONY: all clean view help

# Default target
all: $(PDF)

# Compile the paper
$(PDF): $(TEX)
	@echo "Compiling LaTeX document..."
	pdflatex $(TEX)
	bibtex $(PAPER) || true
	pdflatex $(TEX)
	pdflatex $(TEX)
	@echo "Done! PDF created: $(PDF)"

# Quick compile (single pass, for quick previews)
quick: $(TEX)
	pdflatex $(TEX)

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lot *.lof *.synctex.gz
	@echo "Cleaned!"

# Clean everything including PDF
cleanall: clean
	rm -f $(PDF)
	@echo "All files cleaned!"

# View the PDF (requires a PDF viewer)
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v open > /dev/null; then \
		open $(PDF); \
	else \
		echo "Please open $(PDF) manually"; \
	fi

# Help message
help:
	@echo "PNA Paper Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make          - Compile the paper (full build with references)"
	@echo "  make quick    - Quick compile (single pass)"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make cleanall - Remove all generated files including PDF"
	@echo "  make view     - Compile and open the PDF"
	@echo "  make help     - Show this help message"

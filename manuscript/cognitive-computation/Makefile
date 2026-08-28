MAIN = cognitive_computation_manuscript
FIGURES = figure1_activity_variability_updrs.pdf \
	figure2_long_awakenings_executive.pdf \
	figure3_wake_bouts_rbdq.pdf

.PHONY: all manuscript audit-supplement clean submission-zip

all: manuscript

manuscript:
	latexmk -pdf $(MAIN).tex

audit-supplement:
	cd supplement && latexmk -pdf cognitive_computation_supplement.tex

clean:
	latexmk -C $(MAIN).tex
	cd supplement && latexmk -C cognitive_computation_supplement.tex

submission-zip:
	zip -j -FS $(MAIN)-submission.zip $(MAIN).tex references.bib sn-jnl.cls sn-vancouver-num.bst $(FIGURES)

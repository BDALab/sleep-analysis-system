MAIN = cognitive_computation_manuscript
SUPPLEMENT = supplement/cognitive_computation_supplement
FIGURES = figure1_activity_variability_updrs.pdf \
	figure2_long_awakenings_executive.pdf \
	figure3_wake_bouts_rbdq.pdf

.PHONY: all manuscript supplement clean submission-zip

all: manuscript supplement

manuscript:
	latexmk -pdf $(MAIN).tex

supplement:
	cd supplement && latexmk -pdf cognitive_computation_supplement.tex

clean:
	latexmk -C $(MAIN).tex
	cd supplement && latexmk -C cognitive_computation_supplement.tex

submission-zip:
	zip -j -FS $(MAIN)-submission.zip $(MAIN).tex references.bib sn-jnl.cls sn-vancouver-num.bst $(FIGURES)

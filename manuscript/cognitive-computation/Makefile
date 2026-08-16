MAIN = cognitive_computation_manuscript
FIGURES = figure1_activity_variability_updrs.pdf \
	figure2_long_awakenings_executive.pdf \
	figure3_wake_bouts_rbdq.pdf

.PHONY: all clean submission-zip

all:
	latexmk -pdf $(MAIN).tex

clean:
	latexmk -C $(MAIN).tex

submission-zip:
	zip -j -FS $(MAIN)-submission.zip $(MAIN).tex references.bib sn-jnl.cls sn-vancouver-num.bst $(FIGURES)

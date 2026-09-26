MAIN = cognitive_computation_manuscript
FIGURES = figure1_activity_variability_updrs.pdf \
	figure2_long_awakenings_executive.pdf \
	figure3_wake_bouts_rbdq.pdf \
	study_workflow.pdf \
	main_analysis_results.pdf

.PHONY: all manuscript figures workflow-figure results-figure audit-supplement clean submission-zip

all: figures manuscript

figures: workflow-figure results-figure

workflow-figure:
	latexmk -pdf study_workflow.tex

results-figure:
	latexmk -pdf main_analysis_results.tex

manuscript:
	latexmk -pdf $(MAIN).tex

audit-supplement:
	cd supplement && latexmk -pdf cognitive_computation_supplement.tex

clean:
	latexmk -C $(MAIN).tex
	latexmk -C study_workflow.tex
	latexmk -C main_analysis_results.tex
	cd supplement && latexmk -C cognitive_computation_supplement.tex

submission-zip:
	zip -j -FS $(MAIN)-submission.zip $(MAIN).tex references.bib sn-jnl.cls sn-vancouver-num.bst $(FIGURES)

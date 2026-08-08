MAIN = cognitive_computation_manuscript

.PHONY: all clean submission-zip

all:
	latexmk -pdf $(MAIN).tex

clean:
	latexmk -C $(MAIN).tex

submission-zip:
	zip -j $(MAIN)-submission.zip $(MAIN).tex references.bib sn-jnl.cls sn-vancouver-num.bst

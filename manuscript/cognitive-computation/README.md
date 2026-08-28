# Cognitive Computation manuscript draft

This folder contains two complementary resources:

- `springer-nature-latex-template-dec-2024.zip`: the unmodified official
  Springer Nature journal-article template package (version 3.1, December 2024).
- `cognitive_computation_manuscript.tex`: a complete editable first draft of
  the HC-versus-preDLB actigraphy article, adapted to the current *Cognitive
  Computation* instructions for original research.
- `figure1_activity_variability_updrs.pdf`,
  `figure2_long_awakenings_executive.pdf`, and
  `figure3_wake_bouts_rbdq.pdf`: canonical WASO-corrected vector figures used by
  the manuscript.
- `supplement/` and `analysis/tripod_classification_audit.py`: detailed working
  classification-audit outputs retained for reproducibility and possible
  reviewer queries. They are not part of the planned submission package.

There is no separate LaTeX class published specifically for *Cognitive
Computation*. Springer Nature states that its universal LaTeX authoring template
can be used for any Springer Nature journal. However, the journal-specific page
currently says manuscripts "should be typed in Word" and does not explicitly
mention LaTeX. The included files therefore use the official Springer template,
but it is prudent to confirm LaTeX acceptance with the editorial office before
submission if the upload system does not offer a LaTeX manuscript type.

## Journal-specific settings already applied

- Springer Nature `sn-jnl` class, December 2024 release
- `pdflatex` compilation
- `referee` mode for double spacing
- numbered Vancouver bibliography style and square-bracket citations
- structured abstract with the required headings, 150--250 words total
- 4--6 keywords
- original-research order: Introduction, Methods, Results, Discussion,
  Conclusion, Acknowledgements, References
- declarations scaffold, including the mandatory Data Availability statement
- no more than three numbered heading levels

Regular papers may contain up to 10,000 words. Short Papers or Letters are
limited to 3,000 words including references, three figures, and one table.

## Compile

From this directory, run:

```sh
make
```

This compiles the manuscript only. To rebuild the internal classification audit
document separately, run `make audit-supplement`.

or directly:

```sh
latexmk -pdf cognitive_computation_manuscript.tex
```

For a clean submission archive containing the referenced figure files, run
`make submission-zip`. Springer
advises keeping all uploaded LaTeX source, bibliography, style, and figure files
in one directory and compiling with `pdflatex` before upload.

## Author queries still open

The draft deliberately leaves bracketed queries where the repository cannot
support a reliable statement. Before submission, complete the author list and
affiliations, recruitment and diagnostic criteria, device placement and wear
protocol, exact clinical instruments, ethics and consent, funding, and
data/code access statements. The detailed classification audit should remain an
internal reproducibility resource unless a reviewer specifically requests it.

## Authoritative links checked 8 August 2026

- Journal submission guidelines:
  <https://link.springer.com/journal/12559/submission-guidelines>
- Springer Nature LaTeX author support:
  <https://www.springernature.com/gp/authors/campaigns/latex-author-support>
- Journal submission system:
  <https://www.editorialmanager.com/cogn/>

The journal instructions take precedence over this starter and over the generic
Springer Nature template.

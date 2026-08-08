# Cognitive Computation LaTeX starter

This folder contains two complementary resources:

- `springer-nature-latex-template-dec-2024.zip`: the unmodified official
  Springer Nature journal-article template package (version 3.1, December 2024).
- `cognitive_computation_manuscript.tex`: a compact starter adapted to the
  current *Cognitive Computation* instructions for original research.

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

or directly:

```sh
latexmk -pdf cognitive_computation_manuscript.tex
```

For a clean submission archive after adding all referenced figure files, run
`make submission-zip` and update the Makefile's file list if needed. Springer
advises keeping all uploaded LaTeX source, bibliography, style, and figure files
in one directory and compiling with `pdflatex` before upload.

## Authoritative links checked 8 August 2026

- Journal submission guidelines:
  <https://link.springer.com/journal/12559/submission-guidelines>
- Springer Nature LaTeX author support:
  <https://www.springernature.com/gp/authors/campaigns/latex-author-support>
- Journal submission system:
  <https://www.editorialmanager.com/cogn/>

The journal instructions take precedence over this starter and over the generic
Springer Nature template.

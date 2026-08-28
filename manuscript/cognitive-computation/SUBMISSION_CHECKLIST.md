# Cognitive Computation submission checklist

## Article format

- [ ] Select the correct article type.
- [ ] Keep a regular paper at or below 10,000 words.
- [ ] Use no more than three heading levels.
- [ ] Keep the structured abstract between 150 and 250 words.
- [ ] Use the abstract headings `Background / introduction`, `Methods`,
      `Results`, and `Conclusions`.
- [ ] Supply 4--6 keywords.
- [ ] Define abbreviations at first mention and use SI units.

## Title page and authorship

- [ ] Provide a concise title, every author's full name and affiliation, and an
      active email address for the corresponding author.
- [ ] Verify author spelling, order, affiliations, and corresponding author
      before submission; the journal restricts later authorship changes.
- [ ] Add ORCID identifiers in the submission system when available.
- [ ] Add an author-contribution statement.

## Reporting and declarations

- [ ] Report ethics approval and informed consent for human-participant work.
- [ ] Include funding and financial/non-financial competing-interest statements.
- [ ] Include the mandatory Data Availability statement for original research.
- [ ] Add code and materials availability statements where applicable.
- [ ] Ensure the methods document participant-level splitting and leakage
      controls for any machine-learning analysis.
- [ ] Run and report the prespecified age-adjusted sensitivity models rather
      than relying only on the exploratory covariate-imbalance rule.
- [ ] Freeze the canonical analysis outputs, tag the exact code commit used to
      produce them, and update the Code Availability statement.

## Project details to complete

- [ ] Confirm the full author list, order, affiliations, corresponding email,
      and CRediT contributions with the supervisor before submission.
- [ ] Add centres, recruitment dates, eligibility criteria, operational preDLB
      criteria, visit intervals, and the relationship among source cohorts.
- [x] Add the clinical device model, 25-Hz acquisition rate, left-wrist side,
      wear/non-wear protocol, seven-night duration, calibration statement, and
      valid-night definition. Public development-source rates are reported
      separately (Newcastle 85.7 Hz; DREAMT E4 32 Hz in the supplied 64-Hz
      aligned table).
- [ ] Specify the UPDRS version/section, RBDq version/range, cognitive tests,
      score harmonisation, and whether instruments matched across cohorts.
- [ ] Replace every bracketed ethics, consent, funding, conflict, data access,
      code access, and acknowledgement placeholder.

## References, tables, and figures

- [ ] Cite references in numerical order and use Vancouver format.
- [ ] Include only cited, published, or accepted works in the reference list.
- [ ] Cite every table and figure in consecutive order.
- [ ] Make captions self-contained and identify reused material and permissions.
- [ ] Check figures remain interpretable in grayscale for print.

## Files and final checks

- [ ] Compile locally with `pdflatex`/`latexmk` and resolve errors and warnings.
- [ ] Keep the main `.tex`, `.bib`, `.bst`, `.cls`, and figures in one directory
      for upload; do not use absolute paths in `\includegraphics`.
- [ ] Upload all editable source files and the compiled PDF if requested.
- [ ] Confirm with the editorial office that LaTeX is accepted if Editorial
      Manager does not provide a LaTeX file designation.

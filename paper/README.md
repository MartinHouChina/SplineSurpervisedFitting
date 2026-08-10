# Computer-Aided Design manuscript

This directory is an anonymized Elsevier `elsarticle` manuscript framework for
submission to *Computer-Aided Design*.

## Files

- `manuscript.tex`: anonymized manuscript submitted for review;
- `title_page.tex`: separate non-anonymized title page required by the journal;
- `sections/`: section drafts and experiment placeholders;
- `references.bib`: verified core references plus clearly marked literature TODOs;
- `highlights.txt`: required 3--5 highlights, each within 85 characters;
- `cover_letter.tex`: optional cover-letter skeleton.

## Build

With a TeX distribution containing `elsarticle`:

```text
latexmk -pdf manuscript.tex
latexmk -pdf title_page.tex
latexmk -pdf cover_letter.tex
```

or run `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` for the manuscript.

## Before submission

1. Train and evaluate the implemented v5 canonical ordinal count-conditioned decoder.
2. Replace every `TODO` and every bracketed numerical placeholder.
3. Add recent CAD/CAGD literature after checking titles, venues, pages and DOIs.
4. Keep author names, affiliations and acknowledgements out of `manuscript.tex`.
5. Upload `title_page.tex` separately because the journal uses double-anonymized review.
6. Deposit code/data or replace the data-availability placeholder with a reason.
7. Review the AI-use declaration and retain only a truthful final statement.

The implementation corresponding to Section 4 is in
`src/spline_fitting/models/count_conditioned_knot_head.py`. Current v3 values
are included only as a diagnostic baseline and must not be presented as results
for the proposed method.

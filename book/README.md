# The fuzzyclust book — authoring conventions

`main.tex` includes `preamble.tex` and one file per chapter from `chapters/`.
Build: `book/build.sh` (cover artwork, figures, generated appendices, PDF in `book/_build/`).
EPUB edition: `python book/epub/make_epub.py` after `book/build.sh` (needs tex4ebook, biber and dvisvgm;
output in `book/_build_epub/`). It is generated from the same chapters with an e-book preamble.

## Chapter template (follow exactly)

```latex
\chapter{Title}\label{ch:slug}
\begin{roadmap}
What this chapter covers, why it matters, how it connects to the previous one (3–5 sentences).
\end{roadmap}

\section{Problem and idea}            % motivation in plain words, one TikZ concept figure
\section{Theory}                      % unified notation (preamble macros: \X \xk \vi \U \mik \Ai \Fi \Dsq \J ...)
  ...
  \begin{algobox}{Name of the algorithm}{slug}
    \begin{algorithm}[H] ... \end{algorithm}   % or an enumerated list for short procedures
  \end{algobox}
  \begin{summarybox}{Name}
    3–6 bullets: what it does, what it assumes, when to use it, main parameters.
  \end{summarybox}
\section{Using it in fuzzyclust}      % \begin{apibox}{fuzzyclust.module.Class} parameters \end{apibox}
                                      % + code snippets via \pysnippet{examples/<sec>/plot_xx.py}{tag}
\section{Examples}                    % one subsection per example script; figures via \exfig; discuss parameter effects
\section{Further reading}             % 3–8 references with \cite{} (BibTeX keys in references.bib)
```

Rules
* Theory and experiments are kept apart (Theory vs. Examples sections).
* Index $i,j$ = clusters, $k$ = samples, $\U$ is $c\times N$ in formulas; say once per chapter that the code stores it transposed.
* Every algorithm gets exactly one `algobox` and one `summarybox`.
* Every example script of the chapter is referenced at least once with `\exfig`; captions are descriptive sentences ("GK adapts the shape of the clusters, FCM cuts across them"), not labels. A caption of several sentences gets a short form for the list of figures: `\exfig[width]{section/script_n}[First sentence.]{Full caption.}{label}` (manual figures and tables: `\caption[short]{long}`).
* Every `algobox` is followed by `\index{}` entries for the algorithm name, its abbreviation and its class (`FCM@\code{FCM}`); add them when a new algorithm box is written.
* Spelling is American (-ize, color, center, neighbor, modeling); code identifiers, file names, labels and BibTeX keys are never changed (`plot_10_initialisation_restarts.py` keeps its name).
* Concept figures (TikZ) are drawn at natural font size inside the 360 pt text width: no `\resizebox`; stack panels vertically or use `scale=` on the coordinates instead; `\code{}` breaks lines after underscores, so long identifiers may be used in running text.
* Snippet tags: in the example scripts wrap the lines to quote between `# [snippet:tag]` and `# [/snippet:tag]`.
* Figures: `book/build_figures.py` writes `figures/<section>/<script>_<n>.pdf`; `n` is the 1-based figure number inside the script.
* Citations: add BibTeX entries to `references.bib` (key style `author:year` or `author:year:topic`); only cite works you have actually seen (the uploaded papers, standard textbooks, the original method papers).
* No MATLAB code; mention the MATLAB toolbox only in the migration appendix.

## Design

The visual identity is defined once, in `cover/brand.tex`, and shared by the
book, the cover, the documentation (`docs/_static/custom.css`, logo, banner)
and the README:

* colors: Pannon blue `#285780` for headings, rules and links; navy `#14304A`
  for dark backgrounds; the Okabe–Ito triple blue/vermilion/green of the logo,
  which is also the palette of every data figure (`fuzzyclust.plotting.PALETTE`);
* type: Libertinus Serif and Libertinus Math for text and formulas, Raleway for
  headings and display, IBM Plex Mono for code;
* boxes: one geometry (colored rule on the left, light tint, title in Raleway);
  only the color tells the kind (algorithm, summary, hint, API, roadmap);
* the cover artwork (`cover/make_cover_art.py`) is a Gustafson–Kessel run of the
  package: samples in the blend of their membership colors, membership contours
  as lines. The same script draws the banner of the README and the docs.

Page layout rules: algorithm boxes are floats; headings and listings carry a
`\needspace`, so they never end a page; widows and orphans are forbidden. Do
not add `\clearpage` or `\vspace` in the chapters.

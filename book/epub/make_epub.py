"""Build the EPUB edition of the book from the LaTeX source.

The PDF sources stay untouched: everything is staged in book/_build_epub/,
where an e-book preamble is derived from preamble.tex (standard fonts, plain
text boxes, PNG figures), the chapters get the small changes tex4ht needs, and
tex4ebook (TeX Live) produces an EPUB 3 with MathML formulas. The algorithm
boxes and the TikZ figures become SVG pictures; the figures of the example
scripts are converted from book/figures/*.pdf to PNG.

Requirements: book/figures (python build_figures.py), cover/cover.pdf,
tex4ebook, biber, dvisvgm and poppler-utils.

Usage: python book/epub/make_epub.py   ->  book/_build_epub/fuzzyclust-book.epub
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
BOOK = HERE.parent
STAGE = BOOK / "_build_epub"


def make_preamble() -> str:
    s = (BOOK / "preamble.tex").read_text()
    for line in [r"\usepackage[lining]{libertinus}", r"\usepackage{libertinust1math}",
                 r"\usepackage[type1,lining]{raleway}", r"\usepackage[scale=0.86]{plex-mono}",
                 r"\usepackage[letterspace=120]{microtype}", r"\usepackage{pdfpages}",
                 r"\usepackage{titlesec,titletoc}", r"\usepackage{fancyhdr,emptypage}",
                 r"\usepackage{needspace}", r"\usepackage{makeidx}", r"\usepackage{imakeidx}",
                 r"\makeindex[columns=2]", r"\preto\section{\Needspace{5\baselineskip}}",
                 r"\preto\subsection{\Needspace{4\baselineskip}}",
                 r"\AtBeginEnvironment{lstlisting}{\Needspace{5\baselineskip}}"]:
        if line + "\n" not in s:
            sys.exit(f"preamble.tex changed, cannot find: {line}")
        s = s.replace(line + "\n", "")
    i = s.index("% --- chapter and section headings")
    s = s[:i] + s[s.index("% --- captions", i):]
    s = s.replace(r"\input{cover/brand}",
                  "\\usepackage{lmodern}\n\\usepackage{amssymb}\n"
                  "\\newcommand{\\fcdisplay}{\\sffamily}\\newcommand{\\fcblack}{\\bfseries}"
                  "\\newcommand{\\fcbold}{\\bfseries}\\newcommand{\\fcsemi}{\\bfseries}"
                  "\\newcommand{\\fcmedium}{}\\newcommand{\\fclightface}{}"
                  "\\newcommand{\\plexmono}{\\ttfamily}\n"
                  "\\newcommand{\\Needspace}[1]{}\\newcommand{\\needspace}[1]{}\n\\input{cover/brand}")
    s = s.replace(r"\includegraphics[width=#1]{figures/#2.pdf}",
                  r"\includegraphics[width=#1]{figures/#2.png}")
    s = s.replace("{../#1}", "{../../#1}")  # code listings: the stage is one level deeper
    s = s.replace("fcbase=fcbrand, unbreakable, float=tbp,", "fcbase=fcbrand, unbreakable,")
    i = s.index(r"% summary box at the end of a theory section")
    j = s.index(r"% --- TikZ styles")
    s = s[:i] + (
        "% text boxes as plain environments (wrapped in classed divs by epub.cfg)\n"
        "\\newenvironment{summarybox}[1]{\\par\\noindent\\textbf{In a nutshell: #1}\\par}{\\par}\n"
        "\\newenvironment{hintbox}[1]{\\par\\noindent\\textbf{#1}\\par}{\\par}\n"
        "\\newenvironment{roadmap}{\\par}{\\par}\n"
        "\\newenvironment{apibox}[1]{\\par\\noindent\\textbf{\\code{#1}}\\par}{\\par}\n\n") + s[j:]
    return s + "\n\\renewcommand{\\index}[1]{}\n"




def make_main() -> str:
    main = (BOOK / "main.tex").read_text()
    version = re.search(r"\\newcommand\{\\fcversion\}\{([^}]*)\}", main).group(1)
    date = re.search(r"\\newcommand\{\\fcdate\}\{([^}]*)\}", main).group(1)
    bibs = "\n".join(line for line in main.splitlines() if line.startswith(r"\addbibresource"))
    body = []
    for c in re.findall(r"\\input\{(chapters/[^}]+)\}", main):
        if c.startswith("chapters/A_"):
            body.append(r"\appendix")
        body.append(rf"\input{{{c}}}")
    return "\n".join([
        r"\documentclass[11pt]{book}", r"\input{preamble_epub}",
        rf"\newcommand{{\fcversion}}{{{version}}}", rf"\newcommand{{\fcdate}}{{{date}}}", bibs,
        r"\title{Fuzzy Clustering and Data Analysis in Python: The fuzzyclust Toolbox}",
        r"\author{J\'anos Abonyi}", rf"\date{{Version {version}, {date}}}",
        r"\begin{document}", (HERE / "front_epub.tex").read_text(), *body,
        r"\printbibliography[heading=bibintoc]", r"\end{document}", ""])


SCRIPT_RE = re.compile(
    r"([_^])\\(text|textup|mathrm|mathbf|mathsf|mathcal|boldsymbol|operatorname|frac|tfrac|sqrt|bm)\b")


def brace_scripts(s: str) -> str:
    """_\\text{x} -> _{\\text{x}}: tex4ht's \\text and friends need the braces."""
    out, i = [], 0
    while (m := SCRIPT_RE.search(s, i)):
        out.append(s[i:m.start()])
        k, ok = m.end(), True
        for _ in range(2 if m.group(2) in ("frac", "tfrac") else 1):
            while k < len(s) and s[k] == " ":
                k += 1
            if k >= len(s) or s[k] != "{":
                ok = False
                break
            depth = 0
            while k < len(s):
                depth += {"{": 1, "}": -1}.get(s[k], 0)
                k += 1
                if depth == 0:
                    break
        if not ok:
            out.append(s[m.start():m.end()])
            i = m.end()
            continue
        out.append(m.group(1) + "{\\" + m.group(2) + s[m.end():k] + "}")
        i = k
    out.append(s[i:])
    return "".join(out)


def tex4ht_safe(s: str) -> str:
    """Source changes that tex4ht needs and the PDF does not."""
    s = brace_scripts(s)
    # \T is a superscript macro; after a subscript tex4ht only nests the scripts
    # correctly when it sees the ^ itself
    s = re.sub(r"\\T(?![A-Za-z])", r"^{\\mathsf{T}}", s)
    # inline code inside math text: plain typewriter instead of \lstinline
    s = re.sub(r"\\text\{\\py\{([^}]*)\}\}",
               lambda m: r"\text{\texttt{" + m.group(1).replace("_", r"\_") + "}}", s)
    return s


def figure_png(pdf: pathlib.Path) -> None:
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    width_pt = float(re.search(r"Page size:\s+([\d.]+)", info).group(1))
    target = STAGE / "figures" / pdf.relative_to(BOOK / "figures").with_suffix("")
    target.parent.mkdir(parents=True, exist_ok=True)
    px = min(1100, round(width_pt / 72 * 150))
    subprocess.run(["pdftocairo", "-png", "-scale-to-x", str(px), "-scale-to-y", "-1", "-singlefile",
                    str(pdf), str(target)], check=True)
    png = target.with_suffix(".png")
    img = Image.open(png).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                                   dither=Image.Dither.NONE)
    img.save(png, optimize=True)


FIG_RE = re.compile(r"<figure class='figure'>.*?</figure>", re.S)
ID_RE = re.compile(r"<span class='id'>\s*(Figure|Table)\s*([\d.A-Z]+)", re.S)
IMG_RE = re.compile(r"<img\b[^>]*?/>", re.S)
QOP_RE = re.compile(r"(<mstyle class='qopname'>)(.*?)(</mstyle>)", re.S)
TEXT_RE = re.compile(r"(^|>)([^<]+)")


def tidy_and_pack(src: pathlib.Path, out: pathlib.Path) -> None:
    def fix_img(tag: str, label: str) -> str:
        if "class='algobox'" in tag:
            return tag
        tag = re.sub(r"\s(width|height)='\d+'", "", tag)
        tag = re.sub(r"alt='[^']*'", f"alt='{label}'", tag, flags=re.S)
        if "figures/" in tag and "class=" not in tag:
            tag = tag.replace("<img ", "<img class='exfig' ", 1)
        return tag

    def fix_figure(m: re.Match) -> str:
        idm = ID_RE.search(m.group(0))
        label = f"{idm.group(1)} {idm.group(2).rstrip('.')}" if idm else "Figure"
        return IMG_RE.sub(lambda t: fix_img(t.group(0), label), m.group(0))

    for page in sorted((src / "OEBPS").glob("*.xhtml")):
        s = page.read_text(encoding="utf-8")
        t = FIG_RE.sub(fix_figure, s)
        t = re.sub(r"<img alt='PIC' height='\d+' src='(figures/[^']+)' width='\d+' />",
                   r"<img alt='' class='exfig' src='\1' />", t)
        # a figcaption must be the first child of its figure: move the anchor into it
        t = re.sub(r"(<figure class='float'>)\s*(<a id='[^']*'></a>)\s*(<figcaption class='caption'>)",
                   r"\1\3\2", t)
        # operator names (arg max, ...) need their text inside <mi>
        t = QOP_RE.sub(lambda m: m.group(1) + TEXT_RE.sub(
            lambda x: x.group(1) + (f"<mi>{x.group(2).strip()}</mi>" if x.group(2).strip() else x.group(2)),
            m.group(2)) + m.group(3), t)
        if t != s:
            page.write_text(t, encoding="utf-8")
    opf = src / "OEBPS" / "content.opf"
    s = opf.read_text(encoding="utf-8")
    s = s.replace("href='cover_epub.jpg' id='cover-image' media-type='image/png'",
                  "href='cover_epub.jpg' id='cover-image' media-type='image/jpeg'")
    s = re.sub(r"<dc:identifier id='dcidid'>[^<]*</dc:identifier>",
               "<dc:identifier id='dcidid'>https://complexsystemsintelligence.github.io/fuzzyclust/"
               "fuzzyclust-book.epub</dc:identifier>\n<dc:publisher>University of Pannonia</dc:publisher>\n"
               "<dc:rights>CC BY 4.0, https://creativecommons.org/licenses/by/4.0/</dc:rights>\n"
               "<dc:subject>fuzzy clustering</dc:subject>\n<dc:subject>data analysis</dc:subject>\n"
               "<dc:subject>Python</dc:subject>\n"
               "<dc:description>Companion book of the fuzzyclust Python toolbox.</dc:description>", s)
    for page in (src / "OEBPS").glob("*.xhtml"):  # the mathml property only where there is math
        if "<math" not in page.read_text(encoding="utf-8"):
            s = s.replace(f"href='{page.name}' id=", f"href='{page.name}' data-nomath='1' id=")
    s = re.sub(r"(href='[^']+') data-nomath='1' (id='[^']+' media-type='application/xhtml\+xml') "
               r"properties='mathml'", r"\1 \2", s)
    s = s.replace(" data-nomath='1'", "")
    opf.write_text(s, encoding="utf-8")
    uid = re.search(r"<dc:identifier id='dcidid'>([^<]*)<", s).group(1)
    for ncx in (src / "OEBPS").glob("*.ncx"):
        t = ncx.read_text(encoding="utf-8")
        ncx.write_text(re.sub(r"<meta content='[^']*' name='dtb:uid' />",
                              f"<meta content='{uid}' name='dtb:uid' />", t), encoding="utf-8")
    with zipfile.ZipFile(out, "w") as z:
        z.write(src / "mimetype", "mimetype", compress_type=zipfile.ZIP_STORED)
        for f in sorted(src.rglob("*")):
            if f.is_file() and f.name != "mimetype" and f.suffix != ".epub":
                z.write(f, f.relative_to(src).as_posix(), compress_type=zipfile.ZIP_DEFLATED)


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()
    for name in ["chapters", "bib", "cover"]:
        shutil.copytree(BOOK / name, STAGE / name)
    shutil.copy(BOOK / "references.bib", STAGE)
    for name in ["epub.cfg", "epub.css", "main_epub.mk4"]:
        shutil.copy(HERE / name, STAGE)
    for tex in (STAGE / "chapters").glob("*.tex"):
        tex.write_text(tex4ht_safe(tex.read_text()))
    (STAGE / "preamble_epub.tex").write_text(make_preamble())
    (STAGE / "main_epub.tex").write_text(make_main())
    subprocess.run(["pdftocairo", "-jpeg", "-jpegopt", "quality=88", "-f", "1", "-l", "1",
                    "-scale-to-y", "1600", "-scale-to-x", "-1", "-singlefile",
                    str(BOOK / "cover" / "cover.pdf"), str(STAGE / "cover_epub")], check=True)
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(figure_png, sorted((BOOK / "figures").rglob("*.pdf"))))
    subprocess.run(["tex4ebook", "-f", "epub3", "-c", "epub.cfg", "-e", "main_epub.mk4", "main_epub.tex"],
                   cwd=STAGE, check=True)
    out = STAGE / "fuzzyclust-book.epub"
    tidy_and_pack(STAGE / "main_epub-epub3", out)
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

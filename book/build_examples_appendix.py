"""Generate book/chapters/C_examples.tex from the gallery scripts.

Every ``examples/<section>/plot_*.py`` contributes one table row: its title (first
line of the module docstring) and a reference to the first book figure it produced.  Run after adding
or renaming an example:  ``python book/build_examples_appendix.py``.
"""

from __future__ import annotations

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
OUT = ROOT / "book" / "chapters" / "C_examples.tex"

SECTION_TITLES = {
    "01_clustering": ("Core partitioning algorithms and validity", "ch:partitioning"),
    "02_visualization": ("Visualization of fuzzy clusters", "ch:visualization"),
    "03_models": ("Model identification", "ch:models"),
    "04_timeseries": ("Time-series analysis", "ch:timeseries"),
    "05_graph": ("Graph-based clustering", "ch:graph"),
    "06_extensions": ("Possibilistic, noise, kernel, and shape-prototype extensions", "ch:robust_shape"),
    "07_relational": ("Relational, categorical, and mixed data", "ch:relational"),
    "08_data_realities": ("Incomplete, very large, and image data", "ch:data_realities"),
    "09_beyond_unsupervised": (
        "Partial supervision, co-clustering, collaboration, and ensembles",
        "ch:beyond",
    ),
}


GREEK = {
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "γ": r"$\gamma$",
    "δ": r"$\delta$",
    "ε": r"$\varepsilon$",
    "η": r"$\eta$",
    "θ": r"$\theta$",
    "λ": r"$\lambda$",
    "μ": r"$\mu$",
    "ρ": r"$\rho$",
    "σ": r"$\sigma$",
    "τ": r"$\tau$",
    "φ": r"$\varphi$",
    "ω": r"$\omega$",
    "–": "--",
    "—": "---",
    "×": r"$\times$",
    "≥": r"$\geq$",
    "≤": r"$\leq$",
    "²": r"$^2$",
    "−": "$-$",
    "→": r"$\rightarrow$",
    "…": r"\ldots{}",
    "κ": r"$\kappa$",
    "ξ": r"$\xi$",
    "π": r"$\pi$",
    "Δ": r"$\Delta$",
}


def tex_escape(s: str) -> str:
    s = s.replace("\\", r"\textbackslash{}")
    for ch in "&%$#_{}":
        s = s.replace(ch, "\\" + ch)
    s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    for k, v in GREEK.items():
        s = s.replace(k, v)
    return s


def rst_to_tex(s: str) -> str:
    """Minimal conversion of the reST used in the docstrings to LaTeX."""
    s = re.sub(r":math:`([^`]*)`", lambda m: "$" + m.group(1) + "$", s)
    s = re.sub(r":class:`~?([^`]*)`", lambda m: r"\code{" + tex_escape(m.group(1).split(".")[-1]) + "}", s)
    s = re.sub(r":func:`~?([^`]*)`", lambda m: r"\code{" + tex_escape(m.group(1).split(".")[-1]) + "}", s)
    s = re.sub(r":mod:`~?([^`]*)`", lambda m: r"\code{" + tex_escape(m.group(1)) + "}", s)
    s = re.sub(r"\$\$(.*?)\$\$", lambda m: "\\[" + m.group(1) + "\\]", s)
    parts = re.split(r"(\\\[.*?\\\]|\$[^$]*\$|``[^`]*``|\\code\{[^}]*\})", s)
    out = []
    for p in parts:
        if p.startswith("$") or p.startswith("\\code") or p.startswith("\\["):
            out.append(p)
        elif p.startswith("``"):
            code = tex_escape(p[2:-2]).replace("(", "(\\allowbreak{}").replace(",", ",\\allowbreak{}")
            out.append(r"\code{" + code + "}")
        else:
            p = tex_escape(p)
            p = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", p)
            p = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", p)
            out.append(p)
    return "".join(out)


def describe(script: pathlib.Path) -> tuple[str, str]:
    tree = ast.parse(script.read_text())
    doc = ast.get_docstring(tree) or script.stem
    lines = doc.strip().splitlines()
    title = lines[0].strip()
    body = "\n".join(lines[1:]).strip()
    body = re.sub(r"^=+\s*\n", "", body)  # underline of the title
    paragraphs = []
    pars = [p for p in body.split("\n\n") if p.strip()]
    i = 0
    while i < len(pars):
        par = pars[i]
        stripped = par.strip()
        if stripped.startswith(".. math::"):
            first = stripped.splitlines()[0][len(".. math::") :].strip()
            expr_lines = ([first] if first else []) + stripped.splitlines()[1:]
            if not expr_lines and i + 1 < len(pars):  # expression in the next (indented) block
                i += 1
                expr_lines = pars[i].splitlines()
            i += 1
            continue  # formulas are left to the chapters and the scripts themselves
        elif stripped.startswith(".."):  # any other directive ends the intro
            break
        elif any(re.match(r"^[=\-]{3,}( +[=\-]{3,})*$", line.strip()) for line in stripped.splitlines()):
            i += 1
            continue  # reST tables are not reproduced here
        elif all(
            line.lstrip().startswith(("* ", "- ")) or line.startswith("  ") for line in stripped.splitlines()
        ):
            items = []
            for line in stripped.splitlines():
                if line.lstrip().startswith(("* ", "- ")):
                    items.append(line.lstrip()[2:].strip())
                elif items:
                    items[-1] += " " + line.strip()
            paragraphs.append("@@ITEMIZE@@" + "@@ITEM@@".join(items) + "@@END@@")
        else:
            paragraphs.append(" ".join(line.strip() for line in stripped.splitlines()))
        i += 1
    return title, " ".join(paragraphs)


def figure_labels() -> dict[str, list[str]]:
    """Map 'section/script' to the labels of the book figures made from it, in order of appearance."""
    found: dict[str, list[str]] = {}
    chapters = sorted((ROOT / "book" / "chapters").glob("[0-9]*.tex"))
    for tex in chapters:
        text = tex.read_text()
        for m in re.finditer(r"\\exfig(?:\[[^\]]*\])?\{([^}]*)\}(?:\[(?:[^\[\]]|\[[^\]]*\])*\])?\{", text):
            path = m.group(1)
            key = re.sub(r"_\d+$", "", path)
            # the label is the last brace group of the macro call
            depth, j, start = 0, m.end() - 1, None
            groups = []
            while j < len(text) and len(groups) < 2:
                ch = text[j]
                if ch == "{":
                    if depth == 0:
                        start = j + 1
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        groups.append(text[start:j])
                elif depth == 0 and not ch.isspace():
                    break
                j += 1
            if len(groups) == 2:
                found.setdefault(key, []).append("fig:" + groups[1])
    return found


def main() -> None:
    labels = figure_labels()
    chunks = [
        r"\chapter{The example gallery}\label{app:examples}",
        r"\begin{roadmap}",
        "Every algorithm of the toolbox has at least one runnable example script. The scripts generate their "
        "own data, explain the method in their comments, and end with a study of the main parameters. "
        "The same "
        "scripts produce the gallery of the online documentation, the Jupyter notebooks, and every computed "
        "figure and code listing of this book. The tables below list them by folder of \\code{examples/}, "
        "with the first figure of the book that each script produced.",
        r"\end{roadmap}",
        "",
    ]
    for folder in sorted(p for p in EXAMPLES.iterdir() if p.is_dir()):
        title, label = SECTION_TITLES.get(folder.name, (folder.name, None))
        ref = f" The methods are described in \\cref{{{label}}}." if label else ""
        short = title if len(title) <= 40 else title.split(",")[0].split(" and ")[0]
        chunks.append(f"\\section[{tex_escape(short)}]{{{tex_escape(title)}}}")
        chunks.append(f"Folder \\code{{examples/{tex_escape(folder.name)}/}}.{ref}")
        chunks.append("")
        chunks.append(
            r"{\small\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.40\textwidth}"
            r">{\raggedright\arraybackslash}p{0.44\textwidth}>{\raggedleft\arraybackslash}p{0.09\textwidth}@{}}"
        )
        chunks.append(r"\toprule Script & Topic & Figure \\\midrule\endfirsthead")
        chunks.append(r"\toprule Script & Topic & Figure \\\midrule\endhead")
        chunks.append(r"\bottomrule\endfoot")
        for script in sorted(folder.glob("plot_*.py")):
            t, _ = describe(script)
            figs = labels.get(f"{folder.name}/{script.stem}", [])
            fig = f"\\ref{{{figs[0]}}}" if figs else "---"
            chunks.append(f"\\code{{{tex_escape(script.name)}}} & {rst_to_tex(t)} & {fig} \\\\[2pt]")
        chunks.append(r"\end{longtable}}")
        chunks.append("")
    OUT.write_text("\n".join(chunks) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

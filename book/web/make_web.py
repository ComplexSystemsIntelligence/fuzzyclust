# ruff: noqa: E501
"""Build the web edition of the book from the LaTeX source.

The chapters are converted with tex4ht exactly as for the EPUB edition (see
book/epub/make_epub.py: same staging, same e-book preamble, MathML formulas,
SVG algorithm boxes and TikZ figures, PNG figures of the example scripts).
The pages are then laid out as a reading site in the look of the fuzzyclust
documentation: one page per chapter, the table of contents on the left, the
sections of the chapter on the right, previous/next links, and next to every
figure of an example script a link that opens the script as a notebook in
Google Colab.

Requirements: as for the EPUB (book/figures, cover/cover.pdf, tex4ht, biber,
dvisvgm, poppler-utils) plus beautifulsoup4.

Usage: python book/web/make_web.py  ->  book/_build_web/site/ (served as /book/)
"""

from __future__ import annotations

import html
import pathlib
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup

HERE = pathlib.Path(__file__).resolve().parent
BOOK = HERE.parent
sys.path.insert(0, str(BOOK / "epub"))
import make_epub as ep  # noqa: E402

STAGE = BOOK / "_build_web"
SITE = STAGE / "site"
REPO = "ComplexSystemsIntelligence/fuzzyclust"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/gh-pages/notebooks/auto_examples"
TITLE = "Fuzzy Clustering and Data Analysis in Python"
SUBTITLE = "The fuzzyclust toolbox"


# --------------------------------------------------------------------------- build
def stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()
    for name in ["chapters", "bib", "cover"]:
        shutil.copytree(BOOK / name, STAGE / name)
    shutil.copy(BOOK / "references.bib", STAGE)
    for tex in (STAGE / "chapters").glob("*.tex"):
        tex.write_text(ep.tex4ht_safe(tex.read_text()))
    (STAGE / "preamble_epub.tex").write_text(ep.make_preamble())
    main = ep.make_main().replace("This EPUB edition", "This web edition")
    (STAGE / "main_web.tex").write_text(main)
    (STAGE / "main_web.mk4").write_text((BOOK / "epub" / "main_epub.mk4").read_text())
    cfg = (BOOK / "epub" / "epub.cfg").read_text()
    cfg = "\n".join(line for line in cfg.splitlines() if "CoverMetadata" not in line and "AddCss" not in line)
    (STAGE / "web.cfg").write_text(cfg.replace(r"\Preamble{xhtml,mathml}", r"\Preamble{xhtml,mathml,html5,2}"))
    ep.STAGE = STAGE  # figure_png writes into the stage of the module
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(ep.figure_png, sorted((BOOK / "figures").rglob("*.pdf"))))
    subprocess.run(["pdftocairo", "-png", "-f", "1", "-l", "1", "-scale-to-x", "520", "-scale-to-y", "-1",
                    "-singlefile", str(BOOK / "cover" / "cover.pdf"), str(STAGE / "cover")], check=True)
    subprocess.run(["make4ht", "-c", "web.cfg", "-e", "main_web.mk4", "main_web.tex"], cwd=STAGE, check=True)


# --------------------------------------------------------------------------- layout
def slug(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return "-".join(text.split("-")[:5])


def page_list() -> list[dict]:
    """Pages in reading order with their new file names and titles."""
    order = ["main_webli1.html"] + [f"main_webch{i}.html" for i in range(1, 13)] + \
        [f"main_webap{i}.html" for i in range(1, 4)] + ["main_webli2.html"]
    pages = []
    for old in order:
        soup = BeautifulSoup((STAGE / old).read_text(encoding="utf-8"), "lxml")
        head = soup.find(["h2"], class_=re.compile("chapterHead|likechapterHead|appendixHead"))
        mark = head.find("span", class_="titlemark") if head else None
        number = ""
        if mark:
            number = mark.get_text(" ", strip=True).replace("Chapter", "").replace("Appendix", "").strip()
            mark.extract()
        title = head.get_text(" ", strip=True) if head else old
        name = f"{number.lower()}-{slug(title)}.html" if number else f"{slug(title)}.html"
        if number.isdigit():
            name = f"{int(number):02d}-{slug(title)}.html"
        pages.append({"old": old, "new": name, "number": number, "title": title})
    return pages


def nav_html(pages: list[dict], current: str | None) -> str:
    items = []
    for p in pages:
        num = f"<span class='num'>{p['number']}</span>" if p["number"] else "<span class='num'></span>"
        cls = " class='current'" if p["new"] == current else ""
        items.append(f"<li{cls}><a href='{p['new']}'>{num}{html.escape(p['title'])}</a></li>")
    return "<nav class='toc' aria-label='Contents'><p class='toc-title'>Contents</p><ol>" + "".join(items) + \
        "</ol></nav>"


def header_html() -> str:
    return (
        "<header class='topbar'><a class='brand' href='index.html'><img src='logo.svg' alt='' width='34' "
        f"height='34' /><span><b>{TITLE}</b><small>{SUBTITLE}</small></span></a>"
        "<nav class='toplinks'><a href='../'>Documentation</a><a href='../auto_examples/index.html'>Examples</a>"
        "<a href='../fuzzyclust-book.pdf'>PDF</a><a href='../fuzzyclust-book.epub'>EPUB</a>"
        f"<a href='https://github.com/{REPO}'>GitHub</a></nav></header>")


FOOTER = ("<footer class='foot'>© 2026 János Abonyi, University of Pannonia · licensed under "
          "<a href='https://creativecommons.org/licenses/by/4.0/'>CC BY 4.0</a> · "
          "web edition of the book, generated from the same LaTeX source as the "
          "<a href='../fuzzyclust-book.pdf'>PDF</a> · software: "
          f"<a href='https://github.com/{REPO}'>fuzzyclust</a> (MIT) · "
          "<code>pip install fuzzyclust</code></footer>")


def wrap(title: str, body: str, left: str, right: str, extra_class: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)} · {TITLE}</title>
<link rel="icon" href="../_static/favicon.png" />
<link rel="stylesheet" href="main_web.css" />
<link rel="stylesheet" href="web.css" />
</head>
<body class="{extra_class}">
{header_html()}
<div class="layout">
<aside class="left">{left}</aside>
<main class="content">
{body}
</main>
<aside class="right">{right}</aside>
</div>
{FOOTER}
</body>
</html>
"""


def colab_links(soup: BeautifulSoup) -> None:
    seen = set()
    for fig in soup.find_all("figure", class_="figure"):
        img = fig.find("img", src=re.compile(r"^figures/"))
        if not img:
            continue
        m = re.match(r"figures/([^/]+)/(.+)_\d+\.png$", img["src"])
        if not m or m.group(2) in seen:
            continue
        seen.add(m.group(2))
        folder, script = m.groups()
        box = soup.new_tag("p", attrs={"class": "runbox"})
        box.append(BeautifulSoup(
            f"<a class='colab' href='{COLAB}/{folder}/{script}.ipynb'>Open in Colab</a>"
            f"<a href='../auto_examples/{folder}/{script}.html'>Example <code>{script}.py</code> in the gallery</a>",
            "html.parser"))
        fig.append(box)


def tidy(soup: BeautifulSoup) -> None:
    for div in soup.find_all("div", class_="crosslinks"):
        div.decompose()
    for img in soup.find_all("img"):
        if img.get("src", "").startswith("figures/"):
            for attr in ("width", "height"):
                img.attrs.pop(attr, None)
            img["class"] = "exfig"
            fig = img.find_parent("figure")
            cap = fig.find("span", class_="id") if fig else None
            img["alt"] = cap.get_text(strip=True).rstrip(".") if cap else ""
        elif img.get("src", "").endswith(".svg") and "algobox" not in img.get("class", []):
            fig = img.find_parent("figure")
            cap = fig.find("span", class_="id") if fig else None
            img["alt"] = cap.get_text(strip=True).rstrip(".") if cap else "Diagram"


def build_site() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir()
    pages = page_list()
    rename = {p["old"]: p["new"] for p in pages}
    rename["main_web.html"] = "index.html"
    href_re = re.compile(r"\b(main_web(?:ch\d+|ap\d+|li\d+)?\.html)")

    for i, p in enumerate(pages):
        text = (STAGE / p["old"]).read_text(encoding="utf-8")
        text = href_re.sub(lambda m: rename.get(m.group(1), m.group(1)), text)
        soup = BeautifulSoup(text, "lxml")
        tidy(soup)
        colab_links(soup)
        body = soup.body
        sections = []
        for h in body.find_all("h3", class_=re.compile("sectionHead|likesectionHead")):
            anchor = h.find("a", id=True)
            if anchor:
                sections.append(f"<li><a href='#{anchor['id']}'>{html.escape(h.get_text(' ', strip=True))}</a></li>")
        right = ("<nav class='onpage' aria-label='On this page'><p class='toc-title'>On this page</p><ul>"
                 + "".join(sections) + "</ul></nav>") if sections else ""
        prev_p = pages[i - 1] if i > 0 else None
        next_p = pages[i + 1] if i + 1 < len(pages) else None
        pager = "<nav class='pager'>"
        pager += (f"<a class='prev' href='{prev_p['new']}'><small>Previous</small>{html.escape(prev_p['title'])}</a>"
                  if prev_p else "<a class='prev' href='index.html'><small>Previous</small>Title page</a>")
        pager += (f"<a class='next' href='{next_p['new']}'><small>Next</small>{html.escape(next_p['title'])}</a>"
                  if next_p else "<span></span>")
        pager += "</nav>"
        content = body.decode_contents() + pager
        title = f"{p['number']} {p['title']}".strip()
        (SITE / p["new"]).write_text(wrap(title, content, nav_html(pages, p["new"]), right), encoding="utf-8")

    # title page with the cover, the links and the full table of contents
    front = BeautifulSoup((STAGE / "main_web.html").read_text(encoding="utf-8"), "lxml")
    imprint = ""
    for p in front.body.find_all("p"):
        t = p.get_text(" ", strip=True)
        if t.startswith(("©", "Software:", "The research", "This web edition")):
            imprint += str(p)
    toc_items = []
    for p in pages:
        num = f"<span class='num'>{p['number']}</span>" if p["number"] else "<span class='num'></span>"
        toc_items.append(f"<li><a href='{p['new']}'>{num}{html.escape(p['title'])}</a></li>")
    index = f"""<section class="hero">
<img class="cover" src="cover.png" alt="Cover of {TITLE}" width="260" />
<div>
<p class="kicker">Open book · version 1.0 · November 2026</p>
<h1>{TITLE}</h1>
<p class="sub">{SUBTITLE}</p>
<p class="author">János Abonyi · University of Pannonia, Veszprém</p>
<p class="lead">Fuzzy clustering assigns every observation a degree of membership in every group instead of
forcing it into one. This book develops the main families of fuzzy clustering algorithms in one notation and
shows, for each of them, how to run it in Python with the <code>fuzzyclust</code> package and how its parameters
change the result. Every figure of a computed result comes from an example script that opens in Google Colab.</p>
<p class="buttons"><a class="main" href="{pages[1]['new']}">Start reading</a><a href="../fuzzyclust-book.pdf">PDF</a>
<a href="../fuzzyclust-book.epub">EPUB</a><a href="../">Software documentation</a></p>
</div>
</section>
<h2 class="toc-head">Contents</h2>
<ol class="fulltoc">{''.join(toc_items)}</ol>
<div class="imprint">{imprint}</div>"""
    (SITE / "index.html").write_text(wrap("Title page", index, nav_html(pages, None), "", "home"), encoding="utf-8")

    for f in STAGE.glob("main_web*.svg"):
        shutil.copy(f, SITE)
    shutil.copytree(STAGE / "figures", SITE / "figures",
                    ignore=shutil.ignore_patterns("*.pdf", "*.bb", "*.xbb"))
    css = (STAGE / "main_web.css").read_text()
    css = css.replace("body{ margin:1em auto; max-width:80ch; padding:0 .62em; }", "")
    (SITE / "main_web.css").write_text(css)
    shutil.copy(STAGE / "cover.png", SITE)
    shutil.copy(HERE / "web.css", SITE)
    shutil.copy(HERE / "logo.svg", SITE)
    print(f"{SITE}: {len(pages) + 1} pages")


if __name__ == "__main__":
    if "--layout-only" not in sys.argv:
        stage()
    build_site()

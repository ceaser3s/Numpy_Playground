"""Mini-DSL used by the catalog files.

    cat("input", "Input (read)", "Loading data into pandas", "📥", data="none")
    fn("pd.read_csv", "One-line description.", '''code''', reads="How to read the output", params=["name - meaning"], tip="...")

data      : "raw"   -> `df` is the DIRTY table      "clean" -> `df` is the CLEANED table      "none" -> snippet builds its own data
status    : "removed" | "deprecated" | "renamed"  (the code shows the modern replacement)
needs     : extra browser packages: matplotlib, scipy, jinja2, lxml, pyarrow, sqlalchemy, openpyxl, tabulate
local     : if set, the snippet cannot run in a browser (message explains why) - code is shown but Run is disabled
"""
import textwrap

CATEGORIES = []
ENTRIES = []
_cur = None


def cat(cid, title, blurb, icon, data="clean", intro=None):
    global _cur
    _cur = dict(id=cid, title=title, blurb=blurb, icon=icon, data=data, intro=intro or blurb)
    CATEGORIES.append(_cur)


def use(cid):
    """Continue adding entries to an already defined category."""
    global _cur
    _cur = next(c for c in CATEGORIES if c["id"] == cid)


def fn(name, desc, code, data=None, reads=None, params=None, tip=None, status=None, needs=None, local=None, alt=None):
    assert _cur is not None, "call cat() first"
    ENTRIES.append(dict(
        id=f"{_cur['id']}:{name}", cat=_cur["id"], name=name, desc=" ".join(desc.split()),
        code=textwrap.dedent(code).strip("\n"), data=data or _cur["data"], reads=reads, params=params, tip=tip,
        status=status, needs=list(needs or []), local=local, alt=alt,
    ))

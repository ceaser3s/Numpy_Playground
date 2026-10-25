# runner.py - the Python side of NumPy Practice Lab (runs inside Pyodide in the browser,
# but is plain CPython/NumPy code, so it can also be tested on a normal machine).
#
# JS calls:  init(...) -> run(...) -> page(...) / dataset_page(...) / dataset_summary() / check(...)
# Every public function returns a JSON string.
import ast
import base64
import contextlib
import html as _html
import io
import json
import re
import time
import traceback
import warnings

import numpy as np

MAX_COLS = 40
MAX_TEXT = 20000
STATE = {"DATA": {}, "RESULTS": {}, "PLAY": None}
DATASETS = {
    "students": ("data/students.csv", "student_cols"),
    "sensor": ("data/sensor.csv", "sensor_cols"),
    "houses": ("data/houses.csv", "house_cols"),
}


# ------------------------------------------------------------------------------------------------ setup
def _make_image():
    """64x64 grey image: dark-to-light gradient, a bright disc, a bright square and a little noise."""
    rg = np.random.default_rng(7)
    y, x = np.mgrid[0:64, 0:64]
    im = 40 + 1.2 * x
    im = np.where((x - 20) ** 2 + (y - 22) ** 2 < 11 ** 2, 220, im)
    im = np.where((abs(x - 46) < 9) & (abs(y - 44) < 9), 250, im)
    return np.clip(im + rg.normal(0, 6, im.shape), 0, 255).round().astype(np.uint8)


def init(base=""):
    """Load the CSVs once (they are already written into the Pyodide file system)."""
    t0 = time.perf_counter()
    data = {}
    for name, (path, colvar) in DATASETS.items():
        with open(base + path, encoding="utf-8") as f:
            cols = f.readline().strip().split(",")
        conv = {cols.index("gender"): lambda s: 0.0 if s.strip() == "M" else 1.0} if "gender" in cols else None
        arr = np.genfromtxt(base + path, delimiter=",", skip_header=1, converters=conv, encoding="utf-8")
        data[name] = (arr, cols)
    data["img"] = (_make_image(), None)
    STATE["DATA"] = data
    STATE.update(RESULTS={}, PLAY=None)
    return json.dumps({"numpy": np.__version__, "ms": round((time.perf_counter() - t0) * 1000),
                       "shapes": {k: list(v[0].shape) for k, v in data.items()}})


def _namespace():
    d = STATE["DATA"]
    ns = {"np": np, "io": io, "re": re, "json": json}
    for name, (path, colvar) in DATASETS.items():
        ns[name] = d[name][0].copy()
        ns[colvar] = list(d[name][1])
    ns["img"] = d["img"][0].copy()
    try:
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.figsize": (7.4, 3.8), "axes.spines.top": False, "axes.spines.right": False,
                             "axes.grid": True, "grid.alpha": .25, "figure.dpi": 100, "font.size": 9})
        ns["plt"] = plt
    except Exception:  # matplotlib not loaded yet
        pass
    return ns


# ------------------------------------------------------------------------------------------------ formatting
def _cell(x):
    """-> (text, css_class). css_class: '' | 'num' | 'na' | 'mk'."""
    if isinstance(x, (np.bool_, bool)):
        return ("True" if x else "False"), ""
    if isinstance(x, (np.floating, float)):
        if np.isnan(x):
            return "nan", "na"
        if np.isinf(x):
            return ("inf" if x > 0 else "-inf"), "num"
        return f"{float(x):.6g}", "num"
    if isinstance(x, (np.integer, int)):
        return str(int(x)), "num"
    if isinstance(x, (np.complexfloating, complex)):
        return f"{x.real:.4g}{x.imag:+.4g}j", "num"
    if isinstance(x, (np.datetime64, np.timedelta64)):
        return ("NaT", "na") if np.isnat(x) else (str(x), "")
    if isinstance(x, bytes):
        return repr(x), ""
    return str(x), ""


def _td(x, masked=False):
    if masked:
        return '<td class="mk"><span>--</span></td>'
    t, c = _cell(x)
    t = _html.escape(t)
    if c == "na":
        return f'<td class="na"><span>{t}</span></td>'
    return f'<td class="{c}">{t}</td>' if c else f"<td>{t}</td>"


def _table(rows_hdr, col_hdr, cells, corner=""):
    """cells: list of rows; each row a list of already-built <td> strings."""
    head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in col_hdr)
    body = "".join(f"<tr><th>{_html.escape(str(r))}</th>{''.join(cs)}</tr>" for r, cs in zip(rows_hdr, cells))
    return f'<table class="pdt"><thead><tr><th>{_html.escape(corner)}</th>{head}</tr></thead><tbody>{body}</tbody></table>'


def _facts(a):
    return {"shape": list(a.shape), "dtype": str(a.dtype), "ndim": int(a.ndim), "size": int(a.size), "itemsize": int(a.itemsize),
            "nbytes": int(a.nbytes), "c_contig": bool(a.flags.c_contiguous), "owns": a.base is None}


def _array_payload(obj, start=0, n=25, col_names=None):
    """HTML table (+ paging info) for any ndarray. Works for 1-D, 2-D, N-D, structured and masked arrays."""
    mask = None
    if isinstance(obj, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(obj)
        a = np.asarray(obj.data)
    else:
        a = np.asarray(obj)
    start, n = max(0, int(start)), max(1, int(n))
    out = {"start": start, "n": n, "layout": "col", "plane": None}
    if a.dtype.names and a.ndim == 1:  # structured / record array: fields are columns
        names = list(a.dtype.names)
        sub = a[start:start + n]
        cells = [[_td(rec[f]) for f in names[:MAX_COLS]] for rec in sub]
        out.update(html=_table(range(start, start + len(sub)), names[:MAX_COLS], cells), total_rows=int(a.shape[0]), total_cols=len(names),
                   cols_cut=len(names) > MAX_COLS)
        return out
    if a.ndim >= 3:
        idx = (0,) * (a.ndim - 2)
        out["plane"] = "a[" + ", ".join(str(i) for i in idx) + ", :, :]"
        a = a[idx]
        mask = mask[idx] if mask is not None else None
    if a.ndim == 0:
        a = a.reshape(1)
    if a.ndim == 1:
        if a.shape[0] <= 20 and start == 0:
            m = mask if mask is not None else np.zeros(a.shape, bool)
            cells = [[_td(v, bool(k)) for v, k in zip(a.tolist() if a.dtype.kind in "biufcUS" else a, m)]]
            out.update(html=_table(["value"], range(a.shape[0]), cells, "idx"), total_rows=1, total_cols=int(a.shape[0]), cols_cut=False, layout="row")
            return out
        sub = a[start:start + n]
        m = (mask[start:start + n] if mask is not None else np.zeros(sub.shape, bool))
        cells = [[_td(v, bool(k))] for v, k in zip(sub, m)]
        out.update(html=_table(range(start, start + len(sub)), ["value"], cells, "idx"), total_rows=int(a.shape[0]), total_cols=1, cols_cut=False)
        return out
    rows, cols = a.shape
    sub = a[start:start + n, :MAX_COLS]
    msub = mask[start:start + n, :MAX_COLS] if mask is not None else np.zeros(sub.shape, bool)
    cells = [[_td(v, bool(k)) for v, k in zip(r, mr)] for r, mr in zip(sub, msub)]
    chdr = (col_names[:MAX_COLS] if col_names else range(min(cols, MAX_COLS)))
    out.update(html=_table(range(start, start + len(sub)), chdr, cells, ""), total_rows=int(rows), total_cols=int(cols), cols_cut=cols > MAX_COLS)
    return out


def _clip(text):
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + f"\n... [{len(text) - MAX_TEXT:,} more characters hidden]"


def _repr(a):
    with np.printoptions(threshold=400, edgeitems=4, linewidth=100):
        return _clip(repr(a))


def _array_insights(a):
    ins = []
    if isinstance(a, np.ma.MaskedArray):
        ins.append(f"A <b>masked array</b>: {int(np.ma.getmaskarray(a).sum())} of {a.size} values are masked (shown as <code>--</code>) and are ignored by calculations.")
        return ins
    nd = a.ndim
    if a.dtype.names:
        ins.append(f"A <b>structured array</b> with fields <code>{', '.join(a.dtype.names)}</code>: every element is a record.")
        return ins
    if nd == 0:
        ins.append("A 0-dimensional array (one number wrapped in an array).")
    elif nd == 1:
        ins.append(f"A <b>1-D array</b> with {a.shape[0]:,} elements.")
    elif nd == 2:
        ins.append(f"A <b>2-D array</b>: {a.shape[0]:,} rows x {a.shape[1]:,} columns (shape <code>{a.shape}</code>).")
    else:
        ins.append(f"A <b>{nd}-D array</b> with shape <code>{a.shape}</code> - the table shows only the first 2-D plane; the NumPy repr tab shows everything.")
    ins.append(f"Every element is stored as <code>{a.dtype}</code> ({a.itemsize} byte{'s' if a.itemsize != 1 else ''} each, {a.nbytes:,} bytes in total).")
    if a.base is not None:
        ins.append("This array is a <b>view</b> of another array's memory - changing it also changes the original. Use <code>.copy()</code> to detach it.")
    if a.dtype.kind == "f" and a.size:
        k = int(np.isnan(a).sum())
        if k:
            ins.append(f"Contains <b>{k:,} nan</b> value{'s' if k != 1 else ''} (not-a-number = missing). Use the <code>nan*</code> functions or a mask to skip them.")
    return ins


def _describe(obj, rid, page_size, depth=0):
    """-> (result payload, insights)"""
    cname = type(obj).__name__
    if obj is None:
        return {"kind": "none"}, []
    if isinstance(obj, np.ndarray) and obj.ndim == 0 and not isinstance(obj, np.ma.MaskedArray):
        return {"kind": "scalar", "text": repr(obj.item()) if obj.dtype.kind in "US" else str(obj.item()), "type": f"0-d ndarray, dtype {obj.dtype}"}, \
            ["A <b>0-dimensional array</b>: one value inside an ndarray wrapper. <code>.item()</code> gives the plain Python number."]
    if isinstance(obj, np.ndarray):
        STATE["RESULTS"][rid] = obj
        payload = _array_payload(obj, 0, page_size)
        payload.update(kind="array", facts=_facts(obj), repr=_repr(obj), label=("masked array" if isinstance(obj, np.ma.MaskedArray) else "ndarray"))
        ins = _array_insights(obj)
        if obj.shape and obj.ndim <= 2 and obj.shape[0] > page_size:
            ins.append(f"Showing {page_size} rows at a time - use the pager under the table.")
        return payload, ins
    if isinstance(obj, np.generic) or isinstance(obj, (bool, int, float, complex, str, bytes)):
        t = type(obj)
        tname = f"{t.__module__}.{t.__name__}" if isinstance(obj, np.generic) else t.__name__
        txt = repr(obj) if isinstance(obj, (str, bytes)) else str(obj)
        ins = [f"A single value of type <code>{tname}</code>."]
        if isinstance(obj, np.generic):
            ins.append("NumPy scalars like this behave like 0-d arrays: they have <code>.dtype</code>, <code>.shape</code> and all array methods.")
        return {"kind": "scalar", "text": txt, "type": tname}, ins
    if depth == 0 and isinstance(obj, (tuple, list)) and 1 <= len(obj) <= 16 and any(isinstance(x, np.ndarray) for x in obj):
        fields = getattr(obj, "_fields", None)
        items = []
        for i, x in enumerate(obj):
            res, ins = _describe(x, f"{rid}#{i}", page_size, depth + 1)
            items.append({"label": fields[i] if fields else f"[{i}]", "result": res, "insights": ins})
        what = f"named tuple <code>{cname}</code>" if fields else f"{cname} of {len(obj)} items"
        return {"kind": "multi", "items": items}, [f"A {what}; each item is shown separately below."]
    if isinstance(obj, (dict, list, tuple, set, frozenset)):
        import pprint
        txt = pprint.pformat(obj, width=100, compact=True, sort_dicts=False)
        return {"kind": "text", "text": _clip(txt)}, [f"A Python <code>{cname}</code> with {len(obj):,} items."]
    if cname in ("poly1d", "Polynomial", "Chebyshev", "Legendre", "Hermite", "Laguerre", "HermiteE"):
        return {"kind": "text", "text": _clip(str(obj))}, [f"A polynomial object (<code>{cname}</code>) - call it like a function: <code>p(3)</code>."]
    return {"kind": "text", "text": _clip(repr(obj))}, [f"Returned an object of type <code>{cname}</code>."]


def _is_axes(obj):
    n = type(obj).__name__
    return n.startswith("Axes") or (isinstance(obj, np.ndarray) and obj.size and type(obj.flat[0]).__name__.startswith("Axes")) \
        or (isinstance(obj, (list, tuple)) and obj and type(obj[0]).__name__.startswith(("Axes", "Line2D", "Bar", "Patch", "Text", "Polygon", "PathCollection", "QuadMesh")))


def _grab_figures():
    figs = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return figs
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
        figs.append(base64.b64encode(buf.getvalue()).decode())
        plt.close(fig)
    return figs


HINTS = [
    (r"could not be broadcast together|cannot be broadcast|shape mismatch: .* cannot be broadcast",
     "The two shapes are not compatible. Broadcasting compares shapes from the <b>right</b>: each pair of sizes must be equal or one of them must be 1. Print <code>a.shape</code> and <code>b.shape</code>; fix with <code>reshape</code>, <code>[:, None]</code> or <code>np.newaxis</code>."),
    (r"cannot reshape array of size", "The new shape must hold exactly the same number of elements. Use <code>-1</code> for one dimension to let NumPy work it out, e.g. <code>a.reshape(3, -1)</code>."),
    (r"truth value of an array .* is ambiguous", "An array cannot be used as a single True/False. Use <code>a.any()</code> / <code>a.all()</code>, and combine conditions with <code>&amp;</code>, <code>|</code>, <code>~</code> (with parentheses) instead of <code>and</code>/<code>or</code>/<code>not</code>."),
    (r"index \d+ is out of bounds|out of bounds for axis", "The index is bigger than the axis length. Remember indices start at 0, so the last valid index is <code>len - 1</code> (or use <code>-1</code>)."),
    (r"axis \d+ is out of bounds for array of dimension", "The array has fewer dimensions than that axis number. Check <code>a.ndim</code> - axes are numbered 0, 1, 2 ..."),
    (r"too many indices for array", "You gave more indices than the array has dimensions. Check <code>a.ndim</code> / <code>a.shape</code>."),
    (r"setting an array element with a sequence|inhomogeneous shape", "The nested lists have different lengths, so NumPy cannot build a regular array. Make all rows the same length, or use <code>dtype=object</code> on purpose."),
    (r"could not convert string to float|invalid literal for int", "A text value cannot be turned into a number. Clean it first, or read it with <code>np.genfromtxt</code> which turns bad cells into <code>nan</code>."),
    (r"Cannot cast ufunc .* output|UFuncTypeError", "In-place operations keep the left array's dtype, so a float result cannot be written into an int array. Use <code>a = a + 0.5</code> (new array) or <code>a.astype(float)</code> first."),
    (r"Singular matrix", "The matrix has no inverse (its determinant is 0 - rows/columns are linearly dependent). Use <code>np.linalg.pinv</code> or <code>np.linalg.lstsq</code> instead."),
    (r"all the input array dimensions .* must match exactly|all the input arrays must have same number of dimensions",
     "<code>concatenate</code> needs equal shapes in every dimension except the one you join along. Check the shapes with <code>.shape</code> or use <code>np.hstack / np.vstack</code> after reshaping."),
    (r"has no attribute", "NumPy arrays and functions have exact names - check spelling, and whether you need <code>np.xxx(a)</code> or <code>a.xxx()</code>."),
    (r"No module named|ModuleNotFound", "This feature needs an extra Python package that could not be loaded in the browser."),
    (r"Integers to negative integer powers", "Integer arrays cannot be raised to negative powers. Convert first: <code>a.astype(float) ** -1</code>."),
    (r"must be (a )?(positive|non-negative)|negative dimensions", "A size/count argument must be zero or positive."),
    (r"Object arrays cannot be loaded|allow_pickle", "Loading pickled objects is disabled by default for safety. Save plain numeric arrays instead."),
]


def _hint(exc):
    msg = f"{type(exc).__name__}: {exc}"
    for pat, hint in HINTS:
        if re.search(pat, msg, re.I | re.S):
            return hint
    return None


# ------------------------------------------------------------------------------------------------ public API
def _execute(code, ns, rid, page_size):
    """Run a snippet in ns; return the JSON-ready dict used by run()."""
    out = {"ok": True}
    t0 = time.perf_counter()
    stdout = io.StringIO()
    value = None
    try:
        tree = ast.parse(code, filename="<snippet>")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", ResourceWarning)
            with np.errstate(all="warn"), np.printoptions():  # printoptions/errstate set inside a card never leak
                exec(compile(tree, "<snippet>", "exec"), ns)
                if last is not None:
                    value = eval(compile(last, "<snippet>", "eval"), ns)
        seen, ws = set(), []
        for w in caught:
            text = f"{w.category.__name__}: {str(w.message).strip()[:300]}"
            if text not in seen:
                seen.add(text)
                ws.append(text)
        out["warnings"] = ws[:4]
    except BaseException as exc:  # noqa - show every failure to the learner
        out["ok"] = False
        line = None
        for fs in traceback.extract_tb(exc.__traceback__):
            if fs.filename == "<snippet>":
                line = fs.lineno
        out["error"] = {"type": type(exc).__name__, "message": str(exc)[:600], "line": line, "hint": _hint(exc)}
        out["warnings"] = []
    out["stdout"] = _clip(stdout.getvalue())
    out["figures"] = _grab_figures() if out["ok"] else []
    if out["ok"]:
        if out["figures"] and (value is None or _is_axes(value)):
            value = None
        try:
            out["result"], out["insights"] = _describe(value, rid, int(page_size))
        except Exception as exc:  # rendering problem, not the learner's fault
            out["result"] = {"kind": "text", "text": _clip(repr(value))}
            out["insights"] = [f"(Could not draw this result as a table: {type(exc).__name__}: {exc})"]
        if value is None and not out["figures"] and not out["stdout"].strip():
            out["insights"] = ["The code ran fine but did not return a value - put an expression on the last line (for example <code>a.shape</code>) or use <code>print(...)</code> to see something."]
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out, value


def run(code, rid="r", page_size=25, persist=False):
    """Execute a snippet and return everything the UI needs to show the result."""
    if persist:
        if STATE["PLAY"] is None or code == "#reset":
            STATE["PLAY"] = _namespace()
            if code == "#reset":
                return json.dumps({"ok": True, "result": {"kind": "none"}, "warnings": [], "stdout": "", "figures": [], "insights": [], "ms": 0, "vars": []})
        ns = STATE["PLAY"]
    else:
        ns = _namespace()
    out, _ = _execute(code, ns, rid, page_size)
    if persist:
        hidden = ("np", "io", "re", "json", "plt")
        out["vars"] = sorted(k for k, v in ns.items() if not k.startswith("_") and k not in hidden and not callable(v) and not isinstance(v, type(np)))[:40]
    return json.dumps(out, default=str)


def _find(rid):
    if "#" in rid:
        base, i = rid.split("#", 1)
        return STATE["RESULTS"].get(f"{base}#{i}")
    return STATE["RESULTS"].get(rid)


def page(rid, start, n):
    a = _find(rid)
    if a is None:
        return json.dumps({"error": "result expired - run the card again"})
    return json.dumps(_array_payload(a, start, n))


def dataset_page(name, start=0, n=25, sort_col=None, ascending=True):
    arr, cols = STATE["DATA"][name]
    a = arr
    if sort_col in (cols or []):
        k = cols.index(sort_col)
        col = a[:, k]
        order = np.argsort(col, kind="stable")
        nn = int(np.isnan(col).sum())
        if not ascending:
            good = order[: len(order) - nn][::-1]
            order = np.concatenate([good, order[len(order) - nn:]])
        a = a[order]
    payload = _array_payload(a, start, n, col_names=cols)
    payload.update(total_all=int(arr.shape[0]), columns=cols)
    return json.dumps(payload)


def dataset_summary():
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("students", "sensor", "houses"):
            arr, cols = STATE["DATA"][name]
            out[name] = {"rows": int(arr.shape[0]), "cols": int(arr.shape[1]), "nan": int(np.isnan(arr).sum()),
                         "columns": [{"name": c, "min": float(np.nanmin(arr[:, i])), "max": float(np.nanmax(arr[:, i])), "mean": float(np.nanmean(arr[:, i])),
                                      "nan": int(np.isnan(arr[:, i]).sum())} for i, c in enumerate(cols)]}
    return json.dumps(out)


# ------------------------------------------------------------------------------------------------ projects: encode / check
def encode(value):
    """Python value -> JSON-safe description used for 'expected output'."""
    a = np.asarray(value)
    if a.dtype.kind in "fc":
        data = np.where(np.isnan(a.real), 0, a).tolist() if a.dtype.kind == "f" else a.tolist()
        nan_mask = np.isnan(a).tolist() if a.dtype.kind == "f" else None
        if a.dtype.kind == "c":
            data = [[x.real, x.imag] for x in a.ravel().tolist()]
        return {"kind": "array", "dtype": str(a.dtype), "shape": list(a.shape), "data": data, "nan": nan_mask}
    return {"kind": "array", "dtype": str(a.dtype), "shape": list(a.shape), "data": a.tolist(), "nan": None}


def decode(exp):
    a = np.array(exp["data"], dtype=exp["dtype"] if exp["dtype"][0] != "c" else float)
    if exp.get("nan") is not None:
        a = np.where(np.array(exp["nan"]), np.nan, a)
    return a.reshape(exp["shape"])


def expected_text(value):
    with np.printoptions(threshold=60, edgeitems=3, linewidth=90, precision=4, suppress=True):
        a = np.asarray(value)
        return np.array2string(a, separator=", ") if a.ndim else str(a.item())


def check(code, expected_json, var="answer", rtol=1e-5, atol=1e-8):
    """Run the learner's code and compare the variable `var` with the expected value."""
    exp = json.loads(expected_json)
    ns = _namespace()
    out, _ = _execute(code, ns, "chk", 25)
    res = {"ok": False, "ran": out["ok"], "stdout": out.get("stdout", ""), "error": out.get("error"), "message": ""}
    if not out["ok"]:
        res["message"] = "Your code raised an error - fix it first (see the message above), then press Check again."
        return json.dumps(res)
    if var not in ns:
        res["message"] = f"Your code must create a variable named `{var}` holding the final result."
        return json.dumps(res)
    got = np.asarray(ns[var])
    want = decode(exp)
    res["got"] = expected_text(got)
    if got.shape != want.shape:
        res["message"] = f"The shape is different: expected {want.shape}, you got {got.shape}."
        return json.dumps(res)
    try:
        if want.dtype.kind in "fc" or got.dtype.kind in "fc":
            good = bool(np.allclose(got.astype(float), want.astype(float), rtol=rtol, atol=atol, equal_nan=True))
        else:
            good = bool(np.array_equal(got, want))
    except Exception as e:
        res["message"] = f"The values could not be compared ({type(e).__name__}: {e}). Is `{var}` a number or array?"
        return json.dumps(res)
    if good:
        res["ok"] = True
        note = ""
        if got.dtype != want.dtype and got.dtype.kind != want.dtype.kind:
            note = f" (Note: the dtype is {got.dtype}; the reference solution gives {want.dtype}.)"
        res["message"] = "Correct - your answer matches the expected output." + note
    else:
        with np.errstate(all="ignore"):
            if want.dtype.kind in "fiub" and got.dtype.kind in "fiub":
                bad = int((~np.isclose(got.astype(float), want.astype(float), rtol=rtol, atol=atol, equal_nan=True)).sum())
                res["message"] = f"The shape is right but {bad} of {want.size} values differ from the expected output. Compare your output with the expected output below."
            else:
                res["message"] = "The values differ from the expected output."
    return json.dumps(res)

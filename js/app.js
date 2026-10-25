/* NumPy Practice Lab - runs real NumPy in the browser with Pyodide. No build step, no server code. */
(function () {
  "use strict";
  const CAT = window.CATALOG, PROJ = window.PROJECTS || [];
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(kid));
    return el;
  }
  const fmt = (n) => Number(n).toLocaleString("en-US");
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (_) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (_) {} },
  };

  /* ------------------------------------------------------------------ Python engine */
  const BASES = ["https://cdn.jsdelivr.net/pyodide/v314.0.7/full/", "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/"];
  const Engine = { py: null, runner: null, info: null, ready: null, chain: Promise.resolve(), loaded: new Set(), failed: null };

  function setEngine(state, text, pct) {
    const box = $("#engine");
    box.className = "engine " + (state || "");
    $("#engineText").textContent = text;
    if (pct != null) $("#engineBar").style.width = pct + "%";
  }
  function loadScript(src) {
    return new Promise((ok, bad) => {
      const s = document.createElement("script");
      s.src = src; s.onload = ok; s.onerror = () => bad(new Error("could not load " + src));
      document.head.append(s);
    });
  }
  async function fetchText(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(url + " -> HTTP " + r.status);
    return r.text();
  }

  async function bootEngine() {
    let lastErr; const errors = [];
    if (location.protocol === "file:") {
      const e = new Error("This page was opened as a file (file://). Browsers block that - serve it with `python -m http.server` or open the GitHub Pages link instead.");
      Engine.failed = e; setEngine("fail", "Cannot run from file:// - use the GitHub Pages link or python -m http.server", 0); throw e;
    }
    for (const base of BASES) {
      try {
        setEngine("", "Downloading Python…", 8);
        if (!window.loadPyodide) await loadScript(base + "pyodide.js");
        const py = await window.loadPyodide({ indexURL: base });
        setEngine("", "Loading NumPy…", 45);
        await py.loadPackage(["numpy"]);
        setEngine("", "Loading the datasets…", 75);
        py.FS.mkdirTree("/home/pyodide/data");
        py.FS.mkdirTree("/home/pyodide/py");
        const files = ["data/students.csv", "data/sensor.csv", "data/houses.csv", "py/runner.py"];
        const texts = await Promise.all(files.map(fetchText));
        files.forEach((f, i) => py.FS.writeFile("/home/pyodide/" + f, texts[i]));
        py.runPython("import os, sys\nos.environ['MPLBACKEND']='agg'\nsys.path.insert(0,'/home/pyodide/py')\nimport runner");
        const runner = py.pyimport("runner");
        const info = JSON.parse(runner.init("/home/pyodide/"));
        Engine.py = py; Engine.runner = runner; Engine.info = info; Engine.base = base;
        setEngine("ready", "NumPy " + info.numpy + " ready", 100);
        document.dispatchEvent(new Event("engine-ready"));
        return info;
      } catch (e) {
        lastErr = e; errors.push(base + " -> " + (e && e.message || e)); console.warn("engine attempt failed on", base, e);
        window.loadPyodide = undefined;
      }
    }
    Engine.failed = lastErr; Engine.errors = errors;
    setEngine("fail", "Engine failed: " + String(lastErr && lastErr.message || lastErr).slice(0, 90), 0);
    $("#engineText").title = errors.join("\n");
    if (!$("#retry")) $("#engine").append(h("button", { id: "retry", class: "btn small", onclick: () => location.reload() }, "Retry"));
    throw lastErr;
  }
  window.NumpyLab = { Engine, execute: (...a) => execute(...a), check: (...a) => checkAnswer(...a) };
  Engine.ready = bootEngine();
  Engine.ready.catch(() => {});

  function enqueue(fn) {
    const p = Engine.chain.then(fn);
    Engine.chain = p.catch(() => {});
    return p;
  }
  const NEED_RULES = [
    [/\bplt\.|matplotlib|\.imshow\(|pyplot/, "matplotlib"],
    [/scipy/, "scipy"],
  ];
  async function ensurePackages(code, needs) {
    const want = new Set(needs || []);
    for (const [re, name] of NEED_RULES) if (re.test(code)) want.add(name);
    for (const n of [...want].filter((n) => !Engine.loaded.has(n))) {
      try { await Engine.py.loadPackage(n); Engine.loaded.add(n); } catch (e) { console.warn("package", n, "failed", e); }
    }
  }
  async function execute(code, rid, pageSize, persist, needs) {
    await Engine.ready;
    return enqueue(async () => {
      await ensurePackages(code, needs);
      await new Promise((r) => setTimeout(r, 15)); // let the spinner paint
      return JSON.parse(Engine.runner.run(code, rid, pageSize, !!persist));
    });
  }
  async function checkAnswer(code, expected, rtol, atol) {
    await Engine.ready;
    return enqueue(async () => {
      await ensurePackages(code, []);
      return JSON.parse(Engine.runner.check(code, JSON.stringify(expected), "answer", rtol, atol));
    });
  }
  const pyCall = (name, ...args) => Engine.ready.then(() => enqueue(async () => JSON.parse(Engine.runner[name](...args))));

  /* ------------------------------------------------------------------ result rendering */
  function makePager(payload, load) {
    const box = h("div", { class: "pager" });
    const total = payload.total_rows, start = payload.start, n = payload.n;
    const end = Math.min(total, start + n);
    const mk = (label, to, dis) => h("button", { class: "btn small", disabled: dis, onclick: () => load(to, n) }, label);
    const sel = h("select", { "aria-label": "Rows per page", onchange: (e) => load(0, +e.target.value) },
      [10, 25, 50, 100, 250].map((v) => h("option", { value: v, selected: v === n }, v + " per page")));
    box.append(mk("« First", 0, start === 0), mk("‹ Prev", Math.max(0, start - n), start === 0),
      h("span", {}, total ? `rows ${fmt(start + 1)}–${fmt(end)} of ${fmt(total)}` : "no rows"),
      mk("Next ›", start + n, end >= total), mk("Last »", Math.max(0, Math.floor((total - 1) / n) * n), end >= total),
      h("span", { class: "sp" }), sel);
    return box;
  }
  function renderTable(host, payload, loader, opts = {}) {
    host.innerHTML = "";
    const wrap = h("div", { class: "tablewrap", tabindex: "0", html: payload.html });
    $$("td", wrap).forEach((td) => { if (td.textContent.length > 40) td.title = td.textContent; });
    if (opts.sortable) $$("thead tr:first-child th", wrap).slice(1).forEach((th, i) => { th.classList.add("sortable"); th.dataset.col = (opts.columns && opts.columns[i]) || th.textContent; th.title = "Click to sort"; });
    host.append(wrap);
    if (payload.plane) host.append(h("div", { class: "hint", style: "margin-top:6px" }, "N-dimensional array: showing the first 2-D plane, " + payload.plane));
    if (payload.total_rows > Math.min(payload.n, 10) || payload.start > 0) {
      host.append(makePager(payload, async (s, n) => {
        const p = await loader(s, n);
        if (p.error) { host.append(h("div", { class: "warn" }, p.error)); return; }
        renderTable(host, p, loader, opts);
        if (opts.onPage) opts.onPage(p);
      }));
    }
    if (payload.cols_cut) host.append(h("div", { class: "warn" }, "This array has " + payload.total_cols + " columns - only the first ones are drawn. Slice with a[:, 40:] to see the others."));
  }

  function factChips(f) {
    const c = (k, v, tip) => h("span", { class: "fact", title: tip }, h("i", {}, k), " " + v);
    return h("div", { class: "facts" },
      c("shape", "(" + f.shape.join(", ") + (f.shape.length === 1 ? "," : "") + ")", "a.shape - length along each axis"),
      c("dtype", f.dtype, "a.dtype - the data type shared by all elements"),
      c("ndim", f.ndim, "a.ndim - number of axes"),
      c("size", fmt(f.size), "a.size - total element count"),
      c("nbytes", fmt(f.nbytes), "a.nbytes - bytes used by the elements"),
      c("layout", f.c_contig ? "C-contiguous" : "not C-contiguous", "a.flags.c_contiguous"),
      c("memory", f.owns ? "owns its data" : "view of another array", "a.base is None -> owns its data"));
  }
  function renderArrayResult(body, res, rid) {
    const wrapper = h("div");
    const tabs = h("div", { class: "tabs mini", role: "tablist" });
    const tbl = h("div"), rep = h("pre", { class: "text", hidden: true }, res.repr || "");
    const t1 = h("button", { class: "tab", role: "tab", "aria-selected": "true", onclick: () => sel(true) }, "Table view");
    const t2 = h("button", { class: "tab", role: "tab", "aria-selected": "false", onclick: () => sel(false) }, "print(a)  /  repr");
    function sel(tv) { t1.setAttribute("aria-selected", tv); t2.setAttribute("aria-selected", !tv); tbl.hidden = !tv; rep.hidden = tv; }
    tabs.append(t1, t2);
    wrapper.append(h("div", { class: "meta" }, h("b", {}, res.label || "ndarray")), res.facts ? factChips(res.facts) : null, tabs, tbl, rep);
    renderTable(tbl, res, async (s, n) => pyCall("page", rid, s, n));
    body.append(wrapper);
  }
  function renderResult(body, res, rid) {
    if (res.kind === "array") renderArrayResult(body, res, rid);
    else if (res.kind === "multi") {
      res.items.forEach((it, i) => {
        const box = h("div", { class: "multi" }, h("div", { class: "multilabel" }, it.label));
        body.append(box);
        renderResult(box, it.result, rid + "#" + i);
        if (it.insights && it.insights.length) box.append(h("ul", { class: "mini-ins" }, it.insights.map((t) => h("li", { html: t }))));
      });
    } else if (res.kind === "scalar") body.append(h("div", { class: "scalar" }, res.text, h("small", {}, res.type)));
    else if (res.kind === "text") body.append(h("pre", { class: "text" }, res.text));
  }

  function renderOutput(box, r, ctx) {
    box.innerHTML = "";
    const body = h("div", { class: "outbody" });
    box.append(h("div", { class: "cell" }, h("div", { class: "gut" }, r.ok ? "Out" : "Err"), body));
    if (!r.ok) {
      const e = r.error;
      let line = e.line;
      if (line && ctx.lineShift) line = line > ctx.lineShift ? line - ctx.lineShift : null;
      body.append(h("div", { class: "err", role: "alert" },
        h("h4", {}, e.type + (line ? " on line " + line : "")),
        h("pre", {}, e.message || "(no message)"),
        e.hint ? h("div", { class: "fix", html: "<b>Try this:</b> " + e.hint }) : null));
      if (r.stdout && r.stdout.trim()) body.append(h("pre", { class: "text stdout" }, r.stdout));
      return;
    }
    const res = r.result || { kind: "none" };
    body.append(h("div", { class: "meta" }, h("span", {}, r.ms + " ms")));
    (r.warnings || []).forEach((w) => body.append(h("div", { class: "warn" }, "⚠ " + w)));
    if (r.stdout && r.stdout.trim()) body.append(h("pre", { class: "text stdout" }, r.stdout));
    renderResult(body, res, ctx.rid);
    if (r.figures && r.figures.length) {
      const f = h("div", { class: "figs" });
      r.figures.forEach((b) => f.append(h("img", { alt: "Chart produced by the code", src: "data:image/png;base64," + b })));
      body.append(f);
    }
    if (ctx.noExplain) return;
    const ex = h("div", { class: "explain" }, h("h4", {}, "What you're looking at"));
    let any = false;
    if (ctx.entry && ctx.entry.reads) { ex.append(h("p", {}, ctx.entry.reads)); any = true; }
    if (r.insights && r.insights.length) { ex.append(h("ul", {}, r.insights.map((t) => h("li", { html: t })))); any = true; }
    if (ctx.entry && ctx.entry.params && ctx.entry.params.length) {
      ex.append(h("p", {}, h("b", {}, "Useful options")), h("ul", {}, ctx.entry.params.map((t) => {
        const i = t.indexOf(" - ");
        return h("li", {}, i > 0 ? [h("code", {}, t.slice(0, i)), " – " + t.slice(i + 3)] : t);
      }))); any = true;
    }
    if (ctx.entry && ctx.entry.tip) { ex.append(h("p", { class: "tip" }, "Tip: " + ctx.entry.tip)); any = true; }
    if (r.figures && r.figures.length && !(r.insights && r.insights.length)) { ex.append(h("p", {}, "A chart drawn with matplotlib from your NumPy arrays. Change the data or options and run again.")); any = true; }
    if (!any && ctx.entry) { ex.append(h("p", {}, "In short: " + ctx.entry.desc)); any = true; }
    if (any) body.append(ex);
  }

  function wireEditor(ta, go) {
    ta.addEventListener("keydown", (ev) => {
      if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") { ev.preventDefault(); go(); }
      else if (ev.key === "Tab" && !ev.shiftKey) { ev.preventDefault(); ta.setRangeText("    ", ta.selectionStart, ta.selectionEnd, "end"); }
    });
  }
  const engineDown = (err) => h("div", { class: "err" }, h("h4", {}, "The Python engine is not available"), h("pre", {}, String(err && err.message || err)), h("div", { class: "fix" }, "Check your internet connection (the engine loads from a CDN) and reload the page."));

  /* ------------------------------------------------------------------ function cards */
  let uid = 0;
  const STATUS = { removed: "removed in NumPy 2", deprecated: "deprecated", renamed: "renamed" };
  function buildCard(e) {
    const rid = "c" + ++uid;
    const lines = e.code.split("\n").length;
    const ta = h("textarea", { spellcheck: "false", rows: Math.min(Math.max(lines, 2), 22), "aria-label": "Python code for " + e.name, autocapitalize: "off", autocomplete: "off" });
    ta.value = e.code;
    const out = h("div", { class: "out", hidden: true, "aria-live": "polite" });
    const run = h("button", { class: "run", onclick: go }, "Run");
    const reset = h("button", { class: "tbtn", hidden: true, onclick: () => { ta.value = e.code; fit(); reset.hidden = true; } }, "Reset code");
    const copy = h("button", { class: "tbtn", onclick: async () => { try { await navigator.clipboard.writeText(ta.value); copy.textContent = "Copied"; setTimeout(() => (copy.textContent = "Copy"), 1200); } catch (_) { ta.select(); } } }, "Copy");
    const fit = () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 560) + "px"; };
    ta.addEventListener("input", () => { reset.hidden = ta.value === e.code; fit(); });
    wireEditor(ta, go);
    async function go() {
      if (run.disabled) return;
      run.disabled = true; run.classList.add("busy"); run.lastChild.textContent = Engine.py ? "Running" : "Loading Python…";
      out.hidden = false;
      out.innerHTML = '<div class="loading">' + (Engine.py ? "Running…" : "Python is still starting – your code will run as soon as it is ready.") + "</div>";
      try { renderOutput(out, await execute(ta.value, rid, 25, false, e.needs), { rid, entry: e }); }
      catch (err) { out.innerHTML = ""; out.append(engineDown(err)); }
      finally { run.disabled = false; run.classList.remove("busy"); run.lastChild.textContent = "Run"; }
    }
    const used = ["students", "sensor", "houses", "img"].filter((v) => new RegExp("\\b" + v + "\\b").test(e.code));
    const id = "fn-" + e.id.replace(/[^\w.-]+/g, "_");
    const card = h("article", { class: "fn", id, "data-cat": e.cat },
      h("div", { class: "fn-head" },
        h("h3", { class: "fn-name" }, h("a", { href: "#" + id }, e.name)),
        e.status ? h("span", { class: "chip " + e.status }, STATUS[e.status] || e.status) : null,
        e.local ? h("span", { class: "chip local" }, "runs locally only") : null,
        used.map((v) => h("span", { class: "chip data", title: "uses the built-in " + v + " data" }, v))),
      h("p", { class: "fn-desc" }, e.desc),
      e.alt ? h("p", { class: "alt" }, "Use instead: " + e.alt) : null);
    card.append(h("div", { class: "cell" }, h("div", { class: "gut" }, "In"), h("div", { class: "codewrap" }, ta)));
    if (e.local) card.append(h("div", { class: "localnote" }, e.local), h("div", { class: "codebar" }, copy));
    else card.append(h("div", { class: "codebar" }, run, copy, reset, h("span", { class: "hint" }, "edit the code, then Ctrl+Enter")), out);
    requestAnimationFrame(fit);
    card._text = (e.name + " " + e.desc + " " + e.cat).toLowerCase();
    card._fit = fit;
    return card;
  }

  /* ------------------------------------------------------------------ navigation */
  const START = [["overview", "🏠", "Overview"], ["dataset", "🗂️", "Datasets"], ["playground", "🧑‍💻", "Playground"]];
  function progress() { return store.get("np-done", {}); }
  function doneCount(p) { const d = progress(); return p.steps.filter((s) => d[s.id]).length; }
  function buildNav() {
    const nav = $("#navscroll");
    nav.append(h("div", { class: "navgroup" }, "Start here"));
    START.forEach(([id, ic, label]) => nav.append(h("a", { class: "navlink", href: "#" + id, "data-target": id }, h("span", { class: "ic" }, ic), label)));
    nav.append(h("div", { class: "navgroup" }, "Practice projects"));
    nav.append(h("a", { class: "navlink", href: "#projects", "data-target": "projects" }, h("span", { class: "ic" }, "🎯"), "All projects"));
    PROJ.forEach((p) => nav.append(h("a", { class: "navlink sub", href: "#proj-" + p.id, "data-target": "proj-" + p.id }, h("span", { class: "ic" }, p.id === "students" ? "🎓" : p.id === "sensor" ? "📡" : "🏠"), p.title, h("span", { class: "n", id: "pn-" + p.id }, doneCount(p) + "/" + p.steps.length))));
    nav.append(h("div", { class: "navgroup" }, "Functions by category"));
    CAT.categories.forEach((c) => nav.append(h("a", { class: "navlink", href: "#cat-" + c.id, "data-target": "cat-" + c.id }, h("span", { class: "ic" }, c.icon), c.title.replace(/ \(.*\)$/, ""), h("span", { class: "n", id: "n-" + c.id }, c.count))));
    nav.addEventListener("click", (ev) => { if (ev.target.closest("a")) document.body.classList.remove("navopen"); });
  }

  /* ------------------------------------------------------------------ Overview */
  async function buildOverview(sec) {
    const total = CAT.entries.length, steps = PROJ.reduce((n, p) => n + p.steps.length, 0);
    sec.append(h("div", { class: "hero" },
      h("h1", {}, "Think in arrays. Run every NumPy function."),
      h("p", { class: "lead", html: `<b>${total} runnable examples</b> – every function, attribute and idiom of NumPy, from <code>a.shape</code> to FFTs – plus <b>${PROJ.length} guided projects</b> (${steps} steps) whose answers you check against the expected output. Real NumPy runs right here in your browser.` }),
      h("div", { class: "rawstrip", id: "rawstrip" }, h("div", { class: "cap" }, h("b", {}, "students.csv"), "loading…")),
      h("div", { class: "cta-row" },
        h("a", { class: "btn primary", href: "#cat-basics" }, "Start with the basics"),
        h("a", { class: "btn", href: "#projects" }, "Try a project"),
        h("a", { class: "btn", href: "#dataset" }, "Browse the data"))));
    sec.append(h("div", { class: "grid2" },
      h("div", { class: "panel" }, h("h2", {}, "A suggested path"),
        h("ol", { class: "steps" },
          h("li", {}, h("a", { href: "#cat-basics" }, "ndarray basics"), " and ", h("a", { href: "#cat-dtypes" }, "data types"), " – what an array is."),
          h("li", {}, h("a", { href: "#cat-creation" }, "Creating"), ", ", h("a", { href: "#cat-indexing" }, "indexing"), " and ", h("a", { href: "#cat-shape" }, "reshaping"), "."),
          h("li", {}, h("a", { href: "#cat-math" }, "Maths"), ", ", h("a", { href: "#cat-stats" }, "statistics"), " and ", h("a", { href: "#cat-broadcast" }, "broadcasting"), " – the heart of NumPy."),
          h("li", {}, "Try ", h("a", { href: "#proj-students" }, "Project 1"), " to apply it."),
          h("li", {}, h("a", { href: "#cat-linalg" }, "Linear algebra"), ", ", h("a", { href: "#cat-random" }, "random"), ", ", h("a", { href: "#cat-fft" }, "FFT"), " and the rest."),
          h("li", {}, "Finish with Projects 2 and 3."))),
      h("div", { class: "panel" }, h("h2", {}, "How to use it"),
        h("ol", { class: "steps" },
          h("li", {}, "Pick a category on the left – or press ", h("code", {}, "/"), " to search."),
          h("li", {}, h("b", {}, "Run"), " a card. Arrays appear as a table, with shape and dtype chips."),
          h("li", {}, "Read the yellow box – it explains the result."),
          h("li", {}, "Edit the code and run again. Nothing can break."),
          h("li", {}, "Ready-made data: ", h("code", {}, "students"), ", ", h("code", {}, "sensor"), ", ", h("code", {}, "houses"), ", ", h("code", {}, "img"), ".")),
        h("div", { class: "stats" },
          h("div", { class: "stat" }, h("b", {}, CAT.categories.length), h("span", {}, "categories")),
          h("div", { class: "stat" }, h("b", {}, total), h("span", {}, "functions")),
          h("div", { class: "stat" }, h("b", {}, PROJ.length), h("span", {}, "projects"))))));
    try {
      const rows = (await (await fetch("data/students.csv")).text()).split("\n").slice(0, 400).map((l) => l.split(","));
      const head = rows[0], pick = [rows[1], rows[2]].concat(rows.slice(3).filter((r) => r.some((c) => c === "")).slice(0, 5));
      const t = h("table", {}, h("thead", {}, h("tr", {}, head.map((w) => h("th", {}, w)))),
        h("tbody", {}, pick.map((r) => h("tr", {}, r.map((c) => h("td", { html: c.trim() === "" ? '<span class="badv">nan</span>' : esc(c) }))))));
      const box = $("#rawstrip"); box.innerHTML = "";
      box.append(h("div", { class: "cap" }, h("b", {}, "students.csv"), "· real rows; blank cells load as nan in NumPy"),
        h("div", { class: "scroll" }, t),
        h("div", { class: "legend", html: '<span><span class="badv">nan</span> missing value (not-a-number)</span><span>np.genfromtxt turns the file into a 600 × 11 array</span>' }));
    } catch (_) { $("#rawstrip").innerHTML = ""; }
  }

  /* ------------------------------------------------------------------ Datasets */
  async function buildDataset(sec) {
    let meta = {};
    try { meta = await (await fetch("data/datasets.json")).json(); } catch (_) {}
    const state = { name: "students", start: 0, n: 25, sort: null, asc: true };
    const host = h("div"), note = h("div", { class: "dsnote" }), colsBox = h("div", { class: "dscols" });
    const tabs = h("div", { class: "tabs", role: "tablist" });
    const names = [["students", "students"], ["sensor", "sensor"], ["houses", "houses"]];
    const info = h("span", { class: "hint" });
    names.forEach(([k, label]) => tabs.append(h("button", { class: "tab", role: "tab", "aria-selected": k === state.name ? "true" : "false", "data-k": k, onclick: () => { state.name = k; state.start = 0; state.sort = null; $$(".tab", tabs).forEach((b) => b.setAttribute("aria-selected", b.dataset.k === k ? "true" : "false")); load(); } }, label)));
    async function load() {
      const m = meta[state.name];
      note.innerHTML = m ? `<p><code>${esc(state.name)}</code> – ${esc(m.desc)}</p>` : "";
      colsBox.innerHTML = "";
      if (m) Object.entries(m.columns).forEach(([c, d], i) => colsBox.append(h("span", { class: "pill", title: d }, h("b", {}, i + ": " + c), d ? " – " + d : "")));
      host.innerHTML = '<div class="loading">Loading…</div>';
      const loader = (s, n) => { state.start = s; state.n = n; return pyCall("dataset_page", state.name, s, n, state.sort, state.asc); };
      const p = await loader(state.start, state.n);
      renderTable(host, p, loader, { sortable: true, columns: p.columns });
      info.textContent = `${fmt(p.total_all)} rows × ${p.columns.length} columns · ${state.name}[row, col]`;
      $$("th.sortable", host).forEach((th) => th.addEventListener("click", () => { state.asc = state.sort === th.dataset.col ? !state.asc : true; state.sort = th.dataset.col; state.start = 0; load(); }));
    }
    sec.append(h("header", { class: "cathead" }, h("span", { class: "big" }, "🗂️"), h("div", {}, h("h2", {}, "Datasets"), h("p", {}, "Three ready-made arrays (plus a 64 × 64 image, img). In your code they are plain NumPy arrays named students, sensor and houses; blanks are nan. Click a header to sort."))));
    sec.append(h("div", { class: "panel" }, tabs, note, colsBox, h("div", { class: "toolbar" }, info), host));
    document.addEventListener("engine-ready", load, { once: true });
    host.innerHTML = '<div class="loading">Waiting for the Python engine…</div>';
  }

  /* ------------------------------------------------------------------ Playground */
  function buildPlayground(sec) {
    sec.append(h("header", { class: "cathead" }, h("span", { class: "big" }, "🧑‍💻"), h("div", {}, h("h2", {}, "Playground"), h("p", {}, "Free practice. Variables stay alive between runs. Ready-made: np, students, sensor, houses, img, student_cols, sensor_cols, house_cols. The last line is displayed."))));
    const ta = h("textarea", { class: "playta", spellcheck: "false", "aria-label": "Playground code" });
    ta.value = `# columns of students: ${"{"}student_cols${"}"}\nmarks = students[:, 5:9]\nmean_by_subject = np.nanmean(marks, axis=0)\nmean_by_subject`;
    ta.value = "# students columns: id, age, gender, height, weight, math, physics, chemistry, english, attendance, study_hours\nmarks = students[:, 5:9]\nnp.nanmean(marks, axis=0)";
    const out = h("div", { class: "out", hidden: true, style: "margin:12px 0 0" });
    const vars = h("div", { class: "vars" });
    const run = h("button", { class: "run", onclick: go }, "Run");
    const reset = h("button", { class: "btn small", onclick: async () => { await execute("#reset", "pg", 25, true); vars.textContent = "Variables reset."; } }, "Reset variables");
    wireEditor(ta, go);
    async function go() {
      run.disabled = true; run.classList.add("busy"); out.hidden = false; out.innerHTML = '<div class="loading">Running…</div>';
      try {
        const r = await execute(ta.value, "pg" + ++uid, 25, true, []);
        renderOutput(out, r, { rid: "pg" + uid, entry: null });
        if (r.vars) vars.textContent = "Variables: " + r.vars.join(", ");
      } catch (e) { out.innerHTML = ""; out.append(engineDown(e)); }
      run.disabled = false; run.classList.remove("busy");
    }
    sec.append(h("div", { class: "panel" }, ta, h("div", { class: "toolbar", style: "margin-top:10px" }, run, reset, h("span", { class: "hint" }, "Ctrl+Enter to run")), vars, out));
  }

  /* ------------------------------------------------------------------ Projects */
  function updateProgress(p) {
    const n = doneCount(p), el = $("#pn-" + p.id);
    if (el) el.textContent = n + "/" + p.steps.length;
    const b = $("#pbar-" + p.id); if (b) { b.style.width = (100 * n / p.steps.length) + "%"; $("#ptxt-" + p.id).textContent = n + " of " + p.steps.length + " steps solved"; }
    const c = $("#pc-" + p.id); if (c) c.textContent = n + "/" + p.steps.length + " solved";
  }
  function buildStep(p, s, preludeLines) {
    const ta = h("textarea", { spellcheck: "false", "aria-label": "Your code for step " + s.n, autocapitalize: "off", autocomplete: "off" });
    ta.value = s.starter;
    const fit = () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 520) + "px"; };
    ta.addEventListener("input", fit);
    const full = () => p.prelude + "\n" + ta.value;
    const out = h("div", { class: "out", hidden: true }), res = h("div", { class: "checkres", hidden: true, role: "status" });
    const hintBox = h("div", { class: "explain", hidden: true }, h("h4", {}, "Hint"), h("p", {}, s.hint));
    const solBox = h("div", { class: "solbox", hidden: true }, h("h4", {}, "Reference solution"), h("pre", {}, s.solution), h("button", { class: "btn small", onclick: () => { ta.value = s.solution; fit(); } }, "Copy into editor"));
    const badge = h("span", { class: "stepdone", hidden: !progress()[s.id] }, "✓ solved");
    const run = h("button", { class: "run", onclick: runIt }, "Run");
    const chk = h("button", { class: "btn primary", onclick: checkIt }, "Check answer");
    async function runIt() {
      if (run.disabled) return;
      run.disabled = true; run.classList.add("busy"); out.hidden = false; out.innerHTML = '<div class="loading">Running…</div>';
      try { renderOutput(out, await execute(full() + "\nglobals().get('answer')", "p" + ++uid, 25, false, []), { rid: "p" + uid, noExplain: true, lineShift: preludeLines }); }
      catch (e) { out.innerHTML = ""; out.append(engineDown(e)); }
      run.disabled = false; run.classList.remove("busy");
    }
    async function checkIt() {
      chk.disabled = true; res.hidden = false; res.className = "checkres"; res.textContent = "Checking…";
      try {
        const r = await checkAnswer(full(), s.expected, s.rtol, s.atol);
        res.className = "checkres " + (r.ok ? "ok" : "bad");
        res.innerHTML = "";
        res.append(h("b", {}, r.ok ? "✓ " : "✗ "), r.message);
        if (!r.ok && r.error) res.append(h("pre", {}, r.error.type + ": " + r.error.message));
        if (!r.ok && r.got) res.append(h("div", { class: "got" }, h("b", {}, "Your answer:"), h("pre", {}, r.got)));
        if (r.ok) { const d = progress(); d[s.id] = 1; store.set("np-done", d); badge.hidden = false; updateProgress(p); }
      } catch (e) { res.className = "checkres bad"; res.textContent = String(e && e.message || e); }
      chk.disabled = false;
    }
    wireEditor(ta, runIt);
    requestAnimationFrame(fit);
    return h("article", { class: "pstep", id: "step-" + s.id },
      h("div", { class: "pstep-head" }, h("span", { class: "pnum" }, s.n), h("h3", {}, s.title), badge),
      h("p", { class: "ptask" }, s.task),
      h("div", { class: "cell" }, h("div", { class: "gut" }, "In"), h("div", { class: "codewrap" }, ta)),
      h("div", { class: "codebar" }, run, chk,
        h("button", { class: "tbtn", onclick: () => { hintBox.hidden = !hintBox.hidden; } }, "Hint"),
        h("button", { class: "tbtn", onclick: () => { solBox.hidden = !solBox.hidden; } }, "Solution"),
        h("button", { class: "tbtn", onclick: () => { ta.value = s.starter; fit(); } }, "Reset"),
        h("span", { class: "hint" }, "Ctrl+Enter runs · store the result in answer")),
      h("div", { class: "expected" }, h("div", { class: "explabel" }, "Expected output ", h("span", { class: "chip data" }, "answer: " + (s.shape.length ? "array " + "(" + s.shape.join(", ") + (s.shape.length === 1 ? "," : "") + ") " : "number ") + s.dtype)), h("pre", {}, s.expected_text)),
      res, out, hintBox, solBox);
  }
  function buildProjectIndex(sec) {
    sec.append(h("header", { class: "cathead" }, h("span", { class: "big" }, "🎯"), h("div", {}, h("h2", {}, "Practice projects"), h("p", {}, "Real tasks on real data. Every step shows the expected output. Write code that stores the result in a variable called answer, press Check, and the page compares it with the reference – floats within a small tolerance. Your progress is saved in this browser."))));
    const grid = h("div", { class: "pgrid" });
    PROJ.forEach((p) => grid.append(h("a", { class: "pcard", href: "#proj-" + p.id },
      h("div", { class: "plevel " + p.level.toLowerCase() }, p.level, " · ~" + p.minutes + " min"),
      h("h3", {}, p.title), h("p", {}, p.story),
      h("div", {}, p.learn.map((l) => h("span", { class: "pill" }, l))),
      h("div", { class: "pfoot" }, p.steps.length + " steps · dataset: ", h("code", {}, p.dataset), h("span", { class: "sp", id: "pc-" + p.id }, doneCount(p) + "/" + p.steps.length + " solved")))));
    sec.append(grid);
  }
  function buildProject(sec, p) {
    const preludeLines = p.prelude.split("\n").length;
    sec.append(h("header", { class: "cathead" }, h("span", { class: "big" }, p.id === "students" ? "🎓" : p.id === "sensor" ? "📡" : "🏠"),
      h("div", {}, h("h2", {}, p.title), h("p", {}, p.story)), h("span", { class: "cnt" }, p.level + " · " + p.steps.length + " steps")));
    sec.append(h("div", { class: "pbarwrap" }, h("div", { class: "pbar" }, h("i", { id: "pbar-" + p.id })), h("span", { id: "ptxt-" + p.id }, "")));
    sec.append(h("details", { class: "panel prelude", open: true }, h("summary", {}, "Data provided for this project (already available in every step)"), h("pre", {}, p.prelude)));
    p.steps.forEach((s) => sec.append(buildStep(p, s, preludeLines)));
    updateProgress(p);
  }

  /* ------------------------------------------------------------------ Search */
  const qbox = $("#q"), qclear = $("#qclear");
  function applySearch() {
    const q = qbox.value.trim().toLowerCase();
    qclear.hidden = !q;
    const terms = q.split(/\s+/).filter(Boolean);
    let shown = 0;
    CAT.categories.forEach((c) => {
      let n = 0;
      $$(".fn", $("#cat-" + c.id)).forEach((card) => { const ok = terms.every((t) => card._text.includes(t)); card.hidden = !ok; if (ok) n++; });
      $("#n-" + c.id).textContent = q ? n : c.count;
      $('[data-target="cat-' + c.id + '"]').classList.toggle("dim", !!q && n === 0);
      $("#cat-" + c.id + " .cnt").textContent = (q ? n + " of " : "") + c.count + " functions";
      shown += n;
    });
    let empty = $("#noresults");
    if (!empty) { empty = h("div", { class: "empty", id: "noresults" }, "No function matches. Try a shorter word, e.g. “sort”, “linalg”, “cumsum”."); $("#cats").prepend(empty); }
    empty.hidden = !(q && shown === 0);
  }
  qbox.addEventListener("input", applySearch);
  qclear.addEventListener("click", () => { qbox.value = ""; applySearch(); qbox.focus(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) { e.preventDefault(); qbox.focus(); qbox.select(); }
    if (e.key === "Escape" && document.activeElement === qbox) { qbox.value = ""; applySearch(); qbox.blur(); }
  });

  /* ------------------------------------------------------------------ boot UI */
  function init() {
    buildNav();
    const main = $("#main");
    const sec = (id) => h("section", { class: "sec", id });
    const sOverview = sec("overview"), sDataset = sec("dataset"), sPlay = sec("playground"), sProj = sec("projects");
    main.append(sOverview, sDataset, sPlay, sProj);
    buildOverview(sOverview); buildDataset(sDataset); buildPlayground(sPlay); buildProjectIndex(sProj);
    PROJ.forEach((p) => { const s = sec("proj-" + p.id); main.append(s); buildProject(s, p); });
    main.append(h("div", { id: "cats" }));
    const cats = $("#cats"), byCat = {};
    CAT.entries.forEach((e) => (byCat[e.cat] = byCat[e.cat] || []).push(e));
    CAT.categories.forEach((c) => {
      const s = h("section", { class: "sec", id: "cat-" + c.id },
        h("header", { class: "cathead" }, h("span", { class: "big" }, c.icon), h("div", {}, h("h2", {}, c.title), h("p", {}, c.intro || c.blurb)), h("span", { class: "cnt" }, c.count + " functions")));
      (byCat[c.id] || []).forEach((e) => s.append(buildCard(e)));
      cats.append(s);
    });
    const links = new Map($$(".navlink").map((a) => [a.dataset.target, a]));
    const io = new IntersectionObserver((ents) => {
      ents.forEach((en) => { if (en.isIntersecting) { links.forEach((a) => a.classList.remove("on")); const a = links.get(en.target.id); if (a) { a.classList.add("on"); a.scrollIntoView({ block: "nearest" }); } } });
    }, { rootMargin: "-70px 0px -70% 0px" });
    $$("main section.sec").forEach((s) => io.observe(s));
    if (location.hash.startsWith("#fn-")) {
      const el = document.getElementById(location.hash.slice(1));
      if (el) setTimeout(() => { el.scrollIntoView({ block: "center" }); el.classList.add("flash"); }, 100);
    }
    $$("main .fn").forEach((c) => c._fit && c._fit());
  }
  $("#burger").addEventListener("click", () => { const o = document.body.classList.toggle("navopen"); $("#burger").setAttribute("aria-expanded", o); });
  $("#scrim").addEventListener("click", () => document.body.classList.remove("navopen"));
  init();
})();

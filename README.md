# NumPy Practice Lab

A practice portal that runs **real NumPy in your browser** (Pyodide). 944 runnable examples in 27 categories - every function, attribute and idiom -
plus **3 guided projects** (31 steps) where each step shows the expected output and a **Check answer** button.

## Put it on GitHub Pages
```bash
cd numpy-practice-lab
git init
git add .
git commit -m "NumPy Practice Lab"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```
Then on GitHub: **Settings -> Pages -> Source: Deploy from a branch -> main / (root) -> Save**.
After a minute the site is at `https://<your-user>.github.io/<your-repo>/`.

## Run it locally
Browsers block `fetch` from `file://` (double-clicking index.html will NOT work), so serve the folder:
```bash
python -m http.server 8000      # then open http://localhost:8000
```

## Projects
| Project | Level | Steps | Skills |
|---|---|---|---|
| Student marks analytics | Beginner | 11 | slicing, nan-aware stats, masks, argsort, digitize, corrcoef, z-scores |
| Sensor cleaning & anomaly detection | Intermediate | 10 | -999 sentinels, np.interp, convolve, diff, bincount group-by |
| Linear regression from scratch | Advanced | 10 | design matrix, linalg.solve, R2/RMSE, gradient descent, train/test split |

Write code that stores the result in a variable called `answer`; Check compares it with the reference (floats within a small tolerance).
Progress is saved in your browser (localStorage).

## What is inside
| Path | Purpose |
|---|---|
| `index.html`, `css/`, `js/app.js` | the website (no build step) |
| `js/catalog.js`, `js/projects.js` | all examples and projects (generated) |
| `data/` | `students.csv` (600 x 11), `sensor.csv` (10,000 x 5), `houses.csv` (400 x 4), `datasets.json` |
| `py/runner.py` | Python glue: runs a snippet, returns tables/charts/errors, checks project answers |
| `tools/` | generators and tests (only needed if you want to change things) |

## Notes
- First load downloads Python + NumPy from the jsdelivr CDN (about 10 MB, then cached). Internet is required. matplotlib loads the first time a chart card runs.
- The browser runs NumPy 2.4.x. A few things cannot run in a browser (`np.memmap`); the card says why.
- Functions removed in NumPy 2 (`np.float_`, `np.in1d`, `np.row_stack`, `np.trapz`...) appear with a "removed" badge and the modern replacement.

## Change things (needs Python + NumPy)
```bash
python tools/generate_data.py             # regenerate the data (seeded, reproducible)
python tools/test_catalog.py              # run every example
python tools/build_catalog.py             # rebuild js/catalog.js after editing tools/catalog_src/*.py
python tools/build_projects.py            # re-run every solution, rebuild js/projects.js with expected outputs
```

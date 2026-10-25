from ._core import cat, fn

# =====================================================================================================  15  FFT
cat("fft", "Fourier transforms (np.fft)", "Move between time and frequency: fft, rfft, 2-D/N-D transforms, frequencies and shifts.", "🌊",
    intro="The FFT splits a signal into sine waves. fft() turns samples over time into amplitudes per frequency; ifft() converts back. "
          "For real signals use rfft (half the output, same information).")

fn("np.fft.fft", "Discrete Fourier transform of a signal.", '''
fs, n = 64, 64
t = np.arange(n) / fs
sig = np.sin(2 * np.pi * 5 * t)
spec = np.abs(np.fft.fft(sig))
(int(np.argmax(spec[:n // 2])), np.round(spec[:8], 2))
''', reads="A 5 Hz sine peaks at bin 5. The height 32 = n / 2.")
fn("np.fft.ifft", "Inverse transform.", '''
x = np.array([1.0, 2.0, 3.0, 4.0])
np.allclose(np.fft.ifft(np.fft.fft(x)).real, x)
''')
fn("np.fft.rfft / irfft", "Real-input transform: returns only the non-negative frequencies.", '''
x = np.cos(2 * np.pi * 3 * np.arange(32) / 32)
r = np.fft.rfft(x)
(len(r), np.allclose(np.fft.irfft(r, n=32), x))
''')
fn("np.fft.fftfreq", "The frequency of every FFT bin.", '''
np.fft.fftfreq(8, d=1 / 8)
''', reads="Positive frequencies first, then the negative ones.")
fn("np.fft.rfftfreq", "Frequencies for rfft.", '''
np.fft.rfftfreq(8, d=1 / 8)
''')
fn("np.fft.fftshift / ifftshift", "Put zero frequency in the centre.", '''
f = np.fft.fftfreq(8, d=1 / 8)
(np.fft.fftshift(f), np.fft.ifftshift(np.fft.fftshift(f)))
''')
fn("np.fft.fft2 / ifft2", "2-D transform (images).", '''
F = np.fft.fft2(img.astype(float))
(F.shape, np.allclose(np.fft.ifft2(F).real, img))
''', reads="img is a 64x64 grey image included in this lab.")
fn("np.fft.fftn / ifftn / rfft2 / rfftn", "N-dimensional versions.", '''
a = np.random.default_rng(0).random((4, 4, 4))
(np.fft.fftn(a).shape, np.fft.rfft2(a[0]).shape, np.fft.rfftn(a).shape)
''')
fn("np.fft.hfft / ihfft", "Transforms of Hermitian-symmetric signals.", '''
np.round(np.fft.hfft(np.array([1.0, 2.0, 3.0])), 3)
''')
fn("filter noise with FFT", "Remove high frequencies from a noisy signal.", '''
rng = np.random.default_rng(1)
t = np.linspace(0, 1, 256, endpoint=False)
noisy = np.sin(2 * np.pi * 4 * t) + 0.5 * rng.normal(size=256)
F = np.fft.rfft(noisy)
F[10:] = 0
clean = np.fft.irfft(F, n=256)
(float(np.std(noisy - np.sin(2 * np.pi * 4 * t))), float(np.std(clean - np.sin(2 * np.pi * 4 * t))))
''', reads="The error against the true sine drops a lot after cutting frequencies above bin 10.")
fn("dominant frequency of sensor data", "Find the daily cycle in the temperature sensor.", '''
t = np.nan_to_num(sensor[:, 1], nan=np.nanmean(sensor[:, 1]))
t = t - t.mean()
f = np.fft.rfftfreq(len(t), d=60)
p = np.abs(np.fft.rfft(t))
period_hours = 1 / f[np.argmax(p)] / 3600
round(float(period_hours), 1)
''', reads="About 24 hours: the daily temperature cycle.")
fn("plot a spectrum", "Chart of frequency vs amplitude (needs matplotlib).", '''
t = np.arange(512) / 512
sig = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
f = np.fft.rfftfreq(512, 1 / 512)
plt.plot(f, np.abs(np.fft.rfft(sig)) / 256)
plt.xlim(0, 100); plt.xlabel("Hz"); plt.title("Two tones: 20 Hz and 60 Hz")
''', needs=["matplotlib"])
fn("np.fft.set_workers / config", "FFT thread options (NumPy 2).", '''
hasattr(np.fft, "fft")
''')

# =====================================================================================================  16  POLYNOMIALS
cat("poly", "Polynomials", "np.poly1d, polyfit, roots and the numpy.polynomial package (Polynomial, Chebyshev, Legendre ...).", "📈",
    intro="Two APIs exist: the old np.poly1d / np.polyfit family (highest power FIRST) and the newer numpy.polynomial package (lowest power FIRST, recommended). Watch the coefficient order!")

fn("np.poly1d", "A polynomial you can call like a function.", '''
p = np.poly1d([1, -3, 2])
(str(p), p(5), p.roots, p.order)
''', reads="Coefficients highest power first: x^2 - 3x + 2, roots 2 and 1.")
fn("np.polyval", "Evaluate a polynomial.", '''
np.polyval([2, 0, -1], np.array([0, 1, 2, 3]))
''', reads="2x^2 - 1 at several x.")
fn("np.polyfit", "Least-squares polynomial fit.", '''
x = np.arange(10)
y = 3 * x ** 2 - 2 * x + 5 + np.random.default_rng(0).normal(0, 0.5, 10)
np.round(np.polyfit(x, y, 2), 2)
''', reads="Recovers roughly [3, -2, 5].")
fn("np.polyfit (deg=1 trend line)", "Straight-line fit to the house data.", '''
slope, intercept = np.polyfit(houses[:, 0], houses[:, 3], 1)
(round(slope, 4), round(intercept, 2))
''')
fn("np.roots", "Roots of a polynomial.", '''
np.roots([1, -6, 11, -6])
''', reads="x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3).")
fn("np.poly", "Coefficients from roots.", '''
np.poly([1, 2, 3])
''')
fn("np.polyder / np.polyint", "Derivative and integral.", '''
c = [3, 2, 1]
(np.polyder(c), np.polyint(c))
''')
fn("np.polyadd / polysub / polymul / polydiv", "Polynomial arithmetic.", '''
a, b = [1, 2], [1, -1]
{"add": np.polyadd(a, b), "sub": np.polysub(a, b), "mul": np.polymul(a, b), "div": np.polydiv([1, 0, -1], b)}
''')
fn("np.poly1d arithmetic", "Poly objects support + - * ** and calculus.", '''
p, q = np.poly1d([1, 1]), np.poly1d([1, -1])
(str(p * q), str(p ** 2), str(p.deriv()), str(p.integ()))
''')
fn("np.polynomial.Polynomial", "Modern polynomial class (lowest power first).", '''
from numpy.polynomial import Polynomial as P
p = P([2, -3, 1])
(p, p(3), p.roots(), p.degree())
''', reads="[2, -3, 1] means 2 - 3x + x^2 - the REVERSE of poly1d's order.")
fn("Polynomial.fit", "Fit with automatic domain scaling (more stable).", '''
from numpy.polynomial import Polynomial as P
x = np.linspace(0, 10, 30)
y = 1 + 2 * x - 0.3 * x ** 2
p = P.fit(x, y, 2)
np.round(p.convert().coef, 3)
''')
fn("Polynomial.deriv / integ", "Calculus on polynomials.", '''
from numpy.polynomial import Polynomial as P
p = P([1, 2, 3])
(p.deriv().coef, p.integ().coef)
''')
fn("np.polynomial.polynomial.polyfit / polyval", "Functional interface of the new package.", '''
from numpy.polynomial import polynomial as pp
x = np.arange(6)
c = pp.polyfit(x, 1 + 2 * x, 1)
(np.round(c, 6), pp.polyval(2, c))
''')
fn("np.polynomial.polynomial.polyroots / polyfromroots", "Roots <-> coefficients.", '''
from numpy.polynomial import polynomial as pp
(pp.polyfromroots([1, 2]), pp.polyroots([2, -3, 1]))
''')
fn("np.polynomial.chebyshev", "Chebyshev series (good for approximation).", '''
from numpy.polynomial import chebyshev as C
x = np.linspace(-1, 1, 50)
c = C.chebfit(x, np.exp(x), 5)
float(np.max(np.abs(C.chebval(x, c) - np.exp(x))))
''', reads="A degree-5 Chebyshev fit of exp(x) is accurate to about 1e-5.")
fn("np.polynomial.legendre", "Legendre series.", '''
from numpy.polynomial import legendre as L
L.leg2poly([0, 0, 1])
''', reads="Legendre P2(x) = (3x^2 - 1) / 2 -> coefficients -0.5, 0, 1.5.")
fn("np.polynomial.hermite / hermite_e / laguerre", "Other orthogonal families.", '''
from numpy.polynomial import hermite as H, laguerre as La
(H.herm2poly([0, 0, 1]), La.lag2poly([0, 0, 1]))
''')
fn("np.polynomial.polyutils (mapdomain)", "Rescale between intervals.", '''
from numpy.polynomial import polyutils as pu
pu.mapdomain(np.array([0, 5, 10]), [0, 10], [-1, 1])
''')
fn("plot a polynomial fit", "Data and fitted curve (needs matplotlib).", '''
x = np.linspace(0, 10, 25)
y = 0.5 * x ** 2 - 3 * x + 2 + np.random.default_rng(3).normal(0, 2, 25)
c = np.polyfit(x, y, 2)
plt.scatter(x, y)
plt.plot(x, np.polyval(c, x), color="C3")
plt.title("quadratic fit")
''', needs=["matplotlib"])

# =====================================================================================================  17  MASKED ARRAYS
cat("ma", "Masked arrays (np.ma)", "Arrays with a mask that hides invalid or missing entries from calculations.", "🎭",
    intro="A masked array pairs data with a boolean mask: True = 'ignore this value'. Statistics skip masked entries. Unlike nan this works for integers too and can carry a different fill value.")

fn("np.ma.masked_array", "Create from data and a mask.", '''
m = np.ma.masked_array([1, 2, 3, 4], mask=[0, 1, 0, 0])
(m, m.mean(), m.sum(), m.count())
''', reads="The masked 2 is ignored: mean = (1+3+4)/3.")
fn("np.ma.array", "Same, shorter name.", '''
np.ma.array([10, 20, 30], mask=[False, False, True])
''')
fn("np.ma.masked_where", "Mask by a condition.", '''
a = np.array([1, 5, -3, 8, -1])
np.ma.masked_where(a < 0, a)
''')
fn("np.ma.masked_equal / masked_less / masked_greater ...", "Mask by simple comparison.", '''
a = np.array([1, -999, 3, -999, 5])
(np.ma.masked_equal(a, -999), np.ma.masked_greater(a, 2), np.ma.masked_inside(a, 0, 3))
''', reads="Perfect for sentinel values like -999.")
fn("np.ma.masked_invalid", "Mask nan and inf.", '''
m = np.ma.masked_invalid(np.array([1.0, np.nan, 3.0, np.inf]))
(m, m.mean())
''')
fn("np.ma.masked_values", "Mask floats equal to a value (with tolerance).", '''
np.ma.masked_values(np.array([1.0, 1.1, 0.999999999, 2.0]), 1.0)
''')
fn("np.ma.masked_outside / masked_inside", "Mask a range / its complement.", '''
a = np.arange(10)
(np.ma.masked_outside(a, 3, 6), np.ma.masked_inside(a, 3, 6))
''')
fn("np.ma.masked_object / masked_not_equal", "Mask generic objects or everything except a value.", '''
np.ma.masked_not_equal(np.array([1, 2, 1, 3]), 1)
''')
fn("m.mask / np.ma.getmask / getmaskarray", "Read the mask.", '''
m = np.ma.masked_array([1, 2, 3], mask=[0, 1, 0])
(m.mask, np.ma.getmask(m), np.ma.getmaskarray(np.ma.array([1, 2])))
''')
fn("m.data / np.ma.getdata", "The underlying values (masked ones included).", '''
m = np.ma.masked_array([1, 2, 3], mask=[0, 1, 0])
(m.data, np.ma.getdata(m))
''')
fn("m.filled", "Replace masked values with something to get a normal array.", '''
m = np.ma.masked_array([1, 2, 3], mask=[0, 1, 0])
(m.filled(-1), m.filled(0), m.fill_value)
''')
fn("m.compressed", "1-D array of only the valid values.", '''
np.ma.masked_array([1, 2, 3, 4], mask=[0, 1, 1, 0]).compressed()
''')
fn("np.ma.mean / std / var / sum / min / max", "Statistics ignore masked entries.", '''
m = np.ma.masked_invalid(students[:, 5])
{"count": int(m.count()), "mean": round(float(m.mean()), 2), "std": round(float(m.std()), 2), "min": m.min(), "max": m.max()}
''')
fn("np.ma.median", "Median of valid values.", '''
np.ma.median(np.ma.masked_equal([5, -1, 3, -1, 9], -1))
''')
fn("m.count", "Number of valid (unmasked) values.", '''
m = np.ma.masked_array(np.arange(10), mask=np.arange(10) % 3 == 0)
(m.count(), m.size)
''')
fn("np.ma.is_masked", "Is any value masked?", '''
(np.ma.is_masked(np.ma.masked_array([1, 2], [0, 1])), np.ma.is_masked(np.ma.array([1, 2])))
''')
fn("np.ma.masked (constant)", "The special 'masked' singleton.", '''
m = np.ma.array([1.0, 2.0, 3.0])
m[1] = np.ma.masked
m
''')
fn("m.harden_mask / soften_mask", "Prevent assignment from unmasking values.", '''
m = np.ma.array([1, 2, 3], mask=[0, 1, 0])
m.harden_mask()
m[1] = 99
(m, m.hardmask)
''')
fn("np.ma.set_fill_value", "Change what filled() uses.", '''
m = np.ma.array([1, 2, 3], mask=[0, 1, 0])
np.ma.set_fill_value(m, -777)
m.filled()
''')
fn("np.ma.concatenate / np.ma.stack", "Join masked arrays keeping masks.", '''
a = np.ma.array([1, 2], mask=[0, 1])
b = np.ma.array([3, 4], mask=[1, 0])
(np.ma.concatenate([a, b]), np.ma.stack([a, b]).shape)
''')
fn("np.ma.where / np.ma.choose", "Condition-based selection with masks.", '''
a = np.ma.array([1, 2, 3, 4], mask=[0, 0, 1, 0])
np.ma.where(a > 2, a, 0)
''')
fn("np.ma.corrcoef / np.ma.cov", "Correlation ignoring masked values.", '''
x = np.ma.masked_invalid(students[:, 5])
y = np.ma.masked_invalid(students[:, 6])
np.round(np.ma.corrcoef(x, y, allow_masked=True), 3)
''')
fn("np.ma.apply_along_axis", "Apply functions across masked data.", '''
m = np.ma.masked_invalid(students[:5, 3:6])
np.ma.apply_along_axis(lambda r: r.mean(), 1, m)
''')
fn("np.ma.sort / argsort", "Masked values go last.", '''
m = np.ma.array([3, 1, 2, 9], mask=[0, 1, 0, 0])
(np.ma.sort(m), np.ma.argsort(m))
''')
fn("np.ma.average", "Weighted average of unmasked values.", '''
np.ma.average(np.ma.array([1, 2, 3, 4], mask=[0, 0, 0, 1]), weights=[1, 1, 2, 5])
''')
fn("np.ma.clump_masked / clump_unmasked", "Find consecutive masked runs.", '''
m = np.ma.masked_array(np.arange(10), mask=[0, 1, 1, 0, 0, 0, 1, 0, 0, 0])
(np.ma.clump_masked(m), np.ma.clump_unmasked(m))
''')
fn("np.ma.notmasked_edges / notmasked_contiguous", "Find edges of valid regions.", '''
m = np.ma.masked_array(np.arange(8), mask=[1, 0, 0, 1, 0, 0, 0, 1])
np.ma.notmasked_contiguous(m)
''')
fn("np.ma.mask_rows / mask_cols", "Mask whole rows/columns that contain a masked value.", '''
m = np.ma.masked_array(np.arange(9).reshape(3, 3), mask=np.eye(3, dtype=bool))
np.ma.mask_rows(m)
''')
fn("np.ma.dot / np.ma.outer", "Products honouring masks.", '''
a = np.ma.array([1, 2, 3], mask=[0, 0, 1])
np.ma.dot(a, np.array([1, 1, 1]), strict=False)
''')
fn("np.ma.masked_array arithmetic", "Operations keep masks.", '''
a = np.ma.array([1, 2, 3], mask=[0, 1, 0])
b = np.ma.array([10, 20, 30], mask=[0, 0, 1])
a + b
''', reads="The result is masked where EITHER input is masked.")
fn("np.ma (sensor example)", "Treat -999 error codes as masked, then summarise.", '''
h = np.ma.masked_equal(sensor[:, 2], -999)
h = np.ma.masked_invalid(h)
{"readings": int(h.count()), "masked": int(np.ma.getmaskarray(h).sum()), "mean humidity": round(float(h.mean()), 2)}
''')

# =====================================================================================================  18  STRUCTURED
cat("structured", "Structured & record arrays", "Arrays of records with named, typed fields - like a tiny table.", "🗃️",
    intro="A structured array stores one record per element, each with named fields of possibly different types. np.recarray adds attribute-style access (a.name). "
          "For serious table work pandas is better, but structured arrays are fast and common in file formats.")

fn("structured dtype definition", "List of (name, type) pairs.", '''
dt = np.dtype([("name", "U10"), ("age", "i4"), ("score", "f8")])
(dt, dt.names, dt.itemsize)
''')
fn("np.array(..., dtype=structured)", "Create records from tuples.", '''
dt = [("name", "U10"), ("age", "i4"), ("score", "f8")]
people = np.array([("Ann", 31, 88.5), ("Bob", 25, 92.0), ("Cy", 40, 79.5)], dtype=dt)
people
''')
fn("field access: a['name']", "Select a field as a normal array.", '''
dt = [("name", "U10"), ("age", "i4"), ("score", "f8")]
p = np.array([("Ann", 31, 88.5), ("Bob", 25, 92.0)], dtype=dt)
(p["name"], p["age"].mean(), p[0], p[0]["score"])
''')
fn("multiple fields: a[['a','b']]", "Select several fields (a view).", '''
dt = [("name", "U10"), ("age", "i4"), ("score", "f8")]
p = np.array([("Ann", 31, 88.5), ("Bob", 25, 92.0)], dtype=dt)
p[["name", "score"]]
''')
fn("filter records", "Boolean masks work on a field.", '''
dt = [("name", "U10"), ("age", "i4"), ("score", "f8")]
p = np.array([("Ann", 31, 88.5), ("Bob", 25, 92.0), ("Cy", 40, 79.5)], dtype=dt)
p[p["age"] > 30]["name"]
''')
fn("np.sort (order=)", "Sort by fields.", '''
dt = [("name", "U10"), ("age", "i4")]
p = np.array([("Ann", 31), ("Bob", 25), ("Cy", 31)], dtype=dt)
np.sort(p, order=["age", "name"])
''')
fn("np.genfromtxt (names=True)", "Read a CSV into a structured array.", '''
t = np.genfromtxt("data/houses.csv", delimiter=",", names=True)
(t.dtype.names, t["price_k"][:3], t.shape)
''')
fn("np.genfromtxt (dtype=None)", "Let NumPy guess a type per column (text + numbers).", '''
t = np.genfromtxt("data/students.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
(t.dtype.names[:4], t["gender"][:5], t[0])
''')
fn("np.rec.array / np.recarray", "Record arrays: attribute access.", '''
r = np.rec.array([("Ann", 31), ("Bob", 25)], dtype=[("name", "U10"), ("age", "i4")])
(r.name, r.age.mean(), type(r).__name__)
''')
fn("np.rec.fromarrays", "Build records from column arrays.", '''
np.rec.fromarrays([np.array(["a", "b"]), np.array([1.5, 2.5])], names="label,value")
''')
fn("np.rec.fromrecords", "Build from a list of tuples.", '''
np.rec.fromrecords([(1, "x"), (2, "y")], names="id,tag")
''')
fn("a.dtype.fields", "Field names with (dtype, offset).", '''
dt = np.dtype([("a", "i4"), ("b", "f8")])
dict(dt.fields)
''', reads="Offsets show where each field sits inside a record: b starts at byte 8 because of alignment.")
fn("np.dtype(align=True)", "C-style padding between fields.", '''
(np.dtype([("a", "i1"), ("b", "f8")]).itemsize, np.dtype([("a", "i1"), ("b", "f8")], align=True).itemsize)
''')
fn("nested structured dtype", "A field can itself be a record.", '''
dt = np.dtype([("id", "i4"), ("pos", [("x", "f4"), ("y", "f4")])])
a = np.zeros(2, dtype=dt)
a["pos"]["x"] = [1.5, 2.5]
a
''')
fn("subarray fields", "A field holding a small array.", '''
dt = np.dtype([("name", "U5"), ("rgb", "u1", (3,))])
a = np.array([("red", (255, 0, 0)), ("teal", (0, 128, 128))], dtype=dt)
a["rgb"]
''')
fn("np.lib.recfunctions.append_fields", "Add a field.", '''
from numpy.lib import recfunctions as rfn
a = np.array([(1, 2.0), (3, 4.0)], dtype=[("a", "i4"), ("b", "f8")])
rfn.append_fields(a, "c", [10, 20], usemask=False)
''')
fn("np.lib.recfunctions.rename_fields / drop_fields", "Rename or remove fields.", '''
from numpy.lib import recfunctions as rfn
a = np.array([(1, 2.0)], dtype=[("a", "i4"), ("b", "f8")])
(rfn.rename_fields(a, {"a": "id"}).dtype.names, rfn.drop_fields(a, "b").dtype.names)
''')
fn("np.lib.recfunctions.structured_to_unstructured", "Records -> plain 2-D array.", '''
from numpy.lib import recfunctions as rfn
a = np.array([(1, 2.0), (3, 4.0)], dtype=[("a", "i4"), ("b", "f8")])
(rfn.structured_to_unstructured(a), rfn.unstructured_to_structured(np.array([[1, 2], [3, 4]]), names=["x", "y"]))
''')
fn("np.lib.recfunctions.merge_arrays / join_by", "Combine record arrays.", '''
from numpy.lib import recfunctions as rfn
a = np.array([(1, 10.0), (2, 20.0)], dtype=[("id", "i4"), ("v", "f8")])
b = np.array([(1, "x"), (2, "y")], dtype=[("id", "i4"), ("tag", "U1")])
rfn.join_by("id", a, b)
''')
fn("a.view with fields", "Reinterpret a structured array as plain numbers.", '''
a = np.array([(1, 2), (3, 4)], dtype=[("a", "i4"), ("b", "i4")])
a.view("i4").reshape(-1, 2)
''')
fn("np.zeros with structured dtype", "Preallocate records.", '''
np.zeros(3, dtype=[("t", "f8"), ("tag", "U4")])
''')
fn("np.concatenate on records", "Join record arrays with the same dtype.", '''
dt = [("a", "i4"), ("b", "f8")]
np.concatenate([np.array([(1, 2.0)], dtype=dt), np.array([(3, 4.0)], dtype=dt)])
''')
fn("np.unique on a field", "Distinct values of a record field.", '''
t = np.genfromtxt("data/students.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
np.unique(t["gender"], return_counts=True)
''')

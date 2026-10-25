from ._core import cat, fn

# =====================================================================================================  22  MEMORY
cat("memory", "Memory, views & performance", "Views vs copies, strides, memory order, shared memory, and why NumPy is fast.", "⚡",
    intro="Understanding views and copies is the single most important NumPy skill for avoiding bugs. Basic slicing, reshape, transpose and ravel return VIEWS (shared memory); "
          "boolean/fancy indexing, flatten, astype and arithmetic return COPIES.")

fn("view vs copy: slicing", "A slice is a view: writes go through to the original.", '''
a = np.arange(6)
v = a[:3]
v[:] = 0
a
''')
fn("view vs copy: fancy indexing", "Fancy indexing makes a copy: the original is safe.", '''
a = np.arange(6)
c = a[[0, 1, 2]]
c[:] = 0
a
''')
fn("view vs copy: boolean indexing", "Boolean indexing copies, but assigning THROUGH the mask modifies the original.", '''
a = np.arange(6)
sel = a[a > 2]
sel[:] = 0           # modifies only the copy
a[a > 2] = 99        # modifies a
(sel, a)
''')
fn("np.shares_memory", "Do two arrays overlap in memory? (exact)", '''
a = np.arange(10)
(np.shares_memory(a, a[2:5]), np.shares_memory(a, a.copy()), np.shares_memory(a[::2], a[1::2]))
''', reads="The even and odd positions share no bytes even though they come from one array.")
fn("np.may_share_memory", "Quick approximate check (bounds only).", '''
a = np.arange(10)
(np.may_share_memory(a[::2], a[1::2]), np.shares_memory(a[::2], a[1::2]))
''', reads="may_share_memory says True (their memory ranges overlap) but they never touch the same element.")
fn("a.base / a.flags.owndata", "Who owns the memory?", '''
a = np.arange(6)
b = a.reshape(2, 3)
(a.flags.owndata, b.flags.owndata, b.base is a)
''')
fn("a.copy(order=...)", "Copy with a chosen memory layout.", '''
a = np.arange(6).reshape(2, 3)
f = a.copy(order="F")
(a.flags.c_contiguous, f.flags.f_contiguous, np.array_equal(a, f))
''')
fn("C order vs F order", "Row-major (C) and column-major (Fortran) storage.", '''
a = np.arange(6).reshape(2, 3)
(a.ravel(order="K"), a.T.ravel(order="K"), a.strides, a.T.strides)
''', reads="order='K' reads memory in its stored order. Iterating along the fastest-varying axis is quicker (cache-friendly).")
fn("np.lib.stride_tricks.as_strided", "Build custom views by choosing shape and strides (DANGEROUS: no bounds checks).", '''
from numpy.lib.stride_tricks import as_strided
a = np.arange(8, dtype=np.int64)
as_strided(a, shape=(5, 4), strides=(8, 8))
''', reads="Row i is a[i:i+4] - overlapping windows with zero copying. Prefer sliding_window_view, which is safe.")
fn("np.lib.stride_tricks.sliding_window_view", "Safe overlapping windows.", '''
from numpy.lib.stride_tricks import sliding_window_view as swv
x = np.arange(10)
(swv(x, 4)[:3], swv(x, 4).mean(axis=1))
''')
fn("sliding_window_view (2-D patches)", "Image patches.", '''
from numpy.lib.stride_tricks import sliding_window_view as swv
p = swv(img, (3, 3))
(p.shape, p[0, 0].shape)
''', reads="Every 3x3 patch of the 64x64 image: (62, 62, 3, 3).")
fn("memory of different dtypes", "Smaller dtype = less memory.", '''
n = 1_000_000
{t: np.zeros(n, dtype=t).nbytes / 1e6 for t in ("int8", "int32", "float32", "float64", "complex128")}
''', reads="Megabytes for one million elements.")
fn("downcast to save memory", "Use the smallest safe dtype.", '''
ages = students[:, 1]
small = ages.astype(np.uint8)
(ages.nbytes, small.nbytes, np.array_equal(ages, small))
''')
fn("timing: loop vs vectorised", "Why vectorisation matters.", '''
import time
x = np.arange(200_000, dtype=float)
t0 = time.perf_counter(); s = 0.0
for v in x: s += v * v
t1 = time.perf_counter(); s2 = (x * x).sum(); t2 = time.perf_counter()
{"python loop (ms)": round((t1 - t0) * 1000, 1), "numpy (ms)": round((t2 - t1) * 1000, 3), "same result": bool(np.isclose(s, s2))}
''', reads="NumPy is typically 50-200 times faster here because the loop runs in compiled C.")
fn("timing: list vs array", "Memory and speed of a Python list vs an array.", '''
import sys
lst = list(range(100000))
arr = np.arange(100000)
(sys.getsizeof(lst) + sum(sys.getsizeof(i) for i in lst[:100]) * 1000, arr.nbytes)
''')
fn("avoid temporaries with out=", "Reuse memory for big calculations.", '''
a = np.arange(5.0)
b = np.empty_like(a)
np.multiply(a, 2, out=b)
np.add(b, 1, out=b)
b
''')
fn("in-place ops (+=)", "Modify without allocating a new array.", '''
a = np.ones(5)
id_before = id(a)
a += 1
(a, id(a) == id_before)
''')
fn("a.flags.writeable", "Make an array read-only.", '''
a = np.arange(3)
a.flags.writeable = False
try:
    a[0] = 5
except ValueError as e:
    msg = str(e)
msg
''')
fn("a.setflags", "Change flags.", '''
a = np.arange(3)
a.setflags(write=False)
a.flags.writeable
''')
fn("np.ndarray.ctypes data pointer", "Where the data lives (address).", '''
a = np.arange(4)
(a.ctypes.data == a.__array_interface__["data"][0], a[1:].ctypes.data - a.ctypes.data)
''', reads="The slice a[1:] starts exactly one int64 (8 bytes) after a.")
fn("np.ascontiguousarray before heavy work", "Contiguous data is faster to process.", '''
a = np.random.default_rng(0).random((1000, 1000))[:, ::2]
b = np.ascontiguousarray(a)
(a.flags.c_contiguous, b.flags.c_contiguous)
''')
fn("np.lib.stride_tricks.broadcast_arrays (memory)", "Broadcasting does not copy.", '''
a = np.arange(3)
b = np.broadcast_to(a, (1000, 3))
(b.shape, b.strides, a.nbytes)
''', reads="stride 0 along the first axis: all 1000 rows read the same memory.")
fn("np.einsum vs loops (speed)", "Contractions without Python loops.", '''
a = np.random.default_rng(0).random((200, 200))
np.allclose(np.einsum("ij,jk->ik", a, a), a @ a)
''')
fn("np.fromiter vs list (memory)", "Build from a generator without a big temporary list.", '''
np.fromiter((i * i for i in range(10)), dtype=np.int64, count=10)
''')
fn("np.empty vs np.zeros", "zeros is a little slower because it writes memory.", '''
import time
t0 = time.perf_counter(); np.empty(5_000_000); t1 = time.perf_counter(); np.zeros(5_000_000); t2 = time.perf_counter()
(round((t1 - t0) * 1000, 3), round((t2 - t1) * 1000, 3))
''')
fn("np.cumsum vs python loop (speed)", "Prefix sums.", '''
a = np.arange(1, 8)
np.cumsum(a)
''')
fn("np.set_printoptions vs memory", "Printing summarises big arrays; the data is still all there.", '''
a = np.arange(1_000_000)
(len(repr(a)) < 200, a.size)
''')
fn("pickle / copy.deepcopy of arrays", "Deep copies of containers holding arrays.", '''
import copy
d = {"x": np.arange(3)}
e = copy.deepcopy(d)
e["x"][0] = 99
(d["x"], e["x"])
''')
fn("np.ndarray.__array_priority__ / subclassing", "Create a subclass that tags an array.", '''
class Tagged(np.ndarray):
    def __new__(cls, a, tag):
        o = np.asarray(a).view(cls)
        o.tag = tag
        return o
    def __array_finalize__(self, obj):
        self.tag = getattr(obj, "tag", None)

t = Tagged([1, 2, 3], "meters")
(t.tag, (t * 2).tag, type(t[1:]).__name__)
''')

# =====================================================================================================  23  WINDOWS / SIGNAL
cat("signal", "Windows, smoothing & interpolation", "Window functions, moving averages, convolution, resampling and simple filters.", "〰️",
    intro="Small building blocks for signal processing and smoothing: window functions for FFTs, convolution for filters, interp for resampling.")

fn("np.hanning / hamming / blackman", "Window functions that taper a signal to zero at the edges.", '''
(np.round(np.hanning(8), 3), np.round(np.hamming(8), 3), np.round(np.blackman(8), 3))
''')
fn("np.bartlett / np.kaiser", "Triangular and Kaiser windows.", '''
(np.round(np.bartlett(7), 3), np.round(np.kaiser(7, 5), 3))
''')
fn("window before FFT", "Tapering reduces spectral leakage.", '''
t = np.arange(100) / 100
x = np.sin(2 * np.pi * 10.5 * t)
a = np.abs(np.fft.rfft(x)); b = np.abs(np.fft.rfft(x * np.hanning(100)))
(round(float(a[1:].max() / a.sum()), 3), round(float(b[1:].max() / b.sum()), 3))
''', reads="With the window the energy is more concentrated at the peak.")
fn("moving average (convolve)", "Smooth noisy data with a box filter.", '''
t = sensor[:200, 1]
t = np.nan_to_num(t, nan=np.nanmean(t))
sm = np.convolve(t, np.ones(9) / 9, mode="same")
(np.round(t[:5], 2), np.round(sm[:5], 2))
''')
fn("np.convolve (modes)", "full, same and valid output lengths.", '''
x, k = np.arange(1, 6), np.array([1, 0, -1])
{m: np.convolve(x, k, mode=m) for m in ("full", "same", "valid")}
''')
fn("convolution as polynomial multiplication", "(1 + 2x)(3 + 4x).", '''
np.convolve([1, 2], [3, 4])
''', reads="Coefficients 3, 10, 8 -> 3 + 10x + 8x^2.")
fn("np.correlate (lag detection)", "Find how far a pattern is shifted.", '''
a = np.array([0, 0, 1, 2, 1, 0, 0, 0.0])
b = np.array([0, 0, 0, 0, 1, 2, 1, 0.0])
c = np.correlate(b, a, mode="full")
int(np.argmax(c)) - (len(a) - 1)
''', reads="b is a shifted by 2 samples.")
fn("np.interp (resample)", "Re-sample data onto a new grid.", '''
x = np.array([0, 2, 4, 6.0])
y = x ** 2
np.interp(np.linspace(0, 6, 7), x, y)
''')
fn("np.interp (fill nan gaps)", "Linear interpolation across missing values.", '''
y = np.array([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
idx = np.arange(len(y)); ok = ~np.isnan(y)
np.interp(idx, idx[ok], y[ok])
''')
fn("np.interp (period)", "Interpolate on circular data.", '''
np.interp(370, [0, 90, 180, 270], [0, 1, 0, -1], period=360)
''')
fn("exponential smoothing", "A recursive filter (loop version).", '''
x = np.array([1.0, 2, 3, 10, 3, 2, 1])
y = np.empty_like(x); y[0] = x[0]
for i in range(1, len(x)):
    y[i] = 0.5 * x[i] + 0.5 * y[i - 1]
np.round(y, 2)
''')
fn("np.cumsum moving window", "Fast rolling sum via cumulative sums.", '''
x = np.arange(10.0)
c = np.cumsum(np.concatenate([[0], x]))
c[3:] - c[:-3]
''')
fn("sliding_window_view moving stats", "Rolling min / max / std.", '''
from numpy.lib.stride_tricks import sliding_window_view as swv
x = np.array([3, 1, 4, 1, 5, 9, 2, 6.0])
w = swv(x, 3)
(w.min(axis=1), w.max(axis=1), np.round(w.std(axis=1), 2))
''')
fn("2-D convolution by hand (blur)", "Average each 3x3 neighbourhood of the image.", '''
from numpy.lib.stride_tricks import sliding_window_view as swv
blur = swv(img.astype(float), (3, 3)).mean(axis=(-1, -2))
(img.shape, blur.shape, round(float(img.std()), 2), round(float(blur.std()), 2))
''', reads="Blurring lowers the noise: the standard deviation of the blurred image is smaller.")
fn("edge detection (Sobel by hand)", "Differences highlight edges.", '''
from numpy.lib.stride_tricks import sliding_window_view as swv
f = img.astype(float)
kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1.0]])
gx = (swv(f, (3, 3)) * kx).sum(axis=(-1, -2))
gx.shape, float(np.abs(gx).max())
''')
fn("show the image", "Display the built-in 64x64 image (needs matplotlib).", '''
plt.imshow(img, cmap="gray")
plt.title("img (uint8, 64x64)")
plt.axis("off")
''', needs=["matplotlib"])
fn("np.histogram of pixel values", "Brightness distribution.", '''
counts, edges = np.histogram(img, bins=8, range=(0, 256))
counts
''')
fn("downsample an image by averaging", "Reshape trick for block means.", '''
small = img.reshape(16, 4, 16, 4).mean(axis=(1, 3))
small.shape
''', reads="Each 4x4 block of the 64x64 image becomes one pixel.")
fn("threshold an image (mask)", "Segment bright objects.", '''
mask = img > 200
(mask.sum(), mask.mean().round(3))
''')
fn("np.gradient on an image", "Gradients along both axes.", '''
gy, gx = np.gradient(img.astype(float))
(gy.shape, round(float(np.abs(gx).mean()), 3))
''')

# =====================================================================================================  24  UTILITIES
cat("utils", "Constants, iteration & utilities", "pi, e, inf, nan; iterators; testing helpers; info and typing helpers.", "🧰",
    intro="The small helpers that round out NumPy: mathematical constants, iterating over arrays, testing utilities, introspection.")

fn("np.pi / np.e / np.euler_gamma", "Constants.", '''
(np.pi, np.e, np.euler_gamma)
''')
fn("np.inf / np.nan / np.NINF", "Special floating-point values.", '''
(np.inf, -np.inf, np.nan, np.inf - np.inf, np.inf * 0, 1 / np.inf)
''', reads="nan appears whenever the maths is undefined. np.NINF and np.PINF were removed in 2.0: use -np.inf and np.inf.")
fn("np.newaxis", "An alias of None used to add axes.", '''
(np.newaxis is None, np.arange(3)[:, np.newaxis].shape)
''')
fn("np.nan is not nan", "nan != nan, so test with isnan.", '''
x = np.nan
(x == x, np.isnan(x), x is np.nan)
''')
fn("np.ndenumerate", "Index and value together.", '''
[(idx, int(v)) for idx, v in np.ndenumerate(np.array([[5, 6], [7, 8]]))]
''')
fn("np.ndindex", "All indices of a shape.", '''
list(np.ndindex(2, 2))
''')
fn("np.nditer (read / write)", "Iterate and modify in place.", '''
a = np.arange(6).reshape(2, 3)
for x in np.nditer(a, op_flags=["readwrite"]):
    x[...] = x * 10
a
''')
fn("np.nditer (multi-array)", "Iterate several arrays in lock-step with broadcasting.", '''
a = np.arange(3); b = np.array([[10], [20]])
[int(x + y) for x, y in np.nditer([a, b])]
''')
fn("np.nditer (external_loop, order)", "Chunked, memory-order iteration.", '''
a = np.arange(6).reshape(2, 3)
[c.tolist() for c in np.nditer(a, flags=["external_loop"], order="F")]
''')
fn("np.flatiter (a.flat)", "Flat iterator object.", '''
a = np.arange(6).reshape(2, 3)
it = a.flat
(next(it), next(it), it.index, it.coords)
''')
fn("np.iterable", "Is it iterable?", '''
(np.iterable([1, 2]), np.iterable(5), np.iterable("abc"))
''')
fn("np.testing.assert_allclose", "Tolerance-based assertion.", '''
np.testing.assert_allclose(np.array([1.0, 2.0]), np.array([1.0 + 1e-9, 2.0]), rtol=1e-6)
"ok - arrays are close"
''')
fn("np.testing.assert_array_almost_equal", "Compare to a number of decimals.", '''
np.testing.assert_array_almost_equal(np.array([0.123456]), np.array([0.123457]), decimal=4)
"ok"
''')
fn("np.testing.assert_equal / assert_raises", "General assertions.", '''
np.testing.assert_equal({"a": np.arange(3)}, {"a": np.arange(3)})
with np.testing.assert_raises(ZeroDivisionError):
    1 / 0
"ok"
''')
fn("np.testing.assert_array_less", "All elements strictly smaller.", '''
np.testing.assert_array_less(np.array([1, 2]), np.array([2, 3]))
"ok"
''')
fn("np.info / np.lookfor", "Built-in documentation.", '''
np.info(np.sum)
''')
fn("np.add.__doc__ (help text)", "Docstrings are always available.", '''
np.mean.__doc__.strip().splitlines()[0]
''')
fn("np.who (variables)", "REMOVED in NumPy 2.0: list your arrays with a comprehension over globals().", '''
a = np.arange(3); b = np.zeros((2, 2))
{k: v.shape for k, v in list(globals().items()) if isinstance(v, np.ndarray) and not k.startswith("_")}
''', status="removed", alt="[k for k, v in globals().items() if isinstance(v, np.ndarray)]")
fn("np.get_include / np.__config__", "Build helpers.", '''
isinstance(np.get_include(), str)
''')
fn("np.typing.NDArray", "Type hints for arrays.", '''
import numpy.typing as npt
def double(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return x * 2
double(np.array([1.0, 2.0]))
''')
fn("np.lib.NumpyVersion", "Compare versions safely.", '''
v = np.lib.NumpyVersion(np.__version__)
(v >= "2.0.0", v.major)
''')
fn("np.dtype.type vs np.dtype", "Scalar type vs dtype object.", '''
(np.float64, np.dtype(np.float64), np.dtype("float64").type is np.float64)
''')
fn("np.ufunc list (np.core / np._core)", "How many ufuncs exist?", '''
sum(isinstance(getattr(np, n), np.ufunc) for n in dir(np))
''')
fn("np.ma / np.char / np.fft ... namespaces", "NumPy's sub-packages.", '''
[n for n in ("linalg", "fft", "random", "ma", "char", "strings", "polynomial", "lib", "testing", "typing", "rec", "ctypeslib", "dtypes", "exceptions") if hasattr(np, n)]
''')
fn("np.exceptions", "NumPy 2 exception classes live here.", '''
[n for n in dir(np.exceptions) if n[0].isupper()]
''')
fn("np.exceptions.AxisError", "Raised for a bad axis.", '''
try:
    np.zeros((2, 2)).sum(axis=5)
except np.exceptions.AxisError as e:
    msg = str(e)
msg
''')
fn("np.exceptions.ComplexWarning / VisibleDeprecationWarning", "Warning categories.", '''
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    np.array([1 + 2j]).astype(float)
[x.category.__name__ for x in w]
''')
fn("np.random.default_rng for reproducible tests", "Seeds make tests deterministic.", '''
np.array_equal(np.random.default_rng(0).random(3), np.random.default_rng(0).random(3))
''')
fn("np.show_runtime", "Runtime information (SIMD features, BLAS).", '''
len(np.show_runtime.__doc__) > 0
''')
fn("np.lib.scimath.sqrt", "Square root that returns complex for negative input.", '''
np.emath.sqrt(np.array([4, -4]))
''')
fn("np.emath (log, power)", "Domain-extended functions.", '''
(np.emath.log(-1), np.emath.logn(2, 8), np.emath.power(-8, 1 / 3))
''')
fn("np.ctypeslib", "Connect arrays with C code.", '''
np.ctypeslib.as_ctypes_type(np.dtype("int32"))
''')
fn("np.array_api / array API standard names", "NumPy 2 supports the standard names.", '''
(np.concat is np.concatenate, np.acos is np.arccos, np.permute_dims is np.transpose)
''')

# =====================================================================================================  25  PITFALLS
cat("pitfalls", "Common pitfalls & gotchas", "Mistakes everyone makes once - and how to avoid them.", "🪤",
    intro="Each card here shows a trap that surprises beginners. Run it, read the explanation, remember the fix.")

fn("views change the original", "Slice, then modify -> original changes.", '''
a = np.array([1, 2, 3, 4])
b = a[1:3]
b[0] = 100
a
''', tip="Use .copy() when you need an independent array.")
fn("a = b does not copy", "Two names, one array.", '''
a = np.array([1, 2, 3])
b = a
b[0] = 99
(a, a is b)
''')
fn("integer division surprises", "/ is float, // floors, int() truncates.", '''
{"7 / 2": 7 / 2, "7 // 2": 7 // 2, "-7 // 2": -7 // 2, "int(-7 / 2)": int(-7 / 2)}
''')
fn("integer overflow", "Fixed-size integers wrap around silently.", '''
np.array([127], dtype=np.int8) + np.int8(1)
''')
fn("float equality", "Never compare floats with ==.", '''
a = np.arange(0, 1, 0.1)
(a[3] == 0.3, np.isclose(a[3], 0.3))
''')
fn("nan comparisons", "nan == nan is False and nan makes max/sum nan.", '''
a = np.array([1.0, np.nan, 3.0])
(a == np.nan, np.isnan(a), a.max(), np.nanmax(a))
''')
fn("boolean 'and'/'or' on arrays", "Use & and | with parentheses.", '''
a = np.array([1, 5, 9])
try:
    r = (a > 2) and (a < 8)
except ValueError as e:
    r = "ValueError: " + str(e)[:50]
(r, (a > 2) & (a < 8))
''')
fn("operator precedence with &", "Without parentheses & binds tighter than >.", '''
a = np.array([1, 5, 9])
try:
    r = a > 2 & a < 8
except (TypeError, ValueError) as e:
    r = type(e).__name__ + ": " + str(e)[:70]
r
''')
fn("shape (n,) vs (n,1)", "A 1-D array is neither a row nor a column.", '''
v = np.array([1, 2, 3])
(v.shape, (v * v[:, None]).shape, v @ v)
''', reads="v * v[:, None] broadcasts to 3x3 - often NOT what you meant.")
fn("shape (3,) + (3,1) broadcasting trap", "Accidental 2-D result.", '''
a = np.array([1, 2, 3])
b = np.array([[10], [20], [30]])
a + b
''', reads="You may expect 3 numbers but get 9.")
fn("np.sum of a generator", "Use the array method, not Python's sum.", '''
a = np.arange(1_000_000)
(sum(a[:5]), int(np.sum(a)), type(sum(a[:5])).__name__)
''')
fn("append in a loop is slow", "Each np.append copies everything.", '''
import time
t0 = time.perf_counter()
a = np.array([])
for i in range(3000):
    a = np.append(a, i)
t1 = time.perf_counter()
b = np.array([i for i in range(3000)])
t2 = time.perf_counter()
{"append loop (ms)": round((t1 - t0) * 1000, 1), "list then array (ms)": round((t2 - t1) * 1000, 2)}
''')
fn("mutable default: [[0]*3]*3", "Row replication by multiplication shares rows (lists).", '''
bad = [[0] * 3] * 3
bad[0][0] = 1
good = np.zeros((3, 3), dtype=int)
good[0, 0] = 1
(bad, good)
''')
fn("np.array([...]) ragged lists", "Different lengths raise an error in NumPy 2.", '''
try:
    np.array([[1, 2], [3]])
except ValueError as e:
    msg = str(e)[:90]
msg
''')
fn("string truncation", "Assigning a longer string to a fixed-width array cuts it.", '''
s = np.array(["ab", "cd"])
s[0] = "abcdef"
s
''')
fn("np.round half to even", "2.5 rounds to 2, 3.5 to 4.", '''
np.round(np.array([0.5, 1.5, 2.5, 3.5]))
''')
fn("np.float32 accumulation error", "float32 sums lose precision on big arrays.", '''
a = np.full(10_000_000, 0.1, dtype=np.float32)
(float(a.sum()), float(a.sum(dtype=np.float64)))
''', reads="NumPy uses pairwise summation so the error is small; a naive loop would be worse.")
fn("sorting returns a copy", "np.sort(a) does not change a.", '''
a = np.array([3, 1, 2])
np.sort(a)
a
''')
fn("a.sort() returns None", "In-place methods return None.", '''
a = np.array([3, 1, 2])
r = a.sort()
(r, a)
''')
fn("reshape needs matching size", "Wrong total size raises ValueError.", '''
try:
    np.arange(10).reshape(3, 4)
except ValueError as e:
    msg = str(e)
msg
''')
fn("axis confusion", "axis=0 means 'collapse the rows'.", '''
a = np.array([[1, 2, 3], [4, 5, 6]])
(a.sum(axis=0), a.sum(axis=1))
''')
fn("np.mean of ints", "Mean is float, median of even count averages.", '''
a = np.array([1, 2, 3, 4])
(a.mean(), np.median(a), (a / 2).dtype)
''')
fn("np.max vs np.maximum", "One reduces, the other compares element-wise.", '''
a, b = np.array([1, 8]), np.array([5, 2])
(np.max(a), np.maximum(a, b))
''')
fn("list.index vs np.where", "Find positions of ALL matches.", '''
a = np.array([4, 2, 4, 7])
(a.tolist().index(4), np.where(a == 4)[0])
''')
fn("copy of a list inside an object array", "Object arrays hold references.", '''
x = [1, 2]
o = np.empty(2, dtype=object)
o[0] = x; o[1] = x
x.append(3)
o
''')
fn("np.float64 vs float", "NumPy scalars print differently and behave subtly differently.", '''
(type(np.float64(1.5)).__name__, type(float(np.float64(1.5))).__name__, repr(np.float64(1.5)), np.float64(1.5) == 1.5)
''')
fn("0-d array vs scalar indexing", "a[0] on 1-D returns a NumPy scalar, a[0:1] an array.", '''
a = np.array([5, 6, 7])
(type(a[0]).__name__, a[0:1].shape)
''')
fn("negative step and empty slices", "Slices never raise errors, even out of range.", '''
a = np.arange(5)
(a[10:20], a[3:1], a[-10:2])
''')
fn("index out of range (single index)", "A single out-of-range index DOES raise.", '''
try:
    np.arange(5)[7]
except IndexError as e:
    msg = str(e)
msg
''')
fn("fancy assign with duplicates", "a[idx] += 1 counts duplicates only once.", '''
a = np.zeros(3, dtype=int)
a[[0, 0, 1]] += 1
b = np.zeros(3, dtype=int)
np.add.at(b, [0, 0, 1], 1)
(a, b)
''')
fn("np.array(a) vs np.asarray(a)", "array copies by default, asarray does not.", '''
a = np.arange(3)
(np.array(a) is a, np.asarray(a) is a)
''')
fn("a.astype default copies", "astype returns a new array even for the same dtype.", '''
a = np.arange(3)
(a.astype(a.dtype) is a, a.astype(a.dtype, copy=False) is a)
''')
fn("True/False as numbers", "Booleans act as 0/1 in arithmetic.", '''
a = np.array([True, True, False])
(a.sum(), a * 5, a + a, ~a)
''')
fn("np.bool_ vs ~ on ints", "~ on integers is bitwise NOT, not logical NOT.", '''
(~np.array([0, 1, 5]), np.logical_not(np.array([0, 1, 5])))
''')
fn("a.T on 1-D does nothing", "Use [:, None] to make a column.", '''
v = np.arange(3)
(v.T.shape, v[:, None].shape)
''')
fn("NumPy scalar ** negative int", "Integer powers with negative exponents fail.", '''
try:
    np.array([2, 3]) ** -1
except ValueError as e:
    msg = str(e)
msg
''')
fn("sum of float32 vs float64 default dtype", "Default dtype of sums depends on input.", '''
(np.array([1, 2], np.int8).sum().dtype, np.array([1, 2], np.float32).sum().dtype, np.array([True, False]).sum().dtype)
''')

# =====================================================================================================  26  MIGRATION
cat("migration", "NumPy 2 changes (old -> new)", "Names removed or changed in NumPy 2.0 - 2.5 and what to use now.", "🔁",
    intro="Old tutorials use functions that no longer exist. Each card shows the removed name (as a note) and runs the modern replacement.")

_M = [
    ("np.float_, np.complex_, np.cfloat, np.longfloat ...", "Use np.float64, np.complex128, np.longdouble.", "np.float64(1)", "np.float64"),
    ("np.bool8 / np.int0 / np.uint0 / np.object0", "Use np.bool_, np.intp, np.uintp, np.object_.", "(np.bool_, np.intp, np.uintp, np.object_)", "np.bool_ / np.intp / np.uintp"),
    ("np.Inf / np.Infinity / np.PINF / np.NINF / np.NaN / np.NAN", "Use np.inf and np.nan.", "(np.inf, -np.inf, np.nan)", "np.inf / np.nan"),
    ("np.asfarray", "Use np.asarray(a, dtype=float).", "np.asarray([1, 2], dtype=float)", "np.asarray(..., dtype=float)"),
    ("np.alltrue / np.sometrue", "Use np.all / np.any.", "(np.all([True, False]), np.any([True, False]))", "np.all / np.any"),
    ("np.cumproduct / np.product", "Use np.cumprod / np.prod.", "(np.cumprod([1, 2, 3]), np.prod([1, 2, 3]))", "np.cumprod / np.prod"),
    ("np.round_ / np.around alias", "Use np.round.", "np.round(2.567, 1)", "np.round"),
    ("np.in1d", "Use np.isin.", "np.isin([1, 2, 3], [2])", "np.isin"),
    ("np.row_stack", "Use np.vstack.", "np.vstack([[1], [2]])", "np.vstack"),
    ("np.trapz", "Use np.trapezoid.", "np.trapezoid([1, 2, 3])", "np.trapezoid"),
    ("np.msort", "Use np.sort(a, axis=0).", "np.sort([[3, 1], [2, 5]], axis=0)", "np.sort(a, axis=0)"),
    ("np.find_common_type", "Use np.result_type / np.promote_types.", "np.result_type(np.int8, np.float32)", "np.result_type"),
    ("np.issubsctype / np.obj2sctype / np.sctype2char", "Use np.issubdtype / np.dtype(...).char.", "(np.issubdtype(np.int8, np.integer), np.dtype('int8').char)", "np.issubdtype"),
    ("np.source / np.lookfor", "Use help() or the docs website.", "np.info(np.mean) is None", "help(np.mean)"),
    ("np.safe_eval", "Use ast.literal_eval.", "__import__('ast').literal_eval('[1, 2, 3]')", "ast.literal_eval"),
    ("np.deprecate / np.deprecate_with_doc", "Use warnings.warn(..., DeprecationWarning).", "'use warnings.warn'", "warnings.warn"),
    ("np.recfromcsv / np.recfromtxt", "Use np.genfromtxt(..., names=True).", "np.genfromtxt('data/houses.csv', delimiter=',', names=True).dtype.names", "np.genfromtxt"),
    ("np.set_string_function", "Use np.set_printoptions(formatter=...).", "'formatter option'", "np.set_printoptions(formatter=...)"),
    ("np.fastCopyAndTranspose", "Use a.T.copy().", "np.arange(6).reshape(2, 3).T.copy()", "a.T.copy()"),
    ("np.add_newdoc / np.add_docstring", "Use the private module or docstrings in your code.", "'n/a'", "docstrings"),
    ("a.ptp() method", "Use np.ptp(a).", "np.ptp(np.array([3, 9, 1]))", "np.ptp(a)"),
    ("a.newbyteorder()", "Use a.view(a.dtype.newbyteorder()).", "np.arange(3, dtype='<i4').view(np.dtype('<i4').newbyteorder())", "a.view(a.dtype.newbyteorder())"),
    ("a.tostring()", "Use a.tobytes().", "np.arange(2, dtype=np.uint8).tobytes()", "a.tobytes()"),
    ("np.core / np.lib.* private paths", "Use the public namespaces (np.lib.stride_tricks, np._core is private).", "np.lib.stride_tricks.sliding_window_view(np.arange(4), 2).shape", "public np.* names"),
    ("copy=False semantics", "In NumPy 2, copy=False RAISES if a copy is needed; use copy=None for 'copy only if needed'.", "np.array([1, 2, 3], copy=None)", "copy=None"),
    ("np.linalg.det / svd on stacks", "Same API; NumPy 2 adds np.linalg.matrix_transpose / vecdot / vector_norm / matrix_norm.", "np.linalg.vector_norm(np.array([3.0, 4.0]))", "np.linalg.vector_norm"),
    ("np.array_api names", "np.concat, np.acos, np.asin, np.atan, np.permute_dims were added.", "(np.concat([[1], [2]]), np.acos(1.0))", "standard names"),
]
for nm, why, code, alt in _M:
    fn("[removed] " + nm, why, code if "\n" in code else code + "\n", reads="Old name no longer exists in NumPy 2. " + why, status="removed", alt=alt)

fn("NEP 50: Python scalars are 'weak'", "NumPy 2 type promotion changed: a Python int no longer widens an array's dtype.", '''
a = np.array([100], dtype=np.int8)
try:
    r = a + 1000
except OverflowError as e:
    r = "OverflowError: " + str(e)[:60]
(r, (a + 27).dtype, (np.array([1.0], np.float32) + 3.0).dtype)
''', reads="1000 does not fit int8 so it raises in NumPy 2 (old versions silently widened to int16). Python scalars adopt the ARRAY's dtype.")
fn("repr changes: np.float64(3.0)", "Scalar repr now shows the type in NumPy 2.", '''
(repr(np.float64(3.0)), str(np.float64(3.0)), repr(np.int64(5)))
''')
fn("default integer on Windows is int64", "NumPy 2 uses 64-bit default ints everywhere.", '''
np.array([1, 2, 3]).dtype
''')
fn("copy keyword", "np.array(copy=False) can now raise.", '''
a = np.arange(3)
try:
    np.array([1, 2, 3], copy=False)
except ValueError as e:
    msg = str(e)[:80]
msg
''')

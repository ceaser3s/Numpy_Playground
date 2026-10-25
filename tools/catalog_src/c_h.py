from ._core import cat, fn, use

# ---------------------------------------------------------------- completeness: remaining public names, grouped by topic
use("basics")
fn("a.mT", "NumPy 2: transpose of the LAST two axes (works for stacks of matrices; .T would reverse all axes).", '''
a = np.zeros((5, 2, 3))
(a.T.shape, a.mT.shape)
''')
fn("a.conjugate / np.conjugate", "Complex conjugate (a - bi).", '''
z = np.array([1 + 2j, 3 - 4j])
(z.conjugate(), np.conjugate(z))
''')
fn("a.getfield / a.setfield", "Read or write a typed field/slice of the raw bytes.", '''
a = np.array([1.5, 2.5])
a.getfield(np.float64, 0)
''')
fn("a.dumps / np.loads", "Pickle an array into bytes (dumps was removed in NumPy 2: use pickle).", '''
import pickle
b = pickle.dumps(np.arange(3))
pickle.loads(b)
''', status="removed", alt="pickle.dumps(a)")
fn("a.device / a.to_device / np.from_dlpack", "Array-API device support and DLPack exchange.", '''
a = np.arange(3)
(a.device, np.from_dlpack(a))
''')
fn("np.isfortran", "Is the array Fortran-ordered (and not C)?", '''
(np.isfortran(np.asfortranarray(np.arange(6).reshape(2, 3))), np.isfortran(np.arange(6).reshape(2, 3)))
''')
fn("np.little_endian", "True when this machine stores the least-significant byte first.", '''
np.little_endian
''')
fn("np.ScalarType", "Tuple of all NumPy and Python scalar types.", '''
len(np.ScalarType)
''')
fn("np.True_ / np.False_", "The NumPy boolean scalars.", '''
(np.True_, np.False_, type(np.True_).__name__)
''')
fn("np.mintypecode", "Smallest type code that can hold all the given codes.", '''
np.mintypecode(["f", "d"])
''')
fn("np.typename", "DEPRECATED (NumPy 2.5): description of a type code.", '''
np.dtype("d").name
''', status="deprecated", alt="np.dtype(code).name")
fn("np.getbufsize / np.setbufsize", "Size of the internal ufunc buffer.", '''
old = np.getbufsize()
np.setbufsize(old)
old
''')
fn("np.geterrcall / np.seterrcall", "Run your own function when a floating-point error occurs.", '''
def handler(kind, flag):
    print("floating point problem:", kind)

old = np.seterrcall(handler)
with np.errstate(divide="call"):
    np.array([1.0]) / 0
np.seterrcall(old)
''')

use("dtypes")
fn("np.void / np.flexible / np.unsignedinteger / np.complexfloating", "Abstract dtype classes for issubdtype checks.", '''
{"uint8 is unsignedinteger": np.issubdtype(np.uint8, np.unsignedinteger), "complex64 is complexfloating": np.issubdtype(np.complex64, np.complexfloating), "str_ is flexible": np.issubdtype(np.str_, np.flexible), "void is flexible": np.issubdtype(np.void, np.flexible)}
''')
fn("np.csingle / cdouble / clongdouble / np.long / np.ulong", "More type aliases.", '''
{n: np.dtype(getattr(np, n)).name for n in ("csingle", "cdouble", "clongdouble", "long", "ulong")}
''')

use("math")
fn("np.acosh / asinh / atanh", "NumPy 2 short names of the inverse hyperbolic functions.", '''
(np.acosh(2.0), np.asinh(1.0), np.atanh(0.5))
''')
fn("np.bitwise_not / invert / bitwise_invert", "Bitwise NOT has three names.", '''
a = np.array([0, 1, 5])
(np.bitwise_not(a), np.invert(a), np.bitwise_invert(a))
''')
fn("np.bitwise_or / bitwise_xor", "Function forms of | and ^.", '''
(np.bitwise_or(12, 10), np.bitwise_xor(12, 10))
''')
fn("np.bitwise_left_shift / bitwise_right_shift", "Array-API names for << and >>.", '''
(np.bitwise_left_shift(1, 4), np.bitwise_right_shift(64, 3))
''')
fn("np.logaddexp2", "log2(2**a + 2**b) computed stably.", '''
np.logaddexp2(np.array([1.0, 1000.0]), np.array([1.0, 1000.0]))
''')
fn("np.pow", "NumPy 2 alias of np.power.", '''
np.pow(np.array([2, 3]), 3)
''')
fn("np.matvec / np.vecmat", "NumPy 2.2: matrix-vector and vector-matrix products over stacks.", '''
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 1])
(np.matvec(A, v), np.vecmat(v, A))
''')
fn("a.conj() / np.conj", "Short names of conjugate.", '''
np.conj(np.array([1 + 1j]))
''')

use("indexing")
fn("np.fill_diagonal", "Set the diagonal in place.", '''
a = np.zeros((3, 3), dtype=int)
np.fill_diagonal(a, 7)
a
''')
fn("np.diag_indices_from / tril_indices_from / triu_indices_from", "Indices based on an existing array's shape.", '''
a = np.arange(9).reshape(3, 3)
(a[np.diag_indices_from(a)], a[np.tril_indices_from(a, k=-1)], a[np.triu_indices_from(a, k=1)])
''')
fn("np.index_exp", "Like np.s_ but returns a tuple (safe to use as an index).", '''
ix = np.index_exp[1:3, ::2]
(ix, np.arange(20).reshape(4, 5)[ix])
''')
fn("np.nested_iters", "Nested iteration over chosen axes.", '''
a = np.arange(6).reshape(2, 3)
outer, inner = np.nested_iters(a, [[0], [1]])
[[int(x) for x in inner] for _ in outer]
''')
fn("np.lib.Arrayterator", "Iterate over a big array in buffer-sized blocks.", '''
from numpy.lib import Arrayterator
a = np.arange(12).reshape(3, 4)
[blk.shape for blk in Arrayterator(a, 6)]
''')

use("fft")
fn("np.fft.irfft2 / irfftn", "Inverse real-input transforms in 2-D and N-D.", '''
a = np.random.default_rng(0).random((4, 6))
(np.allclose(np.fft.irfft2(np.fft.rfft2(a), s=a.shape), a), np.allclose(np.fft.irfftn(np.fft.rfftn(a), s=a.shape, axes=(0, 1)), a))
''')

use("random")
fn("rng.standard_gamma / standard_exponential", "Standard (unit-scale) distributions.", '''
rng = np.random.default_rng(30)
(np.round(rng.standard_gamma(2.0, 3), 3), np.round(rng.standard_exponential(3), 3))
''')
fn("rng.noncentral_chisquare / noncentral_f / logseries", "More distributions.", '''
rng = np.random.default_rng(31)
(np.round(rng.noncentral_chisquare(3, 2, 3), 3), np.round(rng.noncentral_f(3, 20, 2, 3), 3), rng.logseries(0.6, 3))
''')
fn("np.random.PCG64DXSM / BitGenerator", "Extra bit generator and the base class.", '''
g = np.random.Generator(np.random.PCG64DXSM(5))
(type(g.bit_generator).__name__, issubclass(np.random.PCG64DXSM, np.random.BitGenerator))
''')
fn("np.random.get_state / set_state (legacy)", "Save and restore the global legacy generator.", '''
np.random.seed(0)
st = np.random.get_state()
a = np.random.rand(2)
np.random.set_state(st)
(a, np.random.rand(2))
''')
fn("np.random.random_integers", "REMOVED (deprecated long ago): use integers / randint.", '''
np.random.default_rng(0).integers(1, 7, 5, endpoint=True)
''', status="removed", alt="rng.integers(low, high, endpoint=True)")
fn("np.random.get_bit_generator / set_bit_generator", "Swap the engine behind the legacy global functions.", '''
bg = np.random.get_bit_generator()
type(bg).__name__
''')

use("ma")
fn("np.ma.masked_greater_equal / masked_less_equal", "Mask by inclusive comparison.", '''
a = np.arange(6)
(np.ma.masked_greater_equal(a, 4), np.ma.masked_less_equal(a, 1))
''')
fn("np.ma.masked_all / masked_all_like", "Completely masked arrays (a blank canvas).", '''
m = np.ma.masked_all((2, 3))
m[0, 1] = 5
(m, np.ma.masked_all_like(np.zeros(2)).count())
''')
fn("np.ma.make_mask / make_mask_none / mask_or / nomask", "Build and combine mask arrays.", '''
m1 = np.ma.make_mask([1, 0, 0])
m2 = np.ma.make_mask([0, 0, 1])
(np.ma.mask_or(m1, m2), np.ma.make_mask_none((2,)), np.ma.nomask)
''')
fn("np.ma.fix_invalid", "Mask nan/inf and replace them by a fill value in one step.", '''
np.ma.fix_invalid(np.array([1.0, np.nan, np.inf]), fill_value=0)
''')
fn("np.ma.count_masked / isMaskedArray / is_mask / isarray", "Inspect masked arrays.", '''
m = np.ma.array([1, 2, 3], mask=[0, 1, 1])
(np.ma.count_masked(m), np.ma.isMaskedArray(m), np.ma.is_mask(m.mask), np.ma.isarray(m))
''')
fn("np.ma.default_fill_value / minimum_fill_value / maximum_fill_value / common_fill_value", "Fill-value rules per dtype.", '''
(np.ma.default_fill_value(1.5), np.ma.default_fill_value(3), np.ma.minimum_fill_value(np.array([1])), np.ma.maximum_fill_value(np.array([1])))
''')
fn("np.ma.anom / anomalies", "Deviation from the mean, ignoring masked.", '''
np.ma.anom(np.ma.array([1.0, 2.0, 3.0, 10.0], mask=[0, 0, 0, 1]))
''')
fn("np.ma.allequal / allclose", "Compare masked arrays.", '''
a = np.ma.array([1, 2, 3], mask=[0, 0, 1])
b = np.ma.array([1, 2, 99], mask=[0, 0, 1])
(np.ma.allequal(a, b), np.ma.allclose(a, b))
''')
fn("np.ma.compress_rows / compress_cols / compress_rowcols / compress_nd", "Drop rows or columns that contain masked values.", '''
m = np.ma.array(np.arange(9).reshape(3, 3), mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
(np.ma.compress_rows(m), np.ma.compress_cols(m))
''')
fn("np.ma.flatnotmasked_edges / flatnotmasked_contiguous", "Edges of valid stretches in a 1-D mask.", '''
m = np.ma.array(np.arange(8), mask=[1, 0, 0, 1, 0, 0, 0, 1])
(np.ma.flatnotmasked_edges(m), np.ma.flatnotmasked_contiguous(m))
''')
fn("np.ma.mr_ / np.ma.masked_print_option / MaskedArray / MaskType", "Helpers and classes.", '''
(np.ma.mr_[[1, 2], 3].mask.tolist(), str(np.ma.masked_print_option), np.ma.MaskedArray.__name__, np.ma.MaskType.__name__)
''')
fn("np.ma.innerproduct / outerproduct / bitwise_or / conjugate", "More masked functions.", '''
a = np.ma.array([1, 2, 3], mask=[0, 0, 1])
(np.ma.outerproduct(a, [1, 2]), np.ma.innerproduct(a, [1, 1, 1]))
''')
fn("np.ma.flatten_mask / flatten_structured_array / fromflex / mvoid", "Structured masked data helpers.", '''
m = np.ma.array([(1, 2.0), (3, 4.0)], mask=[(0, 1), (0, 0)], dtype=[("a", int), ("b", float)])
(np.ma.flatten_mask(m.mask), type(m[0]).__name__)
''')
fn("np.ma.MAError / MaskError / masked_singleton / ids", "Exceptions and singletons.", '''
(issubclass(np.ma.MaskError, np.ma.MAError), np.ma.masked_singleton is np.ma.masked, len(np.ma.ids(np.ma.array([1]))))
''')

use("poly")
fn("np.polynomial.Hermite / HermiteE / Laguerre", "Classes for the other orthogonal polynomial families.", '''
from numpy.polynomial import Hermite, HermiteE, Laguerre
(Hermite([0, 0, 1])(2.0), HermiteE([0, 0, 1])(2.0), Laguerre([0, 0, 1])(2.0))
''')
fn("np.polynomial.set_default_printstyle", "Show polynomials with ascii or unicode powers.", '''
from numpy.polynomial import Polynomial as P, set_default_printstyle
set_default_printstyle("ascii")
s = str(P([1, 2, 3]))
set_default_printstyle("unicode")
s
''')

use("strings")
fn("np.char.rsplit / narray / asnarray", "Remaining legacy char functions.", '''
np.char.rsplit(np.array(["a b c"]), maxsplit=1)
''')

use("utils")
fn("np.testing.assert_ / assert_almost_equal / assert_approx_equal", "Basic assertions.", '''
np.testing.assert_(1 < 2)
np.testing.assert_almost_equal(0.30000001, 0.3, decimal=6)
np.testing.assert_approx_equal(123456, 123457, significant=5)
"ok"
''')
fn("np.testing.assert_raises_regex", "Check that an error with a matching message is raised.", '''
with np.testing.assert_raises_regex(ValueError, "broadcast"):
    np.ones(3) + np.ones(4)
"ok - the expected error was raised"
''')
fn("np.testing.assert_warns / suppress_warnings / assert_no_warnings", "DEPRECATED in NumPy 2.4: use the standard warnings module (or pytest).", '''
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    warnings.warn("demo warning", RuntimeWarning)
(len(w), w[0].category.__name__)
''', status="deprecated", alt="warnings.catch_warnings / pytest.warns")
fn("np.testing.assert_array_almost_equal_nulp / assert_array_max_ulp", "Float comparison in units in the last place.", '''
np.testing.assert_array_max_ulp(np.array([1.0]), np.nextafter(np.array([1.0]), 2.0), maxulp=2)
"within 2 ulp"
''')
fn("np.testing.assert_string_equal / print_assert_equal", "Text comparison helpers.", '''
np.testing.assert_string_equal("abc", "abc")
"ok"
''')
fn("np.testing flags (IS_WASM, IS_64BIT ...)", "Platform flags used by NumPy's own tests.", '''
{"IS_WASM": np.testing.IS_WASM, "IS_64BIT": np.testing.IS_64BIT, "HAS_REFCOUNT": np.testing.HAS_REFCOUNT}
''')

# ---------------------------------------------------------------- recipes: short real-world patterns
cat("recipes", "Recipes & idioms", "Short, practical NumPy patterns you will use again and again.", "🍳",
    intro="Small complete solutions to everyday tasks - each one is vectorised (no Python loops). Good starting points for your own code.")

fn("count how many values fall in a range", "Mask + sum.", '''
age = students[:, 1]
np.sum((age >= 20) & (age <= 22))
''')
fn("replace outliers with the median", "Clip values beyond 3 standard deviations.", '''
x = houses[:, 3].copy()
z = (x - x.mean()) / x.std()
x[np.abs(z) > 3] = np.median(x)
(int((np.abs(z) > 3).sum()), x.max())
''')
fn("fill nan with the column mean", "Per-column imputation.", '''
s = students[:, 3:9].copy()
means = np.nanmean(s, axis=0)
idx = np.where(np.isnan(s))
s[idx] = means[idx[1]]
(int(np.isnan(s).sum()), np.round(means, 2))
''', reads="np.where(np.isnan(s)) gives (rows, cols); means[cols] picks each missing cell's column mean.")
fn("drop rows that contain nan", "Keep only complete rows.", '''
s = students[:, 3:9]
clean = s[~np.isnan(s).any(axis=1)]
(s.shape, clean.shape)
''')
fn("one-hot encoding", "Turn labels into indicator columns.", '''
labels = np.array([2, 0, 1, 2])
np.eye(3, dtype=int)[labels]
''')
fn("group sums without a loop", "Total per group with bincount.", '''
g = students[:, 2].astype(int)   # gender 0/1
m = np.nan_to_num(students[:, 5])
np.bincount(g, weights=m) / np.bincount(g)
''', reads="Average math score for gender code 0 (M) and 1 (F). (blank scores are counted as 0 here - see the project for the careful version)")
fn("group statistics with masks", "Mean per category using boolean masks.", '''
g = students[:, 2]
{"M": round(float(np.nanmean(students[g == 0, 5])), 2), "F": round(float(np.nanmean(students[g == 1, 5])), 2)}
''')
fn("rank with ties (dense)", "Rank values 1..k.", '''
a = np.array([30, 10, 20, 10])
np.unique(a, return_inverse=True)[1] + 1
''')
fn("running maximum / drawdown", "Peak-to-trough decline.", '''
p = np.array([100, 120, 90, 130, 80, 140.0])
peak = np.maximum.accumulate(p)
(peak, np.round((p - peak) / peak, 3))
''')
fn("detect runs of consecutive True", "Find start and length of each run.", '''
b = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
d = np.diff(np.concatenate([[0], b.astype(int), [0]]))
starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
np.column_stack([starts, ends - starts])
''', reads="Columns: start index, run length.")
fn("find peaks in a signal", "Local maxima by comparing neighbours.", '''
y = np.array([0, 2, 1, 4, 3, 3, 5, 2.0])
np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
''')
fn("cosine similarity matrix", "Compare every row with every other.", '''
x = houses[:4, :3]
n = x / np.linalg.norm(x, axis=1, keepdims=True)
np.round(n @ n.T, 4)
''')
fn("pairwise distances", "Euclidean distances between all rows.", '''
x = houses[:4, :3]
np.round(np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)), 1)
''')
fn("nearest neighbour", "Closest other point for each point.", '''
x = houses[:6, :2]
d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
np.fill_diagonal(d, np.inf)
d.argmin(axis=1)
''')
fn("normalise to unit length", "Divide each row by its norm.", '''
v = np.array([[3.0, 4.0], [5.0, 12.0]])
v / np.linalg.norm(v, axis=1, keepdims=True)
''')
fn("matrix of all pairs (cartesian product)", "Every combination of two lists.", '''
a, b = np.array([1, 2, 3]), np.array([10, 20])
np.stack(np.meshgrid(a, b, indexing="ij"), -1).reshape(-1, 2)
''')
fn("cumulative percentage (Pareto)", "Cumulative share of sorted contributions.", '''
v = np.array([50, 30, 10, 7, 3.0])
np.round(np.cumsum(np.sort(v)[::-1]) / v.sum() * 100, 1)
''')
fn("checkerboard pattern", "Indices parity.", '''
i, j = np.indices((5, 5))
(i + j) % 2
''')
fn("circle mask on a grid", "Select pixels inside a disc.", '''
y, x = np.ogrid[-5:6, -5:6]
(x ** 2 + y ** 2 <= 16).astype(int)
''')
fn("binning values into labels", "Bucket ages.", '''
age = np.array([17, 25, 34, 52, 71])
labels = np.array(["<18", "18-34", "35-59", "60+"])
labels[np.digitize(age, [18, 35, 60])]
''')
fn("compare two arrays in a table", "Side-by-side view with column_stack.", '''
a = np.arange(5); b = a ** 2
np.column_stack([a, b, b - a])
''')
fn("polynomial features", "Add squares and interactions.", '''
x = houses[:3, :2]
np.column_stack([x, x ** 2, x[:, 0] * x[:, 1]])
''')
fn("weighted random choice", "Sample categories by probability.", '''
rng = np.random.default_rng(0)
s = rng.choice(["A", "B", "C"], 10000, p=[0.5, 0.3, 0.2])
np.unique(s, return_counts=True)
''')
fn("shuffle two arrays together", "Same permutation for X and y.", '''
rng = np.random.default_rng(1)
X = np.arange(5)[:, None] * 10; y = np.arange(5)
p = rng.permutation(5)
(X[p].ravel(), y[p])
''')
fn("batching an array", "Split a long array into mini-batches.", '''
a = np.arange(10)
[b.tolist() for b in np.array_split(a, 3)]
''')
fn("flatten a list of arrays", "Concatenate quickly.", '''
np.concatenate([np.arange(2), np.arange(3), np.arange(1)])
''')
fn("most common value (mode) of each column", "Column-wise mode.", '''
a = np.array([[1, 2], [1, 3], [2, 3], [1, 3]])
[int(np.bincount(c).argmax()) for c in a.T]
''')
fn("percent change", "Relative change between consecutive values.", '''
p = np.array([100, 110, 99, 120.0])
np.round(np.diff(p) / p[:-1] * 100, 2)
''')
fn("moving window argmax", "Position of the max in each window.", '''
from numpy.lib.stride_tricks import sliding_window_view as swv
swv(np.array([1, 5, 2, 8, 3, 4]), 3).argmax(axis=1)
''')
fn("lookup table with fancy indexing", "Map codes to values.", '''
lut = np.array([0, 10, 20, 30])
codes = np.array([3, 0, 2, 2, 1])
lut[codes]
''')
fn("unique rows count", "How many distinct rows?", '''
a = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
np.unique(a, axis=0).shape[0]
''')
fn("matrix with a given condition number (random)", "Generate test matrices.", '''
rng = np.random.default_rng(0)
U, _ = np.linalg.qr(rng.normal(size=(4, 4)))
V, _ = np.linalg.qr(rng.normal(size=(4, 4)))
A = U @ np.diag([10, 5, 2, 1.0]) @ V.T
round(float(np.linalg.cond(A)), 3)
''')
fn("save a figure of data", "Scatter of the regression data (needs matplotlib).", '''
plt.scatter(houses[:, 0], houses[:, 3], s=8, alpha=.6)
plt.xlabel("area_sqft"); plt.ylabel("price_k")
plt.title("houses: price vs area")
''', needs=["matplotlib"])

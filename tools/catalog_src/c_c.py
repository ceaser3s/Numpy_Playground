from ._core import cat, fn

# =====================================================================================================  7  ARITHMETIC & MATH
cat("math", "Arithmetic & math functions", "Element-wise operators and universal functions: powers, roots, logs, rounding, extremes.", "➕",
    intro="Universal functions (ufuncs) apply an operation to EVERY element at once, in fast compiled code. "
          "The operators + - * / // % ** are ufuncs in disguise. No loops needed.")

fn("a + b  (np.add)", "Element-wise addition. Also works with a single number (broadcasting).", '''
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
(a + b, np.add(a, b), a + 100)
''', reads="Each element is added to the element in the same position; the number 100 is stretched to match.")
fn("a - b  (np.subtract)", "Element-wise subtraction.", '''
np.subtract(np.array([10, 20, 30]), np.array([1, 2, 3]))
''')
fn("a * b  (np.multiply)", "Element-wise multiplication - NOT matrix multiplication.", '''
a = np.array([[1, 2], [3, 4]])
(a * a, a @ a)
''', reads="* multiplies matching elements. @ (matmul) is the matrix product - very different!")
fn("a / b  (np.divide / true_divide)", "True division: always returns floats.", '''
np.array([7, 8, 9]) / 2
''')
fn("a // b  (np.floor_divide)", "Division rounded DOWN (toward -infinity).", '''
a = np.array([7, -7, 8])
(a // 2, np.floor_divide(a, 2))
''', reads="-7 // 2 is -4, not -3: floor rounds toward minus infinity.")
fn("a % b  (np.mod / np.remainder)", "Remainder with the sign of the divisor.", '''
a = np.array([7, -7, 8])
(a % 3, np.remainder(a, -3))
''')
fn("np.fmod", "Remainder with the sign of the DIVIDEND (like C).", '''
(np.fmod(np.array([7, -7]), 3), np.mod(np.array([7, -7]), 3))
''')
fn("np.divmod", "Quotient and remainder in one call.", '''
np.divmod(np.array([7, 10, 25]), 4)
''')
fn("a ** b  (np.power)", "Raise to a power.", '''
a = np.array([1, 2, 3, 4])
(a ** 2, 2 ** a, np.power(a, 0.5))
''')
fn("np.float_power", "Power computed in float64 (avoids integer overflow / negative-power errors).", '''
np.float_power(np.array([2, 3]), -1)
''')
fn("np.negative / np.positive", "Unary minus / plus.", '''
(np.negative([1, -2, 3]), -np.array([1, -2, 3]))
''')
fn("np.abs / np.absolute", "Absolute value (also the magnitude of complex numbers).", '''
(np.abs(np.array([-3, 4, -5])), np.abs(3 + 4j))
''')
fn("np.fabs", "Absolute value, always as float.", '''
np.fabs(np.array([-3, 4]))
''')
fn("np.sign", "-1, 0 or +1 according to the sign.", '''
np.sign(np.array([-5.5, 0, 12]))
''')
fn("np.reciprocal", "1 / x.", '''
np.reciprocal(np.array([2.0, 4.0, 0.5]))
''')
fn("np.sqrt", "Square root (nan for negatives, with a warning).", '''
(np.sqrt(np.array([4, 9, 2])), np.sqrt(np.array([-1.0])))
''')
fn("np.cbrt", "Cube root (works for negatives).", '''
np.cbrt(np.array([8, -27, 2]))
''')
fn("np.square", "x * x.", '''
np.square(np.array([1, -2, 3]))
''')
fn("np.exp", "e ** x.", '''
np.exp(np.array([0, 1, 2]))
''')
fn("np.exp2", "2 ** x.", '''
np.exp2(np.array([0, 3, 10]))
''')
fn("np.expm1", "exp(x) - 1, accurate for tiny x.", '''
x = 1e-12
(np.expm1(x), np.exp(x) - 1)
''', reads="The naive version loses most digits; expm1 keeps them.")
fn("np.log", "Natural logarithm.", '''
np.log(np.array([1, np.e, 10]))
''')
fn("np.log10 / np.log2", "Base-10 and base-2 logarithms.", '''
(np.log10(np.array([1, 10, 1000])), np.log2(np.array([1, 8, 1024])))
''')
fn("np.log1p", "log(1 + x), accurate for tiny x.", '''
np.log1p(np.array([1e-10, 1.0]))
''')
fn("np.logaddexp", "log(exp(a) + exp(b)) without overflow.", '''
with np.errstate(over="ignore"):
    naive = np.log(np.exp(1000) + np.exp(1000))
(np.logaddexp(1000, 1000), naive)
''', reads="The naive formula overflows to inf; logaddexp is stable.")
fn("np.hypot", "sqrt(a**2 + b**2) - the hypotenuse.", '''
np.hypot(np.array([3, 5]), np.array([4, 12]))
''')
fn("np.round / np.around", "Round to a number of decimals (halves go to the nearest EVEN number).", '''
a = np.array([0.5, 1.5, 2.5, 3.14159])
(np.round(a), np.round(a, 2), np.round(1234.5, -2))
''', reads="Banker's rounding: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2. Negative decimals round to tens, hundreds ...")
fn("np.rint", "Round to the nearest integer (as float).", '''
np.rint(np.array([-1.5, 1.5, 2.5]))
''')
fn("np.floor", "Round DOWN.", '''
np.floor(np.array([-1.5, 1.5, 2.9]))
''')
fn("np.ceil", "Round UP.", '''
np.ceil(np.array([-1.5, 1.2, 2.0]))
''')
fn("np.trunc", "Drop the decimals (round toward zero).", '''
np.trunc(np.array([-1.7, 1.7]))
''')
fn("np.fix", "DEPRECATED (NumPy 2.5): same as np.trunc.", '''
np.trunc(np.array([-1.7, 1.7]))
''', status="deprecated", alt="np.trunc")
fn("np.clip", "Limit values to a range.", '''
np.clip(np.array([-5, 3, 12, 8]), 0, 10)
''', reads="Anything below 0 becomes 0, anything above 10 becomes 10.", params=["a_min, a_max - either may be None", "out - write the result in place"])
fn("np.maximum / np.minimum", "Element-wise larger / smaller of two arrays (nan propagates).", '''
a, b = np.array([1, 8, 3]), np.array([4, 2, 9])
(np.maximum(a, b), np.minimum(a, b))
''', reads="Do not confuse with np.max (reduces one array to a single number).")
fn("np.fmax / np.fmin", "Like maximum/minimum but IGNORE nan.", '''
a, b = np.array([1, np.nan]), np.array([np.nan, 2])
(np.maximum(a, b), np.fmax(a, b))
''')
fn("np.gcd / np.lcm", "Greatest common divisor / least common multiple.", '''
(np.gcd(np.array([12, 18, 100]), 8), np.lcm(4, np.array([6, 10])))
''')
fn("np.heaviside", "Step function: 0 for x<0, given value at 0, 1 for x>0.", '''
np.heaviside(np.array([-2, 0, 3]), 0.5)
''')
fn("np.copysign", "Magnitude of the first, sign of the second.", '''
np.copysign(np.array([3.0, 3.0]), np.array([-1.0, 1.0]))
''')
fn("np.signbit", "True where the sign bit is set (detects -0.0).", '''
np.signbit(np.array([-1.0, 0.0, -0.0, 2.0]))
''')
fn("np.nextafter / np.spacing", "Next representable float / gap size.", '''
(np.nextafter(1.0, 2.0) - 1.0, np.spacing(1.0), np.spacing(1e10))
''', reads="Floats get sparser as numbers grow: the gap near 1e10 is far bigger than near 1.")
fn("np.frexp / np.ldexp", "Split a float into mantissa and exponent, and back.", '''
m, e = np.frexp(np.array([8.0, 0.75]))
(m, e, np.ldexp(m, e))
''')
fn("np.modf", "Fractional and integer parts.", '''
np.modf(np.array([3.75, -2.5]))
''')
fn("np.sinc", "Normalised sinc: sin(pi x) / (pi x).", '''
np.sinc(np.array([0, 0.5, 1]))
''')
fn("np.i0", "Modified Bessel function of order 0.", '''
np.i0(np.array([0.0, 1.0, 2.0]))
''')
fn("np.cumulative_sum", "NumPy 2.1 array-API name for cumsum (with include_initial).", '''
np.cumulative_sum(np.array([1, 2, 3, 4]), include_initial=True)
''')
fn("np.cumsum", "Running total.", '''
a = np.array([[1, 2, 3], [4, 5, 6]])
(np.cumsum(a), np.cumsum(a, axis=0), np.cumsum(a, axis=1))
''', reads="Without axis the array is flattened first.")
fn("np.cumprod", "Running product.", '''
np.cumprod(np.array([1, 2, 3, 4, 5]))
''')
fn("np.cumulative_prod", "NumPy 2.1 name for cumprod.", '''
np.cumulative_prod(np.array([1, 2, 3, 4]))
''')
fn("np.prod", "Product of all elements.", '''
(np.prod(np.arange(1, 6)), np.prod(np.array([[1, 2], [3, 4]]), axis=0))
''')
fn("np.sum", "Sum of elements, optionally along an axis.", '''
a = np.array([[1, 2, 3], [4, 5, 6]])
{"all": a.sum(), "axis=0 (down columns)": a.sum(axis=0), "axis=1 (across rows)": a.sum(axis=1), "keepdims": a.sum(axis=1, keepdims=True)}
''', reads="axis=0 collapses the ROWS (gives one value per column); axis=1 collapses the COLUMNS (one value per row). keepdims=True keeps the shape broadcast-friendly.",
   params=["axis - int, tuple or None", "keepdims=True - keep reduced axes with length 1", "dtype - accumulator type", "where - boolean mask of elements to include", "initial - starting value"])
fn("np.sum (where= / initial=)", "Only sum selected elements.", '''
a = np.array([1, -2, 3, -4, 5])
(np.sum(a, where=a > 0), np.sum(a, initial=100))
''')
fn("np.sum (dtype overflow)", "Small ints accumulate in a bigger type by default.", '''
a = np.full(300, 200, dtype=np.uint8)
(np.sum(a), np.sum(a, dtype=np.uint8), a.cumsum()[-1])
''', reads="300 x 200 = 60000 does not fit in uint8, but sum() automatically uses uint64. Forcing dtype=uint8 wraps around.")
fn("np.add.reduce / np.multiply.accumulate", "Every ufunc can reduce or accumulate (details in Ufuncs & broadcasting).", '''
a = np.array([1, 2, 3, 4])
(np.add.reduce(a), np.multiply.accumulate(a))
''')
fn("np.angle / np.conj / np.real / np.imag", "Complex-number helpers.", '''
z = np.array([3 + 4j, -1j])
{"abs": np.abs(z), "angle (rad)": np.angle(z), "conj": np.conj(z), "real": np.real(z), "imag": np.imag(z)}
''')
fn("np.real_if_close", "Drop a negligible imaginary part.", '''
np.real_if_close(np.array([1 + 1e-15j, 2 + 0j]))
''')
fn("np.roots of unity (complex exp)", "Complex exponentials build the unit circle.", '''
n = 4
np.round(np.exp(2j * np.pi * np.arange(n) / n), 3)
''')
fn("in-place operators (+=, *=)", "Update an array without creating a new one.", '''
a = np.array([1.0, 2.0, 3.0])
b = a
a *= 10
(a, b)
''', reads="b changed too because += / *= modify the SAME array object. a = a * 10 would create a new one.")
fn("out= argument", "Write the result into an existing array (saves memory).", '''
a = np.arange(5.0)
out = np.empty(5)
np.multiply(a, 3, out=out)
out
''')
fn("where= argument (ufunc)", "Only compute where the mask is True.", '''
a = np.array([1.0, 4.0, 9.0, 16.0])
out = np.full(4, -1.0)
np.sqrt(a, out=out, where=a > 5)
out
''', reads="Untouched positions keep the value that was already in out.")
fn("np.errstate", "Control what happens on divide-by-zero / overflow / invalid.", '''
with np.errstate(divide="ignore", invalid="ignore"):
    r = np.array([1.0, 0.0, -1.0]) / 0.0
r
''', reads="1/0 = inf, -1/0 = -inf, 0/0 = nan. errstate silences the warnings (or can turn them into errors with 'raise').")
fn("np.seterr / np.geterr", "Set error handling globally.", '''
old = np.geterr()
np.seterr(all="ignore")
r = np.array([1.0]) / 0
np.seterr(**old)
(r, old)
''')
fn("np.isnan-safe math with np.nan", "nan spreads through calculations.", '''
a = np.array([1.0, np.nan, 3.0])
(a + 1, a.sum(), np.nansum(a))
''')
fn("integer division by zero", "Integer / 0 gives 0 with a warning (not an error).", '''
with np.errstate(divide="ignore"):
    r = np.array([5, 0]) // 0
r
''')
fn("np.floor_divide vs int()", "floor_divide rounds down; astype(int) truncates.", '''
a = np.array([-3.5, 3.5])
(a // 1, a.astype(int))
''')
fn("np.bitwise_and / or / xor / invert", "Bit-level operations on integers.", '''
a, b = np.array([12, 10]), np.array([10, 6])
{"and": a & b, "or": a | b, "xor": a ^ b, "not": ~a}
''', reads="12 = 1100, 10 = 1010 -> and = 1000 = 8, or = 1110 = 14, xor = 0110 = 6.")
fn("np.left_shift / right_shift", "Shift bits (multiply / divide by powers of 2).", '''
a = np.array([1, 2, 3])
(a << 3, np.array([64, 100]) >> 2)
''')
fn("np.bitwise_count", "NumPy 2.1: number of 1-bits (popcount).", '''
np.bitwise_count(np.array([0, 1, 7, 255], dtype=np.uint8))
''')
fn("np.unpackbits / np.packbits", "Convert between bytes and bit arrays.", '''
b = np.unpackbits(np.array([5, 255], dtype=np.uint8))
(b, np.packbits(b))
''')
fn("np.binary_repr / np.base_repr", "Number to a binary / any-base text.", '''
(np.binary_repr(10), np.binary_repr(-5, width=8), np.base_repr(255, base=16))
''')

# =====================================================================================================  8  TRIG
cat("trig", "Trigonometry & hyperbolic", "Angles in radians, trig functions and their inverses.", "📐",
    intro="NumPy trig functions expect RADIANS. 180 degrees = pi radians. Use np.deg2rad / np.rad2deg to convert.")

fn("np.pi / np.e", "Constants.", '''
(np.pi, np.e)
''')
fn("np.sin / np.cos / np.tan", "Basic trig functions (input in radians).", '''
x = np.array([0, np.pi / 6, np.pi / 4, np.pi / 2])
{"sin": np.round(np.sin(x), 4), "cos": np.round(np.cos(x), 4), "tan": np.round(np.tan(x[:3]), 4)}
''')
fn("np.arcsin / arccos / arctan", "Inverse trig (result in radians).", '''
np.rad2deg(np.array([np.arcsin(0.5), np.arccos(0.5), np.arctan(1.0)]))
''')
fn("np.arctan2", "Angle of the point (x, y) in the correct quadrant.", '''
np.rad2deg(np.arctan2(np.array([1, 1, -1, -1]), np.array([1, -1, -1, 1])))
''', reads="Unlike arctan(y/x), arctan2(y, x) knows which quadrant the point is in. Note: first argument is y!")
fn("np.sinh / cosh / tanh", "Hyperbolic functions.", '''
x = np.array([-1.0, 0.0, 1.0])
(np.sinh(x), np.cosh(x), np.tanh(x))
''', reads="tanh squashes any input into (-1, 1) - used as a neural-network activation.")
fn("np.arcsinh / arccosh / arctanh", "Inverse hyperbolic functions.", '''
(np.arcsinh(1.0), np.arccosh(2.0), np.arctanh(0.5))
''')
fn("np.deg2rad / np.radians", "Degrees to radians.", '''
np.deg2rad(np.array([0, 90, 180, 360]))
''')
fn("np.rad2deg / np.degrees", "Radians to degrees.", '''
np.rad2deg(np.array([0, np.pi / 2, np.pi]))
''')
fn("np.unwrap", "Remove 2*pi jumps from a phase sequence.", '''
p = np.array([0, 3, 6, 9, 12]) % (2 * np.pi)
(np.round(p, 2), np.round(np.unwrap(p), 2))
''')
fn("np.acos / asin / atan / atan2", "NumPy 2 short aliases of the inverse functions.", '''
(np.acos(0.0), np.asin(1.0), np.atan(1.0), np.atan2(1.0, 1.0))
''')
fn("Pythagorean identity check", "sin^2 + cos^2 = 1 for every angle.", '''
x = np.linspace(0, 2 * np.pi, 7)
np.allclose(np.sin(x) ** 2 + np.cos(x) ** 2, 1)
''')
fn("sine wave sampling", "Typical use: sample a signal.", '''
t = np.linspace(0, 1, 9)
np.round(np.sin(2 * np.pi * 2 * t), 3)
''', reads="Two full cycles over one second (frequency 2 Hz), sampled at 9 points.")
fn("polar <-> cartesian", "Convert (r, angle) to (x, y) and back.", '''
r, th = np.array([1.0, 2.0]), np.array([0, np.pi / 2])
x, y = r * np.cos(th), r * np.sin(th)
(np.round(x, 6), np.round(y, 6), np.hypot(x, y), np.arctan2(y, x))
''')
fn("plot of sin and cos", "A quick chart (needs matplotlib).", '''
x = np.linspace(0, 2 * np.pi, 200)
plt.plot(x, np.sin(x), label="sin")
plt.plot(x, np.cos(x), label="cos")
plt.legend()
plt.title("sin and cos")
''', needs=["matplotlib"])

# =====================================================================================================  9  STATISTICS
cat("stats", "Statistics & aggregation", "Mean, median, spread, percentiles, correlation, histograms, counting, nan-aware versions.", "📊",
    intro="Aggregations collapse many values into fewer. The AXIS argument says which dimension disappears: axis=0 -> down the rows (one result per column); axis=1 -> across the columns (one result per row). "
          "Missing values (nan) poison normal functions - use the nan-versions. Datasets available: students, sensor, houses.")

fn("np.mean", "Average.", '''
a = np.array([[1, 2, 3], [4, 5, 6]])
{"all": np.mean(a), "per column": a.mean(axis=0), "per row": a.mean(axis=1)}
''')
fn("np.average", "Weighted average.", '''
np.average([70, 80, 90], weights=[1, 1, 3])
''', reads="(70 + 80 + 3 x 90) / 5 = 84. np.mean cannot take weights.")
fn("np.median", "Middle value - not affected by outliers.", '''
a = np.array([1, 2, 3, 4, 1000])
(np.mean(a), np.median(a))
''')
fn("np.std", "Standard deviation (population by default, ddof=0).", '''
a = np.array([2, 4, 4, 4, 5, 5, 7, 9])
(np.std(a), np.std(a, ddof=1))
''', reads="ddof=1 gives the SAMPLE standard deviation (divides by n-1) - what pandas uses by default.", params=["ddof - delta degrees of freedom (0 population, 1 sample)"])
fn("np.var", "Variance (std squared).", '''
a = np.array([2, 4, 4, 4, 5, 5, 7, 9])
(np.var(a), np.var(a, ddof=1))
''')
fn("np.min / np.max", "Smallest / largest value.", '''
a = np.array([[3, 9, 1], [7, 2, 8]])
(a.min(), a.max(), a.max(axis=0), a.min(axis=1))
''')
fn("np.amin / np.amax", "Older names of min / max (still available).", '''
(np.amin(np.array([4, 2, 8])), np.amax(np.array([4, 2, 8])))
''')
fn("np.ptp", "Peak-to-peak: max - min (range).", '''
np.ptp(np.array([[1, 10], [4, 6]]), axis=1)
''')
fn("np.argmin / np.argmax", "POSITION of the smallest / largest value.", '''
a = np.array([3, 9, 1, 9, 5])
(np.argmin(a), np.argmax(a), np.argmax(np.array([[1, 5], [7, 2]]), axis=1))
''', reads="For ties the FIRST position wins.")
fn("np.percentile", "Value below which a given % of the data falls.", '''
a = np.arange(1, 101)
np.percentile(a, [25, 50, 75, 90])
''', reads="Percentiles use 0-100; np.quantile uses 0-1.", params=["q - percentile(s) 0-100", "method - 'linear' (default), 'lower', 'higher', 'nearest', 'midpoint'", "axis"])
fn("np.quantile", "Same as percentile with fractions.", '''
np.quantile(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [0.1, 0.5, 0.9])
''')
fn("np.percentile (method=)", "Different definitions of a percentile.", '''
a = np.array([1, 2, 3, 4])
{m: float(np.percentile(a, 40, method=m)) for m in ("linear", "lower", "higher", "nearest", "midpoint")}
''')
fn("np.nanmean", "Mean ignoring nan.", '''
a = students[:, 5]
(np.mean(a), np.nanmean(a))
''', reads="students[:, 5] is the math column with blanks. The plain mean is nan; nanmean skips the blanks.")
fn("np.nansum / nanprod", "Sum / product ignoring nan.", '''
a = np.array([1.0, np.nan, 3.0])
(np.nansum(a), np.nanprod(a))
''')
fn("np.nanmedian", "Median ignoring nan.", '''
np.nanmedian(students[:, 6])
''')
fn("np.nanstd / nanvar", "Spread ignoring nan.", '''
(np.nanstd(students[:, 5]), np.nanvar(students[:, 5]))
''')
fn("np.nanmin / nanmax", "Extremes ignoring nan.", '''
(np.nanmin(students[:, 3]), np.nanmax(students[:, 3]))
''')
fn("np.nanargmin / nanargmax", "Positions of extremes ignoring nan.", '''
(np.nanargmin(students[:, 5]), np.nanargmax(students[:, 5]))
''')
fn("np.nanpercentile / nanquantile", "Percentiles ignoring nan.", '''
(np.nanpercentile(students[:, 5], [25, 50, 75]), np.nanquantile(students[:, 5], 0.9))
''')
fn("np.nancumsum / nancumprod", "Running totals treating nan as 0 / 1.", '''
(np.nancumsum(np.array([1, np.nan, 3])), np.nancumprod(np.array([2, np.nan, 3])))
''')
fn("np.nan_to_num", "Replace nan / inf with numbers.", '''
np.nan_to_num(np.array([1.0, np.nan, np.inf, -np.inf]), nan=0.0, posinf=999, neginf=-999)
''')
fn("np.count_nonzero", "Count non-zero / True elements.", '''
a = np.array([[0, 1, 2], [0, 0, 3]])
(np.count_nonzero(a), np.count_nonzero(a, axis=1), np.count_nonzero(a > 1))
''')
fn("np.corrcoef", "Correlation matrix (-1 to +1).", '''
np.round(np.corrcoef(houses[:, 0], houses[:, 3]), 3)
''', reads="Off-diagonal ~ +0.9: bigger houses cost more. The diagonal is always 1 (a variable with itself).")
fn("np.corrcoef (many variables)", "Rows are variables by default (rowvar=True).", '''
np.round(np.corrcoef(houses.T), 2)
''', reads="houses.T makes each COLUMN a variable: area, bedrooms, age, price.")
fn("np.cov", "Covariance matrix.", '''
np.round(np.cov(houses[:, 0], houses[:, 3]), 1)
''')
fn("np.correlate", "Cross-correlation of two 1-D sequences.", '''
np.correlate([1, 2, 3], [0, 1, 0.5], mode="full")
''')
fn("np.histogram", "Count values in equal-width bins.", '''
counts, edges = np.histogram(students[:, 5][~np.isnan(students[:, 5])], bins=5, range=(0, 100))
(counts, edges)
''', reads="5 bins of width 20: [0,20) [20,40) ... [80,100]. The LAST bin includes its right edge.", params=["bins - number or edges", "range", "density=True - probability density", "weights"])
fn("np.histogram_bin_edges", "Only the bin edges.", '''
np.histogram_bin_edges(np.arange(100), bins="auto")
''')
fn("np.histogram2d", "2-D histogram (counts in a grid).", '''
H, xe, ye = np.histogram2d(houses[:, 0], houses[:, 3], bins=4)
H
''')
fn("np.histogramdd", "Histogram in several dimensions.", '''
H, edges = np.histogramdd(houses[:, [0, 2, 3]], bins=(3, 3, 3))
H.shape
''')
fn("np.bincount", "Count occurrences of each non-negative integer.", '''
np.bincount(np.array([0, 1, 1, 3, 3, 3]))
''', reads="Index = value, entry = how many times it occurred. Very fast for small integer data.")
fn("np.bincount (weights)", "Sum weights per integer label (a fast group-by).", '''
labels = np.array([0, 1, 0, 2, 1, 0])
values = np.array([10.0, 20, 30, 40, 50, 60])
np.bincount(labels, weights=values)
''', reads="Total value per label: label 0 -> 10+30+60 = 100, label 1 -> 70, label 2 -> 40.")
fn("np.digitize", "Which bin does each value belong to?", '''
np.digitize(np.array([0.2, 6.4, 3.0, 1.6]), bins=[0.0, 1.0, 2.5, 4.0, 10.0])
''')
fn("np.unique (counts)", "Frequency table.", '''
vals, counts = np.unique(students[:, 1], return_counts=True)
np.column_stack([vals, counts])
''', reads="Ages and how many students have each age.")
fn("np.mode via unique", "NumPy has no mode(): use unique + argmax.", '''
v, c = np.unique(students[:, 1], return_counts=True)
v[np.argmax(c)]
''')
fn("np.any / np.all", "Is at least one / are all conditions True?", '''
a = np.array([[1, 2, 3], [4, 5, 6]])
(np.any(a > 5), np.all(a > 0), np.all(a > 3, axis=1), np.any(a > 5, axis=0))
''')
fn("axis explained with 3-D data", "Which dimension disappears.", '''
a = np.arange(24).reshape(2, 3, 4)
{"sum axis=0": a.sum(axis=0).shape, "axis=1": a.sum(axis=1).shape, "axis=2": a.sum(axis=2).shape, "axis=(0,2)": a.sum(axis=(0, 2)).shape}
''', reads="The reduced axis vanishes from the shape: (2,3,4) with axis=1 -> (2,4).")
fn("keepdims for broadcasting", "keepdims=True lets you subtract the mean directly.", '''
a = np.array([[1.0, 2, 3], [10, 20, 30]])
a - a.mean(axis=1, keepdims=True)
''', reads="Each row centred on its own mean. Without keepdims the shapes (2,3) and (2,) would not broadcast.")
fn("standardise (z-score)", "(x - mean) / std for every column.", '''
x = houses[:, :3]
z = (x - x.mean(axis=0)) / x.std(axis=0)
np.round(np.column_stack([z.mean(axis=0), z.std(axis=0)]), 6)
''', reads="After standardising, every column has mean 0 and std 1.")
fn("min-max scaling", "Rescale to 0-1.", '''
x = houses[:, 0]
s = (x - x.min()) / (x.max() - x.min())
(s.min(), s.max())
''')
fn("np.ma mean (alternative to nan)", "Masked statistics (see the Masked arrays category).", '''
np.ma.masked_invalid(students[:, 5]).mean()
''')
fn("np.diff", "Differences between neighbours.", '''
a = np.array([1, 4, 9, 16, 25])
(np.diff(a), np.diff(a, n=2), np.diff(np.array([[1, 3], [6, 10]]), axis=0))
''', reads="n=2 applies diff twice: constant 2 -> second difference of squares.")
fn("np.ediff1d", "Differences of the flattened array, with optional ends.", '''
np.ediff1d(np.array([1, 2, 4, 7]), to_begin=0)
''')
fn("np.gradient", "Numerical derivative (central differences).", '''
x = np.linspace(0, 2, 5)
np.gradient(x ** 2, x)
''', reads="The true derivative of x^2 is 2x = 0, 1, 2, 3, 4 - this gives exactly that inside, one-sided at the ends.")
fn("np.trapezoid", "Area under a curve (trapezoid rule; NumPy 2.0 name).", '''
x = np.linspace(0, np.pi, 1001)
np.trapezoid(np.sin(x), x)
''', reads="The exact area of sin from 0 to pi is 2.")
fn("np.trapz", "REMOVED in NumPy 2.0 - renamed np.trapezoid.", '''
np.trapezoid([1, 2, 3])
''', status="removed", alt="np.trapezoid")
fn("np.interp", "Linear interpolation of a lookup table.", '''
xp, fp = [0, 10, 20], [0, 100, 50]
np.interp([2.5, 10, 15, 30], xp, fp)
''', reads="Values outside the table are clamped to the end values (30 -> 50).")
fn("np.convolve", "Convolution: moving average, smoothing, polynomial products.", '''
x = np.array([1, 2, 3, 4, 5, 6])
np.convolve(x, np.ones(3) / 3, mode="valid")
''', reads="A 3-point moving average. mode 'valid' keeps only positions where the window fully fits.")
fn("moving average with cumsum", "A fast O(n) running mean.", '''
x = np.arange(1.0, 11)
c = np.cumsum(np.insert(x, 0, 0))
(c[4:] - c[:-4]) / 4
''')
fn("np.cross", "Cross product of 3-D vectors.", '''
np.cross(np.array([1, 0, 0]), np.array([0, 1, 0]))
''')
fn("np.nanmean on axis", "Per-column mean ignoring blanks.", '''
np.round(np.nanmean(students[:, 3:], axis=0), 2)
''', reads="Average height, weight, math, physics ... of the 600 students.")
fn("np.apply_along_axis (statistics)", "Apply any 1-D function to rows or columns.", '''
np.apply_along_axis(lambda c: np.nanpercentile(c, 90), 0, students[:, 5:9])
''')
fn("np.cumsum for ranks / ECDF", "Empirical CDF: fraction of values <= x.", '''
x = np.sort(np.array([3, 1, 4, 1, 5, 9, 2, 6]))
np.arange(1, len(x) + 1) / len(x)
''')
fn("np.sum of booleans", "Count how many are True.", '''
a = np.array([3, 8, 1, 9, 4])
(np.sum(a > 3), np.mean(a > 3))
''')
fn("np.around vs round", "np.around is the older name of np.round.", '''
np.around(np.array([2.567, 3.141]), 1)
''')

# =====================================================================================================  10  BROADCASTING & UFUNCS
cat("broadcast", "Broadcasting & ufunc machinery", "The rules that let different shapes work together, plus ufunc methods, vectorize, einsum.", "📡",
    intro="BROADCASTING lets NumPy combine arrays of different shapes without copying. Rule: line the shapes up from the RIGHT; each pair of dimensions must be equal, or one of them must be 1 (or missing). "
          "The size-1 dimension is stretched to match.")

fn("broadcasting: scalar + array", "A number is stretched to every element.", '''
np.array([[1, 2], [3, 4]]) + 10
''')
fn("broadcasting: row + matrix", "A (3,) row is added to every row of a (2, 3) matrix.", '''
a = np.arange(6).reshape(2, 3)
r = np.array([100, 200, 300])
(a, r, a + r)
''', reads="Shapes (2,3) and (3,): align right -> 3 matches 3; the missing leading dimension is stretched.")
fn("broadcasting: column + row", "(3,1) with (1,4) gives a (3,4) table.", '''
col = np.array([[0], [10], [20]])
row = np.array([1, 2, 3, 4])
col + row
''')
fn("broadcasting: incompatible shapes", "Shapes (3,) and (4,) cannot broadcast - the typical error.", '''
try:
    np.ones(3) + np.ones(4)
except ValueError as e:
    msg = str(e)
msg
''', reads="3 vs 4: not equal and neither is 1 -> error.")
fn("broadcasting: step by step", "Work out the result shape by hand.", '''
a = np.zeros((8, 1, 6, 1))
b = np.zeros((7, 1, 5))
np.broadcast(a, b).shape
''', reads="Right-align: (8,1,6,1) vs (_,7,1,5) -> 8, 7, 6, 5.")
fn("broadcasting: normalise columns", "Subtract a per-column mean from a table - no loop.", '''
x = houses[:5, :3]
np.round(x - x.mean(axis=0), 2)
''')
fn("broadcasting: distance matrix", "All pairwise distances between points in one expression.", '''
p = np.array([[0, 0], [3, 4], [6, 8]])
d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=-1))
d
''', reads="p[:, None, :] is (3,1,2), p[None, :, :] is (1,3,2) -> the difference is (3,3,2); summing the last axis gives a 3x3 distance table.")
fn("np.outer", "Outer product of two vectors.", '''
np.outer(np.array([1, 2, 3]), np.array([10, 20]))
''')
fn("np.add.outer (ufunc.outer)", "Apply any ufunc to every pair.", '''
np.add.outer(np.arange(3), np.arange(4))
''')
fn("ufunc.reduce", "Collapse an axis with a binary ufunc.", '''
a = np.arange(1, 7).reshape(2, 3)
(np.add.reduce(a, axis=0), np.multiply.reduce(a, axis=1), np.maximum.reduce(a, axis=None))
''')
fn("ufunc.accumulate", "Running version of reduce.", '''
(np.add.accumulate(np.array([1, 2, 3, 4])), np.maximum.accumulate(np.array([3, 1, 4, 1, 5, 9, 2])))
''', reads="maximum.accumulate = running maximum.")
fn("ufunc.reduceat", "Reduce over slices that start at given positions.", '''
np.add.reduceat(np.arange(10), [0, 4, 7])
''', reads="Sums of a[0:4], a[4:7], a[7:].")
fn("ufunc.at", "Unbuffered in-place operation: repeated indices ARE accumulated.", '''
a = np.zeros(4, dtype=int)
np.add.at(a, [0, 1, 1, 3, 3, 3], 1)
a
''', reads="a[[0,1,1,3,3,3]] += 1 would count each index only once; add.at counts duplicates.")
fn("ufunc attributes", "Every ufunc describes itself.", '''
u = np.add
{"name": u.__name__, "nin": u.nin, "nout": u.nout, "nargs": u.nargs, "identity": u.identity, "ntypes": u.ntypes, "first types": u.types[:4]}
''')
fn("np.frompyfunc", "Turn a Python function into a ufunc (returns object arrays).", '''
f = np.frompyfunc(lambda a, b: a * 10 + b, 2, 1)
f(np.array([1, 2]), np.array([3, 4]))
''')
fn("np.vectorize", "Wrap a scalar Python function so it accepts arrays (convenience, not speed).", '''
def classify(x):
    return "pass" if x >= 50 else "fail"

v = np.vectorize(classify)
v(np.array([30, 50, 80]))
''', reads="np.vectorize is a Python loop in disguise. Prefer np.where / np.select when you can.")
fn("np.vectorize (otypes, signature)", "Give output types to avoid a trial call.", '''
f = np.vectorize(lambda a, b: a + b, otypes=[float])
f([1, 2, 3], 0.5)
''')
fn("np.apply_along_axis", "Call a 1-D function on every row (axis=1) or column (axis=0).", '''
a = np.array([[3, 1, 2], [9, 8, 7]])
np.apply_along_axis(np.sort, 1, a)
''')
fn("np.apply_over_axes", "Apply a function over several axes, keeping dimensions.", '''
np.apply_over_axes(np.sum, np.arange(24).reshape(2, 3, 4), [0, 2]).shape
''')
fn("np.einsum (basics)", "Einstein summation: describe any tensor operation with a short string.", '''
a = np.arange(6).reshape(2, 3)
b = np.arange(12).reshape(3, 4)
{"matrix product 'ij,jk->ik'": np.einsum("ij,jk->ik", a, b), "transpose 'ij->ji'": np.einsum("ij->ji", a), "sum all 'ij->'": np.einsum("ij->", a), "row sums 'ij->i'": np.einsum("ij->i", a)}
''', reads="Letters name axes. Letters missing after '->' are summed over; repeated letters are multiplied together.")
fn("np.einsum (dot, trace, diagonal)", "More einsum recipes.", '''
v = np.array([1, 2, 3]); m = np.arange(9).reshape(3, 3)
{"dot 'i,i->'": np.einsum("i,i->", v, v), "trace 'ii->'": np.einsum("ii->", m), "diagonal 'ii->i'": np.einsum("ii->i", m), "outer 'i,j->ij'": np.einsum("i,j->ij", v, v[:2])}
''')
fn("np.einsum (batch matrix multiply)", "Multiply a stack of matrices.", '''
a = np.random.default_rng(0).integers(0, 5, (2, 2, 3))
b = np.random.default_rng(1).integers(0, 5, (2, 3, 2))
np.einsum("bij,bjk->bik", a, b).shape
''')
fn("np.einsum_path", "Find the cheapest contraction order.", '''
a = np.ones((10, 20)); b = np.ones((20, 30)); c = np.ones((30, 5))
path, info = np.einsum_path("ij,jk,kl->il", a, b, c, optimize="optimal")
path
''')
fn("np.tensordot", "Sum products over chosen axes.", '''
a = np.arange(6).reshape(2, 3)
b = np.arange(12).reshape(3, 4)
np.tensordot(a, b, axes=1)
''')
fn("np.inner", "Inner product (dot over the last axes).", '''
np.inner(np.array([1, 2, 3]), np.array([4, 5, 6]))
''')
fn("np.vdot", "Flattened dot product with complex conjugation.", '''
np.vdot(np.array([1 + 2j, 3]), np.array([4, 5j]))
''')
fn("np.vecdot", "NumPy 2: dot product along the last axis for stacks of vectors.", '''
np.vecdot(np.array([[1, 2], [3, 4]]), np.array([[1, 1], [2, 2]]))
''')
fn("np.kron", "Kronecker product.", '''
np.kron(np.eye(2, dtype=int), np.array([[1, 2], [3, 4]]))
''')
fn("np.trace", "Sum of the diagonal.", '''
np.trace(np.arange(9).reshape(3, 3))
''')
fn("np.matmul / @", "Matrix multiplication (and batches of matrices).", '''
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])
(A @ B, np.matmul(A, B))
''')
fn("np.dot", "Dot product / matrix product depending on dimensions.", '''
(np.dot(np.array([1, 2, 3]), np.array([4, 5, 6])), np.dot(np.eye(2), np.array([5, 6])))
''')
fn("np.newaxis trick: normalise rows", "Divide each row by its own sum.", '''
a = np.array([[1.0, 3.0], [2.0, 2.0], [1.0, 9.0]])
a / a.sum(axis=1)[:, np.newaxis]
''')
fn("np.testing.assert_allclose", "Raise a clear error when arrays differ (use in tests).", '''
np.testing.assert_allclose(0.1 + 0.2, 0.3)
"arrays are close"
''')
fn("np.testing.assert_array_equal", "Exact comparison with a readable report.", '''
try:
    np.testing.assert_array_equal(np.array([1, 2, 3]), np.array([1, 5, 3]))
except AssertionError as e:
    msg = str(e).strip()
msg
''')
fn("np.broadcast (object)", "Iterate over broadcast pairs.", '''
b = np.broadcast(np.array([[1], [2]]), np.array([10, 20, 30]))
(b.shape, b.nd, b.size, list(b)[:3])
''')

# =====================================================================================================  11  LOGIC
cat("logic", "Comparison & logic", "Compare arrays, combine conditions, test for nan/inf, close values, set-style membership.", "⚖️",
    intro="Comparing arrays gives arrays of True/False (masks). Combine masks with & | ~ (bitwise operators with parentheses) or the logical_* functions.")

fn("a == b, a != b, a < b ...", "Element-wise comparison operators.", '''
a = np.array([1, 5, 3])
b = np.array([1, 2, 4])
{"==": a == b, "!=": a != b, "<": a < b, ">=": a >= b}
''')
fn("np.equal / not_equal / greater / ...", "Function forms of the comparison operators.", '''
a = np.array([1, 5, 3])
(np.greater(a, 2), np.less_equal(a, 3))
''')
fn("np.array_equal", "Do two arrays have the same shape AND values?", '''
a = np.array([1, 2, 3])
(np.array_equal(a, [1, 2, 3]), np.array_equal(a, [1, 2]), np.array_equal(np.array([np.nan]), np.array([np.nan]), equal_nan=True))
''', reads="Plain == on arrays gives an array and cannot be used in an if. array_equal returns ONE True/False.")
fn("np.array_equiv", "Equal after broadcasting?", '''
np.array_equiv(np.array([1, 1, 1]), 1)
''')
fn("np.allclose", "True if ALL elements are close (tolerances).", '''
np.allclose(np.array([1.0, 2.0]), np.array([1.0 + 1e-9, 2.0]))
''', params=["rtol - relative tolerance (1e-5)", "atol - absolute tolerance (1e-8)", "equal_nan"])
fn("np.isclose", "Element-wise closeness.", '''
np.isclose(np.array([1.0, 2.0, 3.0]), np.array([1.0000001, 2.1, 3.0]))
''')
fn("np.logical_and / or / xor / not", "Boolean combinations (work on any truthy values).", '''
a = np.array([True, True, False, False])
b = np.array([True, False, True, False])
{"and": np.logical_and(a, b), "or": np.logical_or(a, b), "xor": np.logical_xor(a, b), "not a": np.logical_not(a)}
''')
fn("np.logical_and.reduce", "Combine many masks.", '''
m = [np.array([1, 0, 1]) > 0, np.array([1, 1, 0]) > 0, np.array([1, 1, 1]) > 0]
np.logical_and.reduce(m)
''')
fn("np.isnan / np.isinf / np.isfinite", "Detect special float values.", '''
a = np.array([1.0, np.nan, np.inf, -np.inf])
{"isnan": np.isnan(a), "isinf": np.isinf(a), "isfinite": np.isfinite(a)}
''', reads="nan == nan is False! Always use np.isnan to find missing values.")
fn("np.isposinf / np.isneginf", "Detect +inf or -inf only.", '''
a = np.array([np.inf, -np.inf, 1.0])
(np.isposinf(a), np.isneginf(a))
''')
fn("np.isnat", "Missing dates.", '''
np.isnat(np.array(["2024-01-01", "NaT"], dtype="datetime64[D]"))
''')
fn("np.isin", "Is each element in a list of values?", '''
np.isin(np.array([1, 5, 8, 3]), [3, 5, 7])
''', params=["invert=True - not in", "assume_unique"])
fn("np.in1d", "REMOVED in NumPy 2.4: use np.isin.", '''
np.isin([1, 2, 3], [2])
''', status="removed", alt="np.isin")
fn("np.isreal / iscomplex / isrealobj", "Check for complex values.", '''
z = np.array([1 + 0j, 2 + 3j, 4])
(np.isreal(z), np.iscomplex(z), np.iscomplexobj(z), np.isrealobj(z))
''')
fn("np.isscalar", "Is it a scalar (not an array)?", '''
(np.isscalar(5), np.isscalar(np.float64(5)), np.isscalar(np.array(5)), np.isscalar([5]))
''')
fn("np.issubdtype / np.isdtype (logic)", "Type tests.", '''
np.issubdtype(np.dtype("int16"), np.signedinteger)
''')
fn("np.signbit / np.copysign (logic)", "Sign tests for floats.", '''
np.signbit(np.array([-0.0, 0.0]))
''')
fn("np.greater_equal chained comparison", "Python's 3 < a < 8 does not work on arrays.", '''
a = np.arange(10)
try:
    r = 3 < a < 8
except ValueError as e:
    r = "ValueError: " + str(e)[:60]
(r, (3 < a) & (a < 8))
''', tip="Write (a > 3) & (a < 8), or use np.logical_and(a > 3, a < 8).")
fn("ambiguous truth value", "if array: raises an error - use any() / all().", '''
a = np.array([1, 2, 3])
try:
    if a > 1:
        pass
except ValueError as e:
    msg = str(e)
(msg, bool((a > 1).any()), bool((a > 1).all()))
''')
fn("np.where with conditions", "Combine masks inside where.", '''
a = np.array([5, 15, 25, 35])
np.where((a > 10) & (a < 30), "mid", "other")
''')
fn("np.bitwise_and on booleans", "& and logical_and agree on boolean arrays.", '''
a, b = np.array([True, False]), np.array([True, True])
(a & b, np.logical_and(a, b))
''')
fn("np.count_nonzero for conditions", "Fast count of matches.", '''
np.count_nonzero(students[:, 5] > 80)
''')
fn("np.nanargmax with mask logic", "Combine nan handling with conditions.", '''
m = students[:, 5]
np.sum((m > 50) & ~np.isnan(m))
''')
fn("a is b vs np.array_equal", "Identity vs equality.", '''
a = np.array([1, 2])
b = a.copy()
(a is b, np.array_equal(a, b), np.shares_memory(a, b))
''')
fn("np.char.equal (strings)", "Element-wise string comparison.", '''
np.char.equal(np.array(["a", "b"]), "a")
''')

from ._core import cat, fn

# =====================================================================================================  1  BASICS
cat("basics", "ndarray basics & attributes", "What an array is, and every attribute you can read from it.", "🧱",
    intro="An ndarray is a fixed-size grid of values that all share ONE data type, stored in one block of memory. "
          "That uniformity is why NumPy is fast. Every array carries metadata: shape, dtype, strides ... - read them with attributes (no brackets).")

fn("np.array", "Builds an array from a (nested) Python list. The nesting depth becomes the number of dimensions.", '''
a = np.array([[1, 2, 3], [4, 5, 6]])
a
''', reads="Two inner lists of three numbers become 2 rows x 3 columns. NumPy picked int64 because every item is a whole number.",
   params=["object - list, tuple, or another array", "dtype - force a type, e.g. dtype=float", "copy - True (default) makes new memory", "ndmin - minimum number of dimensions", "order - 'C' row-major or 'F' column-major"])
fn("a.shape", "A tuple with the length of every dimension.", '''
a = np.arange(24).reshape(2, 3, 4)
a.shape
''', reads="(2, 3, 4) means 2 blocks, each with 3 rows and 4 columns. len(a.shape) is the number of dimensions.")
fn("a.ndim", "Number of dimensions (axes).", '''
{"scalar": np.array(5).ndim, "vector": np.array([1, 2]).ndim, "matrix": np.array([[1], [2]]).ndim, "3-D": np.zeros((2, 2, 2)).ndim}
''')
fn("a.size", "Total number of elements (the product of the shape).", '''
a = np.zeros((4, 5, 2))
a.size
''', reads="4 x 5 x 2 = 40 elements.")
fn("a.dtype", "The data type of every element.", '''
{"ints": np.array([1, 2]).dtype, "floats": np.array([1, 2.5]).dtype, "bools": np.array([True, False]).dtype, "text": np.array(["a", "bc"]).dtype}
''', reads="One array = one dtype. <U2 means 'unicode text up to 2 characters'.")
fn("a.itemsize", "Bytes used by ONE element.", '''
{"int8": np.zeros(1, "int8").itemsize, "int64": np.zeros(1, "int64").itemsize, "float32": np.zeros(1, "float32").itemsize, "complex128": np.zeros(1, "complex128").itemsize}
''')
fn("a.nbytes", "Total bytes the data occupies (size x itemsize).", '''
a = np.zeros((1000, 1000))
{"nbytes": a.nbytes, "MB": a.nbytes / 1e6}
''', reads="A million float64 numbers need exactly 8 MB - a Python list of floats would need roughly 4x more.")
fn("a.strides", "Bytes to jump in memory to reach the next element along each axis.", '''
a = np.arange(12, dtype=np.int32).reshape(3, 4)
a.strides
''', reads="(16, 4): moving down one row skips 16 bytes (4 ints x 4 bytes), moving right skips 4 bytes. Transposing swaps these numbers WITHOUT copying data.")
fn("a.T", "Transpose: swaps rows and columns (a view, no copy).", '''
a = np.arange(6).reshape(2, 3)
a.T
''')
fn("a.flat", "A 1-D iterator over all elements; also lets you index as if the array were flat.", '''
a = np.arange(12).reshape(3, 4)
a.flat[5], list(a.flat[:4]), a.flat[[0, 11]]
''')
fn("a.real / a.imag", "Real and imaginary parts of a complex array.", '''
z = np.array([1 + 2j, 3 - 4j])
{"real": z.real, "imag": z.imag}
''')
fn("a.flags", "Memory-layout information: contiguous? owns its data? writeable?", '''
a = np.arange(12).reshape(3, 4)
print(a.flags)
print("transposed C_CONTIGUOUS:", a.T.flags["C_CONTIGUOUS"], "| F_CONTIGUOUS:", a.T.flags["F_CONTIGUOUS"])
''', reads="A transposed array is no longer C-contiguous (row-major) but is F-contiguous (column-major) - same memory, different walk order.")
fn("a.base", "The array whose memory this one borrows (None when it owns its data).", '''
a = np.arange(10)
v = a[2:6]
{"a.base is None": a.base is None, "v.base is a": v.base is a}
''', reads="Slices are VIEWS: v.base points at a. That is why changing v changes a.")
fn("a.data", "A raw memoryview of the buffer.", '''
a = np.array([1, 2, 3], dtype=np.uint8)
bytes(a.data)
''')
fn("a.ctypes", "Helper for passing the array's memory pointer to C code.", '''
a = np.arange(4, dtype=np.int32)
{"data pointer is int": isinstance(a.ctypes.data, int), "shape": tuple(a.ctypes.shape)}
''')
fn("a.__array_interface__", "The dict other libraries use to share this array's memory.", '''
a = np.arange(6, dtype=np.int16).reshape(2, 3)
{k: v for k, v in a.__array_interface__.items() if k != "data"}
''')
fn("a.dtype.name / .kind / .char / .str", "Different ways to name a dtype.", '''
d = np.dtype("float32")
{"name": d.name, "kind": d.kind, "char": d.char, "str": d.str, "byteorder": d.byteorder, "itemsize": d.itemsize}
''', reads="kind is a single letter: b bool, i int, u unsigned, f float, c complex, U text, M datetime, m timedelta, O object.")
fn("a.item", "Copies ONE element out as a plain Python number.", '''
a = np.array([[1.5, 2.5], [3.5, 4.5]])
(a.item(3), a.item(1, 0), type(a.item(0)).__name__)
''')
fn("a.itemset", "REMOVED in NumPy 2.0: assign with a[idx] = value instead.", '''
a = np.zeros(3)
a[1] = 7.0
a
''', status="removed", alt="a[index] = value")
fn("a.tolist", "Converts to (nested) Python lists.", '''
np.arange(6).reshape(2, 3).tolist()
''', reads="The numbers also become plain Python ints/floats.")
fn("a.len / len(a)", "len() gives the size of the FIRST axis only.", '''
a = np.zeros((5, 3))
(len(a), a.size, a.shape[1])
''', reads="Use a.size for the total count. len(a) is easy to misuse on 2-D arrays.")
fn("np.ndim / np.shape / np.size", "Function versions of the attributes - also work on plain lists.", '''
lst = [[1, 2, 3], [4, 5, 6]]
{"ndim": np.ndim(lst), "shape": np.shape(lst), "size": np.size(lst), "size axis 1": np.size(lst, 1)}
''')
fn("a.copy", "A new array with its OWN memory.", '''
a = np.array([1, 2, 3])
b = a.copy()
b[0] = 99
{"a": a, "b": b}
''', reads="Changing the copy leaves the original untouched.")
fn("a.view", "A new array object looking at the SAME memory (optionally reinterpreting the type).", '''
a = np.array([1.0], dtype=np.float32)
a.view(np.uint32)
''', reads="1.0 stored as float32 has the bit pattern 0x3F800000 = 1065353216. No data is converted - only reinterpreted.")
fn("a.fill", "Sets every element to one value, in place.", '''
a = np.empty(5)
a.fill(3.14)
a
''')
fn("a.sum", "Methods can also be called on the array itself.", '''
a = np.arange(10)
(a.sum(), np.sum(a))
''', reads="a.method() and np.method(a) give the same answer; use whichever reads better.")
fn("0-d vs 1-element arrays", "Scalars, 0-d arrays and length-1 arrays are three different things.", '''
{"np.float64(5)": np.float64(5).shape, "np.array(5)": np.array(5).shape, "np.array([5])": np.array([5]).shape}
''', reads="() = no axes at all; (1,) = one axis with one element.")
fn("np.__version__", "The installed NumPy version.", '''
np.__version__
''')
fn("np.show_config", "Prints build information (BLAS library, CPU features).", '''
np.show_config()
''')

# =====================================================================================================  2  DTYPES
cat("dtypes", "Data types & casting", "Every dtype, converting between them, limits, promotion rules.", "🔢",
    intro="Choosing the right dtype controls memory use, speed and precision. Integers overflow silently, floats lose precision - knowing the limits avoids nasty surprises.")

fn("np.int8 / uint8 / ... int64", "Fixed-size integers. The number is the bit width.", '''
rows = []
for t in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64):
    i = np.iinfo(t)
    rows.append((np.dtype(t).name, int(i.min), int(i.max)))
rows
''', reads="uint = unsigned (no negatives, but twice the positive range).")
fn("np.float16 / 32 / 64", "Floating-point types: more bits = more precision.", '''
rows = []
for t in (np.float16, np.float32, np.float64):
    f = np.finfo(t)
    rows.append((np.dtype(t).name, float(f.eps), float(f.max), int(f.precision)))
rows
''', reads="Columns: name, machine epsilon (smallest gap near 1), largest value, decimal digits of precision.")
fn("np.longdouble / float128", "Extended precision (platform dependent).", '''
(np.dtype(np.longdouble).name, np.finfo(np.longdouble).precision)
''', reads="On many desktops this is 80-bit (x86) giving ~18 digits; in the browser it can equal float64.")
fn("np.complex64 / complex128", "Complex numbers (two floats each).", '''
z = np.array([1 + 2j, 3 - 1j])
(z.dtype, np.abs(z), np.conj(z))
''')
fn("np.bool_", "True / False values (1 byte each).", '''
b = np.array([True, False, True])
(b.dtype, b.sum(), b.nbytes)
''', reads="True counts as 1 in sums - the standard trick for counting matches.")
fn("np.str_ / np.bytes_", "Fixed-width text and bytes.", '''
s = np.array(["cat", "elephant"])
(s.dtype, s.itemsize)
''', reads="<U8 = unicode, longest string is 8 characters; each character takes 4 bytes -> itemsize 32.")
fn("np.object_", "A generic array of Python objects (slow, avoid for numbers).", '''
o = np.array([1, "two", [3]], dtype=object)
(o.dtype, o[2])
''')
fn("np.datetime64 / timedelta64", "Dates and durations (see the Date & time category).", '''
np.array(["2024-01-15", "2024-03-01"], dtype="datetime64[D]")
''')
fn("np.dtype", "Creates a dtype object from a name, code or Python type.", '''
[np.dtype("i4"), np.dtype(float), np.dtype("U5"), np.dtype(">f8"), np.dtype("<i2")]
''', reads="'>' means big-endian bytes, '<' little-endian. i4 = 4-byte int.")
fn("a.astype", "Converts to another dtype (always returns a COPY by default).", '''
a = np.array([1.9, -1.9, 2.5])
{"to int (truncates toward 0)": a.astype(int), "to float32": a.astype(np.float32), "to bool": a.astype(bool), "to str": a.astype(str)}
''', reads="astype(int) truncates toward zero, it does NOT round. Use np.round first if you want rounding.",
   params=["dtype - target type", "copy=False - avoid a copy when the type already matches", "casting - 'unsafe' (default), 'safe', 'same_kind'", "order - memory layout of the result"])
fn("np.astype", "Function form of astype (NumPy 2.1+).", '''
np.astype(np.array([1.5, 2.5]), np.int32)
''')
fn("np.can_cast", "Can this type be converted to that one safely?", '''
{"int32 -> int64": np.can_cast(np.int32, np.int64), "int64 -> int32": np.can_cast(np.int64, np.int32), "float64 -> float32, 'same_kind'": np.can_cast(np.float64, np.float32, "same_kind")}
''')
fn("np.promote_types", "The smallest dtype that can hold both types.", '''
[np.promote_types("int8", "uint8"), np.promote_types("int64", "float32"), np.promote_types("float32", "complex64")]
''', reads="int8 + uint8 -> int16 (neither fits the other). int64 + float32 -> float64.")
fn("np.result_type", "The dtype an operation on these inputs would produce.", '''
[np.result_type(np.int8, np.float32), np.result_type(3, np.int8), np.result_type(np.array([1], np.int8), 300)]
''', reads="Python scalars are 'weak' in NumPy 2: 3 does not force a bigger dtype, but 300 is too large for int8 so it widens.")
fn("np.common_type", "Common inexact type of several arrays.", '''
np.common_type(np.array([1, 2], np.int16), np.array([1.5], np.float32))
''')
fn("np.min_scalar_type", "Smallest dtype that can hold a value.", '''
[np.min_scalar_type(10), np.min_scalar_type(300), np.min_scalar_type(-5), np.min_scalar_type(2.5), np.min_scalar_type(2**40)]
''')
fn("np.issubdtype", "Is a dtype a member of a family (integer, floating ...)?", '''
{"int32 is integer": np.issubdtype(np.int32, np.integer), "float32 is integer": np.issubdtype(np.float32, np.integer), "float16 is floating": np.issubdtype(np.float16, np.floating), "uint8 is number": np.issubdtype(np.uint8, np.number)}
''')
fn("np.isdtype", "NumPy 2: check a dtype against a kind name.", '''
{"int64 'integral'": np.isdtype(np.int64, "integral"), "float32 'real floating'": np.isdtype(np.float32, "real floating"), "bool 'numeric'": np.isdtype(np.bool_, "numeric")}
''')
fn("np.iinfo", "Limits of an integer type.", '''
i = np.iinfo(np.int16)
{"bits": i.bits, "min": i.min, "max": i.max}
''')
fn("np.finfo", "Limits and precision of a float type.", '''
f = np.finfo(np.float64)
{"eps": f.eps, "tiny (smallest normal)": f.tiny, "max": f.max, "digits": f.precision, "resolution": f.resolution}
''')
fn("integer overflow", "Integers wrap around silently - no error!", '''
a = np.array([250, 5], dtype=np.uint8)
a + np.uint8(10)
''', reads="250 + 10 = 260, but a uint8 only holds 0-255, so it wraps to 4. Float arrays overflow to inf instead.",
   tip="Pick a wide enough dtype (int64) when summing large counts. np.sum already upcasts small ints for you.")
fn("float precision", "Floats cannot store most decimals exactly.", '''
{"0.1 + 0.2": 0.1 + 0.2, "== 0.3 ?": 0.1 + 0.2 == 0.3, "isclose": bool(np.isclose(0.1 + 0.2, 0.3))}
''', tip="Never compare floats with ==; use np.isclose / np.allclose.")
fn("float32 vs float64 precision", "Fewer bits, fewer correct digits.", '''
x = 16777217
{"float32": float(np.float32(x)), "float64": float(np.float64(x)), "int": x}
''', reads="float32 can only count exactly up to 2**24 = 16,777,216.")
fn("int / float mixing (type promotion)", "Mixed operations pick the wider type.", '''
a = np.array([1, 2, 3])
{"int + int": (a + 1).dtype, "int + float": (a + 1.5).dtype, "int / int": (a / 2).dtype, "int // int": (a // 2).dtype, "bool + bool": (np.array([True]) + np.array([True])).dtype}
''', reads="True division (/) always gives float. bool + bool is logical OR (stays bool) - use .astype(int) to count.")
fn("np.sctypeDict", "REMOVED / reorganised: use np.dtype(name).type instead of the old scalar-type dictionaries.", '''
np.dtype("float32").type
''', status="removed", alt="np.dtype(name).type")
fn("np.float_ / np.int_ / np.bool8 ...", "Several old aliases were removed in NumPy 2.0.", '''
{"np.float_ -> use": np.float64, "np.complex_ -> use": np.complex128, "np.bool8 -> use": np.bool_, "np.object0 -> use": np.object_, "np.unicode_ -> use": np.str_}
''', status="removed", alt="np.float64 / np.complex128 / np.bool_ / np.str_")
fn("np.byte / ubyte / short / intc / int_ / longlong", "C-style names for integer types.", '''
{n: np.dtype(getattr(np, n)).name for n in ("byte", "ubyte", "short", "ushort", "intc", "uintc", "int_", "uint", "longlong", "ulonglong", "intp", "uintp")}
''', reads="intp is the type used for array indices (64-bit on modern machines). int_ is the default integer.")
fn("np.half / single / double", "Float aliases.", '''
{n: np.dtype(getattr(np, n)).name for n in ("half", "single", "double", "longdouble")}
''')
fn("np.typecodes", "The full list of one-letter dtype codes.", '''
np.typecodes
''')
fn("np.dtype.byteorder / newbyteorder", "Endianness: which byte is stored first.", '''
a = np.array([1], dtype=">i4")
(a.dtype.byteorder, a.tobytes(), a.astype("<i4").tobytes())
''', reads="Big-endian stores 00 00 00 01; little-endian stores 01 00 00 00. NumPy converts transparently.")
fn("a.byteswap", "Swaps the byte order of every element.", '''
a = np.array([1, 256], dtype=np.int16)
a.byteswap()
''')
fn("np.finfo(...).eps usage", "eps tells how close two floats can be before they are 'equal'.", '''
eps = np.finfo(float).eps
{"1 + eps == 1": 1 + eps == 1, "1 + eps/2 == 1": 1 + eps / 2 == 1}
''')
fn("np.set_printoptions (dtype view)", "Control how arrays print.", '''
a = np.array([1 / 3, 2 / 3])
with np.printoptions(precision=3):
    s = repr(a)
s
''')

# =====================================================================================================  3  CREATION
cat("creation", "Creating arrays", "Every way to build an array: from data, ranges, constants, grids, random shapes.", "🏗️",
    intro="You rarely type arrays by hand. NumPy can build them from ranges, constants, identity matrices, grids, functions, files or other arrays.")

fn("np.zeros", "Array filled with 0.", '''
np.zeros((2, 3))
''', reads="The default dtype is float64. Pass dtype=int for integer zeros.", params=["shape - int or tuple", "dtype", "order - 'C' or 'F'"])
fn("np.ones", "Array filled with 1.", '''
np.ones((2, 4), dtype=int)
''')
fn("np.full", "Array filled with any constant.", '''
np.full((2, 3), 7.5)
''')
fn("np.empty", "Allocates memory WITHOUT initialising - contents are garbage.", '''
a = np.empty((2, 2))
a.shape
''', reads="Fast, but the values are whatever was in memory. Only use when you will overwrite everything.")
fn("np.arange", "Like range() but returns an array and allows float steps.", '''
{"arange(5)": np.arange(5), "arange(2, 10, 3)": np.arange(2, 10, 3), "arange(0, 1, 0.25)": np.arange(0, 1, 0.25), "arange(10, 0, -3)": np.arange(10, 0, -3)}
''', reads="The stop value is EXCLUDED. With float steps, rounding can make the length surprising - use linspace when you need an exact count.",
   params=["start, stop, step", "dtype"])
fn("np.linspace", "N evenly spaced numbers between two ends (end INCLUDED).", '''
np.linspace(0, 1, 5)
''', reads="5 points from 0 to 1 including both ends. Use endpoint=False to exclude the stop.",
   params=["num - how many points (default 50)", "endpoint=False - exclude stop", "retstep=True - also return the step size", "axis"])
fn("np.linspace (retstep)", "Also returns the spacing.", '''
np.linspace(0, 10, 5, retstep=True)
''')
fn("np.logspace", "Points evenly spaced on a log scale (base ** exponent).", '''
np.logspace(0, 3, 4)
''', reads="Exponents 0,1,2,3 -> 1, 10, 100, 1000.")
fn("np.geomspace", "Points with a constant RATIO between them.", '''
np.geomspace(1, 256, 9)
''', reads="Each value is 2x the previous one.")
fn("np.eye", "2-D array with ones on a diagonal.", '''
np.eye(3, 4, k=1)
''', reads="k=1 shifts the diagonal one place to the right.")
fn("np.identity", "Square identity matrix.", '''
np.identity(3, dtype=int)
''')
fn("np.diag", "Build a diagonal matrix from a vector - or extract a diagonal from a matrix.", '''
v = np.diag([1, 2, 3])
m = np.arange(9).reshape(3, 3)
(v, np.diag(m), np.diag(m, k=1))
''')
fn("np.diagflat", "Flatten the input, then place it on a diagonal.", '''
np.diagflat([[1, 2], [3, 4]])
''')
fn("np.tri", "Lower-triangle matrix of ones.", '''
np.tri(4, dtype=int)
''')
fn("np.tril", "Keep the lower triangle, zero the rest.", '''
np.tril(np.arange(1, 10).reshape(3, 3))
''')
fn("np.triu", "Keep the upper triangle, zero the rest.", '''
np.triu(np.arange(1, 10).reshape(3, 3), k=1)
''')
fn("np.vander", "Vandermonde matrix: powers of each input value.", '''
np.vander([1, 2, 3, 4], 3)
''')
fn("np.zeros_like", "Zeros with the SAME shape and dtype as another array.", '''
a = np.arange(6, dtype=np.float32).reshape(2, 3)
z = np.zeros_like(a)
(z, z.dtype)
''')
fn("np.ones_like", "Ones shaped like another array.", '''
np.ones_like(np.array([[1, 2], [3, 4]]))
''')
fn("np.full_like", "Constant shaped like another array.", '''
np.full_like(np.arange(4), 9)
''')
fn("np.empty_like", "Uninitialised array shaped like another.", '''
np.empty_like(np.zeros((2, 3))).shape
''')
fn("np.asarray", "Convert to an array WITHOUT copying if it already is one.", '''
a = np.array([1, 2, 3])
b = np.asarray(a)
c = np.array(a)
(b is a, c is a)
''', reads="np.asarray avoids needless copies; np.array always copies.")
fn("np.asanyarray", "Like asarray but keeps subclasses (masked arrays, matrices).", '''
m = np.ma.masked_array([1, 2], [0, 1])
(type(np.asarray(m)).__name__, type(np.asanyarray(m)).__name__)
''')
fn("np.ascontiguousarray", "Ensures C-contiguous memory.", '''
a = np.arange(6).reshape(2, 3).T
b = np.ascontiguousarray(a)
(a.flags.c_contiguous, b.flags.c_contiguous)
''')
fn("np.asfortranarray", "Ensures Fortran (column-major) memory.", '''
a = np.arange(6).reshape(2, 3)
np.asfortranarray(a).flags.f_contiguous
''')
fn("np.copy", "Function version of a.copy().", '''
a = np.array([1, 2, 3])
b = np.copy(a)
b[0] = 10
a
''')
fn("np.array (copy=False / ndmin)", "ndmin forces extra leading dimensions.", '''
np.array([1, 2, 3], ndmin=3).shape
''')
fn("np.fromfunction", "Build an array by calling a function with the INDICES.", '''
np.fromfunction(lambda i, j: i * 10 + j, (3, 4), dtype=int)
''', reads="The function receives the row index array i and the column index array j.")
fn("np.fromiter", "Build a 1-D array from any iterable/generator.", '''
np.fromiter((x * x for x in range(6)), dtype=int)
''')
fn("np.fromstring", "Parse numbers out of a text string.", '''
np.fromstring("1 2 3 4.5", sep=" ")
''', reads="Text mode with sep= is supported; the old binary mode was removed.")
fn("np.frombuffer", "Wrap existing bytes as an array (no copy).", '''
np.frombuffer(b"\\x01\\x02\\x03\\x04", dtype=np.uint8)
''')
fn("np.fromregex", "Read structured data with a regular expression.", '''
import io
np.fromregex(io.StringIO("a=1 b=2\\na=3 b=4"), r"a=(\\d+) b=(\\d+)", dtype=[("a", int), ("b", int)])
''')
fn("np.meshgrid", "Coordinate matrices from coordinate vectors (for evaluating f(x, y) on a grid).", '''
x = np.array([0, 1, 2])
y = np.array([10, 20])
X, Y = np.meshgrid(x, y)
(X, Y, X + Y)
''', reads="X repeats x along rows, Y repeats y along columns - together they list every (x, y) pair.", params=["indexing - 'xy' (default) or 'ij'", "sparse=True - memory-saving open grids"])
fn("np.meshgrid (indexing='ij')", "Matrix-style indexing: first axis follows the first input.", '''
I, J = np.meshgrid([0, 1, 2], [0, 1], indexing="ij")
I.shape
''')
fn("np.mgrid", "Dense grid with slice syntax.", '''
y, x = np.mgrid[0:3, 0:4]
(y, x)
''')
fn("np.ogrid", "Open (sparse) grid - uses less memory, broadcasting does the rest.", '''
y, x = np.ogrid[0:3, 0:4]
(y.shape, x.shape, (y * 10 + x))
''')
fn("np.indices", "Index arrays for every axis of a shape.", '''
np.indices((2, 3))
''')
fn("np.r_", "Quick concatenation / ranges with slice syntax.", '''
np.r_[1:4, 10, 20:23, [7, 8]]
''', reads="Slices (1:4) expand like arange; lists and numbers are glued together.")
fn("np.c_", "Glue 1-D arrays as COLUMNS.", '''
np.c_[np.array([1, 2, 3]), np.array([4, 5, 6])]
''')
fn("np.s_", "Builds slice objects you can store and reuse.", '''
sl = np.s_[1:3, ::2]
np.arange(20).reshape(4, 5)[sl]
''')
fn("np.array (dtype=object)", "Ragged lists need dtype=object.", '''
np.array([[1, 2], [3]], dtype=object)
''', reads="Without dtype=object NumPy 2 raises 'inhomogeneous shape' for ragged lists.")
fn("np.repeat / np.tile (for creation)", "Build patterns by repeating.", '''
(np.repeat([1, 2], 3), np.tile([1, 2], 3))
''')
fn("np.identity / eye comparison", "eye accepts non-square shapes, identity does not.", '''
(np.eye(2, 3), np.identity(2))
''')
fn("np.arange (negative step / reverse)", "Count down.", '''
np.arange(5, -1, -1)
''')
fn("np.empty + fill pattern", "Preallocate then fill in a loop (avoid growing arrays!).", '''
out = np.empty(5)
for i in range(5):
    out[i] = i ** 2
out
''', tip="Appending to arrays in a loop copies everything each time. Preallocate, or build a list and convert once.")
fn("np.require", "Make sure an array has the properties you need.", '''
a = np.arange(6.0).reshape(2, 3).T
b = np.require(a, requirements=["C", "W"])
(b.flags.c_contiguous, b.flags.writeable)
''')
fn("np.asarray_chkfinite", "Convert and raise if any nan/inf is present.", '''
try:
    np.asarray_chkfinite([1.0, np.nan])
except ValueError as e:
    msg = str(e)
msg
''')
fn("np.broadcast_to (read-only grid)", "Expand to a bigger shape without copying (read-only).", '''
b = np.broadcast_to(np.array([1, 2, 3]), (4, 3))
(b, b.flags.writeable)
''')
fn("np.loadtxt (creation from file)", "Create from a CSV file (more in Input / output).", '''
np.loadtxt("data/houses.csv", delimiter=",", skiprows=1)[:3]
''')
fn("np.ones / zeros with dtype=bool", "Boolean masks of a fixed shape.", '''
(np.zeros(4, dtype=bool), np.ones((2, 2), dtype=bool))
''')
fn("np.random.default_rng (creation)", "Random arrays are covered in the Random category.", '''
rng = np.random.default_rng(0)
rng.random((2, 3))
''')

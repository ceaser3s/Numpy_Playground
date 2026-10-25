from ._core import cat, fn

# =====================================================================================================  19  STRINGS
cat("strings", "String arrays (np.strings / np.char)", "Vectorised text functions for arrays of strings, plus StringDType.", "🔤",
    intro="NumPy can hold text: dtype <U10 (fixed-width Unicode) or the newer variable-width StringDType. np.strings (NumPy 2) and the older np.char apply a string function to EVERY element. "
          "For heavy text/table work pandas' .str is usually nicer, but these are fast and dependency-free.")

W = 'w = np.array(["  Alice ", "bob", "CHARLIE", "dave smith"])\n'

fn("np.array of strings", "Text arrays have a fixed-width Unicode dtype.", '''
s = np.array(["cat", "elephant", "ox"])
(s, s.dtype, s.itemsize)
''', reads="<U8: unicode strings, longest is 8 chars. Assigning a longer string to an element silently truncates it!")
fn("np.dtypes.StringDType", "Variable-width strings (NumPy 2): no truncation, supports missing values.", '''
from numpy.dtypes import StringDType
s = np.array(["a", "much longer string"], dtype=StringDType())
(s, s.dtype)
''')
fn("StringDType(na_object=...)", "Missing values inside a string array.", '''
from numpy.dtypes import StringDType
s = np.array(["x", None, "z"], dtype=StringDType(na_object=None))
(s, np.isnan(np.array([1.0])))
''')
fn("np.strings.upper / lower", "Change case.", W + '''
(np.strings.upper(w), np.strings.lower(w))
''')
fn("np.strings.capitalize / title / swapcase", "More case styles.", W + '''
(np.strings.capitalize(w), np.strings.title(w), np.strings.swapcase(w))
''')
fn("np.strings.strip / lstrip / rstrip", "Remove spaces (or listed characters).", W + '''
(np.strings.strip(w), np.strings.lstrip(w), np.strings.rstrip(w))
''')
fn("np.strings.str_len", "Length of each string.", W + '''
np.strings.str_len(w)
''')
fn("np.strings.add", "Concatenate element-wise (like +).", '''
np.strings.add(np.array(["a", "b"]), np.array(["1", "2"]))
''')
fn("np.strings.multiply", "Repeat each string.", '''
np.strings.multiply(np.array(["ab", "x"]), 3)
''')
fn("np.strings.replace", "Replace text.", '''
np.strings.replace(np.array(["a-b-c", "x-y"]), "-", "_")
''')
fn("np.strings.find / rfind", "Position of a substring (-1 if absent).", '''
s = np.array(["banana", "apple", "cherry"])
(np.strings.find(s, "an"), np.strings.rfind(s, "an"), np.strings.find(s, "zz"))
''')
fn("np.strings.index / rindex", "Like find but raises if missing.", '''
np.strings.index(np.array(["banana", "bandana"]), "an")
''')
fn("np.strings.count", "How often a substring occurs.", '''
np.strings.count(np.array(["banana", "apple"]), "a")
''')
fn("np.strings.startswith / endswith", "Prefix and suffix tests.", '''
s = np.array(["report.csv", "photo.png", "data.csv"])
(np.strings.endswith(s, ".csv"), np.strings.startswith(s, "d"))
''')
fn("np.strings.isalpha / isdigit / isalnum", "Character-class tests.", '''
s = np.array(["abc", "123", "a1", " ", ""])
{"isalpha": np.strings.isalpha(s), "isdigit": np.strings.isdigit(s), "isalnum": np.strings.isalnum(s), "isspace": np.strings.isspace(s)}
''')
fn("np.strings.islower / isupper / istitle", "Case tests.", '''
s = np.array(["abc", "ABC", "Abc Def"])
(np.strings.islower(s), np.strings.isupper(s), np.strings.istitle(s))
''')
fn("np.strings.isnumeric / isdecimal", "Numeric character tests.", '''
s = np.array(["12", "1.5", "\\u00bd", "x"])
(np.strings.isnumeric(s), np.strings.isdecimal(s))
''')
fn("np.strings.center / ljust / rjust", "Pad to a width.", '''
s = np.array(["a", "bcd"])
(np.strings.center(s, 7, "*"), np.strings.ljust(s, 5, "."), np.strings.rjust(s, 5, "."))
''')
fn("np.strings.zfill", "Left-pad numbers with zeros.", '''
np.strings.zfill(np.array(["7", "42", "-3"]), 4)
''')
fn("np.strings.partition / rpartition", "Split at the first / last separator into three parts.", '''
np.strings.partition(np.array(["key=value", "a=b=c"]), "=")
''')
fn("np.strings.expandtabs", "Replace tabs by spaces.", '''
np.strings.expandtabs(np.array(["a\\tb"]), tabsize=4)
''')
fn("np.strings.translate", "Character mapping.", '''
np.strings.translate(np.array(["hello"]), str.maketrans("el", "ip"))
''')
fn("np.strings.encode / decode", "Text <-> bytes.", '''
b = np.strings.encode(np.array(["caf\\u00e9"]), "utf-8")
(b, np.strings.decode(b, "utf-8"))
''')
fn("np.strings.slice", "Slice every string (NumPy 2.3).", '''
np.strings.slice(np.array(["abcdef", "uvwxyz"]), 1, 4)
''')
fn("np.strings.mod", "printf-style formatting per element.", '''
np.strings.mod(np.array(["%d items", "%d things"]), 5)
''')
fn("np.strings.equal / not_equal / less ...", "Element-wise string comparison.", '''
s = np.array(["apple", "banana"])
(np.strings.equal(s, "apple"), np.strings.less(s, "b"))
''')
fn("a == 'text' (comparison operators)", "Comparison operators work directly on string arrays.", '''
s = np.array(["red", "green", "red", "blue"])
(s == "red", s[s != "red"], np.sum(s == "red"))
''')
fn("np.char.add", "Legacy np.char module (same idea, older API).", '''
np.char.add(np.array(["x", "y"]), "!")
''')
fn("np.char.split", "Split each string into a list (object array).", '''
np.char.split(np.array(["a b c", "d e"]))
''')
fn("np.char.join", "Join characters with a separator.", '''
np.char.join("-", np.array(["abc", "xy"]))
''')
fn("np.char.chararray", "Array subclass with string methods as attributes.", '''
c = np.char.array(["Ab", "cD"])
(c.upper(), c.lower(), type(c).__name__)
''', status="deprecated", alt="plain str array + np.strings.upper(...)")
fn("np.char.compare_chararrays", "Compare with any operator string.", '''
np.char.compare_chararrays(np.array(["a", "b"]), np.array(["a", "c"]), "==", False)
''')
fn("np.char.asarray / np.char.array", "Create char arrays.", '''
np.char.asarray(["a", "bb"]).itemsize
''', status="deprecated", alt="np.array([...]) with a str dtype")
fn("np.unique on strings", "Distinct words and counts.", '''
w = np.array("the cat saw the dog and the cat ran".split())
np.unique(w, return_counts=True)
''')
fn("np.sort on strings", "Alphabetical order.", '''
np.sort(np.array(["pear", "Apple", "banana", "apple"]))
''', reads="Uppercase sorts before lowercase (by character code).")
fn("np.isin on strings", "Membership tests.", '''
np.isin(np.array(["a", "b", "z"]), ["a", "z"])
''')
fn("string -> number: astype(float)", "Convert text numbers.", '''
np.array(["1.5", "2", "3.25"]).astype(float)
''')
fn("number -> string: astype(str)", "Format numbers as text.", '''
np.array([1.5, 2.0, 3.25]).astype(str)
''')
fn("np.format_float_positional / scientific", "Control float -> text conversion.", '''
(np.format_float_positional(1234.5678, precision=2), np.format_float_scientific(1234.5678, precision=2))
''')
fn("np.array2string", "Custom array printing.", '''
np.array2string(np.array([[1.5, 2.25], [3, 4]]), separator=" | ", precision=1)
''')
fn("np.array_repr / np.array_str", "The text of repr(a) and str(a).", '''
a = np.arange(4)
(np.array_repr(a), np.array_str(a))
''')

# =====================================================================================================  20  DATETIME
cat("datetime", "Dates & times (datetime64)", "datetime64 / timedelta64 arithmetic, ranges, business days, formatting.", "📅",
    intro="NumPy stores dates as integers counting units (days, seconds ...) since 1970-01-01, so date arithmetic is fast and vectorised. The unit is part of the dtype: datetime64[D], [h], [s], [ns] ...")

fn("np.datetime64", "A single date or timestamp.", '''
(np.datetime64("2024-03-15"), np.datetime64("2024-03-15T14:30"), np.datetime64("2024-03"), np.datetime64("2024"))
''', reads="The precision comes from how much you write: year, month, day, minute ...")
fn("datetime64 array", "Array of dates.", '''
d = np.array(["2024-01-15", "2024-02-20", "2024-12-31"], dtype="datetime64[D]")
(d, d.dtype)
''')
fn("np.arange for dates", "A sequence of days / months.", '''
(np.arange("2024-01-01", "2024-01-08", dtype="datetime64[D]"), np.arange("2024-01", "2024-07", dtype="datetime64[M]"))
''')
fn("datetime64 - datetime64", "Subtracting dates gives a timedelta64.", '''
a, b = np.datetime64("2024-03-01"), np.datetime64("2024-01-15")
(a - b, (a - b).astype(int))
''')
fn("np.timedelta64", "A duration.", '''
(np.timedelta64(5, "D"), np.timedelta64(90, "m"), np.datetime64("2024-01-30") + np.timedelta64(3, "D"))
''')
fn("timedelta division", "Convert a duration to another unit.", '''
td = np.timedelta64(90, "m")
(td / np.timedelta64(1, "h"), td.astype("timedelta64[s]"))
''')
fn("a.astype('datetime64[...]') (change unit)", "Convert resolution.", '''
d = np.array(["2024-03-15T13:45:10"], dtype="datetime64[s]")
(d.astype("datetime64[D]"), d.astype("datetime64[M]"), d.astype("datetime64[h]"))
''', reads="Going to a coarser unit truncates (floors) the time.")
fn("np.datetime_as_string", "Format dates as text.", '''
d = np.array(["2024-03-15T13:45"], dtype="datetime64[m]")
(np.datetime_as_string(d), np.datetime_as_string(d, unit="D"))
''')
fn("np.datetime_data", "Unit and count of a dtype.", '''
np.datetime_data(np.dtype("datetime64[15m]"))
''')
fn("np.is_busday", "Is it a working day (Mon-Fri)?", '''
np.is_busday(np.array(["2024-03-15", "2024-03-16", "2024-03-17"], dtype="datetime64[D]"))
''')
fn("np.busday_count", "Number of working days between two dates.", '''
np.busday_count("2024-03-01", "2024-04-01")
''')
fn("np.busday_offset", "Move forward/back by working days.", '''
(np.busday_offset("2024-03-15", 5), np.busday_offset("2024-03-16", 0, roll="forward"))
''', reads="Friday 15 March + 5 business days = Friday 22 March. A Saturday rolls forward to Monday.")
fn("np.busdaycalendar", "Custom weekmask and holidays.", '''
cal = np.busdaycalendar(weekmask="Mon Tue Wed Thu", holidays=["2024-03-13"])
np.busday_count("2024-03-11", "2024-03-18", busdaycal=cal)
''')
fn("np.isnat", "Missing dates (NaT).", '''
d = np.array(["2024-01-01", "NaT"], dtype="datetime64[D]")
(d, np.isnat(d))
''')
fn("min / max / sort of dates", "Dates behave like numbers.", '''
d = np.array(["2024-05-01", "2023-12-25", "2024-02-29"], dtype="datetime64[D]")
(d.min(), d.max(), np.sort(d), np.diff(np.sort(d)))
''')
fn("weekday of a date", "Day of week from the integer count.", '''
d = np.array(["2024-03-15", "2024-03-16"], dtype="datetime64[D]")
((d.astype("int64") + 4) % 7, np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])[(d.astype("int64") + 3) % 7])
''', reads="1970-01-01 was a Thursday, so shift the count by 3 to index Mon..Sun.")
fn("year / month from datetime64", "Extract parts by changing units.", '''
d = np.array(["2024-03-15", "2025-11-02"], dtype="datetime64[D]")
(d.astype("datetime64[Y]").astype(int) + 1970, d.astype("datetime64[M]").astype(int) % 12 + 1)
''')
fn("np.timedelta64 arithmetic", "Add and scale durations.", '''
a = np.array([1, 2, 3], dtype="timedelta64[h]")
(a * 2, a.sum(), a.astype("timedelta64[m]"))
''')
fn("np.datetime64('now')", "Current time (depends on when you run it).", '''
type(np.datetime64("now")).__name__
''')
fn("sensor timestamps to datetimes", "Seconds-since-start -> real timestamps.", '''
start = np.datetime64("2024-06-01T00:00:00")
ts = start + sensor[:5, 0].astype("timedelta64[s]")
ts
''')
fn("resample by day with datetime64", "Group readings per day using integer division.", '''
t = sensor[:, 0].astype(np.int64)
day = t // 86400
temp = sensor[:, 1]
np.round([np.nanmean(temp[day == d]) for d in np.unique(day)], 2)
''', reads="Mean temperature of each of the 7 days.")

# =====================================================================================================  21  IO
cat("io", "Input / output", "Read and write text, binary, .npy / .npz, CSV; control printing.", "💾",
    intro="Data files for practice are in the data/ folder (students.csv, sensor.csv, houses.csv). In the browser, files you save live only in memory for this page. "
          "Use np.load / np.save for fast binary storage, loadtxt / genfromtxt for CSV.")

fn("np.loadtxt", "Read a numeric text file into an array.", '''
a = np.loadtxt("data/houses.csv", delimiter=",", skiprows=1)
(a.shape, a[:3])
''', reads="Fast and strict: every cell must be a number.", params=["delimiter", "skiprows", "usecols - pick columns", "dtype", "comments - ignore lines after this char", "converters", "max_rows"])
fn("np.loadtxt (usecols, unpack)", "Pick columns and get each as its own array.", '''
area, price = np.loadtxt("data/houses.csv", delimiter=",", skiprows=1, usecols=(0, 3), unpack=True)
(area[:3], price[:3])
''')
fn("np.genfromtxt", "Read text with missing values (become nan).", '''
a = np.genfromtxt("data/students.csv", delimiter=",", skip_header=1, usecols=(5, 6, 7), filling_values=np.nan)
(a.shape, int(np.isnan(a).sum()))
''', reads="Blank cells are tolerated: that is the main difference from loadtxt.")
fn("np.genfromtxt (names=True)", "Use the header row as field names.", '''
t = np.genfromtxt("data/sensor.csv", delimiter=",", names=True)
(t.dtype.names, t["temperature_c"][:3])
''')
fn("np.genfromtxt (converters / missing_values)", "Handle special tokens.", '''
import io
txt = "1,2,N/A\\n4,-999,6"
np.genfromtxt(io.StringIO(txt), delimiter=",", missing_values=["N/A", "-999"], filling_values=np.nan)
''')
fn("np.savetxt", "Write an array as text.", '''
import io
buf = io.StringIO()
np.savetxt(buf, np.array([[1.5, 2], [3, 4.25]]), fmt="%.2f", delimiter=",", header="a,b", comments="")
buf.getvalue()
''', reads="fmt controls formatting; header becomes the first line.")
fn("np.save / np.load (.npy)", "Fast binary format for ONE array.", '''
import io
buf = io.BytesIO()
np.save(buf, np.arange(6).reshape(2, 3))
buf.seek(0)
np.load(buf)
''', reads="Stores dtype and shape exactly - no parsing needed.")
fn("np.savez / np.load (.npz)", "Several arrays in one archive.", '''
import io
buf = io.BytesIO()
np.savez(buf, a=np.arange(3), b=np.eye(2))
buf.seek(0)
z = np.load(buf)
(list(z.keys()), z["a"], z["b"])
''')
fn("np.savez_compressed", "Same, with zip compression.", '''
import io
b1, b2 = io.BytesIO(), io.BytesIO()
x = np.zeros(10000)
np.savez(b1, x=x); np.savez_compressed(b2, x=x)
(b1.getbuffer().nbytes, b2.getbuffer().nbytes)
''')
fn("np.load (mmap_mode)", "Memory-map big files instead of loading them (needs a real file).", '''
import tempfile, os
p = os.path.join(tempfile.gettempdir(), "demo_mmap.npy")
np.save(p, np.arange(10))
m = np.load(p, mmap_mode="r")
(type(m).__name__, m[3])
''')
fn("a.tofile / np.fromfile", "Raw binary dump and read (no header).", '''
import tempfile, os
p = os.path.join(tempfile.gettempdir(), "demo_raw.bin")
np.arange(5, dtype=np.int32).tofile(p)
np.fromfile(p, dtype=np.int32)
''', reads="No shape or dtype is stored - you must remember them.")
fn("a.tobytes / np.frombuffer", "Bytes round-trip.", '''
b = np.array([1, 2, 3], dtype=np.int16).tobytes()
(b, np.frombuffer(b, dtype=np.int16))
''')
fn("np.memmap", "Array backed by a file on disk.", '''
import tempfile, os
p = os.path.join(tempfile.gettempdir(), "demo.mm")
m = np.memmap(p, dtype="float32", mode="w+", shape=(4,))
m[:] = [1, 2, 3, 4]
m.flush()
np.memmap(p, dtype="float32", mode="r")
''', local="Memory-mapping needs real OS file descriptors, which the browser's Python (WebAssembly) does not provide. Copy this code and run it on your own computer.")
fn("np.fromregex", "Parse structured text with a regex.", '''
import io
np.fromregex(io.StringIO("x=1.5;y=2\\nx=3;y=4"), r"x=([\\d.]+);y=(\\d+)", dtype=[("x", float), ("y", int)])
''')
fn("np.set_printoptions", "Change how arrays print (globally).", '''
a = np.array([1 / 3, 2 / 3, 1e-9])
np.set_printoptions(precision=3, suppress=True)
s = repr(a)
np.set_printoptions(precision=8, suppress=False)
s
''', params=["precision - digits after the point", "suppress=True - no scientific notation", "threshold - max elements before summarising", "linewidth", "edgeitems", "formatter - per-type functions", "floatmode", "sign", "legacy"])
fn("np.printoptions (context manager)", "Temporary print settings.", '''
a = np.random.default_rng(0).random(4)
with np.printoptions(precision=2):
    s = str(a)
s
''')
fn("np.get_printoptions", "Current print settings.", '''
{k: v for k, v in np.get_printoptions().items() if k in ("precision", "threshold", "linewidth", "edgeitems", "suppress")}
''')
fn("np.set_printoptions(threshold=...)", "Print big arrays completely.", '''
with np.printoptions(threshold=20):
    s = str(np.arange(100))
s
''', reads="Arrays longer than threshold are shown with ... in the middle (summarised).")
fn("np.set_printoptions(formatter=...)", "Custom number format.", '''
with np.printoptions(formatter={"float": lambda x: f"${x:,.2f}"}):
    s = str(np.array([1234.5, 99.999]))
s
''')
fn("np.array2string", "Format an array into text with full control.", '''
np.array2string(np.arange(6).reshape(2, 3), separator=", ", prefix="data = ")
''')
fn("a.tolist / a.tostring replacement", "Convert to Python containers (tostring was removed).", '''
a = np.array([1, 2])
(a.tolist(), a.tobytes())
''')
fn("np.ndarray.dump / np.load pickle (allow_pickle)", "Pickled object arrays need allow_pickle=True when loading.", '''
import io
buf = io.BytesIO()
np.save(buf, np.array([{"a": 1}, None], dtype=object), allow_pickle=True)
buf.seek(0)
np.load(buf, allow_pickle=True)
''', tip="Only load pickles you trust - they can execute code.")
fn("np.lib.format (inspect .npy header)", "Read the header without loading data.", '''
import io
buf = io.BytesIO()
np.save(buf, np.zeros((3, 4), dtype=np.float32))
buf.seek(0)
np.lib.format.read_magic(buf)
np.lib.format.read_array_header_1_0(buf)
''')
fn("read CSV text from a string", "Use io.StringIO to practise without files.", '''
import io
np.loadtxt(io.StringIO("1,2,3\\n4,5,6"), delimiter=",", dtype=int)
''')
fn("np.loadtxt (converters, dtype=str)", "Read text columns.", '''
import io
np.loadtxt(io.StringIO("a 1\\nb 2"), dtype=str)
''')
fn("np.fromfile (text, sep)", "Read numbers separated by spaces.", '''
import tempfile, os
p = os.path.join(tempfile.gettempdir(), "demo.txt")
open(p, "w").write("1 2 3 4")
np.fromfile(p, sep=" ")
''')
fn("np.save with allow_pickle=False", "Default safe saving of numeric arrays.", '''
import io
b = io.BytesIO()
np.save(b, np.arange(3), allow_pickle=False)
b.getbuffer().nbytes
''')
fn("np.DataSource", "Open local or remote files transparently (local use).", '''
ds = np.lib.npyio.DataSource(None)
ds.exists("data/houses.csv")
''')
fn("np.load vs np.loadtxt speed", "Binary loads are far faster than text parsing.", '''
import io, time
a = np.random.default_rng(0).random((20000, 5))
b = io.BytesIO(); np.save(b, a); b.seek(0)
t = io.StringIO(); np.savetxt(t, a); t.seek(0)
t0 = time.perf_counter(); np.load(b); t1 = time.perf_counter(); np.loadtxt(t); t2 = time.perf_counter()
{"npy faster by about": round((t2 - t1) / max(t1 - t0, 1e-9))}
''')

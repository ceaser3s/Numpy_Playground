from ._core import cat, fn

# =====================================================================================================  4  INDEXING
cat("indexing", "Indexing, slicing & selection", "Pick elements: positions, slices, conditions, lists of positions, and where-style searches.", "🎯",
    intro="Three kinds of indexing: BASIC (integers and slices - returns a view), BOOLEAN (a True/False mask - returns a copy) and FANCY (arrays of positions - returns a copy). "
          "Understanding which one you are using tells you whether changing the result changes the original.")

fn("a[i] (1-D indexing)", "Position counting starts at 0; negative numbers count from the end.", '''
a = np.array([10, 20, 30, 40, 50])
(a[0], a[2], a[-1], a[-2])
''')
fn("a[i, j] (2-D indexing)", "Row first, then column - separated by a comma.", '''
a = np.arange(1, 13).reshape(3, 4)
(a, a[1, 2], a[-1, -1], a[0][3])
''', reads="a[1, 2] is the efficient form. a[1][2] works too but builds a temporary row array first.")
fn("a[start:stop:step] (slicing)", "Slices include start, exclude stop; step can be negative.", '''
a = np.arange(10)
{"a[2:6]": a[2:6], "a[:3]": a[:3], "a[7:]": a[7:], "a[::2]": a[::2], "a[::-1]": a[::-1], "a[-3:]": a[-3:], "a[1:8:3]": a[1:8:3]}
''')
fn("a[r1:r2, c1:c2] (2-D slice)", "Slice rows and columns at once.", '''
a = np.arange(1, 21).reshape(4, 5)
(a, a[1:3, 2:5], a[:, 0], a[0, :], a[::2, ::2])
''', reads="A bare colon : means 'everything along this axis'. a[:, 0] is the first COLUMN.")
fn("slices are views", "Changing a slice changes the original array!", '''
a = np.arange(6)
s = a[1:4]
s[0] = 99
a
''', reads="No data was copied: s looks at a's memory. Use a[1:4].copy() when you need independence.", tip="This is the #1 source of surprising bugs for beginners.")
fn("a[..., i] (Ellipsis)", "... stands for 'as many colons as needed'.", '''
a = np.arange(24).reshape(2, 3, 4)
(a[..., 0].shape, a[0, ...].shape, a[..., -1])
''')
fn("a[:, np.newaxis]", "Insert a new axis of length 1 (turn a vector into a column).", '''
v = np.array([1, 2, 3])
(v.shape, v[:, np.newaxis].shape, v[np.newaxis, :].shape, v[:, None])
''', reads="np.newaxis is just an alias for None.")
fn("a[mask] (boolean indexing)", "Keep the elements where the mask is True.", '''
a = np.array([5, 12, 7, 20, 3, 15])
mask = a > 6
(mask, a[mask])
''', reads="The result is a flat COPY of the selected values.")
fn("a[(c1) & (c2)] (combine conditions)", "Use & | ~ with parentheses - NOT and / or / not.", '''
a = np.arange(1, 21)
(a[(a > 5) & (a < 12)], a[(a < 4) | (a > 17)], a[~(a % 2 == 0)])
''')
fn("a[mask] = value", "Assign through a mask: replace many values at once.", '''
a = np.array([3, -1, 4, -5, 9])
a[a < 0] = 0
a
''')
fn("a[mask] on 2-D (rows)", "A row-level mask keeps whole rows.", '''
s = students[:6, [0, 5, 6]]
s[s[:, 1] > 50]
''', reads="students columns: id, math, physics. Only students with math > 50 remain.")
fn("a[[i, j, k]] (fancy indexing)", "A list of positions - any order, repeats allowed.", '''
a = np.array([10, 20, 30, 40, 50])
a[[4, 0, 0, 2]]
''', reads="Result is a new array (copy), in the order you asked for.")
fn("a[[rows], [cols]] (paired fancy)", "Two index lists pick PAIRS (r0,c0), (r1,c1) ...", '''
a = np.arange(1, 17).reshape(4, 4)
(a, a[[0, 1, 3], [1, 2, 0]])
''', reads="Elements a[0,1], a[1,2], a[3,0].")
fn("a[np.ix_(rows, cols)]", "Pick a sub-grid: every combination of the given rows and columns.", '''
a = np.arange(1, 26).reshape(5, 5)
a[np.ix_([0, 2, 4], [1, 3])]
''', reads="Compare with paired fancy indexing: ix_ gives a 3x2 block, not 2 pairs.")
fn("np.take", "Pick by position along an axis.", '''
a = np.array([[10, 20, 30], [40, 50, 60]])
(np.take(a, [0, 2], axis=1), np.take(a, [1, 4]), np.take(a, [5, 7], mode="wrap"))
''', params=["indices", "axis - None flattens first", "mode - 'raise', 'wrap' or 'clip' for out-of-range"])
fn("np.take_along_axis", "Pick one element per row/column using an index array (e.g. from argsort/argmax).", '''
a = np.array([[30, 10, 20], [3, 1, 2]])
idx = np.argsort(a, axis=1)
(idx, np.take_along_axis(a, idx, axis=1))
''', reads="Sorts every row independently - the standard use with argsort.")
fn("np.put", "Assign by flat position (in place).", '''
a = np.zeros(6, dtype=int)
np.put(a, [1, 4], [7, 9])
a
''')
fn("np.put_along_axis", "Assign using index arrays along an axis.", '''
a = np.zeros((2, 3), dtype=int)
np.put_along_axis(a, np.array([[0], [2]]), 5, axis=1)
a
''')
fn("np.putmask", "Assign where a mask is True (values repeat if short).", '''
a = np.arange(8)
np.putmask(a, a % 2 == 0, [-1, -2])
a
''')
fn("np.place", "Replace masked positions with values in order.", '''
a = np.arange(8)
np.place(a, a > 4, [100, 200, 300])
a
''')
fn("np.copyto", "Copy values into an existing array (with an optional mask).", '''
dst = np.zeros(5)
np.copyto(dst, [1, 2, 3, 4, 5], where=[True, False, True, False, True])
dst
''')
fn("np.where (condition only)", "Positions where the condition is True.", '''
a = np.array([4, 9, 2, 7, 9])
np.where(a > 5)
''', reads="Returns a tuple with one array per dimension (here 1-D, so a 1-tuple).")
fn("np.where (cond, x, y)", "Vectorised if/else: pick x where True, y where False.", '''
a = np.array([-2, 5, -1, 8])
np.where(a > 0, a, 0)
''', reads="A branch-free replacement for a loop with if/else - one of NumPy's most useful functions.")
fn("np.nonzero", "Indices of non-zero (or True) elements.", '''
a = np.array([[0, 3, 0], [5, 0, 6]])
np.nonzero(a)
''', reads="Two arrays: row indices and column indices of 3, 5, 6.")
fn("np.argwhere", "Same as nonzero but returns an (n, ndim) array of coordinates.", '''
a = np.array([[0, 3, 0], [5, 0, 6]])
np.argwhere(a)
''')
fn("np.flatnonzero", "Indices of non-zeros in the flattened array.", '''
np.flatnonzero(np.array([[0, 3, 0], [5, 0, 6]]))
''')
fn("np.extract", "Pick values by condition (like boolean indexing).", '''
a = np.arange(10)
np.extract(a % 3 == 0, a)
''')
fn("np.compress", "Select slices along an axis using a boolean list.", '''
a = np.arange(12).reshape(3, 4)
(np.compress([True, False, True], a, axis=0), np.compress([False, True, True, False], a, axis=1))
''')
fn("np.choose", "Build an array by picking from several choices per element.", '''
choices = [np.array([0, 0, 0]), np.array([10, 10, 10]), np.array([20, 20, 20])]
np.choose([2, 0, 1], choices)
''')
fn("np.select", "Multi-way np.where: first matching condition wins.", '''
x = np.array([5, 15, 25, 35])
np.select([x < 10, x < 30], ["low", "mid"], default="high")
''', reads="Like an if / elif / else chain.")
fn("np.piecewise", "Apply a different function to each range.", '''
x = np.linspace(-2, 2, 9)
np.piecewise(x, [x < 0, x >= 0], [lambda t: -t, lambda t: t ** 2])
''')
fn("np.diag_indices", "Indices of the main diagonal of an n x n array.", '''
a = np.zeros((4, 4), dtype=int)
a[np.diag_indices(4)] = 1
a
''')
fn("np.tril_indices / triu_indices", "Indices of the lower / upper triangle.", '''
a = np.arange(16).reshape(4, 4)
(a[np.tril_indices(4)], a[np.triu_indices(4, k=1)])
''')
fn("np.mask_indices", "Indices selected by a mask function (tril / triu).", '''
np.mask_indices(3, np.triu)
''')
fn("np.unravel_index", "Flat position -> (row, col, ...).", '''
np.unravel_index(7, (3, 4))
''', reads="Position 7 of a 3x4 array is row 1, column 3.")
fn("np.ravel_multi_index", "(row, col) -> flat position.", '''
np.ravel_multi_index((1, 3), (3, 4))
''')
fn("np.argmax (find the location)", "Position of the biggest value; unravel for N-D.", '''
a = np.array([[3, 9, 1], [7, 2, 8]])
(np.argmax(a), np.unravel_index(np.argmax(a), a.shape))
''')
fn("np.ndenumerate", "Iterate over (index, value) pairs of any array.", '''
list(np.ndenumerate(np.array([[1, 2], [3, 4]])))
''')
fn("np.ndindex", "Iterate over all index tuples of a shape.", '''
list(np.ndindex(2, 3))
''')
fn("np.nditer", "Powerful multi-array iterator.", '''
a = np.arange(6).reshape(2, 3)
[int(x) for x in np.nditer(a)]
''')
fn("np.lib.stride_tricks.sliding_window_view (selection)", "All windows of a given length as a 2-D view.", '''
np.lib.stride_tricks.sliding_window_view(np.arange(7), 3)
''')
fn("np.delete (select by removing)", "Remove elements by index.", '''
np.delete(np.arange(8), [1, 3, 5])
''')
fn("a[::-1] vs np.flip", "Reverse an axis.", '''
a = np.arange(6).reshape(2, 3)
(a[:, ::-1], np.flip(a, axis=1))
''')
fn("a.take vs a[...] speed note", "Different ways to say the same selection.", '''
a = np.arange(10) * 10
idx = np.array([1, 3, 5])
(a[idx], a.take(idx), np.array_equal(a[idx], a.take(idx)))
''')
fn("a.item / a.itemset replacement", "Read one element as a Python number.", '''
a = np.array([[1, 2], [3, 4]])
a.item(1, 1)
''')
fn("boolean mask from isin", "Select rows whose value is in a list.", '''
ages = students[:, 1]
students[np.isin(ages, [17, 18])][:5, :3]
''', reads="isin gives a mask - then boolean indexing selects the rows.")
fn("count and select with masks", "Mask + sum/mean give counts and percentages.", '''
math = students[:, 5]
mask = math >= 60
{"passed": int(np.nansum(mask)), "fraction": float(np.mean(mask[~np.isnan(math)]))}
''', reads="Comparisons with nan are False, so blank scores are not counted as passes.")
fn("negative-step & copy semantics", "Reversed views share memory too.", '''
a = np.arange(5)
r = a[::-1]
r[0] = 100
a
''')

# =====================================================================================================  5  SHAPE
cat("shape", "Shape manipulation", "Reshape, flatten, transpose, add/remove axes, flip, rotate, roll.", "📐",
    intro="Reshaping changes how the same numbers are grouped without changing the numbers themselves. Most reshaping functions return VIEWS, so they are free.")

fn("a.reshape", "Give the same data a new shape (total size must match).", '''
a = np.arange(12)
(a.reshape(3, 4), a.reshape(2, 2, 3).shape, a.reshape(4, -1).shape)
''', reads="-1 means 'work this dimension out for me'. Reshape reads and fills in row-major (C) order by default.",
   params=["newshape - tuple (one -1 allowed)", "order - 'C' (default) or 'F' fill order"])
fn("a.reshape (order='F')", "Fill column by column (Fortran order).", '''
np.arange(6).reshape(2, 3, order="F")
''')
fn("np.reshape", "Function form.", '''
np.reshape(np.arange(6), (3, 2))
''')
fn("a.ravel", "Flatten to 1-D, returning a view when possible.", '''
a = np.arange(6).reshape(2, 3)
r = a.ravel()
r[0] = 99
(r, a)
''', reads="a changed too: ravel returned a view.")
fn("a.flatten", "Flatten to 1-D, ALWAYS copying.", '''
a = np.arange(6).reshape(2, 3)
f = a.flatten()
f[0] = 99
(f, a)
''')
fn("np.ravel (order)", "order='F' flattens column by column.", '''
a = np.arange(6).reshape(2, 3)
(np.ravel(a, order="C"), np.ravel(a, order="F"))
''')
fn("np.resize", "Resize by repeating (or cutting) data - returns a new array.", '''
np.resize(np.array([1, 2, 3]), (2, 4))
''', reads="Missing values are filled by repeating the input again from the start.")
fn("a.resize (in place)", "Changes the array in place (pads with zeros).", '''
a = np.array([1, 2, 3])
a.resize(5, refcheck=False)
a
''')
fn("a.transpose / np.transpose", "Reverse (or permute) the axes.", '''
a = np.arange(24).reshape(2, 3, 4)
(a.transpose().shape, np.transpose(a, (1, 0, 2)).shape, a.transpose(2, 0, 1).shape)
''')
fn("np.permute_dims", "NumPy 2 name for transpose with an explicit axis order.", '''
np.permute_dims(np.zeros((2, 3, 4)), (2, 0, 1)).shape
''')
fn("np.swapaxes", "Swap two axes.", '''
np.swapaxes(np.zeros((2, 3, 4)), 0, 2).shape
''')
fn("np.moveaxis", "Move axes to new positions.", '''
np.moveaxis(np.zeros((2, 3, 4)), 0, -1).shape
''', reads="Axis 0 moved to the end: (3, 4, 2).")
fn("np.rollaxis", "Older way to move an axis (prefer moveaxis).", '''
np.rollaxis(np.zeros((2, 3, 4)), 2).shape
''')
fn("np.matrix_transpose", "Transpose only the LAST two axes (for stacks of matrices).", '''
np.matrix_transpose(np.zeros((5, 2, 3))).shape
''')
fn("np.squeeze", "Remove axes of length 1.", '''
a = np.zeros((1, 3, 1, 2))
(a.shape, np.squeeze(a).shape, np.squeeze(a, axis=0).shape)
''')
fn("np.expand_dims", "Insert a length-1 axis at a position.", '''
v = np.array([1, 2, 3])
(np.expand_dims(v, 0).shape, np.expand_dims(v, 1).shape, np.expand_dims(v, (0, 2)).shape)
''')
fn("np.atleast_1d / 2d / 3d", "Guarantee a minimum number of dimensions.", '''
(np.atleast_1d(5).shape, np.atleast_2d([1, 2, 3]).shape, np.atleast_3d([[1, 2], [3, 4]]).shape)
''')
fn("np.broadcast_to", "Expand an array to a larger shape (read-only view).", '''
np.broadcast_to(np.array([[1], [2], [3]]), (3, 4))
''')
fn("np.broadcast_arrays", "Broadcast several arrays to a common shape.", '''
a, b = np.broadcast_arrays(np.array([[1], [2]]), np.array([10, 20, 30]))
(a, b)
''')
fn("np.broadcast_shapes", "Which shape would broadcasting give?", '''
(np.broadcast_shapes((5, 1), (1, 4)), np.broadcast_shapes((3,), (2, 3), (2, 2, 3)))
''')
fn("np.flip", "Reverse along given axes (default: all).", '''
a = np.arange(6).reshape(2, 3)
(np.flip(a), np.flip(a, axis=0), np.flip(a, axis=1))
''')
fn("np.fliplr / np.flipud", "Mirror left-right / up-down.", '''
a = np.arange(6).reshape(2, 3)
(np.fliplr(a), np.flipud(a))
''')
fn("np.rot90", "Rotate by 90 degrees (counter-clockwise) in a plane.", '''
a = np.arange(6).reshape(2, 3)
(np.rot90(a), np.rot90(a, k=2), np.rot90(a, k=-1))
''')
fn("np.roll", "Shift elements, wrapping around the end.", '''
a = np.arange(1, 7)
(np.roll(a, 2), np.roll(a, -1), np.roll(np.arange(6).reshape(2, 3), 1, axis=1))
''')
fn("np.swapaxes on images", "Convert (H, W, C) to (C, H, W).", '''
img3 = np.zeros((32, 48, 3))
np.moveaxis(img3, -1, 0).shape
''')
fn("np.vsplit shape example", "See splitting for pieces; here: reshape to blocks.", '''
a = np.arange(16).reshape(4, 4)
a.reshape(2, 2, 2, 2).swapaxes(1, 2).reshape(4, 2, 2)
''', reads="Rearranging a 4x4 into four 2x2 tiles with reshape + swapaxes.")
fn("a.T on 1-D arrays", "Transposing a 1-D array does nothing!", '''
v = np.array([1, 2, 3])
(v.T.shape, v[:, None].shape)
''', reads="To get a column vector use v[:, None] or v.reshape(-1, 1).", tip="A frequent beginner trap when doing matrix maths with vectors.")
fn("a.shape = (...)", "DEPRECATED in NumPy 2.5: assigning to .shape reshaped in place. Use reshape (copy=False raises if a copy would be needed).", '''
a = np.arange(6)
a = a.reshape((2, 3), copy=False)
a
''', status="deprecated", alt="a = a.reshape(new_shape)")
fn("np.ndarray.flat vs ravel", "flat is an iterator; ravel makes an array.", '''
a = np.arange(6).reshape(2, 3)
(type(a.flat).__name__, type(a.ravel()).__name__)
''')
fn("np.newaxis for outer operations", "Column x row broadcasting gives a table.", '''
x = np.arange(1, 5)
x[:, None] * x[None, :]
''', reads="A multiplication table with no loops.")
fn("np.apply_along_axis (reshape-free)", "Run a function on every 1-D slice.", '''
a = np.arange(12).reshape(3, 4)
np.apply_along_axis(lambda row: row.max() - row.min(), 1, a)
''')

# =====================================================================================================  6  JOINING
cat("joining", "Joining, splitting & padding", "Concatenate, stack, split, repeat, tile, insert, delete, pad.", "🧩",
    intro="Joining glues arrays together; splitting cuts them apart. Every join needs matching shapes on all axes except the one you join along.")

fn("np.concatenate", "Join arrays along an EXISTING axis.", '''
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
(np.concatenate([a, b], axis=0), np.concatenate([a, a], axis=1))
''', params=["arrays - a sequence", "axis - 0 default; None flattens first"])
fn("np.concat", "NumPy 2 alias of concatenate.", '''
np.concat([np.array([1, 2]), np.array([3])])
''')
fn("np.stack", "Join arrays along a NEW axis.", '''
a, b = np.array([1, 2, 3]), np.array([4, 5, 6])
(np.stack([a, b]).shape, np.stack([a, b], axis=1).shape, np.stack([a, b], axis=1))
''', reads="Two arrays of shape (3,) become (2, 3) with axis=0 or (3, 2) with axis=1.")
fn("np.vstack", "Stack vertically (rows).", '''
np.vstack([np.array([1, 2, 3]), np.array([4, 5, 6])])
''')
fn("np.hstack", "Stack horizontally (columns for 2-D, glue for 1-D).", '''
(np.hstack([np.array([1, 2]), np.array([3])]), np.hstack([np.ones((2, 1)), np.zeros((2, 2))]))
''')
fn("np.dstack", "Stack depth-wise (third axis).", '''
np.dstack([np.ones((2, 2)), np.zeros((2, 2))]).shape
''')
fn("np.column_stack", "1-D arrays become columns of a 2-D array.", '''
np.column_stack([np.array([1, 2, 3]), np.array([10, 20, 30])])
''')
fn("np.block", "Assemble an array from nested blocks (like a block matrix).", '''
A, B = np.eye(2, dtype=int), np.zeros((2, 2), dtype=int)
np.block([[A, B], [B, A * 5]])
''')
fn("np.append", "Return a new array with values added at the end (slow in loops).", '''
np.append([1, 2, 3], [4, 5])
''', tip="Each call copies the whole array. In loops, collect in a list and convert once.")
fn("np.insert", "Insert values before given positions.", '''
np.insert(np.array([1, 2, 5]), 2, [3, 4])
''')
fn("np.delete", "Remove elements/rows/columns by position.", '''
a = np.arange(12).reshape(3, 4)
(np.delete(a, 1, axis=0), np.delete(a, [0, 2], axis=1))
''')
fn("np.split", "Cut into EQUAL parts (or at given positions).", '''
a = np.arange(9)
(np.split(a, 3), np.split(a, [2, 5]))
''', reads="split(a, 3) needs 9 divisible by 3. split(a, [2, 5]) cuts before positions 2 and 5.")
fn("np.array_split", "Like split but allows UNEQUAL parts.", '''
np.array_split(np.arange(10), 3)
''')
fn("np.hsplit", "Split columns.", '''
np.hsplit(np.arange(12).reshape(3, 4), 2)
''')
fn("np.vsplit", "Split rows.", '''
np.vsplit(np.arange(12).reshape(4, 3), 2)
''')
fn("np.dsplit", "Split along the third axis.", '''
np.dsplit(np.arange(16).reshape(2, 2, 4), 2)
''')
fn("np.unstack", "NumPy 2.1: split an array along an axis into a tuple of sub-arrays.", '''
np.unstack(np.arange(6).reshape(2, 3))
''')
fn("np.repeat", "Repeat each ELEMENT.", '''
(np.repeat([1, 2, 3], 2), np.repeat([[1, 2], [3, 4]], 2, axis=0), np.repeat([1, 2], [3, 1]))
''', reads="A list of counts repeats each element a different number of times.")
fn("np.tile", "Repeat the WHOLE array like floor tiles.", '''
(np.tile([1, 2], 3), np.tile([[1, 2], [3, 4]], (2, 2)))
''')
fn("np.pad", "Add a border around an array.", '''
a = np.arange(1, 5).reshape(2, 2)
(np.pad(a, 1), np.pad(a, ((0, 1), (2, 0)), constant_values=9), np.pad(np.arange(1, 4), 2, mode="edge"), np.pad(np.arange(1, 4), 2, mode="reflect"))
''', reads="Modes: constant, edge, reflect, symmetric, wrap, mean, linear_ramp ... used heavily in image processing.",
   params=["pad_width - int, (before, after) or per-axis pairs", "mode", "constant_values"])
fn("np.concatenate with different dtypes", "The result type is promoted.", '''
r = np.concatenate([np.array([1, 2]), np.array([0.5])])
(r, r.dtype)
''')
fn("np.hstack vs column_stack", "They differ for 1-D inputs.", '''
a, b = np.array([1, 2]), np.array([3, 4])
(np.hstack([a, b]), np.column_stack([a, b]))
''')
fn("np.row_stack", "REMOVED in NumPy 2.0 - use np.vstack.", '''
np.vstack([[1, 2], [3, 4]])
''', status="removed", alt="np.vstack")
fn("np.trim_zeros", "Remove leading/trailing zeros.", '''
np.trim_zeros(np.array([0, 0, 3, 0, 5, 0, 0]))
''')
fn("np.unique (remove duplicates)", "See Sorting & sets for all options.", '''
np.unique([3, 1, 3, 2, 1])
''')
fn("np.lib.stride_tricks.sliding_window_view", "Overlapping windows without copying.", '''
w = np.lib.stride_tricks.sliding_window_view(np.arange(8), 4)
(w, w.mean(axis=1))
''', reads="A moving average in one line.")
fn("np.ma.concatenate / np.r_ for masked", "Concatenation that keeps masks (see Masked arrays).", '''
np.ma.concatenate([np.ma.masked_array([1, 2], [0, 1]), np.ma.masked_array([3], [0])])
''')
fn("np.vstack for building a table", "Typical use: stack rows of equal length.", '''
rows = [np.array([i, i ** 2, i ** 3]) for i in range(1, 5)]
np.vstack(rows)
''')

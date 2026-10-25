from ._core import cat, fn

# =====================================================================================================  12  SORTING / SEARCHING / SETS
cat("sorting", "Sorting, searching & sets", "Sort, argsort, lexsort, partition, searchsorted, unique and set operations.", "🔎",
    intro="Sorting returns a sorted COPY (np.sort / a.sort() sorts in place). argsort returns the ORDER (positions), which lets you sort other arrays the same way. "
          "Set functions work on arrays of unique values.")

fn("np.sort", "Sorted copy (ascending by default).", '''
a = np.array([[3, 1, 2], [9, 7, 8]])
(np.sort(a), np.sort(a, axis=0), np.sort(a, axis=None), np.sort(a)[:, ::-1])
''', reads="Default axis=-1 sorts each ROW. For descending order reverse the result with [::-1].",
   params=["axis - default -1; None flattens", "kind - 'quicksort', 'mergesort', 'stable', 'heapsort'", "order - field names for structured arrays", "stable=True (NumPy 2)"])
fn("a.sort (in place)", "Sorts the array itself and returns None.", '''
a = np.array([3, 1, 2])
r = a.sort()
(a, r)
''')
fn("np.argsort", "Positions that WOULD sort the array.", '''
a = np.array([30, 10, 20])
order = np.argsort(a)
(order, a[order])
''', reads="argsort tells 'element 1 is smallest, then 2, then 0'. Use it to reorder related arrays together.")
fn("argsort to sort one array by another", "Sort names by score.", '''
names = np.array(["ana", "raj", "li", "tom"])
score = np.array([72, 95, 88, 60])
order = np.argsort(-score)
(names[order], score[order])
''', reads="Negating the scores gives descending order.")
fn("np.argsort (stable ties)", "kind='stable' keeps the original order of equal values.", '''
a = np.array([2, 1, 2, 1, 2])
(np.argsort(a, kind="stable"), np.argsort(a, kind="quicksort"))
''')
fn("np.lexsort", "Sort by several keys - the LAST key is the primary one.", '''
last = np.array(["Smith", "Jones", "Smith", "Adams"])
first = np.array(["Zed", "Amy", "Bob", "Cy"])
order = np.lexsort((first, last))
list(zip(last[order], first[order]))
''', reads="Sorted by surname first, then by first name.")
fn("sort rows by a column", "Sort a whole table by one column.", '''
t = houses[:6]
t[np.argsort(t[:, 3])]
''', reads="Rows reordered by price (column 3) ascending.")
fn("np.partition", "Partial sort: the k-th element lands in its sorted place, smaller ones before it.", '''
a = np.array([7, 2, 9, 4, 1, 8])
np.partition(a, 2)
''', reads="Position 2 holds the 3rd smallest (4); everything before is smaller, after is larger - but neither side is sorted. Faster than a full sort.")
fn("np.argpartition", "Indices for a partial sort - the fast way to get the top-k.", '''
a = np.array([7, 2, 9, 4, 1, 8])
idx = np.argpartition(a, -3)[-3:]
(idx, np.sort(a[idx]))
''', reads="The three largest values (7, 8, 9) without sorting everything.")
fn("np.searchsorted", "Where would a value be inserted to keep a sorted array sorted? (binary search)", '''
a = np.array([10, 20, 30, 40])
(np.searchsorted(a, 25), np.searchsorted(a, [5, 20, 45]), np.searchsorted(a, 20, side="right"))
''', reads="Fast lookup in sorted data: O(log n) per query.")
fn("np.sort_complex", "Sort complex numbers by real, then imaginary part.", '''
np.sort_complex(np.array([2 + 1j, 1 + 5j, 1 + 2j]))
''')
fn("np.msort", "REMOVED in NumPy 2.0: use np.sort(a, axis=0).", '''
np.sort(np.array([[3, 1], [2, 9]]), axis=0)
''', status="removed", alt="np.sort(a, axis=0)")
fn("np.argmax / argmin (search)", "Position of extremes.", '''
a = np.array([4, 9, 2])
(np.argmax(a), np.argmin(a))
''')
fn("np.unique", "Sorted unique values.", '''
np.unique(np.array([3, 1, 2, 3, 1, 5]))
''')
fn("np.unique (return_*)", "Also get first indices, inverse mapping and counts.", '''
a = np.array([3, 1, 2, 3, 1, 5])
u, first, inv, cnt = np.unique(a, return_index=True, return_inverse=True, return_counts=True)
{"unique": u, "first index": first, "inverse": inv, "counts": cnt, "rebuilt": u[inv]}
''', reads="u[inv] rebuilds the original array - 'inverse' is the label-encoding of each element.")
fn("np.unique (axis=0)", "Unique ROWS.", '''
np.unique(np.array([[1, 2], [3, 4], [1, 2]]), axis=0)
''')
fn("np.unique_counts / unique_values / unique_inverse", "NumPy 2 array-API style named results.", '''
r = np.unique_counts(np.array([1, 1, 2, 3, 3, 3]))
(r.values, r.counts, np.unique_values(np.array([2, 1, 2])))
''')
fn("np.unique_all", "Everything at once.", '''
r = np.unique_all(np.array([5, 3, 5]))
(r.values, r.indices, r.inverse_indices, r.counts)
''')
fn("np.union1d", "Sorted union.", '''
np.union1d([1, 3, 5], [3, 4, 5, 6])
''')
fn("np.intersect1d", "Common values.", '''
np.intersect1d([1, 3, 5, 7], [3, 4, 5, 6])
''')
fn("np.setdiff1d", "Values in the first but not the second.", '''
np.setdiff1d([1, 2, 3, 4, 5], [2, 4])
''')
fn("np.setxor1d", "Values in exactly one of the two.", '''
np.setxor1d([1, 2, 3], [3, 4])
''')
fn("np.isin (membership)", "Mask version of 'in'.", '''
np.isin([1, 2, 3, 4], [2, 4])
''')
fn("np.searchsorted for binning", "Assign values to ranges.", '''
edges = np.array([0, 50, 75, 90])
marks = np.array([30, 55, 80, 95, 75])
np.array(["F", "C", "B", "A"])[np.searchsorted(edges, marks, side="right") - 1]
''', reads="A tiny grade lookup: 75 falls in [75, 90) -> 'B'.")
fn("np.partition for median", "The median needs only a partial sort.", '''
a = np.array([9, 1, 7, 3, 5])
np.partition(a, 2)[2]
''')
fn("top-k largest rows", "Find the 3 most expensive houses.", '''
top = np.argsort(houses[:, 3])[-3:][::-1]
houses[top]
''')
fn("rank values", "Rank = argsort of argsort.", '''
a = np.array([30, 10, 20, 40])
np.argsort(np.argsort(a))
''', reads="0 = smallest, 3 = largest.")
fn("np.flip + argsort for descending", "Descending order of a column.", '''
a = np.array([3, 1, 2])
a[np.argsort(a)[::-1]]
''')
fn("np.nan in sort", "nan values are sorted to the END.", '''
np.sort(np.array([3.0, np.nan, 1.0, 2.0]))
''')
fn("np.sort (structured, order=)", "Sort records by field names.", '''
people = np.array([("Ann", 31), ("Bob", 25), ("Cy", 31)], dtype=[("name", "U5"), ("age", int)])
np.sort(people, order=["age", "name"])
''')
fn("np.count of sorted via searchsorted", "Count values in a range of a sorted array.", '''
a = np.sort(students[:, 1])
np.searchsorted(a, 21, side="right") - np.searchsorted(a, 19, side="left")
''', reads="How many students are aged 19, 20 or 21.")
fn("np.array_equal after sort", "Compare two arrays ignoring order.", '''
np.array_equal(np.sort([3, 1, 2]), np.sort([2, 3, 1]))
''')

# =====================================================================================================  13  LINEAR ALGEBRA
cat("linalg", "Linear algebra", "Matrix products, inverse, determinant, solve, eigenvalues, SVD, QR, norms, least squares.", "🧮",
    intro="np.linalg wraps LAPACK. Most functions also accept STACKS of matrices (shape (..., n, n)) and process each one. "
          "Remember: * is element-wise, @ is the matrix product.")

fn("A @ B (matmul)", "Matrix product.", '''
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
(A @ B, A * B)
''', reads="(A @ B)[i, j] = row i of A dotted with column j of B.")
fn("np.linalg.det", "Determinant (0 means not invertible).", '''
(np.linalg.det(np.array([[1, 2], [3, 4]])), np.linalg.det(np.array([[1, 2], [2, 4]])))
''')
fn("np.linalg.slogdet", "Sign and log of the determinant (stable for big matrices).", '''
sign, logdet = np.linalg.slogdet(np.array([[2.0, 0], [0, 3.0]]))
(sign, logdet, np.exp(logdet))
''')
fn("np.linalg.inv", "Matrix inverse.", '''
A = np.array([[4.0, 7.0], [2.0, 6.0]])
Ai = np.linalg.inv(A)
(np.round(Ai, 3), np.round(A @ Ai, 10))
''', reads="A @ inv(A) is the identity matrix.")
fn("np.linalg.pinv", "Pseudo-inverse (works for singular / non-square matrices).", '''
np.round(np.linalg.pinv(np.array([[1.0, 2], [2, 4], [3, 6]])), 4)
''')
fn("np.linalg.solve", "Solve A x = b (preferred over inv(A) @ b).", '''
A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([9.0, 8.0])
x = np.linalg.solve(A, b)
(x, A @ x)
''', reads="Solves 3x + y = 9 and x + 2y = 8 -> x = 2, y = 3.")
fn("np.linalg.lstsq", "Least-squares solution (best fit when there is no exact one).", '''
x = np.array([0, 1, 2, 3.0])
y = np.array([1, 3, 5, 7.1])
A = np.column_stack([x, np.ones_like(x)])
sol, res, rank, sv = np.linalg.lstsq(A, y, rcond=None)
(np.round(sol, 3), rank)
''', reads="Fits y = m x + c: slope about 2, intercept about 1.")
fn("np.linalg.eig", "Eigenvalues and eigenvectors of a square matrix.", '''
A = np.array([[2.0, 0], [0, 3.0]])
w, v = np.linalg.eig(A)
(w, v)
''', reads="A v = w v: eigenvectors keep their direction when multiplied by A.")
fn("np.linalg.eigh", "Eigen-decomposition for symmetric matrices (faster, real, sorted).", '''
S = np.array([[2.0, 1.0], [1.0, 2.0]])
np.linalg.eigh(S)
''')
fn("np.linalg.eigvals / eigvalsh", "Only the eigenvalues.", '''
(np.linalg.eigvals(np.array([[0.0, 1], [-1, 0]])), np.linalg.eigvalsh(np.array([[2.0, 1], [1, 2]])))
''', reads="The rotation matrix has complex eigenvalues +i and -i.")
fn("np.linalg.svd", "Singular value decomposition: A = U S Vt.", '''
A = np.array([[3.0, 0], [0, 2.0], [0, 0]])
U, S, Vt = np.linalg.svd(A)
(S, U.shape, Vt.shape)
''')
fn("np.linalg.svd (reconstruct)", "Rebuild the matrix from its parts.", '''
A = np.arange(1.0, 7).reshape(3, 2)
U, S, Vt = np.linalg.svd(A, full_matrices=False)
np.allclose(U @ np.diag(S) @ Vt, A)
''')
fn("np.linalg.svdvals", "Only the singular values.", '''
np.linalg.svdvals(np.array([[3.0, 0], [0, 2.0]]))
''')
fn("np.linalg.qr", "QR decomposition (orthogonal Q, upper-triangular R).", '''
A = np.array([[1.0, 2], [3, 4], [5, 6]])
Q, R = np.linalg.qr(A)
(np.round(R, 3), np.allclose(Q @ R, A))
''')
fn("np.linalg.cholesky", "Cholesky factor L of a positive-definite matrix (A = L L.T).", '''
A = np.array([[4.0, 2], [2, 3]])
L = np.linalg.cholesky(A)
(np.round(L, 4), np.allclose(L @ L.T, A))
''')
fn("np.linalg.norm", "Vector or matrix length (norm).", '''
v = np.array([3.0, 4.0])
M = np.array([[1.0, 2], [3, 4]])
{"L2": np.linalg.norm(v), "L1": np.linalg.norm(v, 1), "inf": np.linalg.norm(v, np.inf), "Frobenius": np.linalg.norm(M)}
''')
fn("np.linalg.vector_norm", "NumPy 2: norm of vectors along axes.", '''
np.linalg.vector_norm(np.array([[3.0, 4], [6, 8]]), axis=1)
''')
fn("np.linalg.matrix_norm", "NumPy 2: norm of matrices.", '''
np.linalg.matrix_norm(np.array([[1.0, 2], [3, 4]]), ord="fro")
''')
fn("np.linalg.matrix_rank", "Number of independent rows/columns.", '''
(np.linalg.matrix_rank(np.array([[1, 2], [2, 4]])), np.linalg.matrix_rank(np.eye(3)))
''')
fn("np.linalg.cond", "Condition number: large = numerically unstable.", '''
(np.linalg.cond(np.eye(2)), np.linalg.cond(np.array([[1.0, 1.0], [1.0, 1.0001]])))
''')
fn("np.linalg.matrix_power", "A multiplied by itself n times (negative n uses the inverse).", '''
F = np.array([[1, 1], [1, 0]])
np.linalg.matrix_power(F, 10)
''', reads="Powers of this matrix contain Fibonacci numbers: [[89, 55], [55, 34]].")
fn("np.linalg.multi_dot", "Multiply several matrices in the cheapest order.", '''
np.linalg.multi_dot([np.ones((3, 4)), np.ones((4, 5)), np.ones((5, 2))]).shape
''')
fn("np.linalg.tensorsolve / tensorinv", "Solve / invert tensor equations.", '''
a = np.eye(6).reshape(2, 3, 6)
b = np.arange(6.0).reshape(2, 3)
np.linalg.tensorsolve(a, b)
''')
fn("np.linalg.diagonal / trace", "NumPy 2 helpers for stacks of matrices.", '''
M = np.arange(18).reshape(2, 3, 3)
(np.linalg.diagonal(M), np.linalg.trace(M))
''')
fn("np.linalg.outer / cross", "NumPy 2 linalg namespace versions.", '''
(np.linalg.outer(np.array([1, 2]), np.array([3, 4])), np.linalg.cross(np.array([1, 0, 0]), np.array([0, 0, 1])))
''')
fn("np.linalg.matmul / vecdot", "NumPy 2 array-API names inside linalg.", '''
np.linalg.matmul(np.eye(2), np.array([[1, 2], [3, 4]]))
''')
fn("np.linalg.LinAlgError", "Raised for singular matrices etc.", '''
msg = "no error raised"
try:
    np.linalg.inv(np.array([[1.0, 2], [2, 4]]))      # singular matrix
except np.linalg.LinAlgError as e:
    msg = str(e)
try:
    raise np.linalg.LinAlgError("you can also raise it yourself")
except np.linalg.LinAlgError as e:
    msg2 = str(e)
(msg, msg2)
''')
fn("stack of matrices", "Linalg functions loop over leading dimensions for you.", '''
stack = np.array([[[2.0, 0], [0, 2]], [[1.0, 2], [3, 4]]])
(np.linalg.det(stack), np.linalg.inv(stack).shape)
''')
fn("np.trace / np.diag / np.tril", "Common matrix helpers (also in Creating arrays).", '''
M = np.arange(1, 10).reshape(3, 3)
(np.trace(M), np.diag(M))
''')
fn("solve a system with inverse vs solve", "Same answer; solve is faster and more accurate.", '''
A = np.array([[2.0, 1], [5, 7]])
b = np.array([11.0, 13.0])
(np.linalg.inv(A) @ b, np.linalg.solve(A, b))
''')
fn("rotation matrix", "Rotate points 90 degrees counter-clockwise.", '''
t = np.pi / 2
R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
np.round(R @ np.array([[1, 0], [0, 1], [2, 3]]).T, 6).T
''')
fn("covariance and PCA with eigh", "Principal components of a small dataset.", '''
X = houses[:, :3]
Xc = X - X.mean(axis=0)
w, v = np.linalg.eigh(np.cov(Xc.T))
(np.round(w[::-1] / w.sum(), 4))
''', reads="Share of the variance explained by each principal component.")
fn("normal equation (regression)", "theta = (X'X)^-1 X'y - see the Linear regression project.", '''
X = np.column_stack([np.ones(len(houses)), houses[:, 0]])
theta = np.linalg.solve(X.T @ X, X.T @ houses[:, 3])
np.round(theta, 4)
''', reads="Intercept and price per extra square foot.")
fn("np.matrix / np.asmatrix", "Old 2-D-only matrix class (discouraged: use arrays and @).", '''
m = np.asmatrix([[1, 2], [3, 4]])
(m * m, type(m).__name__)
''', status="deprecated", alt="plain ndarray with @")

# =====================================================================================================  14  RANDOM
cat("random", "Random numbers", "The modern Generator API: distributions, sampling, shuffling, seeds - plus the legacy np.random functions.", "🎲",
    intro="Create a generator with rng = np.random.default_rng(seed). The same seed always reproduces the same numbers - essential for repeatable experiments. "
          "The old np.random.rand/seed functions still work but the Generator API is recommended.")

fn("np.random.default_rng", "Create a random generator (optionally seeded).", '''
rng = np.random.default_rng(42)
(rng.random(3), np.random.default_rng(42).random(3))
''', reads="Two generators with the same seed produce identical numbers.")
fn("rng.random", "Floats uniformly in [0, 1).", '''
rng = np.random.default_rng(0)
rng.random((2, 3))
''')
fn("rng.integers", "Random integers in [low, high).", '''
rng = np.random.default_rng(1)
(rng.integers(0, 10, size=8), rng.integers(1, 7, size=(2, 3), endpoint=True))
''', reads="high is EXCLUDED unless endpoint=True (a dice roll is 1..6 with endpoint=True).")
fn("rng.uniform", "Uniform floats in [low, high).", '''
np.random.default_rng(2).uniform(10, 20, size=5)
''')
fn("rng.normal", "Normal (Gaussian) distribution.", '''
x = np.random.default_rng(3).normal(loc=100, scale=15, size=10000)
(round(x.mean(), 1), round(x.std(), 1))
''', reads="Mean and standard deviation land close to the 100 and 15 you asked for.")
fn("rng.standard_normal", "Mean 0, std 1.", '''
np.random.default_rng(4).standard_normal((2, 3))
''')
fn("rng.choice", "Random sample from a list (with or without replacement, with weights).", '''
rng = np.random.default_rng(5)
(rng.choice(["red", "green", "blue"], size=6), rng.choice(10, size=4, replace=False), rng.choice([0, 1], size=10, p=[0.9, 0.1]))
''', reads="replace=False never repeats an element. p= sets the probabilities.")
fn("rng.shuffle", "Shuffle IN PLACE.", '''
a = np.arange(10)
np.random.default_rng(6).shuffle(a)
a
''')
fn("rng.permutation", "Shuffled COPY (or a random permutation of 0..n-1).", '''
rng = np.random.default_rng(7)
(rng.permutation(5), rng.permutation(np.array([10, 20, 30])))
''')
fn("rng.permuted", "Shuffle independently along an axis.", '''
np.random.default_rng(8).permuted(np.arange(12).reshape(3, 4), axis=1)
''', reads="Each ROW is shuffled on its own.")
fn("rng.binomial", "Number of successes in n trials.", '''
np.random.default_rng(9).binomial(n=10, p=0.5, size=8)
''')
fn("rng.poisson", "Counts of rare events per interval.", '''
np.random.default_rng(10).poisson(lam=3, size=10)
''')
fn("rng.exponential", "Waiting times between events.", '''
np.round(np.random.default_rng(11).exponential(scale=2.0, size=5), 3)
''')
fn("rng.gamma", "Gamma distribution.", '''
np.round(np.random.default_rng(12).gamma(shape=2.0, scale=1.5, size=5), 3)
''')
fn("rng.beta", "Beta distribution on [0, 1].", '''
np.round(np.random.default_rng(13).beta(2, 5, size=5), 3)
''')
fn("rng.chisquare / rng.f / rng.standard_t", "Test-statistic distributions.", '''
rng = np.random.default_rng(14)
(np.round(rng.chisquare(3, 3), 3), np.round(rng.f(4, 6, 3), 3), np.round(rng.standard_t(10, 3), 3))
''')
fn("rng.lognormal", "Log of the values is normal.", '''
np.round(np.random.default_rng(15).lognormal(0, 1, 5), 3)
''')
fn("rng.geometric / hypergeometric / negative_binomial", "Discrete waiting / sampling distributions.", '''
rng = np.random.default_rng(16)
(rng.geometric(0.3, 5), rng.hypergeometric(10, 5, 4, 5), rng.negative_binomial(3, 0.5, 5))
''')
fn("rng.laplace / logistic / gumbel", "Other continuous distributions.", '''
rng = np.random.default_rng(17)
(np.round(rng.laplace(0, 1, 3), 3), np.round(rng.logistic(0, 1, 3), 3), np.round(rng.gumbel(0, 1, 3), 3))
''')
fn("rng.pareto / weibull / rayleigh / triangular", "Heavy-tailed and shaped distributions.", '''
rng = np.random.default_rng(18)
(np.round(rng.pareto(3, 3), 3), np.round(rng.weibull(1.5, 3), 3), np.round(rng.rayleigh(1, 3), 3), np.round(rng.triangular(0, 5, 10, 3), 3))
''')
fn("rng.vonmises / wald / zipf / power / standard_cauchy", "More distributions.", '''
rng = np.random.default_rng(19)
(np.round(rng.vonmises(0, 2, 3), 3), np.round(rng.wald(1, 2, 3), 3), rng.zipf(2.5, 3), np.round(rng.power(3, 3), 3), np.round(rng.standard_cauchy(3), 3))
''')
fn("rng.multinomial", "Several categories at once (like many dice).", '''
np.random.default_rng(20).multinomial(100, [0.2, 0.3, 0.5])
''')
fn("rng.multivariate_normal", "Correlated normal variables.", '''
cov = [[1.0, 0.8], [0.8, 1.0]]
s = np.random.default_rng(21).multivariate_normal([0, 0], cov, size=5000)
np.round(np.corrcoef(s.T), 2)
''', reads="The sample correlation comes out near the 0.8 requested.")
fn("rng.dirichlet", "Random probability vectors that sum to 1.", '''
d = np.random.default_rng(22).dirichlet([1, 1, 1], size=3)
(np.round(d, 3), d.sum(axis=1))
''')
fn("rng.bytes", "Random bytes.", '''
np.random.default_rng(23).bytes(4)
''')
fn("rng.spawn", "Independent child generators (parallel work).", '''
kids = np.random.default_rng(24).spawn(3)
[round(float(k.random()), 4) for k in kids]
''')
fn("np.random.SeedSequence", "Turn seeds into well-mixed entropy.", '''
ss = np.random.SeedSequence(12345)
(ss.entropy, ss.generate_state(2))
''')
fn("np.random.PCG64 / Philox / SFC64 / MT19937", "Bit generators (the engine under a Generator).", '''
[type(np.random.Generator(bg(1))).__name__ + " <- " + bg.__name__ for bg in (np.random.PCG64, np.random.Philox, np.random.SFC64, np.random.MT19937)]
''')
fn("rng.bit_generator.state", "Save and restore the generator state.", '''
rng = np.random.default_rng(25)
st = rng.bit_generator.state
a = rng.random(2)
rng.bit_generator.state = st
(a, rng.random(2))
''', reads="Restoring the state replays exactly the same numbers.")
fn("np.random.seed (legacy)", "Seed the old global generator.", '''
np.random.seed(0)
np.random.rand(3)
''', reads="Works, but a global seed affects all code. Prefer default_rng(seed).")
fn("np.random.rand / randn (legacy)", "Uniform / standard-normal arrays from the global generator.", '''
np.random.seed(1)
(np.random.rand(2, 2), np.random.randn(3))
''')
fn("np.random.randint (legacy)", "Random integers (high excluded).", '''
np.random.seed(2)
np.random.randint(0, 10, size=6)
''')
fn("np.random.RandomState", "A seeded legacy generator object.", '''
rs = np.random.RandomState(3)
rs.randint(0, 100, 5)
''')
fn("np.random.choice / shuffle / permutation (legacy)", "Sampling helpers.", '''
np.random.seed(4)
(np.random.choice([1, 2, 3, 4], 3, replace=False), np.random.permutation(5))
''')
fn("np.random.random_sample / ranf / sample", "Old aliases for uniform [0,1).", '''
np.random.seed(5)
np.random.random_sample(3)
''')
fn("simulation: estimate pi", "Monte-Carlo with random points.", '''
rng = np.random.default_rng(0)
p = rng.random((1_000_000, 2))
4 * np.mean((p ** 2).sum(axis=1) < 1)
''', reads="The fraction of random points inside a quarter circle approaches pi / 4.")
fn("simulation: random walk", "Cumulative sum of random steps.", '''
steps = np.random.default_rng(1).choice([-1, 1], size=1000)
walk = np.cumsum(steps)
(walk[:10], walk.max(), walk.min())
''')
fn("simulation: dice (law of large numbers)", "Relative frequencies approach 1/6.", '''
r = np.random.default_rng(2).integers(1, 7, 60000)
np.round(np.bincount(r)[1:] / r.size, 3)
''')
fn("bootstrap confidence interval", "Resample to estimate uncertainty of the mean.", '''
rng = np.random.default_rng(3)
x = houses[:, 3]
means = rng.choice(x, size=(2000, len(x))).mean(axis=1)
np.round(np.percentile(means, [2.5, 97.5]), 2)
''')
fn("train / test split", "Shuffle indices and cut.", '''
idx = np.random.default_rng(4).permutation(len(houses))
cut = int(0.8 * len(houses))
(houses[idx[:cut]].shape, houses[idx[cut:]].shape)
''')
fn("random mask (dropout)", "Zero out a random fraction.", '''
x = np.ones(10)
m = np.random.default_rng(5).random(10) < 0.3
np.where(m, 0, x)
''')
fn("histogram of normal samples", "A chart of random data (needs matplotlib).", '''
x = np.random.default_rng(6).normal(size=5000)
plt.hist(x, bins=40)
plt.title("5000 samples from a normal distribution")
''', needs=["matplotlib"])

"""Practice projects: each step has a task, starter code, hint, solution. build_projects.py runs every solution and stores the
expected output, so the website can show 'expected output' and check the learner's `answer` variable.

Authoring rules
 * every step is independent: the project's PRELUDE runs first, then the learner's code
 * the learner must create a variable named `answer`
 * keep the maths unambiguous (say 'round to 2 decimals', say how ties are resolved)
"""

PROJECTS = []


def project(**kw):
    kw["steps"] = []
    PROJECTS.append(kw)
    return kw


def step(p, title, task, starter, solution, hint, rtol=1e-6, atol=1e-6):
    p["steps"].append(dict(title=title, task=" ".join(task.split()), starter=starter.strip("\n"), solution=solution.strip("\n"), hint=" ".join(hint.split()),
                           rtol=rtol, atol=atol))


# ======================================================================================== PROJECT 1
p1 = project(
    id="students", title="Student marks analytics", level="Beginner", minutes=35, dataset="students",
    story="A college wants to understand its exam results. You get a table of 600 students with exam marks, attendance and study hours - "
          "and, like real data, some marks are missing (nan). Use slicing, boolean masks, statistics and sorting to answer the dean's questions.",
    learn=["slicing columns", "nan-aware statistics", "boolean masks", "argsort / ranking", "np.digitize bins", "axis arguments", "correlation", "standardising"],
    prelude='''# Provided for you (do not change):
# students = 600 x 11 array. Columns: 0 student_id, 1 age, 2 gender (0=M, 1=F), 3 height_cm, 4 weight_kg,
#            5 math, 6 physics, 7 chemistry, 8 english, 9 attendance_pct, 10 study_hours   (blank cells are nan)
ids = students[:, 0]
gender = students[:, 2]
marks = students[:, 5:9]          # the four subject columns, shape (600, 4)
attendance = students[:, 9]
study = students[:, 10]''')

step(p1, "How many students have a missing mark?",
     "Some of the four subject marks are blank (nan). Count the students (rows of `marks`) that have AT LEAST ONE missing mark. Store the integer in `answer`.",
     "answer = ...   # TODO: how many rows of marks contain at least one nan?",
     "answer = int(np.isnan(marks).any(axis=1).sum())",
     "np.isnan(marks) gives a True/False table. .any(axis=1) asks 'is any value in this ROW True?'. Summing the resulting booleans counts the rows.")
step(p1, "Average mark per subject",
     "Compute the average of each subject (math, physics, chemistry, english), ignoring missing values. Round to 2 decimals. `answer` is an array of 4 numbers.",
     "answer = ...   # TODO: column means that skip nan, then round",
     "answer = np.round(np.nanmean(marks, axis=0), 2)",
     "np.nanmean is the nan-skipping mean. axis=0 gives one result per COLUMN. Wrap with np.round(..., 2).")
step(p1, "Fill the gaps",
     "Replace every missing mark by the average of ITS subject (from the previous step, unrounded). Call the 600x4 result `filled`, and set `answer = np.round(filled[rows][:5], 2)` where rows = the students that had a missing mark (the first five of them).",
     "means = ...    # TODO: per-subject means ignoring nan\nfilled = marks.copy()\nrows = np.isnan(marks).any(axis=1)\n# TODO: put the right mean into every nan cell of filled\nanswer = np.round(filled[rows][:5], 2)",
     "means = np.nanmean(marks, axis=0)\nfilled = marks.copy()\nrows = np.isnan(marks).any(axis=1)\nr, c = np.where(np.isnan(filled))\nfilled[r, c] = means[c]\nanswer = np.round(filled[rows][:5], 2)",
     "np.where(np.isnan(filled)) returns the row numbers and column numbers of the missing cells. means[c] picks the matching subject mean for each missing cell, so filled[r, c] = means[c] fills them all at once.")
FILL = "means = np.nanmean(marks, axis=0)\nfilled = marks.copy()\nrows = np.isnan(marks).any(axis=1)\nr, c = np.where(np.isnan(filled))\nfilled[r, c] = means[c]\n"
step(p1, "Top 5 students",
     "Using the `filled` marks (build them again below), compute each student's TOTAL of the four subjects. Find the student_ids of the 5 highest totals, best first. If two totals are equal the student that appears first in the table wins (use a stable sort). `answer` = array of 5 ids.",
     "# first rebuild `filled` (copy from the previous step)\n" + FILL + "total = ...    # TODO: sum across the 4 subjects\n# TODO: positions of the 5 largest totals (stable), then look up ids\nanswer = ...",
     FILL + "total = filled.sum(axis=1)\norder = np.argsort(-total, kind='stable')[:5]\nanswer = ids[order]",
     "argsort sorts ascending, so sort the NEGATED totals to get the largest first. kind='stable' keeps the original order for ties. Use the order to index into ids.")
step(p1, "Letter grades",
     "Average each student's four filled marks. Grade bands: A = 80 or more, B = 65 to under 80, C = 50 to under 65, D = under 50. `answer` = how many students got [A, B, C, D] (array of 4 counts).",
     FILL + "avg = filled.mean(axis=1)\n# TODO: assign a band number 0..3 to each student, then count each band\nanswer = ...",
     FILL + "avg = filled.mean(axis=1)\nband = np.digitize(avg, [50, 65, 80])        # 0=D, 1=C, 2=B, 3=A\ncounts = np.bincount(band, minlength=4)\nanswer = counts[::-1]",
     "np.digitize(avg, [50, 65, 80]) returns 0 for <50, 1 for 50-65, 2 for 65-80, 3 for >=80. np.bincount counts each number; reverse it to get A first.")
step(p1, "Pass rate",
     "A student PASSES when ALL four filled marks are at least 40. What percentage of the 600 students pass? Round to 1 decimal. `answer` = a single number.",
     FILL + "# TODO: a boolean per student: is every mark >= 40?\nanswer = ...",
     FILL + "passed = (filled >= 40).all(axis=1)\nanswer = round(float(passed.mean() * 100), 1)",
     "(filled >= 40) is a True/False table; .all(axis=1) is True only if the whole row is True. The mean of booleans is the fraction of True.")
step(p1, "Do boys and girls score differently?",
     "Using the filled marks, compute each student's average mark, then the mean of those averages for gender 0 (M) and gender 1 (F). Round to 2 decimals. `answer` = array [mean_M, mean_F].",
     FILL + "avg = filled.mean(axis=1)\n# TODO: use boolean masks gender == 0 and gender == 1\nanswer = ...",
     FILL + "avg = filled.mean(axis=1)\nanswer = np.round(np.array([avg[gender == 0].mean(), avg[gender == 1].mean()]), 2)",
     "A mask like gender == 0 selects the rows of boys. avg[mask].mean() averages only those.")
step(p1, "Does studying help?",
     "Some study_hours are missing. Keep only students whose study_hours is NOT nan, and compute the correlation coefficient between their study_hours and their average filled mark. Round to 4 decimals. `answer` = a single number.",
     FILL + "avg = filled.mean(axis=1)\n# TODO: mask of students with a known study_hours, then np.corrcoef on the two arrays\nanswer = ...",
     FILL + "avg = filled.mean(axis=1)\nok = ~np.isnan(study)\nanswer = round(float(np.corrcoef(study[ok], avg[ok])[0, 1]), 4)",
     "np.corrcoef(x, y) returns a 2x2 matrix; the correlation is the off-diagonal element [0, 1]. Remove nan rows first with a mask.")
step(p1, "Standardise the marks",
     "Turn each SUBJECT column of `filled` into z-scores: (value - column mean) / column standard deviation (population std, the NumPy default). `answer` = the first 3 rows rounded to 3 decimals (a 3x4 array).",
     FILL + "# TODO: z = (filled - mean) / std, computed per column\nanswer = ...",
     FILL + "z = (filled - filled.mean(axis=0)) / filled.std(axis=0)\nanswer = np.round(z[:3], 3)",
     "Broadcasting: a (600,4) array minus a (4,) array subtracts each column's own mean. Use axis=0 for the statistics.")
step(p1, "Top 10 percent",
     "Count the students whose average filled mark is at or above the 90th percentile of all averages (np.percentile with the default method). `answer` = integer count.",
     FILL + "avg = filled.mean(axis=1)\n# TODO\nanswer = ...",
     FILL + "avg = filled.mean(axis=1)\nanswer = int((avg >= np.percentile(avg, 90)).sum())",
     "cut = np.percentile(avg, 90) gives the threshold. Compare with >= and count with sum().")
step(p1, "Attendance matters?",
     "Ignore students whose attendance is missing. Compare the average of the student averages for attendance BELOW 70 and attendance 70 AND ABOVE. Round to 2 decimals. `answer` = array [mean_below_70, mean_70_or_more].",
     FILL + "avg = filled.mean(axis=1)\n# TODO: two masks that also exclude nan attendance\nanswer = ...",
     FILL + "avg = filled.mean(axis=1)\nok = ~np.isnan(attendance)\nlow = ok & (attendance < 70)\nhigh = ok & (attendance >= 70)\nanswer = np.round(np.array([avg[low].mean(), avg[high].mean()]), 2)",
     "Comparisons with nan are False, so attendance >= 70 already excludes nan, but attendance < 70 also does - still, make the nan test explicit with ~np.isnan to be safe.")

# ======================================================================================== PROJECT 2
p2 = project(
    id="sensor", title="Sensor data cleaning & anomaly detection", level="Intermediate", minutes=45, dataset="sensor",
    story="A weather station logs temperature, humidity, pressure and vibration every minute. The raw log contains blanks, an error code (-999), a power cut and a few suspicious spikes. "
          "Clean the data, smooth it, find the anomalies and summarise each day - all with vectorised NumPy.",
    learn=["sentinel values", "np.interp gap filling", "moving average with convolve", "np.diff and runs", "bincount group-by", "integer arithmetic on time", "masks"],
    prelude='''# Provided for you (do not change). Every column is a COPY so you can modify it freely.
# sensor columns: 0 t_sec (seconds since start), 1 temperature_c, 2 humidity_pct, 3 pressure_hpa, 4 vibration_g
t, temp, hum, pres, vib = (sensor[:, i].copy() for i in range(5))''')

step(p2, "How long is the recording?",
     "`t` holds seconds since the start (one reading per minute). How many DAYS does the log cover from the first to the last reading? Round to 3 decimals. `answer` = a number.",
     "answer = ...   # TODO",
     "answer = round(float((t[-1] - t[0]) / 86400), 3)",
     "The span is last minus first, in seconds. A day has 24 * 60 * 60 = 86400 seconds.")
step(p2, "Count the error codes",
     "Humidity uses -999 as an 'error' code. How many readings equal exactly -999? `answer` = integer.",
     "answer = ...   # TODO",
     "answer = int(np.count_nonzero(hum == -999))",
     "(hum == -999) is a boolean array; np.count_nonzero (or sum) counts the True values. nan never equals anything, so blanks are not counted.")
step(p2, "Missing values per column",
     "First replace every -999 in `hum` by nan. Then count the nan values in each of temp, hum, pres, vib. `answer` = array of 4 integers (in that order).",
     "# TODO: replace -999 by nan in hum\nanswer = ...",
     "hum[hum == -999] = np.nan\nanswer = np.array([np.isnan(a).sum() for a in (temp, hum, pres, vib)])",
     "Assign through a mask: hum[hum == -999] = np.nan. Then np.isnan(column).sum() for every column.")
FILLT = "idx = np.arange(len(temp))\nok = ~np.isnan(temp)\ntemp_f = np.interp(idx, idx[ok], temp[ok])\n"
step(p2, "Repair the temperature gaps",
     "Fill every missing temperature by LINEAR interpolation between the nearest valid readings (np.interp over the reading number). Call the result `temp_f`. `answer` = temp_f[5995:6005] rounded to 3 decimals - this slice covers the 40-minute power cut that starts at reading 6000.",
     "idx = np.arange(len(temp))\nok = ~np.isnan(temp)\n# TODO: temp_f = interpolate at ALL positions using only the valid points\nanswer = ...",
     FILLT + "answer = np.round(temp_f[5995:6005], 3)",
     "np.interp(x_new, x_known, y_known): x_new = every index, x_known = idx[ok], y_known = temp[ok].")
step(p2, "Smooth with a moving average",
     "Smooth `temp_f` (rebuild it below) with a 15-minute moving average using np.convolve(..., mode='valid'). `answer` = the first 5 values of the smoothed series, rounded to 3 decimals.",
     FILLT + "# TODO: convolve with a window of fifteen values that each equal 1/15\nanswer = ...",
     FILLT + "smooth = np.convolve(temp_f, np.ones(15) / 15, mode='valid')\nanswer = np.round(smooth[:5], 3)",
     "np.ones(15) / 15 is a box filter. mode='valid' only keeps positions where all 15 values exist, so the output is 14 shorter.")
step(p2, "Find sudden jumps",
     "A reading is suspicious when the temperature jumps by MORE than 3 degrees compared with the previous minute. Using `temp_f`, how many such jumps (consecutive pairs) are there? `answer` = integer.",
     FILLT + "# TODO: absolute differences of consecutive readings, count those > 3\nanswer = ...",
     FILLT + "answer = int((np.abs(np.diff(temp_f)) > 3).sum())",
     "np.diff(temp_f) is the change between neighbours. Take np.abs, compare with 3, and sum the booleans. One spike usually produces TWO jumps (up and back).")
step(p2, "Temperature vs humidity",
     "Replace -999 in humidity by nan first. Using only the readings where BOTH `temp` and `hum` are valid (not nan), compute the correlation coefficient between them. Round to 3 decimals. `answer` = a number.",
     "hum[hum == -999] = np.nan\n# TODO: mask of rows where neither is nan, then np.corrcoef\nanswer = ...",
     "hum[hum == -999] = np.nan\nok = ~np.isnan(temp) & ~np.isnan(hum)\nanswer = round(float(np.corrcoef(temp[ok], hum[ok])[0, 1]), 3)",
     "Combine two masks with &. np.corrcoef(...)[0, 1] is the correlation. It is strongly negative: warm air holds more moisture, so relative humidity falls.")
step(p2, "Daily average temperature",
     "Group the readings by day (day number = t // 86400, integer division). Using the interpolated `temp_f`, compute the mean temperature of each day. Round to 2 decimals. `answer` = array with one value per day.",
     FILLT + "day = t // 86400\n# TODO: mean of temp_f for each distinct day number\nanswer = ...",
     FILLT + "day = (t // 86400).astype(int)\nsums = np.bincount(day, weights=temp_f)\ncounts = np.bincount(day)\nanswer = np.round(sums / counts, 2)",
     "np.bincount(day, weights=values) adds up the values per day number; dividing by np.bincount(day) (the counts) gives the mean. day must be an integer array.")
step(p2, "Vibration alarms",
     "An ALARM is a run of consecutive readings with vibration above 1.0 g (ignore nan). Find the index at which each run STARTS. `answer` = array of start indices in order.",
     "high = ...   # TODO: boolean array vib > 1.0\n# TODO: positions where high switches from False to True\nanswer = ...",
     "high = (vib > 1.0)\nedges = np.diff(np.concatenate([[0], high.astype(int)]))\nanswer = np.where(edges == 1)[0]",
     "Turn the mask into 0/1, put a 0 in front, and take np.diff: +1 marks a start (False -> True), -1 a stop. np.where(edges == 1)[0] gives the start positions.")
step(p2, "The hottest hour of the day",
     "Hour of day = (t // 3600) % 24. Using `temp_f`, find the hour (0-23) with the highest AVERAGE temperature. `answer` = integer.",
     FILLT + "hour = ...   # TODO\nanswer = ...",
     FILLT + "hour = ((t // 3600) % 24).astype(int)\nmean_by_hour = np.bincount(hour, weights=temp_f) / np.bincount(hour)\nanswer = int(np.argmax(mean_by_hour))",
     "Same bincount trick as before, grouped by hour. np.argmax of the 24 averages is the hour number itself because the index equals the hour.")

# ======================================================================================== PROJECT 3
p3 = project(
    id="regression", title="Linear regression from scratch", level="Advanced", minutes=50, dataset="houses",
    story="Predict house prices from area, bedrooms and age using nothing but NumPy: explore the data, standardise it, solve the normal equation, "
          "measure the fit, then train the same model with gradient descent - the foundation of machine learning.",
    learn=["np.corrcoef", "broadcasting", "np.column_stack", "np.linalg.solve", "matrix @", "gradient descent loop", "train / test split", "R-squared and RMSE"],
    prelude='''# Provided for you (do not change). houses = 400 x 4 array: area_sqft, bedrooms, age_years, price_k (price in thousands)
area, beds, age, price = houses.T
features = houses[:, :3]
y = price''')

step(p3, "Know your target",
     "Compute the mean and the (population) standard deviation of the price. Round both to 2 decimals. `answer` = array [mean, std].",
     "answer = ...   # TODO",
     "answer = np.round(np.array([price.mean(), price.std()]), 2)",
     "price.mean() and price.std() - or np.mean / np.std. Put them in an array and round it.")
step(p3, "Which features matter?",
     "Compute the correlation of each feature (area, bedrooms, age) with the price. Round to 3 decimals. `answer` = array of 3 numbers.",
     "answer = ...   # TODO: np.corrcoef with each column of `features`",
     "answer = np.round(np.array([np.corrcoef(features[:, j], y)[0, 1] for j in range(3)]), 3)",
     "np.corrcoef(a, b)[0, 1] is the correlation of two vectors. Loop over the 3 columns with a list comprehension (small loops over columns are fine).")
step(p3, "Standardise the features",
     "Create Z = (features - column mean) / column std for all three feature columns (population std). `answer` = first 3 rows of Z rounded to 3 decimals (3x3 array).",
     "answer = ...   # TODO",
     "Z = (features - features.mean(axis=0)) / features.std(axis=0)\nanswer = np.round(Z[:3], 3)",
     "Broadcasting subtracts the (3,) mean row from every row. Use axis=0.")
step(p3, "Build the design matrix",
     "Create X by putting a column of ones in FRONT of the three features (the intercept column) so X has shape (400, 4). `answer` = X[0] (the first row).",
     "answer = ...   # TODO: np.column_stack([ones, features])",
     "X = np.column_stack([np.ones(len(features)), features])\nanswer = X[0]",
     "np.ones(len(features)) makes the column of ones; np.column_stack glues columns side by side.")
FX = "X = np.column_stack([np.ones(len(features)), features])\n"
step(p3, "Solve the normal equation",
     "Find the coefficients theta that minimise the squared error using the normal equation  (X^T X) theta = X^T y  with np.linalg.solve. `answer` = theta rounded to 4 decimals (intercept first, then area, bedrooms, age).",
     FX + "# TODO: theta = solve(X.T @ X, X.T @ y)\nanswer = ...",
     FX + "theta = np.linalg.solve(X.T @ X, X.T @ y)\nanswer = np.round(theta, 4)",
     "X.T @ X is a 4x4 matrix, X.T @ y a vector of 4 numbers. np.linalg.solve(A, b) solves A theta = b (better than inverting A).")
FT = FX + "theta = np.linalg.solve(X.T @ X, X.T @ y)\n"
step(p3, "Predict a new house",
     "Predict the price (in thousands) of a house with 2000 sq ft, 3 bedrooms and 10 years old, using theta from the normal equation. Round to 2 decimals. `answer` = a number.",
     FT + "new = np.array([1, 2000, 3, 10])   # leading 1 = intercept\n# TODO\nanswer = ...",
     FT + "new = np.array([1, 2000, 3, 10])\nanswer = round(float(new @ theta), 2)",
     "The prediction is the dot product of the feature row (with the leading 1) and theta: new @ theta.")
step(p3, "How good is the fit? (R-squared)",
     "R-squared = 1 - SS_res / SS_tot, where SS_res = sum of squared residuals (y - prediction) and SS_tot = sum of squared deviations of y from its mean. Round to 4 decimals. `answer` = a number.",
     FT + "pred = X @ theta\n# TODO\nanswer = ...",
     FT + "pred = X @ theta\nss_res = ((y - pred) ** 2).sum()\nss_tot = ((y - y.mean()) ** 2).sum()\nanswer = round(float(1 - ss_res / ss_tot), 4)",
     "Two sums of squares; their ratio is the share of variance NOT explained.")
step(p3, "Prediction error (RMSE)",
     "Root mean squared error = sqrt(mean((y - pred)^2)). Round to 3 decimals. `answer` = a number (in thousands).",
     FT + "pred = X @ theta\n# TODO\nanswer = ...",
     FT + "pred = X @ theta\nanswer = round(float(np.sqrt(np.mean((y - pred) ** 2))), 3)",
     "residuals = y - pred; square, mean, square-root.")
step(p3, "Gradient descent",
     "Train the same model by gradient descent on STANDARDISED features: Z = standardised features (population std), Xs = [ones, Z]. Start with theta = zeros(4); repeat 500 times: theta = theta - 0.1 * (Xs.T @ (Xs @ theta - y)) / n. `answer` = theta rounded to 3 decimals.",
     "Z = (features - features.mean(axis=0)) / features.std(axis=0)\nXs = np.column_stack([np.ones(len(Z)), Z])\nn = len(y)\ntheta = np.zeros(4)\n# TODO: loop 500 times\nanswer = np.round(theta, 3)",
     "Z = (features - features.mean(axis=0)) / features.std(axis=0)\nXs = np.column_stack([np.ones(len(Z)), Z])\nn = len(y)\ntheta = np.zeros(4)\nfor _ in range(500):\n    grad = Xs.T @ (Xs @ theta - y) / n\n    theta = theta - 0.1 * grad\nanswer = np.round(theta, 3)",
     "grad = Xs.T @ (Xs @ theta - y) / n is the gradient of half the mean squared error. After 500 small steps theta matches the closed-form answer on standardised data (the first number is the mean price).")
step(p3, "Honest evaluation: train / test split",
     "Fit the normal equation using only the FIRST 320 rows (train) and compute the RMSE on the LAST 80 rows (test). Round to 3 decimals. `answer` = a number.",
     FX + "Xtr, ytr = X[:320], y[:320]\nXte, yte = X[320:], y[320:]\n# TODO: fit on train, evaluate on test\nanswer = ...",
     FX + "Xtr, ytr = X[:320], y[:320]\nXte, yte = X[320:], y[320:]\nth = np.linalg.solve(Xtr.T @ Xtr, Xtr.T @ ytr)\nanswer = round(float(np.sqrt(np.mean((yte - Xte @ th) ** 2))), 3)",
     "Compute theta from the training rows only, then predict the test rows and compute the RMSE. A test error close to the training error means the model generalises.")

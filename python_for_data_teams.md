# Python for Data Teams — Course Reference

## Module 1: Python Basics Refresher
Python is a high-level, interpreted language ideal for data work due to its readable syntax and rich ecosystem.

### Key Concepts
- **Variables and types**: integers, floats, strings, lists, dicts
- **Control flow**: if/else, for loops, while loops, list comprehensions
- **Functions**: def, return, *args, **kwargs, lambda

### Common Pitfalls
- Mutable default arguments in functions (use `None` instead of `[]`)
- Integer division: in Python 3, `5/2 = 2.5`, use `5//2` for floor division
- String formatting: prefer f-strings over `.format()` for readability

---

## Module 2: Pandas for Data Manipulation
Pandas is the foundational library for tabular data in Python.

### Core Operations
```python
import pandas as pd

df = pd.read_csv("data.csv")
df.head()          # first 5 rows
df.describe()      # summary stats
df.info()          # column types and nulls
df.dropna()        # remove null rows
df.fillna(0)       # fill nulls with 0
df.groupby("dept")["salary"].mean()   # group aggregation
```

### Merging DataFrames
- `pd.merge(df1, df2, on="id", how="left")` — SQL-style join
- `pd.concat([df1, df2])` — stack rows vertically

### Performance Tips
- Use `.loc[]` for label-based indexing, `.iloc[]` for position-based
- Avoid row-by-row iteration; prefer vectorized operations
- Use categorical dtype for low-cardinality string columns

---

## Module 3: Data Visualization with Matplotlib
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(categories, values, color="steelblue")
plt.xlabel("Category")
plt.ylabel("Value")
plt.title("My Chart")
plt.tight_layout()
plt.show()
```

### Chart Selection Guide
| Data type | Recommended chart |
|---|---|
| Comparison across categories | Bar chart |
| Trend over time | Line chart |
| Distribution | Histogram |
| Correlation between two variables | Scatter plot |
| Part-to-whole | Pie or stacked bar |

---

## Module 4: Basic Statistics for Analysts
- **Mean**: average value — sensitive to outliers
- **Median**: middle value — robust to outliers
- **Standard deviation**: spread of data around the mean
- **Correlation**: ranges −1 to +1; measures linear relationship between two variables
- **p-value**: probability the observed result happened by chance (< 0.05 = statistically significant)

### Hypothesis Testing
1. State null hypothesis (H₀) and alternative (H₁)
2. Choose significance level α (typically 0.05)
3. Compute test statistic (t-test, chi-square, etc.)
4. Compare p-value to α; reject H₀ if p < α

---

## Frequently Asked Questions

**Q: When should I use Python vs SQL for data work?**
Use SQL for data extraction and transformation directly in the database. Use Python (pandas) for complex transformations, ML, or when you need programmatic control. They complement each other — extract with SQL, analyse with Python.

**Q: What's the difference between `.loc` and `.iloc`?**
`.loc` uses row/column **labels**. `.iloc` uses integer **positions** (0-based index). When in doubt about which index to use after filtering operations, `.iloc` is safer.

**Q: How do I deal with missing data?**
- Drop rows: `df.dropna(subset=["critical_column"])`
- Fill with a value: `df.fillna(df.mean())`
- Forward fill: `df.fillna(method="ffill")`
- The right choice depends on how much data you can afford to lose and whether missing values are random or systematic.

**Q: How do I make my pandas code faster?**
1. Use vectorized operations instead of loops
2. Load only needed columns: `pd.read_csv("f.csv", usecols=["a","b"])`
3. Use appropriate dtypes (e.g., `category` for strings with few unique values)
4. Consider `polars` for very large datasets

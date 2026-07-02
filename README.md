# Correlation of Matrix Rows

## Overview

```cpp
void correlate(int ny, int nx, const float* data, float* result);
```

which computes the Pearson correlation coefficient between every pair of rows in an input matrix.

Only the upper triangular part of the correlation matrix (including the diagonal) is computed, since the correlation matrix is symmetric.

---

## Input

The input matrix consists of:

- **`ny`** rows (vectors)
- **`nx`** columns (elements per vector)

The matrix is stored in row-major order:

```cpp
data[x + y * nx]
```

where:

- `y` is the row index (`0 <= y < ny`)
- `x` is the column index (`0 <= x < nx`)

---

## Output

The output is a correlation matrix stored in the array `result`.

For every pair of rows `(i, j)` satisfying

```text
0 <= j <= i < ny
```

the Pearson correlation coefficient is stored at

```cpp
result[i + j * ny]
```

Only the upper triangle (including the diagonal) is required. Values where `i < j` are left undefined.

---

## Pearson Correlation

Given two rows **A** and **B**, the Pearson correlation coefficient is computed as

\[
r=\frac{\sum_{k=1}^{nx}(A_k-\bar{A})(B_k-\bar{B})}
{\sqrt{\sum_{k=1}^{nx}(A_k-\bar{A})^2}
\sqrt{\sum_{k=1}^{nx}(B_k-\bar{B})^2}}
\]

where

- \(\bar{A}\) is the mean of row A
- \(\bar{B}\) is the mean of row B

The coefficient ranges from:

- **+1** : perfect positive correlation
- **0** : no linear correlation
- **−1** : perfect negative correlation

---

## Notes

- The input and output arrays are preallocated.
- The implementation should **not** perform any memory allocation for `data` or `result`.
- The lower triangular portion of the result matrix is not required.
- Depending on the assignment version, arithmetic may be performed using either **float** or **double** precision.

---

## Example

If the input matrix represents the rows of a grayscale image:

- each row is treated as one vector,
- the output is a correlation matrix describing the similarity between every pair of rows.

Visualization:

- **Red** indicates a strong positive correlation.
- **Blue** indicates a strong negative correlation.
# Data Science Bootcamp Assignments

A structured multi-day curriculum repository covering end-to-end numerical computation, data cleaning pipelines, exploratory data analysis, and reporting.

---

## 📁 Repository Structure

- **[`Day_01_NumPy/`](file:///Day_01_NumPy/)**: Vectorized Student Performance Analysis (5-task marks evaluation, argmax toppers, Boolean masking).
  - [`Day_01_NumPy/day1_numpy_analysis.py`](file:///Day_01_NumPy/day1_numpy_analysis.py): Vectorized Python script implementing the 5-stage numerical pipeline.
  - [`Day_01_NumPy/README.md`](file:///Day_01_NumPy/README.md): Detailed task breakdown, statistical aggregations, and sample output.
- **[`Day_02_Pandas/`](file:///Day_02_Pandas/)**: Data Cleaning, Audit Logging, GroupBy Aggregations, Pivot Tables.
  - [`Day_02_Pandas/day2_cleaning.ipynb`](file:///Day_02_Pandas/day2_cleaning.ipynb): Interactive Jupyter Notebook fixing 6+ data anomalies, executing aggregations, and pivoting revenue metrics.
  - [`Day_02_Pandas/cleaned_data.csv`](file:///Day_02_Pandas/cleaned_data.csv): Fully sanitized, schema-consistent exported dataset.

---

## 🛠️ Environment Setup

Install the required dependencies across all modules:

```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy>=1.24.0`
- `pandas>=2.0.0`

---

## ▶️ Execution Instructions

### Day 1: NumPy Performance Analysis
Navigate to the root directory and run the vectorized student performance script:
```bash
python Day_01_NumPy/day1_numpy_analysis.py
```

### Day 2: Pandas Data Cleaning & Aggregations
Open and run the notebook in Jupyter Notebook, JupyterLab, or VS Code / Antigravity IDE:
```bash
jupyter notebook Day_02_Pandas/day2_cleaning.ipynb
```
Or view the processed dataset directly:
```bash
cat Day_02_Pandas/cleaned_data.csv
```

---

## 📊 Day 2: Data Quality Summary

1. The original dataset contained 11 records plagued by duplicates, mixed date structures, and negative values.
2. Cleaning operations resolved casing mismatches, stripped whitespace, and eliminated currency symbols from price metrics.
3. Missing units were imputed via median calculations, and total revenues were programmatically recomputed.
4. One duplicate row was permanently removed along with an unrecoverable corrupted timestamp record.
5. The final cleaned dataset achieved 100% schema consistency, zero duplicate entries, and full numerical integrity.

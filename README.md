# Data Science Bootcamp Assignments

A structured multi-day curriculum repository covering end-to-end numerical computation, data cleaning pipelines, exploratory data analysis (EDA), and strategic reporting.

---

## 📁 Repository Structure

- **[`Day_01_NumPy/`](file:///Day_01_NumPy/)**: Vectorized Student Performance Analysis (5-task marks evaluation, argmax toppers, Boolean masking).
  - [`Day_01_NumPy/day1_numpy_analysis.py`](file:///Day_01_NumPy/day1_numpy_analysis.py): Vectorized Python script implementing the 5-stage numerical pipeline.
  - [`Day_01_NumPy/README.md`](file:///Day_01_NumPy/README.md): Detailed task breakdown, statistical aggregations, and sample output.
- **[`Day_02_Pandas/`](file:///Day_02_Pandas/)**: Data Cleaning, Audit Logging, GroupBy Aggregations, Pivot Tables.
  - [`Day_02_Pandas/day2_cleaning.ipynb`](file:///Day_02_Pandas/day2_cleaning.ipynb): Interactive Jupyter Notebook fixing 6+ data anomalies, executing aggregations, and pivoting revenue metrics.
  - [`Day_02_Pandas/cleaned_data.csv`](file:///Day_02_Pandas/cleaned_data.csv): Fully sanitized, schema-consistent exported dataset.
- **[`Day_03_Visualizations/`](file:///Day_03_Visualizations/)**: Exploratory Data Analysis (EDA) & Data Visualizations with Matplotlib & Seaborn.
  - [`Day_03_Visualizations/day3_visualizations.py`](file:///Day_03_Visualizations/day3_visualizations.py): Automated end-to-end Python pipeline generating 5 high-resolution charts, Jupyter notebook, and executive findings report.
  - [`Day_03_Visualizations/day3_eda_visualizations.ipynb`](file:///Day_03_Visualizations/day3_eda_visualizations.ipynb): Interactive EDA notebook featuring 5 charts with analytical interpretations.
  - [`Day_03_Visualizations/findings_report.md`](file:///Day_03_Visualizations/findings_report.md): Executive summary outlining 3 key analytical insights, 2 data-driven recommendations, and analytical limitations.
  - [`Day_03_Visualizations/retail_sales_visualization_dataset.csv`](file:///Day_03_Visualizations/retail_sales_visualization_dataset.csv): Retail sales dataset with transactional revenue, category, discount, and profit attributes.
  - **Exported High-Resolution Visual Artifacts (300 DPI)**:
    - `chart1_distribution.png`: Histogram & KDE of transaction sales volume.
    - `chart2_comparison.png`: Categorical comparison of total sales per product category.
    - `chart3_relationship.png`: Bivariate scatter plot of Sales vs. Profit sized by discount bands.
    - `chart4_correlation_heatmap.png`: Correlation matrix of key numeric metrics.
    - `weekly_sales_trend.png`: Chronological weekly aggregate sales timeline.

---

## 🛠️ Environment Setup

Install the required dependencies across all modules:

```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`

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
Or inspect the cleaned dataset directly:
```bash
cat Day_02_Pandas/cleaned_data.csv
```

### Day 3: EDA & Data Visualizations
Run the automated visualization script to regenerate all charts, notebook, and report:
```bash
cd Day_03_Visualizations
python day3_visualizations.py
```
Or explore the notebook interactively:
```bash
jupyter notebook Day_03_Visualizations/day3_eda_visualizations.ipynb
```

---

## 📊 Quality & Analytical Summaries

### Day 2: Data Quality Summary
1. The original dataset contained 11 records plagued by duplicates, mixed date structures, and negative values.
2. Cleaning operations resolved casing mismatches, stripped whitespace, and eliminated currency symbols from price metrics.
3. Missing units were imputed via median calculations, and total revenues were programmatically recomputed.
4. One duplicate row was permanently removed along with an unrecoverable corrupted timestamp record.
5. The final cleaned dataset achieved 100% schema consistency, zero duplicate entries, and full numerical integrity.

### Day 3: Executive Analytical Summary
1. **Revenue Concentration in High-Unit-Price Categories**: Right-skewed distribution where high-ticket Electronics & Furniture generate over 65% of revenue.
2. **Pricing Dominance Over Volume**: Unit Price ($r = 0.84$) correlates almost three times stronger with Sales than Quantity ($r = 0.31$).
3. **Discount Inefficiency**: Discounting shows negative correlation with Profit ($r = -0.15$) and fails to stimulate higher unit volumes ($r = -0.13$).

# Student Marks Performance Analysis using NumPy

A 5-stage vectorized numerical analysis pipeline analyzing 10 students across 5 subjects using high-performance NumPy array operations.

## 📌 Project Overview

This project demonstrates core numerical computing techniques using NumPy in Python. It simulates academic performance data for 10 students across 5 subjects and executes an end-to-end analytics workflow leveraging vectorization, statistical aggregations, indexing, and Boolean masking.

---

## 🚀 Pipeline Tasks

| Stage | Task Name | Description | Key NumPy Methods / Operations |
| :---: | :--- | :--- | :--- |
| **1** | **Marks Matrix Generation** | Generates a 10×5 matrix of marks (0–100) using a seeded pseudo-random generator for reproducibility. | `np.random.default_rng(seed=42)`, `rng.integers()` |
| **2** | **Student Performance Calculation** | Calculates total marks and percentage for each individual student across all 5 subjects. | `np.sum(axis=1)`, vectorized division & multiplication |
| **3** | **Subject-wise Statistical Analysis** | Computes average score (mean) and variance/spread (standard deviation) for each subject. | `np.mean(axis=0)`, `np.std(axis=0)` |
| **4** | **Topper Identification** | Finds the highest score and the index of the highest-scoring student for each subject. | `np.max(axis=0)`, `np.argmax(axis=0)` |
| **5** | **Pass/Fail Classification** | Classifies students as passed only if score is $\ge 40$ in all subjects using Boolean masking. | Boolean indexing (`>=`), `np.all(axis=1)`, `np.where()` |

---

## 🛠️ Requirements & Installation

- Python 3.8+
- NumPy $\ge$ 1.24.0

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Execution Instructions

Run the analysis pipeline directly from your terminal:

```bash
python main.py
```

---

## 📊 Sample Output Preview

```text
==========================================
TASK 1: MARKS MATRIX (10 Students x 5 Subjects)
==========================================
[[...]]

==========================================
TASK 2: STUDENT PERFORMANCE
==========================================
Student 00 -> Total Marks: .../500 | Percentage: ...%
...

==========================================
TASK 3: SUBJECT-WISE STATISTICAL ANALYSIS
==========================================
Subject 0 -> Average Marks: ... | Standard Deviation: ...
...

==========================================
TASK 4: SUBJECT TOPPERS
==========================================
Subject 0 -> Highest Marks: ... (Secured by Student ...)
...

==========================================
TASK 5: PASS / FAIL CLASSIFICATION (Threshold >= 40)
==========================================
Total Students Passed : ...
Passed Student Indices: [...]
Total Students Failed : ...
Failed Student Indices: [...]
==========================================
```

"""
Data Science Bootcamp - NumPy Assignment
Tasks:
1. Marks Matrix using default_rng (10 students x 5 subjects)
2. Total marks and percentage calculation
3. Average marks and standard deviation per subject
4. Subject toppers and indices using np.argmax
5. Pass/Fail identification using Boolean masks
"""

import numpy as np

# Set reproducible seed
rng = np.random.default_rng(seed=42)

# Task 1: Create Marks Matrix (10 x 5)
marks = rng.integers(0, 101, size=(10, 5))

print("==========================================")
print("TASK 1: MARKS MATRIX (10 Students x 5 Subjects)")
print("==========================================")
print(marks)
print("\n")

# Task 2: Student Performance (Total & Percentage)
total_marks = np.sum(marks, axis=1)
percentage = (total_marks / (5 * 100)) * 100

print("==========================================")
print("TASK 2: STUDENT PERFORMANCE")
print("==========================================")
for student_id, (total, pct) in enumerate(zip(total_marks, percentage)):
    print(f"Student {student_id:02d} -> Total Marks: {total:3d}/500 | Percentage: {pct:6.2f}%")
print("\n")

# Task 3: Subject Performance Analysis
subject_avg = np.mean(marks, axis=0)
subject_std = np.std(marks, axis=0)

print("==========================================")
print("TASK 3: SUBJECT-WISE STATISTICAL ANALYSIS")
print("==========================================")
for sub_id, (avg, std) in enumerate(zip(subject_avg, subject_std)):
    print(f"Subject {sub_id} -> Average Marks: {avg:6.2f} | Standard Deviation: {std:6.2f}")
print("\n")

# Task 4: Subject Toppers using np.argmax
highest_marks = np.max(marks, axis=0)
topper_indices = np.argmax(marks, axis=0)

print("==========================================")
print("TASK 4: SUBJECT TOPPERS")
print("==========================================")
for sub_id, (student_idx, top_score) in enumerate(zip(topper_indices, highest_marks)):
    print(f"Subject {sub_id} -> Highest Marks: {top_score:3d} (Secured by Student {student_idx})")
print("\n")

# Task 5: Identify Pass and Fail using Boolean Mask (Threshold >= 40)
PASS_THRESHOLD = 40
pass_mask = marks >= PASS_THRESHOLD
student_passed = np.all(pass_mask, axis=1)

passed_indices = np.where(student_passed)[0]
failed_indices = np.where(~student_passed)[0]

print("==========================================")
print("TASK 5: PASS / FAIL CLASSIFICATION (Threshold >= 40)")
print("==========================================")
print(f"Total Students Passed : {len(passed_indices)}")
print(f"Passed Student Indices: {passed_indices.tolist()}")
print(f"Total Students Failed : {len(failed_indices)}")
print(f"Failed Student Indices: {failed_indices.tolist()}")
print("==========================================")

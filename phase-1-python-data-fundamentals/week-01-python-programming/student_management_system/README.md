# Student Management System (SMS)
### Week 01 Practical Assignment Reference Solution
**Company:** Next Gen Software Hub Pvt. Ltd.  
**Track:** AI & Machine Learning Internship

---

## 📌 Project Overview
The **Student Management System (SMS)** is an enterprise-grade, object-oriented CLI application developed in Python. It manages student academic profiles, calculates overall percentages and letter grades, provides class-level analytics, and ensures data persistence through robust JSON serialization with automated backup.

---

## ✨ Features
- **Student Profile Management:**
  - Add new students with auto-validated ID, Name, Age, Course, and marks.
  - Multi-subject scoring system with boundary checks ($0 \le \text{mark} \le 100$).
- **Search & Query:**
  - Fast search by exact ID or partial student name (case-insensitive).
- **Academic Grading Engine:**
  - Accurate calculation of total marks, percentage, and letter grade ($A+, A, B, C, D, F$).
  - Pass/Fail verification based on configurable threshold ($50\%$).
- **Class Analytics:**
  - Class average percentage
  - Identification of top-performing and lowest-performing students
  - Visual ASCII distribution breakdown across grades
  - Overall class pass rate percentage
- **Safe Persistence Layer:**
  - Automated JSON persistence (`data/students_sample.json`).
  - Pre-save backup creation (`.json.bak`) to prevent data corruption during unexpected failures.
- **Defensive Error Handling:**
  - Comprehensive input sanitation and descriptive error prompts without application crashes.

---

## 📂 Architecture
```
student_management_system/
├── main.py                     # Application entry point
├── models/
│   └── student.py              # Student data class, validation, and grade scale
├── services/
│   └── student_manager.py      # Business logic: CRUD, filters, and analytics
├── storage/
│   └── json_storage.py         # File persistence and backup mechanism
├── cli/
│   └── menu.py                 # Menu formatting, table renderers, input helpers
├── data/
│   └── students_sample.json    # Persistent JSON storage
└── tests/
    └── test_student_manager.py # Automated unit test suite
```

---

## 🚀 How to Run

### Run the Interactive CLI Application
```bash
python main.py
```

### Run Unit Tests
```bash
python -m unittest tests/test_student_manager.py
```
Or using pytest:
```bash
pytest tests/
```

---

## 📊 Sample CLI Output

### Tabular View
```
-----------------------------------------------------------------------------------------
| ID       | Name               | Age | Course                 | Total  | %      | Grade |
-----------------------------------------------------------------------------------------
| NGS-101  | Aarav Sharma       | 21  | AI & Machine Learning  | 274.5  | 91.50  | A+    |
| NGS-102  | Diya Patel         | 22  | AI & Machine Learning  | 245.5  | 81.83  | A     |
| NGS-103  | Rohan Verma        | 20  | Data Analytics         | 207.0  | 69.00  | C     |
| NGS-104  | Ananya Iyer        | 23  | AI & Machine Learning  | 289.0  | 96.33  | A+    |
| NGS-105  | Kabir Mehta        | 21  | Computer Science       | 165.0  | 55.00  | D     |
-----------------------------------------------------------------------------------------
```

### Analytics Report
```
============================================================
          CLASS ACADEMIC ANALYTICS & GRADE REPORT         
============================================================
 Total Students Enrolled : 5
 Class Average Percentage : 78.73%
 Overall Pass Rate       : 100.0%
 Top Performer           : Ananya Iyer (NGS-104) - 96.33% (Grade A+)
 Lowest Performer        : Kabir Mehta (NGS-105) - 55.0% (Grade D)

 Grade Distribution:
   Grade A+ :  2 student(s) | ##
   Grade A  :  1 student(s) | #
   Grade B  :  0 student(s) | 
   Grade C  :  1 student(s) | #
   Grade D  :  1 student(s) | #
   Grade F  :  0 student(s) | 
============================================================
```

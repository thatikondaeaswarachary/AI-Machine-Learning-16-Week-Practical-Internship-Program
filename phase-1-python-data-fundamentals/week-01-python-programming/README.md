# Week 01 — Python Programming for AI/ML
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
1. Understand core Python syntax, primitive types, operators, and control structures.
2. Master object-oriented programming (classes, inheritance, encapsulation, magic methods).
3. Implement defensive programming with robust `try-except` blocks and input validation.
4. Learn persistent file handling using JSON and CSV context managers (`with` statement).
5. Build, test, and document a modular command-line software application.

---

## 📚 Theory Syllabus Checklist
- [x] **Variables & Data Types:** `int`, `float`, `str`, `bool`, type conversions, dynamic typing.
- [x] **Collections:** Lists, Tuples, Dictionaries, Sets, and slicing notation.
- [x] **Control Flow:** `if`, `elif`, `else`, nested conditions, logical operators (`and`, `or`, `not`).
- [x] **Loops & Iteration:** `for`, `while`, `break`, `continue`, `enumerate()`, `zip()`, List comprehensions.
- [x] **Functions:** Defining functions, default parameters, `*args`, `**kwargs`, return values, docstrings.
- [x] **Exception Handling:** `try`, `except Exception as e`, `else`, `finally`, raising `ValueError`.
- [x] **File I/O:** Reading/writing text, JSON with `json.load()` and `json.dump()`.
- [x] **Object-Oriented Programming:** Classes, constructors (`__init__`), instance methods, `@classmethod`, `@staticmethod`.

---

## 🛠️ Practical Assignment: Student Management System
Build a robust, menu-driven CLI **Student Management System** in Python with modular architecture and persistent storage.

### Required Features:
1. **Add Student:** Collect Student ID, Name, Age, Course, and marks for subjects (e.g., Python, Mathematics, Data Science). Validate that marks are between 0 and 100.
2. **View Students:** Display all registered students in a formatted tabular layout with total marks, percentage, and assigned letter grade.
3. **Search Student:** Search by either exact Student ID or partial Name matching (case-insensitive).
4. **Update Student:** Modify existing details (Course, Age, or specific Subject Marks).
5. **Delete Student:** Remove a student record with safety confirmation.
6. **Academic Analytics:**
   - Class average percentage
   - Top-performing student
   - Grade distribution summary (Count of A+, A, B, C, D, F)
7. **Data Persistence:** Store and load all student data from `data/students_sample.json` with automated backup on save.

---

## 📁 Project Architecture
```
week-01-python-programming/
├── README.md                           # This guide
└── student_management_system/
    ├── main.py                         # Application CLI entry point
    ├── README.md                       # SMS User Manual & CLI Demo
    ├── models/
    │   ├── __init__.py
    │   └── student.py                  # Student dataclass & grade logic
    ├── services/
    │   ├── __init__.py
    │   └── student_manager.py          # CRUD & Analytics business logic
    ├── storage/
    │   ├── __init__.py
    │   └── json_storage.py             # JSON file reader/writer & backup
    ├── cli/
    │   ├── __init__.py
    │   └── menu.py                     # Interactive terminal menu & UI
    ├── data/
    │   └── students_sample.json        # Pre-loaded sample student database
    └── tests/
        ├── __init__.py
        └── test_student_manager.py     # Unit tests verifying all operations
```

---

## 🚀 Running the Project
```bash
# Navigate to the project directory
cd "phase-1-python-data-fundamentals/week-01-python-programming/student_management_system"

# Run the interactive CLI application
python main.py

# Run automated unit tests
python -m unittest tests/test_student_manager.py
```

---

## 📋 Submission Checklist
1. All Python source files organized in modular directories.
2. `students_sample.json` pre-populated with at least 5 realistic records.
3. Successful execution of `python -m unittest tests/test_student_manager.py`.
4. Submission report drafted using `templates/weekly_submission_template.md`.

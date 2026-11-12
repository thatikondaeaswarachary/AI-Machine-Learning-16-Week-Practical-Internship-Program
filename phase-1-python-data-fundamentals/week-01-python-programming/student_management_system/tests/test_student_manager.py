"""
Unit Tests for Student Management System
Verifies model validation, CRUD logic, analytics, and JSON persistence.
"""

import unittest
import os
import sys
import shutil

# Ensure student_management_system package root is on sys.path
SMS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SMS_ROOT not in sys.path:
    sys.path.insert(0, SMS_ROOT)

from models.student import Student
from storage.json_storage import JSONStorage
from services.student_manager import StudentManager


class TestStudentModel(unittest.TestCase):

    def test_valid_student_creation(self):
        s = Student("TEST-01", "Alice Smith", 21, "AI & ML", {"Python": 95, "Math": 85})
        self.assertEqual(s.student_id, "TEST-01")
        self.assertEqual(s.name, "Alice Smith")
        self.assertEqual(s.calculate_total(), 180.0)
        self.assertEqual(s.calculate_percentage(), 90.0)
        self.assertEqual(s.calculate_grade(), "A+")
        self.assertTrue(s.is_passed())

    def test_grade_boundaries(self):
        # Grade A (80-89)
        s_a = Student("T-A", "Bob", 20, "CS", {"Sub1": 85.0})
        self.assertEqual(s_a.calculate_grade(), "A")

        # Grade B (70-79)
        s_b = Student("T-B", "Charlie", 20, "CS", {"Sub1": 75.0})
        self.assertEqual(s_b.calculate_grade(), "B")

        # Grade C (60-69)
        s_c = Student("T-C", "David", 20, "CS", {"Sub1": 65.0})
        self.assertEqual(s_c.calculate_grade(), "C")

        # Grade D (50-59)
        s_d = Student("T-D", "Eve", 20, "CS", {"Sub1": 52.0})
        self.assertEqual(s_d.calculate_grade(), "D")

        # Grade F (<50)
        s_f = Student("T-F", "Frank", 20, "CS", {"Sub1": 42.0})
        self.assertEqual(s_f.calculate_grade(), "F")
        self.assertFalse(s_f.is_passed())

    def test_invalid_attributes_raise_errors(self):
        with self.assertRaises(ValueError):
            Student("", "Valid Name", 20, "Course")  # Empty ID

        with self.assertRaises(ValueError):
            Student("ID-1", "A", 20, "Course")  # Name too short

        with self.assertRaises(ValueError):
            Student("ID-1", "Valid Name", 5, "Course")  # Age too young

        with self.assertRaises(ValueError):
            Student("ID-1", "Valid Name", 20, "Course", {"Math": 105})  # Mark > 100


class TestStudentManager(unittest.TestCase):

    TEST_STORAGE_PATH = "tests_scratch/temp_students.json"

    def setUp(self):
        os.makedirs("tests_scratch", exist_ok=True)
        if os.path.exists(self.TEST_STORAGE_PATH):
            os.remove(self.TEST_STORAGE_PATH)
        self.storage = JSONStorage(self.TEST_STORAGE_PATH)
        self.manager = StudentManager(self.storage)

    def tearDown(self):
        if os.path.exists("tests_scratch"):
            shutil.rmtree("tests_scratch")

    def test_add_and_get_student(self):
        student = Student("NGS-201", "John Doe", 22, "AI & ML", {"Python": 90, "Stats": 80})
        self.manager.add_student(student)

        retrieved = self.manager.get_student("NGS-201")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "John Doe")

    def test_duplicate_id_raises_value_error(self):
        s1 = Student("NGS-201", "John Doe", 22, "AI & ML")
        s2 = Student("NGS-201", "Jane Doe", 21, "AI & ML")
        self.manager.add_student(s1)

        with self.assertRaises(ValueError):
            self.manager.add_student(s2)

    def test_search_by_id_and_partial_name(self):
        s1 = Student("NGS-201", "Vikram Rathore", 22, "Data Science")
        s2 = Student("NGS-202", "Pooja Hegde", 23, "AI & ML")
        self.manager.add_student(s1)
        self.manager.add_student(s2)

        # Search by partial name
        results = self.manager.search_students("rath")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].student_id, "NGS-201")

        # Search by ID
        results_id = self.manager.search_students("202")
        self.assertEqual(len(results_id), 1)
        self.assertEqual(results_id[0].name, "Pooja Hegde")

    def test_update_student(self):
        student = Student("NGS-201", "Sam", 20, "CS", {"Python": 75})
        self.manager.add_student(student)

        self.manager.update_student("NGS-201", age=21, marks={"Python": 88})
        updated = self.manager.get_student("NGS-201")
        self.assertEqual(updated.age, 21)
        self.assertEqual(updated.marks["Python"], 88.0)

    def test_delete_student(self):
        student = Student("NGS-201", "Sam", 20, "CS")
        self.manager.add_student(student)
        self.assertEqual(len(self.manager.get_all_students()), 1)

        self.manager.delete_student("NGS-201")
        self.assertEqual(len(self.manager.get_all_students()), 0)
        self.assertIsNone(self.manager.get_student("NGS-201"))

    def test_class_analytics(self):
        s1 = Student("S1", "High Achiever", 20, "AI", {"Sub1": 95, "Sub2": 95})  # 95% -> A+
        s2 = Student("S2", "Average Student", 21, "AI", {"Sub1": 75, "Sub2": 75})  # 75% -> B
        s3 = Student("S3", "Needs Help", 22, "AI", {"Sub1": 40, "Sub2": 40})  # 40% -> F

        self.manager.add_student(s1)
        self.manager.add_student(s2)
        self.manager.add_student(s3)

        analytics = self.manager.get_class_analytics()
        self.assertEqual(analytics["total_students"], 3)
        self.assertEqual(analytics["highest_scorer"].student_id, "S1")
        self.assertEqual(analytics["lowest_scorer"].student_id, "S3")
        self.assertEqual(analytics["average_percentage"], 70.0)
        self.assertEqual(analytics["grade_distribution"]["A+"], 1)
        self.assertEqual(analytics["grade_distribution"]["B"], 1)
        self.assertEqual(analytics["grade_distribution"]["F"], 1)
        self.assertEqual(analytics["pass_rate_percentage"], 66.67)


if __name__ == "__main__":
    unittest.main()

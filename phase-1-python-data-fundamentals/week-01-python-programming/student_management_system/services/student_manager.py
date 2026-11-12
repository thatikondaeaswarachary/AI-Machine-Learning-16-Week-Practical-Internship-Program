"""
Student Manager Service
Contains business logic for CRUD operations and academic analytics.
"""

from typing import List, Optional, Dict, Any

try:
    from ..models.student import Student
    from ..storage.json_storage import JSONStorage
except (ImportError, ValueError):
    from models.student import Student
    from storage.json_storage import JSONStorage


class StudentManager:
    """Orchestrates student record management, searches, and statistical metrics."""

    def __init__(self, storage: JSONStorage):
        self.storage = storage
        self.students: Dict[str, Student] = {}
        self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Populates in-memory dictionary from storage."""
        records = self.storage.load_data()
        self.students.clear()
        for item in records:
            try:
                student = Student.from_dict(item)
                self.students[student.student_id] = student
            except (ValueError, KeyError) as e:
                print(f"[Warning] Skipping corrupted record {item}: {e}")

    def save_to_storage(self) -> bool:
        """Serializes current in-memory student records to storage."""
        records = [student.to_dict() for student in self.students.values()]
        return self.storage.save_data(records)

    def add_student(self, student: Student) -> bool:
        """Enrolls a new student if ID does not already exist."""
        if student.student_id in self.students:
            raise ValueError(f"Student with ID '{student.student_id}' already exists.")
        self.students[student.student_id] = student
        self.save_to_storage()
        return True

    def get_student(self, student_id: str) -> Optional[Student]:
        """Retrieves a single student by ID."""
        return self.students.get(str(student_id).strip())

    def get_all_students(self) -> List[Student]:
        """Returns a list of all enrolled students sorted by ID."""
        return sorted(self.students.values(), key=lambda s: s.student_id)

    def search_students(self, query: str) -> List[Student]:
        """Searches for students by matching ID or partial name (case-insensitive)."""
        clean_query = query.strip().lower()
        if not clean_query:
            return []

        results = []
        for student in self.students.values():
            if clean_query in student.student_id.lower() or clean_query in student.name.lower():
                results.append(student)

        return sorted(results, key=lambda s: s.name)

    def update_student(
        self,
        student_id: str,
        name: Optional[str] = None,
        age: Optional[int] = None,
        course: Optional[str] = None,
        marks: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Updates specific fields of an existing student."""
        student = self.get_student(student_id)
        if not student:
            raise KeyError(f"Student with ID '{student_id}' not found.")

        if name is not None:
            if len(name.strip()) < 2:
                raise ValueError("Student name must be at least 2 characters long.")
            student.name = name.strip()

        if age is not None:
            if not (10 <= age <= 100):
                raise ValueError("Age must be between 10 and 100.")
            student.age = int(age)

        if course is not None:
            if not course.strip():
                raise ValueError("Course title cannot be empty.")
            student.course = course.strip()

        if marks is not None:
            for subject, score in marks.items():
                student.set_mark(subject, score)

        self.save_to_storage()
        return True

    def delete_student(self, student_id: str) -> bool:
        """Deletes a student record by ID."""
        clean_id = str(student_id).strip()
        if clean_id not in self.students:
            raise KeyError(f"Student with ID '{clean_id}' not found.")

        del self.students[clean_id]
        self.save_to_storage()
        return True

    def get_class_analytics(self) -> Dict[str, Any]:
        """Computes statistical metrics across all enrolled students."""
        all_students = self.get_all_students()
        total_count = len(all_students)

        if total_count == 0:
            return {
                "total_students": 0,
                "average_percentage": 0.0,
                "highest_scorer": None,
                "lowest_scorer": None,
                "grade_distribution": {"A+": 0, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0},
                "pass_rate_percentage": 0.0,
            }

        percentages = [s.calculate_percentage() for s in all_students]
        avg_pct = round(sum(percentages) / total_count, 2)

        highest = max(all_students, key=lambda s: s.calculate_percentage())
        lowest = min(all_students, key=lambda s: s.calculate_percentage())

        grade_counts = {"A+": 0, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        passed_count = 0
        for s in all_students:
            grade = s.calculate_grade()
            if grade in grade_counts:
                grade_counts[grade] += 1
            if s.is_passed():
                passed_count += 1

        pass_rate = round((passed_count / total_count) * 100, 2)

        return {
            "total_students": total_count,
            "average_percentage": avg_pct,
            "highest_scorer": highest,
            "lowest_scorer": lowest,
            "grade_distribution": grade_counts,
            "pass_rate_percentage": pass_rate,
        }

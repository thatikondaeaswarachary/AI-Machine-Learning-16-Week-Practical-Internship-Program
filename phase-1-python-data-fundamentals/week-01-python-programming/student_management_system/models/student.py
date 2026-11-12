"""
Student Model
Defines the Student entity with academic scoring, validation, and serialization.
"""

from typing import Dict, Any


class Student:
    """Represents a student enrolled in the institution."""

    GRADE_SCALE = [
        (90.0, "A+"),
        (80.0, "A"),
        (70.0, "B"),
        (60.0, "C"),
        (50.0, "D"),
        (0.0,  "F"),
    ]

    def __init__(
        self,
        student_id: str,
        name: str,
        age: int,
        course: str,
        marks: Dict[str, float] = None,
    ):
        self.student_id = str(student_id).strip()
        self.name = str(name).strip()
        self.age = int(age)
        self.course = str(course).strip()
        self.marks: Dict[str, float] = {}

        self.validate()

        if marks:
            for subject, score in marks.items():
                self.set_mark(subject, score)

    def validate(self) -> None:
        """Validates student attributes."""
        if not self.student_id:
            raise ValueError("Student ID cannot be empty.")
        if not self.name or len(self.name) < 2:
            raise ValueError("Student name must be at least 2 characters long.")
        if not (10 <= self.age <= 100):
            raise ValueError(f"Student age must be between 10 and 100. Got: {self.age}")
        if not self.course:
            raise ValueError("Course title cannot be empty.")

    def set_mark(self, subject: str, score: float) -> None:
        """Adds or updates a subject score after validation (0 to 100)."""
        subject_clean = str(subject).strip()
        if not subject_clean:
            raise ValueError("Subject name cannot be empty.")

        try:
            score_val = float(score)
        except (ValueError, TypeError):
            raise ValueError(f"Mark for '{subject}' must be a numeric value.")

        if not (0.0 <= score_val <= 100.0):
            raise ValueError(f"Mark for '{subject}' must be between 0 and 100. Got: {score_val}")

        self.marks[subject_clean] = round(score_val, 2)

    def calculate_total(self) -> float:
        """Returns the sum of all subject marks."""
        return round(sum(self.marks.values()), 2)

    def calculate_percentage(self) -> float:
        """Calculates percentage average across all enrolled subjects."""
        if not self.marks:
            return 0.0
        return round(self.calculate_total() / len(self.marks), 2)

    def calculate_grade(self) -> str:
        """Determines the letter grade based on percentage."""
        if not self.marks:
            return "N/A"
        pct = self.calculate_percentage()
        for threshold, grade in self.GRADE_SCALE:
            if pct >= threshold:
                return grade
        return "F"

    def is_passed(self, passing_pct: float = 50.0) -> bool:
        """Checks whether the student meets passing threshold."""
        return self.calculate_percentage() >= passing_pct

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Student instance into a dictionary."""
        return {
            "student_id": self.student_id,
            "name": self.name,
            "age": self.age,
            "course": self.course,
            "marks": self.marks,
            "total_marks": self.calculate_total(),
            "percentage": self.calculate_percentage(),
            "grade": self.calculate_grade(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Student":
        """Factory method to construct a Student instance from dictionary data."""
        return cls(
            student_id=data.get("student_id", ""),
            name=data.get("name", ""),
            age=data.get("age", 0),
            course=data.get("course", ""),
            marks=data.get("marks", {}),
        )

    def __str__(self) -> str:
        pct = self.calculate_percentage()
        grade = self.calculate_grade()
        return (
            f"[{self.student_id}] {self.name} | Age: {self.age} | "
            f"Course: {self.course} | Score: {pct}% ({grade})"
        )

    def __repr__(self) -> str:
        return f"<Student(id='{self.student_id}', name='{self.name}', grade='{self.calculate_grade()}')>"

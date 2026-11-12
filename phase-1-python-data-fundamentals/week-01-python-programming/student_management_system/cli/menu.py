"""
CLI Menu Interface
Interactive terminal dashboard for the Student Management System.
"""

import sys
from typing import Dict

try:
    from ..services.student_manager import StudentManager
    from ..models.student import Student
except (ImportError, ValueError):
    from services.student_manager import StudentManager
    from models.student import Student


class CLIApp:
    """Provides a menu-driven terminal interface."""

    def __init__(self, manager: StudentManager):
        self.manager = manager

    def display_banner(self) -> None:
        print("\n" + "=" * 76)
        print("     NEXT GEN SOFTWARE HUB - STUDENT MANAGEMENT SYSTEM (SMS)     ")
        print("                     AI & ML Internship Track                    ")
        print("=" * 76)

    def display_menu(self) -> None:
        print("\n[MAIN MENU]")
        print(" 1. Add New Student")
        print(" 2. View All Students")
        print(" 3. Search Student (by ID or Name)")
        print(" 4. Update Student Details / Marks")
        print(" 5. Delete Student Record")
        print(" 6. Class Academic Analytics & Grade Report")
        print(" 7. Reload Data from Storage")
        print(" 8. Exit System")
        print("-" * 76)

    def run(self) -> None:
        """Main event loop."""
        self.display_banner()
        while True:
            self.display_menu()
            choice = input("Enter choice (1-8): ").strip()

            if choice == "1":
                self.handle_add_student()
            elif choice == "2":
                self.handle_view_students()
            elif choice == "3":
                self.handle_search_student()
            elif choice == "4":
                self.handle_update_student()
            elif choice == "5":
                self.handle_delete_student()
            elif choice == "6":
                self.handle_analytics()
            elif choice == "7":
                self.manager._load_from_storage()
                print("[INFO] Reloaded latest records from storage.")
            elif choice == "8":
                print("\n[INFO] Saving data and exiting. Thank you!")
                sys.exit(0)
            else:
                print("[ERROR] Invalid choice. Please enter a number between 1 and 8.")

    # ------------------ Handlers ------------------

    def handle_add_student(self) -> None:
        print("\n--- Enroll New Student ---")
        student_id = self._prompt_string("Enter Student ID (e.g. NGS-106): ").upper()
        if self.manager.get_student(student_id):
            print(f"[ERROR] Student with ID '{student_id}' already exists!")
            return

        name = self._prompt_string("Enter Full Name: ")
        age = self._prompt_int("Enter Age (10-100): ", min_val=10, max_val=100)
        course = self._prompt_string("Enter Course Name (e.g. AI & Machine Learning): ")

        print("\nEnter Marks for Core Subjects (0 - 100):")
        subjects = ["Python Programming", "Mathematics for ML", "Data Science"]
        marks: Dict[str, float] = {}
        for sub in subjects:
            score = self._prompt_float(f"  > Score in {sub}: ", min_val=0.0, max_val=100.0)
            marks[sub] = score

        # Optional extra subjects
        while True:
            add_more = input("Add additional subject? (y/n): ").strip().lower()
            if add_more == "y":
                sub_name = self._prompt_string("  Enter Subject Name: ")
                score = self._prompt_float(f"  Score in {sub_name}: ", min_val=0.0, max_val=100.0)
                marks[sub_name] = score
            else:
                break

        try:
            student = Student(student_id=student_id, name=name, age=age, course=course, marks=marks)
            self.manager.add_student(student)
            print(f"\n[SUCCESS] Student '{name}' (ID: {student_id}) enrolled successfully!")
            print(f"Total: {student.calculate_total()} | Percentage: {student.calculate_percentage()}% | Grade: {student.calculate_grade()}")
        except Exception as e:
            print(f"[ERROR] Failed to add student: {e}")

    def handle_view_students(self) -> None:
        students = self.manager.get_all_students()
        if not students:
            print("\n[INFO] No student records found.")
            return

        print(f"\nTotal Enrolled Students: {len(students)}")
        self._render_students_table(students)

    def handle_search_student(self) -> None:
        print("\n--- Search Student ---")
        query = self._prompt_string("Enter Student ID or Student Name: ")
        results = self.manager.search_students(query)

        if not results:
            print(f"[INFO] No students found matching '{query}'.")
            return

        print(f"\nFound {len(results)} matching record(s):")
        self._render_students_table(results)

    def handle_update_student(self) -> None:
        print("\n--- Update Student Details ---")
        student_id = self._prompt_string("Enter Student ID to update: ").upper()
        student = self.manager.get_student(student_id)
        if not student:
            print(f"[ERROR] Student with ID '{student_id}' does not exist.")
            return

        print(f"\nCurrent Record: {student.name} | Age: {student.age} | Course: {student.course}")
        print("Leave field blank to keep current value.")

        new_name = input(f"New Name [{student.name}]: ").strip()
        new_age_str = input(f"New Age [{student.age}]: ").strip()
        new_course = input(f"New Course [{student.course}]: ").strip()

        new_age = None
        if new_age_str:
            try:
                new_age = int(new_age_str)
            except ValueError:
                print("[WARNING] Invalid age entered; keeping previous value.")

        update_marks = input("Update subject marks? (y/n): ").strip().lower()
        new_marks = None
        if update_marks == "y":
            new_marks = {}
            for sub, curr_score in student.marks.items():
                score_str = input(f"  Score for {sub} [Current: {curr_score}]: ").strip()
                if score_str:
                    try:
                        new_marks[sub] = float(score_str)
                    except ValueError:
                        print(f"  [Warning] Invalid number. Keeping {curr_score}")
                        new_marks[sub] = curr_score
                else:
                    new_marks[sub] = curr_score

        try:
            self.manager.update_student(
                student_id=student_id,
                name=new_name if new_name else None,
                age=new_age,
                course=new_course if new_course else None,
                marks=new_marks,
            )
            print(f"[SUCCESS] Student '{student_id}' updated successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to update: {e}")

    def handle_delete_student(self) -> None:
        print("\n--- Delete Student Record ---")
        student_id = self._prompt_string("Enter Student ID to delete: ").upper()
        student = self.manager.get_student(student_id)
        if not student:
            print(f"[ERROR] Student with ID '{student_id}' not found.")
            return

        confirm = input(f"Are you sure you want to delete {student.name} ({student_id})? (yes/no): ").strip().lower()
        if confirm in ["yes", "y"]:
            self.manager.delete_student(student_id)
            print(f"[SUCCESS] Student record '{student_id}' deleted.")
        else:
            print("[INFO] Deletion cancelled.")

    def handle_analytics(self) -> None:
        print("\n" + "=" * 60)
        print("          CLASS ACADEMIC ANALYTICS & GRADE REPORT         ")
        print("=" * 60)
        analytics = self.manager.get_class_analytics()
        total = analytics["total_students"]
        if total == 0:
            print("[INFO] No student records available for analytics.")
            return

        print(f" Total Students Enrolled : {total}")
        print(f" Class Average Percentage : {analytics['average_percentage']}%")
        print(f" Overall Pass Rate       : {analytics['pass_rate_percentage']}%")

        top = analytics["highest_scorer"]
        if top:
            print(f" Top Performer           : {top.name} ({top.student_id}) - {top.calculate_percentage()}% (Grade {top.calculate_grade()})")

        low = analytics["lowest_scorer"]
        if low:
            print(f" Lowest Performer        : {low.name} ({low.student_id}) - {low.calculate_percentage()}% (Grade {low.calculate_grade()})")

        print("\n Grade Distribution:")
        for grade, count in analytics["grade_distribution"].items():
            bar = "#" * count
            print(f"   Grade {grade:<2} : {count:>2} student(s) | {bar}")
        print("=" * 60)

    # ------------------ Formatters & Input Helpers ------------------

    def _render_students_table(self, students) -> None:
        header = f"| {'ID':<8} | {'Name':<18} | {'Age':<3} | {'Course':<22} | {'Total':<6} | {'%':<6} | {'Grade':<5} |"
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for s in students:
            print(
                f"| {s.student_id:<8} | {s.name:<18} | {s.age:<3} | {s.course[:22]:<22} | "
                f"{s.calculate_total():<6.1f} | {s.calculate_percentage():<6.2f} | {s.calculate_grade():<5} |"
            )
        print(sep)

    def _prompt_string(self, msg: str) -> str:
        while True:
            val = input(msg).strip()
            if val:
                return val
            print("[ERROR] Input cannot be blank.")

    def _prompt_int(self, msg: str, min_val: int, max_val: int) -> int:
        while True:
            raw = input(msg).strip()
            try:
                val = int(raw)
                if min_val <= val <= max_val:
                    return val
                print(f"[ERROR] Value must be between {min_val} and {max_val}.")
            except ValueError:
                print("[ERROR] Please enter a valid integer.")

    def _prompt_float(self, msg: str, min_val: float, max_val: float) -> float:
        while True:
            raw = input(msg).strip()
            try:
                val = float(raw)
                if min_val <= val <= max_val:
                    return val
                print(f"[ERROR] Value must be between {min_val} and {max_val}.")
            except ValueError:
                print("[ERROR] Please enter a valid decimal number.")

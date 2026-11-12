#!/usr/bin/env python3
"""
Student Management System - Main Entry Point
Next Gen Software Hub Pvt. Ltd. - AI & ML Internship Program
Week 1 Practical Assignment
"""

import os
import sys

# Ensure package imports resolve when executing directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from storage.json_storage import JSONStorage
from services.student_manager import StudentManager
from cli.menu import CLIApp


def main():
    data_dir = os.path.join(CURRENT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    storage_file = os.path.join(data_dir, "students_sample.json")

    # Initialize storage layer
    storage = JSONStorage(storage_file)

    # Initialize service / business logic layer
    manager = StudentManager(storage)

    # Initialize and start user interface
    app = CLIApp(manager)
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\n[INFO] Application interrupted by user. Exiting safely.")
        sys.exit(0)


if __name__ == "__main__":
    main()

"""
JSON Storage Module
Handles file I/O operations, JSON encoding/decoding, and automatic backups.
"""

import json
import os
import shutil
from typing import List, Dict, Any


class JSONStorage:
    """Manages persistent reading and writing of student records to a JSON file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Creates directory and an empty JSON list file if it doesn't exist."""
        dirname = os.path.dirname(self.filepath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump([], f, indent=4)

    def load_data(self) -> List[Dict[str, Any]]:
        """Loads records from the JSON file. Falls back gracefully on corruption."""
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                return []
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Warning] Failed to read {self.filepath}: {e}. Initializing empty list.")
            return []

    def save_data(self, data: List[Dict[str, Any]]) -> bool:
        """Saves data to JSON with an automated backup copy."""
        try:
            # Create a backup before overwriting if the primary file exists and is not empty
            if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
                backup_path = f"{self.filepath}.bak"
                shutil.copy2(self.filepath, backup_path)

            temp_path = f"{self.filepath}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            # Atomic rename
            shutil.move(temp_path, self.filepath)
            return True
        except Exception as e:
            print(f"[Error] Failed to save to {self.filepath}: {e}")
            return False

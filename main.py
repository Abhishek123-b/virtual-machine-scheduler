import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.gui.main_window import VMSchedulerGUI
import tkinter as tk
from tkinter import messagebox

def main():
    """Main entry point for the VM Scheduler application."""
    try:
        app = VMSchedulerGUI()
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
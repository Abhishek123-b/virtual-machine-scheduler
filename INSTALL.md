# Installation Guide

This document provides detailed instructions for setting up the VM Scheduler Simulator.

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Operating System: Windows 10/11, Linux, or macOS

## Installation Steps

### 1. Python Installation

If you don't have Python installed:

a. Windows:
- Download Python from [python.org](https://python.org)
- Run the installer
- Check "Add Python to PATH"
- Verify installation: `python --version`

b. Linux:
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv
```

c. macOS:
```bash
brew install python
```

### 2. Project Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vm-scheduler-simulator.git
cd vm-scheduler-simulator
```

2. Create virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Verify Installation

Run the test script:
```bash
python tests/test_setup.py
```

You should see "Setup successful!" if everything is installed correctly.

## Common Issues

1. **tkinter not found**:
   - Windows: Reinstall Python with tkinter option
   - Linux: `sudo apt-get install python3-tk`
   - macOS: `brew install python-tk`

2. **Permission errors**:
   - Use `sudo` for Linux/macOS
   - Run as administrator on Windows

3. **Dependency conflicts**:
   - Create a new virtual environment
   - Delete existing one if necessary
   ```bash
   rm -rf venv
   python -m venv venv
   ```

## Running the Application

1. Activate virtual environment (if not already active):
```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

2. Start the application:
```bash
python main.py
```

## Development Setup

For developers, additional tools:

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Setup pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest tests/
```

## Troubleshooting

If you encounter issues:

1. Check Python version:
```bash
python --version
```

2. Verify pip installation:
```bash
pip list
```

3. Check virtual environment:
```bash
# Should show virtual environment path
which python
```

4. Common fixes:
- Clear pip cache: `pip cache purge`
- Upgrade pip: `pip install --upgrade pip`
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

## Support

For additional help:
- Create an issue on GitHub
- Check the FAQ in documentation
- Contact: support@vmscheduler.com 
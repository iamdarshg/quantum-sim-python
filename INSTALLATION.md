# Installation Guide for Quantum Lattice Simulator v2.0

## 📁 Package Structure

Create the following directory structure and place the files:

```
quantum_lattice_simulator/
├── setup.py                              # Package installation script
├── requirements.txt                      # Dependencies
├── README.md                            # Documentation
├── demo.py                              # Interactive demo (WORKING!)
└── quantum_lattice/                     # Main package directory
    ├── __init__.py                      # Main package file (rename quantum_lattice_init.py)
    ├── core/
    │   ├── __init__.py                  # Create empty file
    │   ├── simulator.py                 # Main simulator
    │   └── parameters.py                # Configuration
    ├── physics/
    │   ├── __init__.py                  # Create empty file
    │   └── nuclear.py                   # Nuclear database
    ├── numerical/
    │   ├── __init__.py                  # Create empty file
    │   └── methods.py                   # Create empty file (placeholder)
    └── gui/
        ├── __init__.py                  # Create empty file
        └── interface.py                 # 3D GUI
```

## 🛠️ Step-by-Step Setup

### 1. Create Directory Structure
```bash
mkdir -p quantum_lattice_simulator/quantum_lattice/core
mkdir -p quantum_lattice_simulator/quantum_lattice/physics
mkdir -p quantum_lattice_simulator/quantum_lattice/numerical
mkdir -p quantum_lattice_simulator/quantum_lattice/gui
```

### 2. Download and Place Files
- `setup.py` → `quantum_lattice_simulator/setup.py`
- `requirements.txt` → `quantum_lattice_simulator/requirements.txt` 
- `README.md` → `quantum_lattice_simulator/README.md`
- `demo.py` → `quantum_lattice_simulator/demo.py`
- `quantum_lattice_init.py` → `quantum_lattice_simulator/quantum_lattice/__init__.py`
- `parameters.py` → `quantum_lattice_simulator/quantum_lattice/core/parameters.py`
- `simulator.py` → `quantum_lattice_simulator/quantum_lattice/core/simulator.py`
- `nuclear.py` → `quantum_lattice_simulator/quantum_lattice/physics/nuclear.py`
- `interface.py` → `quantum_lattice_simulator/quantum_lattice/gui/interface.py`

### 3. Create Empty __init__.py Files
Create these empty files:
```bash
touch quantum_lattice_simulator/quantum_lattice/core/__init__.py
touch quantum_lattice_simulator/quantum_lattice/physics/__init__.py
touch quantum_lattice_simulator/quantum_lattice/numerical/__init__.py
touch quantum_lattice_simulator/quantum_lattice/gui/__init__.py
```

### 4. Create Simple Placeholder Files
For `quantum_lattice_simulator/quantum_lattice/numerical/methods.py`:
```python
"""Placeholder for numerical methods."""
pass
```

## 🚀 Installation

### 1. Navigate to Package Directory
```bash
cd quantum_lattice_simulator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
 
### 3. Install Package in Development Mode
```bash
pip install -e .
```

### 4. Test Installation
```bash
python demo.py
```

## 🎮 Usage Examples

### Command Line Test
```python
from quantum_lattice import create_simulator
simulator = create_simulator("Au197", "Au197", 200.0)
print(f"Simulator ready: {simulator.params}")
```

### Launch GUI
```python
from quantum_lattice import launch_gui
launch_gui()
```

### List Available Nuclei
```python
from quantum_lattice import list_nuclei
print("Available nuclei:", list_nuclei())
```

## 🐛 Troubleshooting

### Import Errors
If you get import errors:
1. Make sure all `__init__.py` files exist
2. Check that you're in the correct directory
3. Reinstall with: `pip uninstall quantum-lattice-simulator && pip install -e .`

### Missing Dependencies
Install optional dependencies:
```bash
pip install matplotlib plotly numba
```

### GUI Issues
For GUI problems:
- Make sure tkinter is available: `python -c "import tkinter; print('OK')"`
- Install matplotlib: `pip install matplotlib`

## ✅ Verification

After installation, run this test:
```python
import quantum_lattice
quantum_lattice.version_info()
nuclei = quantum_lattice.list_nuclei()
print(f"Found {len(nuclei)} nuclei in database")
```

You should see version information and a list of available nuclei.

## 🎉 Ready to Go!

Once installed, you can:
- Run `python demo.py` for interactive demos
- Launch the GUI with `python demo.py gui`
- Use in your own scripts with `from quantum_lattice import create_simulator`

**The package is now ready for nuclear collision simulations!** 🚀
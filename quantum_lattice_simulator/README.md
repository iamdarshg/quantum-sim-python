# Quantum Lattice Nuclear Collision Simulator v2.0

🔬 **Advanced quantum field theory simulator for nuclear collisions with 3D visualization**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Physics](https://img.shields.io/badge/physics-quantum%20field%20theory-red.svg)]()

## 🚀 Features

### 🎯 Physics
- **Complete Standard Model**: QED + QCD + Electroweak theory
- **Nuclear Database**: 12+ isotopes from H to U238 with experimental parameters
- **Realistic Nuclear Structure**: Woods-Saxon density profiles, shell effects
- **Advanced Collision Dynamics**: IP-Glasma initial conditions, Glauber geometry

### 🎆 3D Visualization
- **Real-time 3D collision visualization** with Matplotlib
- **Particle production animation** showing expanding fireball
- **Nuclear structure display** with individual nucleons
- **Physics plots**: Temperature, energy density, particle multiplicity

### ⚡ Performance
- **Multithreaded computation** with automatic CPU core detection  
- **Optimized algorithms** for maximum accuracy and speed
- **Adaptive time stepping** for optimal accuracy/speed balance
- **Memory-optimized** data structures for large systems

## 📦 Installation

### Quick Install
```bash
pip install -e .
```

### With Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

## 🎮 Quick Start

### 1. GUI Mode (Recommended)
```bash
python demo.py gui
```
Or in Python:
```python
from quantum_lattice import launch_gui
launch_gui()
```

### 2. Command Line
```python
from quantum_lattice import create_simulator

# Create Au+Au collision at 200 GeV
simulator = create_simulator("Au197", "Au197", 200.0)
results = simulator.run_simulation()

print(f"Max temperature: {max(results['observables']['temperature']):.1f} MeV")
```

### 3. Interactive Demo
```bash
python demo.py
```

## 🔬 Available Nuclei

| **Light** | **Medium** | **Heavy** |
|-----------|------------|-----------|
| H, D, He3, He4 | Ca40, Fe56, Cu63 | Au197, Pb208, U238 |
| C12, O16 | And more... | And more... |

View all available nuclei:
```python
from quantum_lattice import list_nuclei
print(list_nuclei())
```

## 🎯 Example Simulations

### RHIC Au+Au at 200 GeV
```python
simulator = create_simulator("Au197", "Au197", 200.0)
results = simulator.run_simulation()
```

### LHC Pb+Pb at 2.76 TeV  
```python
simulator = create_simulator("Pb208", "Pb208", 2760.0)
results = simulator.run_simulation()
```

### Future O+O collisions
```python
simulator = create_simulator("O16", "O16", 100.0)
results = simulator.run_simulation()
```

## 📊 Output & Analysis

The simulator provides:
- **Temperature evolution** and QGP formation detection
- **Energy density** profiles
- **Particle multiplicity** estimates (π, K, p)
- **Pressure** and equation of state
- **3D collision visualization**
- **JSON export** for further analysis

## 🖼️ 3D Visualization Features

- **Nuclear collision animation**: Watch nuclei approach and collide
- **Particle production**: See particles emerge from collision zone  
- **Real-time physics plots**: Temperature, energy, multiplicity
- **Interactive 3D controls**: Zoom, rotate, pan
- **Export capabilities**: Save images and data

## 🏗️ Package Structure

```
quantum_lattice/
├── core/           # Main simulation engine
├── physics/        # Nuclear database & structure  
├── numerical/      # Advanced numerical methods
└── gui/           # 3D visualization interface
```

## 🎓 Educational Use

Perfect for:
- Graduate nuclear/particle physics courses
- Research group demonstrations  
- Physics education and outreach
- Computational physics training

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Additional nuclear isotopes
- Advanced physics improvements
- Performance optimizations
- Visualization enhancements

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 📚 Citation

```bibtex
@software{quantum_lattice_2025,
  title={Quantum Lattice Nuclear Collision Simulator v2.0},
  author={Advanced Physics Simulation Team},
  year={2025},
  url={https://github.com/quantum-lattice/simulator}
}
```

## 🆘 Support

- 📧 Email: physics@quantum-lattice.org
- 🐛 Issues: [GitHub Issues](https://github.com/quantum-lattice/simulator/issues)  
- 💬 Discussions: [GitHub Discussions](https://github.com/quantum-lattice/simulator/discussions)

---

**🚀 Ready to explore the quantum world of nuclear collisions!** 🎆
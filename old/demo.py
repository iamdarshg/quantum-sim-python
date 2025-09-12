#!/usr/bin/env python3
"""
Enhanced Quantum Lattice Nuclear Collision Simulator v2.0
Quick Demo Launcher
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_quantum_simulator import *
    print("🚀 Enhanced Quantum Lattice Nuclear Collision Simulator v2.0")
    print("=" * 65)

    # Quick demo
    print("\n🔬 Quick Demo - Au+Au collision at RHIC energy")

    params = EnhancedSimulationParameters()
    params.nucleus_A = "Au197"
    params.nucleus_B = "Au197" 
    params.collision_energy_gev = 200.0
    params.lattice_sizes = [(16, 16, 16), (24, 24, 24)]  # Small for demo
    params.max_iterations = 50  # Short demo

    print(f"  Collision: {params.nucleus_A} + {params.nucleus_B}")
    print(f"  Energy: {params.collision_energy_gev} GeV")
    print(f"  Lattices: {params.lattice_sizes}")

    simulator = EnhancedQuantumLatticeSimulator(params)
    print("\n✅ Simulator initialized successfully!")
    print("\n🎯 Full GUI available by running:")
    print("   python3 -c \"from enhanced_quantum_simulator import *; EnhancedQuantumLatticeGUI().run()\"")
    print("\n🧪 Validation test available in validation_report.md")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install required packages:")
    print("   pip install numpy scipy matplotlib psutil")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nCheck that all files are present and properly configured.")

print("\n" + "=" * 65)
print("🎉 Enhanced Quantum Lattice Simulator Package Ready!")

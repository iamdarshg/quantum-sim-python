"""
Quantum Lattice Nuclear Collision Simulator v2.0

A comprehensive quantum field theory simulator for nuclear collisions
implementing QED, QCD, and electroweak theory on discrete spacetime.
"""

__version__ = "2.0.0"
__author__ = "Advanced Physics Simulation Team"

# Main imports for easy access
from .core.gui.simulator import *
from .core.gui.parameters import *
from .core.physics.nuclear import *
from .gui.interface import *
# AdvancedSimulatorGUI = SimulatorGUI
# Convenient shortcuts
def create_simulator(nucleus_a="Au197", nucleus_b="Au197", energy_gev=200.0):
    """Quick simulator creation with common defaults."""
    params = SimulationParameters()
    params.nucleus_A = nucleus_a
    params.nucleus_B = nucleus_b
    params.collision_energy_gev = energy_gev
    return QuantumLatticeSimulator(params)

def launch_gui():
    """Launch the graphical interface."""
    gui = SimulatorGUI()
    gui.run()

def list_nuclei():
    """List available nuclei in the database."""
    pass
    # return list(NuclearDatabase.get_available_nuclei())

# Version info
def version_info():
    """Print version and system information."""
    import sys
    print(f"Quantum Lattice Simulator v{__version__}")
    print(f"Python: {sys.version}")
    print(f"System: {sys.platform}")
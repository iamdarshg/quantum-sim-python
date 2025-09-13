"""
Simulation parameters and configuration.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

@dataclass
class SimulationParameters:
    """Configuration parameters for quantum lattice simulation."""
    
    # Collision parameters
    collision_energy_gev: float = 200.0
    impact_parameter_fm: float = 5.0
    nucleus_A: str = "Au197"
    nucleus_B: str = "Au197"
    
    # Lattice parameters  
    lattice_sizes: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [(24, 24, 24), (32, 32, 32)]
    )
    lattice_spacings_fm: List[float] = field(
        default_factory=lambda: [0.15, 0.10]
    )
    time_step_fm: float = 0.01
    
    # Physics parameters
    qcd_coupling: float = 0.118
    qed_coupling: float = 1/137.036
    weak_coupling: float = 0.65379
    higgs_vev_gev: float = 246.22
    
    # Numerical parameters
    max_iterations: int = 1000
    convergence_threshold: float = 1e-8
    rk_order: int = 4  # Runge-Kutta order
    
    # Performance parameters
    use_multithreading: bool = True
    num_threads: int = 0  # 0 = auto-detect
    use_gpu: bool = False
    
    # Analysis parameters
    calculate_flow: bool = True
    calculate_spectra: bool = True
    save_snapshots: bool = False
    
    def validate(self):
        """Validate parameter values."""
        if self.collision_energy_gev <= 0:
            raise ValueError("Collision energy must be positive")
        if self.time_step_fm <= 0:
            raise ValueError("Time step must be positive")
        if not self.lattice_sizes:
            raise ValueError("At least one lattice size must be specified")
        
    def to_dict(self):
        """Convert parameters to dictionary."""
        return {
            'collision_energy_gev': self.collision_energy_gev,
            'nucleus_A': self.nucleus_A,
            'nucleus_B': self.nucleus_B,
            'lattice_sizes': self.lattice_sizes,
            'max_iterations': self.max_iterations
        }
    
    def __str__(self):
        return f"SimulationParameters(E={self.collision_energy_gev} GeV, {self.nucleus_A}+{self.nucleus_B})"
"""
Nuclear structure and database.
"""

import numpy as np
from typing import Dict, List, Tuple
import json

# Complete nuclear database
NUCLEAR_DATA = {
    # Light nuclei
    "H": {"A": 1, "Z": 1, "radius_fm": 0.88, "binding_energy": 0.0, "spin": 0.5},
    "D": {"A": 2, "Z": 1, "radius_fm": 2.14, "binding_energy": 2.22, "spin": 1.0},
    "He3": {"A": 3, "Z": 2, "radius_fm": 1.96, "binding_energy": 7.72, "spin": 0.5},
    "He4": {"A": 4, "Z": 2, "radius_fm": 1.68, "binding_energy": 28.30, "spin": 0.0},
    "C12": {"A": 12, "Z": 6, "radius_fm": 2.47, "binding_energy": 92.16, "spin": 0.0},
    "O16": {"A": 16, "Z": 8, "radius_fm": 2.70, "binding_energy": 127.62, "spin": 0.0},
    
    # Medium nuclei
    "Ca40": {"A": 40, "Z": 20, "radius_fm": 3.48, "binding_energy": 342.05, "spin": 0.0},
    "Fe56": {"A": 56, "Z": 26, "radius_fm": 3.74, "binding_energy": 492.26, "spin": 0.0},
    "Cu63": {"A": 63, "Z": 29, "radius_fm": 3.91, "binding_energy": 551.38, "spin": 1.5},
    
    # Heavy nuclei
    "Au197": {"A": 197, "Z": 79, "radius_fm": 6.38, "binding_energy": 1559.40, "spin": 1.5},
    "Pb208": {"A": 208, "Z": 82, "radius_fm": 6.68, "binding_energy": 1636.45, "spin": 0.0},
    "U238": {"A": 238, "Z": 92, "radius_fm": 7.44, "binding_energy": 1801.69, "spin": 0.0},
}

class NuclearStructure:
    """Nuclear structure with realistic properties."""
    
    def __init__(self, nucleus_name: str):
        if nucleus_name not in NUCLEAR_DATA:
            raise ValueError(f"Unknown nucleus: {nucleus_name}")
        
        self.name = nucleus_name
        self.data = NUCLEAR_DATA[nucleus_name]
        self.A = self.data["A"]
        self.Z = self.data["Z"]
        self.N = self.A - self.Z
        self.radius_fm = self.data["radius_fm"]
        self.binding_energy = self.data["binding_energy"]
        self.spin = self.data["spin"]
        
        # Nuclear parameters
        self.r0 = 1.2  # fm
        self.a = 0.54  # surface diffuseness
        self.rho_0 = 0.17  # nuclear density fm^-3
    
    def woods_saxon_density(self, r: np.ndarray) -> np.ndarray:
        """Woods-Saxon nuclear density profile."""
        return self.rho_0 / (1.0 + np.exp((r - self.radius_fm) / self.a))
    
    def generate_nucleon_positions(self, num_samples: int = None) -> List[Tuple[float, float, float]]:
        """Generate nucleon positions using Monte Carlo."""
        if num_samples is None:
            num_samples = min(self.A, 200)
        
        positions = []
        for _ in range(num_samples):
            # Rejection sampling for Woods-Saxon
            accepted = False
            attempts = 0
            while not accepted and attempts < 1000:
                # Sample sphere uniformly
                r = 3.0 * self.radius_fm * np.random.random()**(1/3)
                theta = np.arccos(2 * np.random.random() - 1)
                phi = 2 * np.pi * np.random.random()
                
                # Calculate acceptance probability
                density = self.woods_saxon_density(np.array([r]))[0]
                prob = density / self.rho_0
                
                if np.random.random() < prob:
                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = r * np.cos(theta)
                    positions.append((x, y, z))
                    accepted = True
                
                attempts += 1
        
        return positions
    
    def __str__(self):
        return f"Nuclear({self.name}: A={self.A}, Z={self.Z}, R={self.radius_fm:.2f} fm)"

class NuclearDatabase:
    """Interface to nuclear database."""
    
    @staticmethod
    def get_available_nuclei() -> List[str]:
        """Get list of available nuclei."""
        return list(NUCLEAR_DATA.keys())
    
    @staticmethod
    def get_nucleus_info(name: str) -> Dict:
        """Get nuclear information."""
        if name not in NUCLEAR_DATA:
            raise ValueError(f"Unknown nucleus: {name}")
        return NUCLEAR_DATA[name].copy()
    
    @staticmethod
    def search_by_mass_range(min_A: int, max_A: int) -> List[str]:
        """Find nuclei in mass range."""
        return [name for name, data in NUCLEAR_DATA.items() 
                if min_A <= data["A"] <= max_A]
    
    @staticmethod
    def get_collision_systems() -> List[Tuple[str, str]]:
        """Get common collision systems."""
        return [
            ("Au197", "Au197"),  # RHIC
            ("Pb208", "Pb208"),  # LHC
            ("Cu63", "Cu63"),    # RHIC
            ("O16", "O16"),      # Future
            ("Ca40", "Ca40"),    # Future
        ]

# Convenience function
def create_nucleus(name: str) -> NuclearStructure:
    """Create nuclear structure object."""
    return NuclearStructure(name)
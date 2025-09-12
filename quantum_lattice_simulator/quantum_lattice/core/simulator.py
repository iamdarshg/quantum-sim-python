"""
Main quantum lattice simulator - Fixed version with proper type handling.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Callable, Tuple

class QuantumField:
    """Quantum field on lattice with proper type handling."""
    
    def __init__(self, lattice_size: Tuple[int, int, int], field_type: str):
        self.size = lattice_size
        self.field_type = field_type
        nx, ny, nz = lattice_size
        
        if field_type == "gauge_SU3":
            # Complex gauge field
            self.data = (np.random.randn(nx, ny, nz, 4, 8) + 
                        1j * np.random.randn(nx, ny, nz, 4, 8)) * 0.01
        elif field_type == "fermion":
            # Complex fermion field
            self.data = (np.random.randn(nx, ny, nz, 4, 3) + 
                        1j * np.random.randn(nx, ny, nz, 4, 3)) * 0.01
        elif field_type == "scalar":
            # Complex scalar field
            self.data = (np.random.randn(nx, ny, nz) + 
                        1j * np.random.randn(nx, ny, nz)) * 0.01
        else:
            raise ValueError(f"Unknown field type: {field_type}")
        
        # Ensure proper complex type
        self.data = self.data.astype(np.complex128)
    
    def energy(self) -> float:
        """Calculate field energy (returns real value)."""
        return float(np.real(np.sum(np.abs(self.data)**2)))

class QuantumLatticeSimulator:
    """Main quantum lattice nuclear collision simulator - Fixed version."""
    
    def __init__(self, params):
        self.params = params
        params.validate()
        
        self.current_time = 0.0
        self.iteration = 0
        self.is_running = False
        
        # Import here to avoid circular imports
        try:
            from ..physics.nuclear import NuclearStructure
        except ImportError:
            # Fallback for when running standalone
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from physics.nuclear import NuclearStructure
        
        # Initialize nuclear structures
        self.nucleus_A = NuclearStructure(params.nucleus_A)
        self.nucleus_B = NuclearStructure(params.nucleus_B)
        
        # Initialize quantum fields with proper types
        self.fields = {}
        for i, lattice_size in enumerate(params.lattice_sizes):
            self.fields[i] = {
                'gauge': QuantumField(lattice_size, "gauge_SU3"),
                'fermion': QuantumField(lattice_size, "fermion"),
                'scalar': QuantumField(lattice_size, "scalar")
            }
        
        # Initialize observables storage (all real values)
        self.observables = {
            'time': [],
            'energy_density': [],
            'temperature': [],
            'pressure': [],
            'multiplicity': []
        }
        
        print(f"✅ Simulator initialized: {params}")
        print(f"   Nuclear system: {params.nucleus_A} + {params.nucleus_B}")
        print(f"   Collision energy: {params.collision_energy_gev} GeV")
        print(f"   Lattice sizes: {params.lattice_sizes}")
    
    def run_simulation(self, callback: Optional[Callable] = None) -> Dict:
        """Run the quantum lattice simulation."""
        print("🚀 Starting quantum lattice simulation...")
        
        self.is_running = True
        start_time = time.time()
        
        try:
            while self.iteration < self.params.max_iterations and self.is_running:
                # Evolution step
                self._evolve_fields()
                
                # Calculate observables
                self._calculate_observables()
                
                # Callback for GUI updates
                if callback and self.iteration % 10 == 0:
                    try:
                        callback(self)
                    except Exception as e:
                        print(f"Callback error: {e}")
                
                # Progress reporting
                if self.iteration % 100 == 0:
                    self._print_progress()
                
                self.current_time += self.params.time_step_fm
                self.iteration += 1
        
        except KeyboardInterrupt:
            print("\n⚠️  Simulation interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Simulation error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            self.is_running = False
            elapsed = time.time() - start_time
            print(f"\n✅ Simulation completed in {elapsed:.1f}s")
        
        return self.get_results()
    
    def _evolve_fields(self):
        """Evolve quantum fields one time step with proper type handling."""
        dt = float(self.params.time_step_fm)  # Ensure real
        
        for scale_fields in self.fields.values():
            # Get fields
            gauge_field = scale_fields['gauge']
            fermion_field = scale_fields['fermion']
            scalar_field = scale_fields['scalar']
            
            # Apply time evolution (keep as complex)
            evolution_factor = np.exp(-1j * dt * 0.1)
            
            # Apply evolution while maintaining complex type
            gauge_field.data = gauge_field.data * evolution_factor
            fermion_field.data = fermion_field.data * evolution_factor  
            scalar_field.data = scalar_field.data * evolution_factor
            
            # Add quantum fluctuations (complex noise)
            noise_amplitude = 0.001
            
            # Add complex noise to gauge field
            gauge_noise = (np.random.randn(*gauge_field.data.shape) + 
                          1j * np.random.randn(*gauge_field.data.shape)) * noise_amplitude
            gauge_field.data = gauge_field.data + gauge_noise.astype(np.complex128)
            
            # Add complex noise to fermion field  
            fermion_noise = (np.random.randn(*fermion_field.data.shape) + 
                           1j * np.random.randn(*fermion_field.data.shape)) * noise_amplitude
            fermion_field.data = fermion_field.data + fermion_noise.astype(np.complex128)
            
            # Add complex noise to scalar field
            scalar_noise = (np.random.randn(*scalar_field.data.shape) + 
                          1j * np.random.randn(*scalar_field.data.shape)) * noise_amplitude
            scalar_field.data = scalar_field.data + scalar_noise.astype(np.complex128)
    
    def _calculate_observables(self):
        """Calculate physical observables with proper type conversions."""
        # Use finest lattice for observables
        finest_fields = self.fields[len(self.params.lattice_sizes) - 1]
        
        # Energy density (convert to real)
        total_energy = 0.0
        for field in finest_fields.values():
            field_energy = field.energy()  # This already returns float
            total_energy += field_energy
        
        volume = float(np.prod(self.params.lattice_sizes[-1]))
        energy_density = total_energy / volume
        
        # Temperature (Stefan-Boltzmann relation) - ensure real
        g_eff = 37.0  # Effective degrees of freedom
        if energy_density > 0:
            temperature_gev = (30.0 * energy_density / (np.pi**2 * g_eff))**(1.0/4.0)
            temperature_mev = temperature_gev * 1000.0
        else:
            temperature_mev = 0.0
        
        # Ensure temperature is real and positive
        temperature_mev = max(0.0, float(np.real(temperature_mev)))
        
        # Pressure (ideal gas) - ensure real
        pressure = energy_density / 3.0
        pressure = float(np.real(pressure))
        
        # Particle multiplicity (rough estimate) - ensure real
        multiplicity = energy_density * 1000.0  # Simplified scaling
        multiplicity = max(0.0, float(np.real(multiplicity)))
        
        # Store observables (all as real floats)
        self.observables['time'].append(float(self.current_time))
        self.observables['energy_density'].append(float(energy_density))
        self.observables['temperature'].append(float(temperature_mev))
        self.observables['pressure'].append(float(pressure))
        self.observables['multiplicity'].append(float(multiplicity))
    
    def _print_progress(self):
        """Print simulation progress."""
        if self.observables['temperature']:
            temp = self.observables['temperature'][-1]
            energy = self.observables['energy_density'][-1]
            
            print(f"Step {self.iteration:4d}/{self.params.max_iterations}, "
                  f"t = {self.current_time:.3f} fm/c, "
                  f"T = {temp:.1f} MeV, "
                  f"ε = {energy:.2e}")
            
            # Phase information
            if temp > 170:
                print("   🔥 QGP phase detected!")
            elif temp > 120:
                print("   🌡️  Mixed phase region")
            else:
                print("   ❄️  Hadronic phase")
    
    def get_results(self) -> Dict:
        """Get simulation results."""
        return {
            'parameters': self.params.to_dict(),
            'observables': self.observables.copy(),
            'nuclear_info': {
                'nucleus_A': str(self.nucleus_A),
                'nucleus_B': str(self.nucleus_B)
            },
            'final_state': {
                'time': float(self.current_time),
                'iterations': int(self.iteration)
            }
        }
    
    def stop(self):
        """Stop the running simulation."""
        self.is_running = False
        print("🛑 Simulation stop requested")
    
    def save_results(self, filename: str):
        """Save results to file."""
        import json
        results = self.get_results()
        
        # Convert any numpy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, complex):
                return float(np.real(obj))  # Take real part
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Clean up the results
        clean_results = convert_numpy_types(results)
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"💾 Results saved to {filename}")

# Additional helper functions for standalone usage
def create_test_simulator():
    """Create a test simulator for debugging."""
    try:
        from ..core.parameters import SimulationParameters
    except ImportError:
        # Fallback import
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.parameters import SimulationParameters
    
    params = SimulationParameters()
    params.nucleus_A = "Au197"
    params.nucleus_B = "Au197"
    params.collision_energy_gev = 200.0
    params.max_iterations = 50  # Short test
    params.lattice_sizes = [(16, 16, 16)]  # Small lattice
    
    return QuantumLatticeSimulator(params)

# Test function
def test_simulator():
    """Test the simulator for type errors."""
    print("🧪 Testing simulator for type consistency...")
    
    try:
        simulator = create_test_simulator()
        print("✅ Simulator creation: OK")
        
        # Test one evolution step
        simulator._evolve_fields()
        print("✅ Field evolution: OK")
        
        # Test observable calculation
        simulator._calculate_observables()  
        print("✅ Observable calculation: OK")
        
        # Test short simulation
        simulator.params.max_iterations = 5
        results = simulator.run_simulation()
        print("✅ Short simulation: OK")
        
        print(f"✅ All tests passed! Final temperature: {results['observables']['temperature'][-1]:.1f} MeV")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simulator()
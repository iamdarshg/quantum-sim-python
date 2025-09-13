"""
Nuclear Force Integration Module
Integrates advanced QCD-based nuclear forces with the existing quantum lattice simulator.

This module provides a drop-in replacement for the old Yukawa potential system
while maintaining full compatibility with all existing features.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import time

# Import the advanced nuclear force models
try:
    from advanced_nuclear_forces import (
        AdvancedNuclearForceSolver,
        create_nuclear_force_solver,
        NucleonState
    )
    ADVANCED_FORCES_AVAILABLE = True
    print("✅ Advanced nuclear forces loaded successfully")
except ImportError as e:
    ADVANCED_FORCES_AVAILABLE = False
    print(f"⚠️ Advanced nuclear forces not available: {e}")
    print("   Falling back to improved Yukawa potential")

class NuclearForceManager:
    """
    Manager class for nuclear force calculations.
    
    Provides unified interface that can use either:
    1. Advanced QCD-based models (when available)
    2. Improved Yukawa potential (fallback)
    3. Hybrid approach (mixing different models)
    """
    
    def __init__(self, force_model: str = "auto"):
        """
        Initialize nuclear force manager.
        
        Args:
            force_model: Force model to use
                - "auto": Automatically select best available
                - "ChiralEFT_N3LO": Chiral EFT at N3LO
                - "Argonne_v18": High-precision phenomenological  
                - "QCD_SumRules": QCD sum rules based
                - "Lattice_QCD": Lattice QCD inspired
                - "improved_yukawa": Enhanced Yukawa potential
                - "hybrid": Mix multiple models
        """
        
        self.force_model = force_model
        self.advanced_solver = None
        self.performance_stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'total_time': 0.0
        }
        
        # Force calculation cache for performance
        self.force_cache = {}
        self.cache_lifetime = 1000  # Number of calls before cache expires
        
        # Initialize the appropriate force calculator
        self._initialize_force_calculator()
        
        print(f"🚀 Nuclear Force Manager initialized")
        print(f"   Model: {self.get_current_model_name()}")
        print(f"   Advanced QCD forces: {'✅ Available' if ADVANCED_FORCES_AVAILABLE else '❌ Not available'}")
    
    def _initialize_force_calculator(self):
        """Initialize the appropriate force calculation system."""
        
        if self.force_model == "auto":
            # Automatically select best available model
            if ADVANCED_FORCES_AVAILABLE:
                self.force_model = "ChiralEFT_N3LO"
                self._setup_advanced_solver()
            else:
                self.force_model = "improved_yukawa"
                self._setup_improved_yukawa()
                
        elif self.force_model.startswith("ChiralEFT") or self.force_model in ["Argonne_v18", "QCD_SumRules", "Lattice_QCD"]:
            if ADVANCED_FORCES_AVAILABLE:
                self._setup_advanced_solver()
            else:
                print(f"⚠️ Advanced model '{self.force_model}' not available, falling back to improved Yukawa")
                self.force_model = "improved_yukawa"
                self._setup_improved_yukawa()
                
        elif self.force_model == "improved_yukawa":
            self._setup_improved_yukawa()
            
        elif self.force_model == "hybrid":
            self._setup_hybrid_approach()
            
        else:
            raise ValueError(f"Unknown force model: {self.force_model}")
    
    def _setup_advanced_solver(self):
        """Setup advanced QCD-based force solver."""
        
        try:
            self.advanced_solver = create_nuclear_force_solver(self.force_model)
            print(f"✅ Advanced nuclear force solver ready: {self.force_model}")
        except Exception as e:
            print(f"❌ Failed to setup advanced solver: {e}")
            print("   Falling back to improved Yukawa")
            self.force_model = "improved_yukawa"
            self._setup_improved_yukawa()
    
    def _setup_improved_yukawa(self):
        """Setup improved Yukawa potential with modern corrections."""
        
        # Enhanced Yukawa parameters based on modern fits
        self.yukawa_params = {
            'g_strong': 14.0,           # Strong coupling
            'm_pion': 0.13957,          # Pion mass (GeV)
            'm_eta': 0.54785,           # Eta meson mass
            'm_rho': 0.77526,           # Rho meson mass
            'm_omega': 0.78265,         # Omega meson mass
            'lambda_cutoff': 1.0,       # Form factor cutoff (GeV)
            'alpha_em': 1.0/137.036,    # Fine structure constant
            'hbar_c': 0.197327          # ħc in GeV·fm
        }
        
        print("✅ Improved Yukawa potential initialized with modern parameters")
    
    def _setup_hybrid_approach(self):
        """Setup hybrid approach mixing different models."""
        
        self.hybrid_weights = {
            'short_range': 'contact',      # r < 0.5 fm
            'medium_range': 'pion_exchange', # 0.5 < r < 2.0 fm  
            'long_range': 'yukawa'         # r > 2.0 fm
        }
        
        if ADVANCED_FORCES_AVAILABLE:
            self._setup_advanced_solver()
        
        self._setup_improved_yukawa()
        
        print("✅ Hybrid nuclear force approach initialized")
    
    def calculate_nuclear_forces(self, particle: Dict, other_particles: List[Dict]) -> np.ndarray:
        """
        Calculate nuclear forces on a particle from all other particles.
        
        This is the main interface that replaces the old Yukawa calculation
        in the simulation engine.
        
        Args:
            particle: Particle to calculate forces on
            other_particles: List of all other particles
            
        Returns:
            Total force vector (3D numpy array)
        """
        
        total_force = np.zeros(3)
        
        for other in other_particles:
            if other is particle:
                continue
            
            # Calculate pairwise force
            pairwise_force = self._calculate_pairwise_force(particle, other)
            total_force += pairwise_force
            
        # Update performance statistics
        self.performance_stats['total_calls'] += 1
        
        return total_force
    
    def _calculate_pairwise_force(self, particle1: Dict, particle2: Dict) -> np.ndarray:
        """Calculate force between two specific particles."""
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(particle1, particle2)
        if cache_key in self.force_cache:
            self.performance_stats['cache_hits'] += 1
            return self.force_cache[cache_key]
        
        # Calculate position difference
        r_vec = np.array(particle1['position']) - np.array(particle2['position'])
        r = np.linalg.norm(r_vec)
        
        if r < 0.1:  # Avoid singularity
            return np.zeros(3)
        
        r_hat = r_vec / r
        
        # Choose calculation method based on active model
        if self.advanced_solver is not None:
            force_magnitude = self._calculate_with_advanced_model(particle1, particle2, r)
        elif self.force_model == "improved_yukawa":
            force_magnitude = self._calculate_with_improved_yukawa(particle1, particle2, r)
        elif self.force_model == "hybrid":
            force_magnitude = self._calculate_with_hybrid_model(particle1, particle2, r)
        else:
            # Default fallback
            force_magnitude = self._calculate_with_improved_yukawa(particle1, particle2, r)
        
        # Apply force direction
        force_vector = force_magnitude * r_hat
        
        # Cache result
        self.force_cache[cache_key] = force_vector
        
        # Clean cache if too large
        if len(self.force_cache) > self.cache_lifetime:
            self._clean_cache()
        
        # Update timing
        self.performance_stats['total_time'] += time.time() - start_time
        
        return force_vector
    
    def _calculate_with_advanced_model(self, p1: Dict, p2: Dict, r: float) -> float:
        """Calculate force using advanced QCD-based models."""
        
        try:
            force_vec = self.advanced_solver.calculate_force_between_nucleons(p1, p2)
            return np.linalg.norm(force_vec)
        except Exception as e:
            print(f"⚠️ Advanced model calculation failed: {e}")
            print("   Falling back to improved Yukawa")
            return self._calculate_with_improved_yukawa(p1, p2, r)
    
    def _calculate_with_improved_yukawa(self, p1: Dict, p2: Dict, r: float) -> float:
        """Calculate force using improved Yukawa potential with modern corrections."""
        
        # Only calculate for nucleons
        if not (p1.get('type') in ['proton', 'neutron'] and 
                p2.get('type') in ['proton', 'neutron']):
            return 0.0
        
        params = self.yukawa_params
        r_fm = r / params['hbar_c']  # Convert to fm
        
        # Multi-meson exchange (beyond single pion)
        force_total = 0.0
        
        # 1. Pion exchange (attractive, long range)
        x_pi = params['m_pion'] * r / params['hbar_c']
        if x_pi > 0.1:  # Avoid numerical issues
            yukawa_pi = (params['g_strong']**2 * params['m_pion']**2) / (4 * np.pi)
            force_pi = -yukawa_pi * (1 + 1/x_pi) * np.exp(-x_pi) / r_fm**2
            force_total += force_pi
        
        # 2. Eta meson exchange (repulsive at medium range)
        x_eta = params['m_eta'] * r / params['hbar_c']
        if x_eta > 0.1:
            yukawa_eta = (params['g_strong']**2 * params['m_eta']**2) / (8 * np.pi)
            force_eta = yukawa_eta * (1 + 1/x_eta) * np.exp(-x_eta) / r_fm**2
            force_total += force_eta
        
        # 3. Vector meson exchange (rho, omega)
        for m_vec in [params['m_rho'], params['m_omega']]:
            x_vec = m_vec * r / params['hbar_c']
            if x_vec > 0.1:
                yukawa_vec = (params['g_strong']**2 * m_vec**2) / (12 * np.pi)
                force_vec = -yukawa_vec * (1 + 1/x_vec) * np.exp(-x_vec) / r_fm**2
                force_total += force_vec * 0.5  # Reduced coupling
        
        # 4. Form factor to regularize short-distance behavior
        form_factor = 1.0 / (1.0 + (r / (params['hbar_c'] / params['lambda_cutoff']))**4)
        force_total *= form_factor
        
        # 5. Electromagnetic force (if both charged)
        if p1.get('charge', 0) != 0 and p2.get('charge', 0) != 0:
            force_em = params['alpha_em'] * params['hbar_c'] * p1['charge'] * p2['charge'] / r**2
            force_total += force_em
        
        # 6. Short-range repulsion (phenomenological)
        if r_fm < 0.5:
            repulsion_strength = 1000.0  # MeV
            force_repulsion = repulsion_strength * np.exp(-4.0 * r_fm) / r_fm**2
            force_total += force_repulsion
        
        return force_total
    
    def _calculate_with_hybrid_model(self, p1: Dict, p2: Dict, r: float) -> float:
        """Calculate force using hybrid approach."""
        
        r_fm = r / 0.197327  # Convert to fm
        
        if r_fm < 0.5:
            # Short range: use contact interaction or advanced model if available
            if self.advanced_solver is not None:
                try:
                    force_vec = self.advanced_solver.calculate_force_between_nucleons(p1, p2)
                    return np.linalg.norm(force_vec)
                except:
                    pass
            # Fallback: strong repulsion
            return 1000.0 * np.exp(-4.0 * r_fm) / r_fm**2
            
        elif r_fm < 2.0:
            # Medium range: pion exchange dominates
            return self._pion_exchange_force(p1, p2, r)
            
        else:
            # Long range: use improved Yukawa
            return self._calculate_with_improved_yukawa(p1, p2, r) * 0.5
    
    def _pion_exchange_force(self, p1: Dict, p2: Dict, r: float) -> float:
        """Pure pion exchange force."""
        
        params = self.yukawa_params
        x = params['m_pion'] * r / params['hbar_c']
        
        if x < 0.1:
            return 0.0
        
        # Isospin factor
        if p1.get('type') == p2.get('type'):
            tau_factor = 1.0  # pp or nn
        else:
            tau_factor = -3.0  # pn
        
        g_piNN = 13.0  # Pion-nucleon coupling
        force_magnitude = (g_piNN**2 * params['m_pion']**2) / (4 * np.pi)
        force_magnitude *= tau_factor * (1 + 1/x) * np.exp(-x) / (r / params['hbar_c'])**2
        
        return -force_magnitude  # Attractive for pn, repulsive for pp/nn
    
    def _get_cache_key(self, p1: Dict, p2: Dict) -> tuple:
        """Generate cache key for force calculation."""
        
        # Use particle IDs and positions (rounded for stability)
        id1 = p1.get('id', 0)
        id2 = p2.get('id', 0)
        pos1 = tuple(np.round(p1['position'], decimals=3))
        pos2 = tuple(np.round(p2['position'], decimals=3))
        
        # Ensure consistent ordering
        if id1 <= id2:
            return (id1, id2, pos1, pos2, self.force_model)
        else:
            return (id2, id1, pos2, pos1, self.force_model)
    
    def _clean_cache(self):
        """Clean force calculation cache."""
        
        # Keep only most recent 50% of entries (simple LRU approximation)
        if len(self.force_cache) > self.cache_lifetime // 2:
            # Clear cache entirely for simplicity
            self.force_cache.clear()
    
    def switch_model(self, new_model: str):
        """Switch to a different nuclear force model."""
        
        print(f"🔄 Switching nuclear force model: {self.force_model} → {new_model}")
        
        old_model = self.force_model
        self.force_model = new_model
        
        try:
            self._initialize_force_calculator()
            self.force_cache.clear()  # Clear cache when switching models
            print(f"✅ Successfully switched to {new_model}")
        except Exception as e:
            print(f"❌ Failed to switch to {new_model}: {e}")
            print(f"   Reverting to {old_model}")
            self.force_model = old_model
            self._initialize_force_calculator()
    
    def get_current_model_name(self) -> str:
        """Get name of currently active model."""
        
        if self.advanced_solver is not None:
            return self.advanced_solver.get_active_model_info()['name']
        else:
            return f"Improved Yukawa Potential ({self.force_model})"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        
        stats = self.performance_stats.copy()
        
        if stats['total_calls'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_calls']
            stats['avg_time_per_call'] = stats['total_time'] / stats['total_calls']
        else:
            stats['cache_hit_rate'] = 0.0
            stats['avg_time_per_call'] = 0.0
            
        stats['cache_size'] = len(self.force_cache)
        stats['model_name'] = self.get_current_model_name()
        
        if self.advanced_solver is not None:
            advanced_stats = self.advanced_solver.get_active_model_info()
            stats.update(advanced_stats)
        
        return stats
    
    def benchmark_models(self) -> Dict[str, Any]:
        """Benchmark all available nuclear force models."""
        
        if not ADVANCED_FORCES_AVAILABLE:
            return {"error": "Advanced models not available for benchmarking"}
        
        print("🔬 Benchmarking nuclear force models...")
        
        # Create test solver for benchmarking
        test_solver = create_nuclear_force_solver()
        benchmark_results = test_solver.benchmark_models(test_cases=50)
        
        # Add improved Yukawa benchmark
        original_model = self.force_model
        
        try:
            self.switch_model("improved_yukawa")
            
            # Generate test cases
            test_particles = []
            for _ in range(50):
                p1 = {
                    'position': np.random.uniform(-5, 5, 3),
                    'momentum': np.random.uniform(-0.5, 0.5, 3),
                    'type': np.random.choice(['proton', 'neutron']),
                    'charge': 1 if np.random.random() > 0.5 else 0,
                    'id': 1
                }
                p2 = {
                    'position': np.random.uniform(-5, 5, 3),
                    'momentum': np.random.uniform(-0.5, 0.5, 3),
                    'type': np.random.choice(['proton', 'neutron']),
                    'charge': 1 if np.random.random() > 0.5 else 0,
                    'id': 2
                }
                test_particles.append((p1, p2))
            
            start_time = time.time()
            forces = []
            
            for p1, p2 in test_particles:
                force = self.calculate_nuclear_forces(p1, [p2])
                forces.append(np.linalg.norm(force))
            
            total_time = time.time() - start_time
            
            benchmark_results["Improved_Yukawa"] = {
                'total_time': total_time,
                'avg_time_per_call': total_time / 50,
                'avg_force_magnitude': np.mean(forces),
                'force_std': np.std(forces),
                'success': True
            }
            
        except Exception as e:
            benchmark_results["Improved_Yukawa"] = {
                'error': str(e),
                'success': False
            }
        finally:
            # Restore original model
            self.switch_model(original_model)
        
        return benchmark_results
    
    def get_available_models(self) -> List[str]:
        """Get list of all available nuclear force models."""
        
        models = ["improved_yukawa", "hybrid"]
        
        if ADVANCED_FORCES_AVAILABLE:
            if self.advanced_solver is None:
                # Create temporary solver to get model list
                temp_solver = create_nuclear_force_solver()
                models.extend(temp_solver.get_available_models())
            else:
                models.extend(self.advanced_solver.get_available_models())
        
        return models

# Integration function for the simulation engine
def create_nuclear_force_manager(config: Dict[str, Any]) -> NuclearForceManager:
    """
    Create nuclear force manager from simulation configuration.
    
    Args:
        config: Simulation configuration dictionary
        
    Returns:
        Configured NuclearForceManager
    """
    
    # Extract force model from config
    force_model = config.get('nuclear_force_model', 'auto')
    
    # Create and return manager
    return NuclearForceManager(force_model)

# Drop-in replacement function for old Yukawa force calculation
def calculate_nuclear_forces_enhanced(particle: Dict, all_particles: List[Dict], 
                                    force_manager: Optional[NuclearForceManager] = None) -> np.ndarray:
    """
    Drop-in replacement for the old nuclear force calculation.
    
    This function can directly replace the old _compute_nuclear_forces method
    in the simulation engine.
    
    Args:
        particle: Particle to calculate forces on
        all_particles: List of all particles
        force_manager: Nuclear force manager (created if None)
        
    Returns:
        Total nuclear force vector
    """
    
    if force_manager is None:
        # Create default manager
        force_manager = NuclearForceManager("auto")
    
    return force_manager.calculate_nuclear_forces(particle, all_particles)

# Export main classes and functions
__all__ = [
    'NuclearForceManager',
    'create_nuclear_force_manager', 
    'calculate_nuclear_forces_enhanced',
    'ADVANCED_FORCES_AVAILABLE'
]
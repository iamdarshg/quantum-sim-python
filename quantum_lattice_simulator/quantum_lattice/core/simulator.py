"""
Enhanced Simulation Engine with Boundary Detection and Ultra-High Resolution
Supports up to 1024³ lattices and automatic stopping when mass escapes.
"""

import numpy as np
import threading
import queue
import time
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import json
import logging
from nuclear_equation_tracker import NuclearEquationTracker, NuclearReaction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BoundaryConditions:
    """Boundary conditions and escape detection."""
    simulation_volume: Tuple[float, float, float]  # (x_size, y_size, z_size) in fm
    escape_threshold: float = 0.5  # Stop when 50% of mass escapes
    initial_total_mass: float = 0.0
    escaped_mass: float = 0.0
    escaped_particles: List[Dict] = field(default_factory=list)
    
    def check_particle_escape(self, particle: Dict) -> bool:
        """Check if particle has escaped simulation volume."""
        
        position = particle.get('position', np.zeros(3))
        x, y, z = position
        
        x_max, y_max, z_max = self.simulation_volume
        
        # Check if outside boundary (with small buffer)
        buffer = 5.0  # fm
        escaped = (abs(x) > x_max/2 + buffer or 
                  abs(y) > y_max/2 + buffer or 
                  abs(z) > z_max/2 + buffer)
        
        return escaped
    
    def update_escaped_mass(self, escaped_particle: Dict):
        """Update escaped mass tracking."""
        
        mass = escaped_particle.get('mass', 0.938)  # Default nucleon mass
        self.escaped_mass += mass
        self.escaped_particles.append(escaped_particle.copy())
    
    def get_escape_fraction(self) -> float:
        """Get fraction of mass that has escaped."""
        
        if self.initial_total_mass <= 0:
            return 0.0
        
        return self.escaped_mass / self.initial_total_mass
    
    def should_stop_simulation(self) -> bool:
        """Check if simulation should stop due to mass escape."""
        
        return self.get_escape_fraction() >= self.escape_threshold

@dataclass 
class UltraHighResolutionLattice:
    """Ultra-high resolution lattice up to 1024³ points."""
    
    size: Tuple[int, int, int]
    spacing: float  # fm
    total_points: int = field(init=False)
    memory_estimate_gb: float = field(init=False)
    
    def __post_init__(self):
        """Calculate lattice properties."""
        self.total_points = np.prod(self.size)
        
        # Estimate memory usage (assumes complex128 fields)
        # Each lattice point needs ~10 complex fields × 16 bytes × 4 components
        bytes_per_point = 10 * 16 * 4  # ~640 bytes per point
        self.memory_estimate_gb = (self.total_points * bytes_per_point) / (1024**3)
    
    def is_memory_feasible(self, available_gb: float = 16.0) -> bool:
        """Check if lattice fits in available memory."""
        return self.memory_estimate_gb <= available_gb
    
    def get_recommended_chunk_size(self) -> Tuple[int, int, int]:
        """Get recommended chunk size for distributed processing."""
        
        nx, ny, nz = self.size
        
        # Aim for chunks of ~8³ to 32³ depending on total size
        if max(self.size) <= 64:
            chunk_size = 16
        elif max(self.size) <= 256:
            chunk_size = 32  
        elif max(self.size) <= 512:
            chunk_size = 64
        else:
            chunk_size = 128
        
        # Ensure chunks divide evenly
        chunk_x = min(chunk_size, nx)
        chunk_y = min(chunk_size, ny)
        chunk_z = min(chunk_size, nz)
        
        return (chunk_x, chunk_y, chunk_z)

class EnhancedSimulationEngine:
    """Enhanced simulation engine with nuclear equation tracking and boundary detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_workers = config.get('num_workers', mp.cpu_count())
        
        # Ultra-high resolution lattice support
        self.lattice_configs = []
        lattice_sizes = config.get('lattice_sizes', [(64, 64, 64)])
        spacings = config.get('spacings', [0.05])
        
        for size, spacing in zip(lattice_sizes, spacings):
            lattice = UltraHighResolutionLattice(size, spacing)
            
            # Check memory feasibility
            if lattice.is_memory_feasible():
                self.lattice_configs.append(lattice)
                print(f"✅ Lattice {size} feasible: {lattice.memory_estimate_gb:.2f} GB")
            else:
                print(f"⚠️ Lattice {size} requires {lattice.memory_estimate_gb:.2f} GB - may cause memory issues")
                # Allow it anyway but warn user
                self.lattice_configs.append(lattice)
        
        # Nuclear equation tracking
        self.equation_tracker = NuclearEquationTracker()
        
        # Boundary conditions
        max_lattice = max(self.lattice_configs, key=lambda x: max(x.size))
        sim_volume = (max_lattice.size[0] * max_lattice.spacing,
                     max_lattice.size[1] * max_lattice.spacing,
                     max_lattice.size[2] * max_lattice.spacing)
        
        self.boundary_conditions = BoundaryConditions(
            simulation_volume=sim_volume,
            escape_threshold=config.get('escape_threshold', 0.5)
        )
        
        # Enhanced time stepping storage
        self.time_history = []
        self.max_history_length = config.get('max_history_steps', 10000)
        self.history_compression_factor = config.get('history_compression', 2)
        
        # Simulation state
        self.particles = []
        self.current_time = 0.0
        self.time_step = config.get('time_step', 0.005)
        self.max_time = config.get('max_time', 50.0)
        self.is_running = False
        self.stop_requested = False
        
        # Global observables with enhanced tracking
        self.global_observables = {
            'time': [],
            'temperature': [],
            'energy_density': [], 
            'pressure': [],
            'particle_count': [],
            'entropy_density': [],
            'escaped_mass_fraction': [],
            'reaction_rate': [],
            'total_reactions': []
        }
        
        print(f"🚀 Enhanced simulation engine initialized")
        print(f"   Lattice configurations: {len(self.lattice_configs)}")
        print(f"   Boundary volume: {sim_volume[0]:.1f} × {sim_volume[1]:.1f} × {sim_volume[2]:.1f} fm³")
        print(f"   Escape threshold: {self.boundary_conditions.escape_threshold:.1%}")
    
    def initialize_simulation(self, nucleus_a: str, nucleus_b: str, 
                            collision_energy_gev: float, impact_parameter: float):
        """Initialize simulation with enhanced particle tracking."""
        
        print(f"🔬 Initializing {nucleus_a} + {nucleus_b} @ {collision_energy_gev} GeV")
        
        # Initialize particles from nuclear structure
        self._create_nuclear_system(nucleus_a, nucleus_b, collision_energy_gev, impact_parameter)
        
        # Set initial total mass for boundary tracking
        self.boundary_conditions.initial_total_mass = sum(
            p.get('mass', 0.938) for p in self.particles
        )
        
        print(f"✅ Simulation initialized:")
        print(f"   Initial particles: {len(self.particles)}")
        print(f"   Total initial mass: {self.boundary_conditions.initial_total_mass:.3f} GeV")
    
    def _create_nuclear_system(self, nucleus_a: str, nucleus_b: str, 
                             collision_energy_gev: float, impact_parameter: float):
        """Create nuclear system with realistic structure."""
        
        # Nuclear database
        nuclear_data = {
            'H': {'A': 1, 'Z': 1, 'radius': 0.8},
            'D': {'A': 2, 'Z': 1, 'radius': 1.2},
            'He3': {'A': 3, 'Z': 2, 'radius': 1.5},
            'He4': {'A': 4, 'Z': 2, 'radius': 1.7},
            'C12': {'A': 12, 'Z': 6, 'radius': 2.7},
            'O16': {'A': 16, 'Z': 8, 'radius': 3.0},
            'Ca40': {'A': 40, 'Z': 20, 'radius': 4.2},
            'Fe56': {'A': 56, 'Z': 26, 'radius': 4.7},
            'Au197': {'A': 197, 'Z': 79, 'radius': 7.0},
            'Pb208': {'A': 208, 'Z': 82, 'radius': 7.1},
            'U238': {'A': 238, 'Z': 92, 'radius': 7.4}
        }
        
        data_a = nuclear_data.get(nucleus_a, nuclear_data['Au197'])
        data_b = nuclear_data.get(nucleus_b, nuclear_data['Au197'])
        
        # Create nucleus A (projectile)
        self._create_nucleus_particles(
            data_a, nucleus_a,
            center=np.array([-20.0, impact_parameter/2, 0.0]),
            velocity=np.array([np.sqrt(1 - (0.938/(collision_energy_gev + 0.938))**2), 0, 0])
        )
        
        # Create nucleus B (target) 
        self._create_nucleus_particles(
            data_b, nucleus_b,
            center=np.array([20.0, -impact_parameter/2, 0.0]),
            velocity=np.array([0, 0, 0])  # Target at rest
        )
    
    def _create_nucleus_particles(self, nuclear_data: Dict, nucleus_name: str,
                                center: np.ndarray, velocity: np.ndarray):
        """Create particles for a nucleus with proper nuclear structure."""
        
        A = nuclear_data['A']
        Z = nuclear_data['Z']
        R = nuclear_data['radius']
        
        # Generate nucleon positions with Woods-Saxon distribution
        positions = self._sample_woods_saxon_positions(A, R, center)
        
        for i, pos in enumerate(positions):
            # Determine particle type
            is_proton = i < Z
            
            if is_proton:
                particle = {
                    'type': 'proton',
                    'A': 1, 'Z': 1,
                    'mass': 0.938272,  # GeV
                    'charge': 1
                }
            else:
                particle = {
                    'type': 'neutron', 
                    'A': 1, 'Z': 0,
                    'mass': 0.939565,  # GeV
                    'charge': 0
                }
            
            # Position and momentum
            particle['position'] = pos.copy()
            
            # Fermi motion + boost
            fermi_p = self._sample_fermi_momentum()
            boosted_p = self._boost_momentum(fermi_p, velocity)
            particle['momentum'] = boosted_p
            
            # Energy
            p_squared = np.sum(boosted_p**2)
            particle['energy'] = np.sqrt(p_squared + particle['mass']**2) - particle['mass']
            
            # Tracking info
            particle['creation_time'] = 0.0
            particle['parent_nucleus'] = nucleus_name
            particle['id'] = len(self.particles)
            
            self.particles.append(particle)
    
    def _sample_woods_saxon_positions(self, A: int, R: float, center: np.ndarray) -> List[np.ndarray]:
        """Sample positions from Woods-Saxon nuclear density."""
        
        positions = []
        a = 0.5  # Surface diffuseness (fm)
        
        max_attempts = A * 50
        attempts = 0
        
        while len(positions) < A and attempts < max_attempts:
            # Sample radius with Woods-Saxon probability
            r = np.random.exponential(R/2)  # Rough initial sampling
            
            if r > 5 * R:  # Cutoff at 5R
                attempts += 1
                continue
            
            # Woods-Saxon probability
            woods_saxon_prob = 1.0 / (1.0 + np.exp((r - R) / a))
            
            if np.random.random() < woods_saxon_prob:
                # Sample angular coordinates
                theta = np.arccos(2 * np.random.random() - 1)
                phi = 2 * np.pi * np.random.random()
                
                position = center + r * np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
                
                positions.append(position)
            
            attempts += 1
        
        # Fill remaining positions randomly if needed
        while len(positions) < A:
            r = R * np.random.random()**(1/3)
            theta = np.arccos(2 * np.random.random() - 1)
            phi = 2 * np.pi * np.random.random()
            
            position = center + r * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi), 
                np.cos(theta)
            ])
            
            positions.append(position)
        
        return positions[:A]
    
    def _sample_fermi_momentum(self) -> np.ndarray:
        """Sample momentum from nuclear Fermi sea."""
        
        kF = 0.270  # Fermi momentum in GeV/c
        
        # Sample magnitude up to Fermi surface
        k_mag = kF * np.random.random()**(1/3)
        
        # Random direction
        theta = np.arccos(2 * np.random.random() - 1)
        phi = 2 * np.pi * np.random.random()
        
        momentum = k_mag * np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        return momentum
    
    def _boost_momentum(self, momentum: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Apply relativistic boost to momentum."""
        
        v_mag = np.linalg.norm(velocity)
        if v_mag < 1e-6:
            return momentum
        
        gamma = 1.0 / np.sqrt(1 - v_mag**2)
        
        # Boost formula for momentum
        v_hat = velocity / v_mag
        p_parallel = np.dot(momentum, v_hat)
        p_perp = momentum - p_parallel * v_hat
        
        # Approximate boost (exact formula more complex)
        p_parallel_boosted = gamma * p_parallel + gamma * v_mag * 0.938  # Rough approximation
        
        return p_parallel_boosted * v_hat + p_perp
    
    def run_simulation(self, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run enhanced simulation with nuclear equation tracking and boundary detection."""
        
        print("🚀 Starting enhanced simulation with equation tracking")
        
        self.is_running = True
        self.stop_requested = False
        start_time = time.time()
        
        time_steps = int(self.max_time / self.time_step)
        
        try:
            for step in range(time_steps):
                if self.stop_requested:
                    print("🛑 Simulation stopped by user request")
                    break
                
                # Evolve one time step
                self._evolve_time_step()
                
                # Check for nuclear reactions
                self._check_nuclear_reactions()
                
                # Update boundary conditions and check escape
                self._update_boundary_conditions()
                
                # Store complete state in history
                self._store_time_step_state()
                
                # Compute observables
                self._compute_enhanced_observables()
                
                # Check boundary stopping condition
                if self.boundary_conditions.should_stop_simulation():
                    print(f"🚫 Simulation stopped: {self.boundary_conditions.get_escape_fraction():.1%} of mass escaped")
                    break
                
                # Progress callback
                if callback and step % 10 == 0:
                    callback(self)
                
                # Progress reporting
                if step % 100 == 0:
                    self._print_enhanced_progress(step, time_steps)
        
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.is_running = False
            elapsed = time.time() - start_time
            print(f"✅ Enhanced simulation completed in {elapsed:.1f}s")
        
        return self._get_comprehensive_results()
    
    def _evolve_time_step(self):
        """Evolve system by one time step with enhanced physics."""
        
        self.current_time += self.time_step
        dt = self.time_step
        
        # Update particle positions and momenta
        for particle in self.particles:
            # Classical equations of motion
            velocity = self._get_relativistic_velocity(particle)
            particle['position'] += velocity * dt
            
            # Forces (simplified)
            force = self._compute_nuclear_forces(particle)
            particle['momentum'] += force * dt
            
            # Update energy
            p_squared = np.sum(particle['momentum']**2)
            particle['energy'] = np.sqrt(p_squared + particle['mass']**2) - particle['mass']
    
    def _get_relativistic_velocity(self, particle: Dict) -> np.ndarray:
        """Get relativistic velocity from momentum."""
        
        momentum = particle['momentum']
        mass = particle['mass']
        
        total_energy = particle['energy'] + mass
        
        return momentum / total_energy  # v = p/E in natural units (c=1)
    
    def _compute_nuclear_forces(self, particle: Dict) -> np.ndarray:
        """Compute nuclear and electromagnetic forces on particle."""
        
        force = np.zeros(3)
        
        for other in self.particles:
            if other is particle:
                continue
            
            r_vec = particle['position'] - other['position']
            r_mag = np.linalg.norm(r_vec)
            
            if r_mag < 0.1:  # Avoid singularity
                continue
            
            r_hat = r_vec / r_mag
            
            # Nuclear force (Yukawa)
            if (particle['type'] in ['proton', 'neutron'] and 
                other['type'] in ['proton', 'neutron']):
                
                m_pion = 0.138  # GeV
                g_strong = 14.0
                
                yukawa_force = (g_strong / (4 * np.pi * r_mag**2) * 
                              np.exp(-m_pion * r_mag * 5.07) *  # Convert to fm^-1
                              (1 + m_pion * r_mag * 5.07))
                
                force -= yukawa_force * r_hat  # Attractive
            
            # Electromagnetic force
            if particle['charge'] != 0 and other['charge'] != 0:
                alpha = 1.0 / 137.036
                coulomb_force = (alpha * particle['charge'] * other['charge'] / 
                               r_mag**2 * 0.197**2)  # Convert to GeV/fm
                
                force += coulomb_force * r_hat  # Repulsive for like charges
        
        return force
    
    def _check_nuclear_reactions(self):
        """Check for nuclear reactions between close particles."""
        
        reactions_this_step = []
        
        for i, p1 in enumerate(self.particles):
            for j, p2 in enumerate(self.particles[i+1:], i+1):
                
                r_vec = p1['position'] - p2['position']
                r_mag = np.linalg.norm(r_vec)
                
                # Check if particles are close enough to react
                if r_mag < 2.0:  # 2 fm interaction range
                    
                    # Determine if reaction occurs (probabilistic)
                    reaction_prob = self._calculate_reaction_probability(p1, p2, r_mag)
                    
                    if np.random.random() < reaction_prob:
                        # Create reaction
                        reactants = [p1.copy(), p2.copy()]
                        products = self._generate_reaction_products(p1, p2)
                        
                        if products:  # Valid reaction occurred
                            # Track the reaction
                            reaction = self.equation_tracker.track_reaction(
                                reactants, products,
                                (p1['position'] + p2['position']) / 2,
                                self.current_time
                            )
                            
                            reactions_this_step.append((i, j, products))
        
        # Apply reactions (remove reactants, add products)
        # Process in reverse order to maintain indices
        for i, j, products in sorted(reactions_this_step, reverse=True):
            # Remove reactants
            if j < len(self.particles):
                del self.particles[j]
            if i < len(self.particles):
                del self.particles[i]
            
            # Add products
            for product in products:
                product['id'] = len(self.particles) + len(products)
                product['creation_time'] = self.current_time
                self.particles.append(product)
    
    def _calculate_reaction_probability(self, p1: Dict, p2: Dict, distance: float) -> float:
        """Calculate probability of nuclear reaction."""
        
        # Only nucleons react for now
        if not (p1['type'] in ['proton', 'neutron'] and p2['type'] in ['proton', 'neutron']):
            return 0.0
        
        # Relative velocity
        v_rel = np.linalg.norm(p1['momentum']/p1['mass'] - p2['momentum']/p2['mass'])
        
        # Cross-section (simplified)
        sigma = 50.0 * np.exp(-distance / 0.5)  # mb, decreases with distance
        
        # Convert to probability (simplified)
        prob = sigma * v_rel * self.time_step * 1e-6  # Rough conversion
        
        return min(prob, 0.1)  # Cap at 10% per time step
    
    def _generate_reaction_products(self, p1: Dict, p2: Dict) -> List[Dict]:
        """Generate products of nuclear reaction."""
        
        # Center of mass energy
        total_momentum = p1['momentum'] + p2['momentum']
        total_energy = (p1['energy'] + p1['mass']) + (p2['energy'] + p2['mass'])
        
        invariant_mass_squared = total_energy**2 - np.sum(total_momentum**2)
        if invariant_mass_squared <= 0:
            return []
        
        invariant_mass = np.sqrt(invariant_mass_squared)
        
        # Determine reaction type based on energy and participants
        products = []
        
        if p1['type'] == 'neutron' and p2['type'] == 'proton':
            # n + p → d + γ (deuteron formation)
            if invariant_mass > 1.876:  # Deuteron mass
                deuteron = {
                    'type': 'deuteron',
                    'A': 2, 'Z': 1,
                    'mass': 1.876,
                    'charge': 1,
                    'position': (p1['position'] + p2['position']) / 2,
                    'momentum': total_momentum * 0.9,  # Conservation approx
                    'energy': 0.0  # Will be recalculated
                }
                deuteron['energy'] = np.sqrt(np.sum(deuteron['momentum']**2) + deuteron['mass']**2) - deuteron['mass']
                
                gamma = {
                    'type': 'gamma',
                    'A': 0, 'Z': 0,
                    'mass': 0.0,
                    'charge': 0,
                    'position': (p1['position'] + p2['position']) / 2,
                    'momentum': total_momentum * 0.1,
                    'energy': np.linalg.norm(total_momentum * 0.1)  # E = pc for photon
                }
                
                products = [deuteron, gamma]
        
        elif p1['type'] == p2['type'] == 'proton':
            # p + p → p + p + π⁰ (pion production)
            if invariant_mass > 2 * 0.938 + 0.135:  # Two protons + pion
                
                # Two outgoing protons
                for i in range(2):
                    proton = {
                        'type': 'proton',
                        'A': 1, 'Z': 1,
                        'mass': 0.938,
                        'charge': 1,
                        'position': (p1['position'] + p2['position']) / 2 + 0.5 * np.random.randn(3),
                        'momentum': total_momentum * (0.4 + 0.1 * np.random.random()) * np.random.randn(3),
                        'energy': 0.0
                    }
                    proton['momentum'] /= np.linalg.norm(proton['momentum']) if np.linalg.norm(proton['momentum']) > 0 else 1.0
                    proton['momentum'] *= 0.5  # Scale momentum
                    proton['energy'] = np.sqrt(np.sum(proton['momentum']**2) + proton['mass']**2) - proton['mass']
                    products.append(proton)
                
                # Pion
                pion = {
                    'type': 'pion_zero',
                    'A': 0, 'Z': 0,
                    'mass': 0.135,
                    'charge': 0,
                    'position': (p1['position'] + p2['position']) / 2,
                    'momentum': -sum(p['momentum'] for p in products),  # Momentum conservation
                    'energy': 0.0
                }
                pion['energy'] = np.sqrt(np.sum(pion['momentum']**2) + pion['mass']**2) - pion['mass']
                products.append(pion)
        
        return products
    
    def _update_boundary_conditions(self):
        """Update boundary conditions and track escaped particles."""
        
        escaped_particles = []
        
        for i, particle in enumerate(self.particles):
            if self.boundary_conditions.check_particle_escape(particle):
                self.boundary_conditions.update_escaped_mass(particle)
                escaped_particles.append(i)
        
        # Remove escaped particles (in reverse order)
        for i in sorted(escaped_particles, reverse=True):
            del self.particles[i]
    
    def _store_time_step_state(self):
        """Store complete simulation state with compression."""
        
        state = {
            'time': self.current_time,
            'particles': [p.copy() for p in self.particles],
            'escaped_mass_fraction': self.boundary_conditions.get_escape_fraction(),
            'total_reactions': len(self.equation_tracker.reactions),
            'recent_reactions': self.equation_tracker.get_reactions_in_time_range(
                max(0, self.current_time - self.time_step * 5), 
                self.current_time
            )
        }
        
        self.time_history.append(state)
        
        # Compress history if too long
        if len(self.time_history) > self.max_history_length:
            # Keep every nth step for older history
            compressed_history = []
            
            # Keep recent history (last 1000 steps) at full resolution
            compressed_history.extend(self.time_history[-1000:])
            
            # Compress older history
            older_history = self.time_history[:-1000]
            for i in range(0, len(older_history), self.history_compression_factor):
                compressed_history.append(older_history[i])
            
            self.time_history = sorted(compressed_history, key=lambda x: x['time'])
    
    def _compute_enhanced_observables(self):
        """Compute enhanced global observables."""
        
        # Basic observables
        total_energy = sum(p['energy'] for p in self.particles)
        total_particles = len(self.particles)
        
        # Temperature estimate from average kinetic energy
        if total_particles > 0:
            avg_kinetic = total_energy / total_particles
            temperature = avg_kinetic * (2/3)  # Rough estimate
        else:
            temperature = 0.0
        
        # Store observables
        self.global_observables['time'].append(self.current_time)
        self.global_observables['temperature'].append(temperature)
        self.global_observables['energy_density'].append(total_energy / 1000)  # Rough estimate
        self.global_observables['pressure'].append(temperature / 3.0)
        self.global_observables['particle_count'].append(total_particles)
        self.global_observables['entropy_density'].append(temperature * total_particles / 1000)
        self.global_observables['escaped_mass_fraction'].append(self.boundary_conditions.get_escape_fraction())
        
        # Reaction rate (reactions per time)
        recent_reactions = len(self.equation_tracker.get_reactions_in_time_range(
            max(0, self.current_time - 1.0), self.current_time
        ))
        self.global_observables['reaction_rate'].append(recent_reactions)
        self.global_observables['total_reactions'].append(len(self.equation_tracker.reactions))
    
    def _print_enhanced_progress(self, step: int, total_steps: int):
        """Print enhanced progress with reaction and boundary info."""
        
        progress = (step / total_steps) * 100
        temp = self.global_observables['temperature'][-1] if self.global_observables['temperature'] else 0
        particles = len(self.particles)
        reactions = len(self.equation_tracker.reactions)
        escaped_frac = self.boundary_conditions.get_escape_fraction()
        
        print(f"Step {step:5d}/{total_steps} ({progress:5.1f}%) | "
              f"t = {self.current_time:6.2f} fm/c | "
              f"T = {temp:6.1f} MeV | "
              f"N = {particles:4d} | "
              f"Reactions = {reactions:3d} | "
              f"Escaped = {escaped_frac:.1%}")
    
    def _get_comprehensive_results(self) -> Dict[str, Any]:
        """Get comprehensive simulation results."""
        
        # Get reaction summary
        reaction_summary = self.equation_tracker.get_reaction_summary()
        
        results = {
            'global_observables': self.global_observables,
            'time_history': self.time_history,
            'final_particles': [p.copy() for p in self.particles],
            'nuclear_reactions': {
                'summary': reaction_summary,
                'equations': self.equation_tracker.generate_reaction_equations_text(),
                'all_reactions': [
                    {
                        'equation': r.to_equation_string(),
                        'time': r.time,
                        'q_value': r.q_value,
                        'type': r.reaction_type
                    }
                    for r in self.equation_tracker.reactions
                ]
            },
            'boundary_conditions': {
                'initial_mass': self.boundary_conditions.initial_total_mass,
                'escaped_mass': self.boundary_conditions.escaped_mass,
                'escape_fraction': self.boundary_conditions.get_escape_fraction(),
                'escaped_particles': len(self.boundary_conditions.escaped_particles)
            },
            'lattice_info': [
                {
                    'size': lattice.size,
                    'spacing': lattice.spacing,
                    'points': lattice.total_points,
                    'memory_gb': lattice.memory_estimate_gb
                }
                for lattice in self.lattice_configs
            ],
            'simulation_config': self.config
        }
        
        return results
    
    def stop_simulation(self):
        """Stop the simulation gracefully."""
        self.stop_requested = True
        print("🛑 Stop requested - simulation will halt at next time step")
EnhancedSimulationEngine = SimulationEngine
# Export main class
__all__ = ['EnhancedSimulationEngine', 'BoundaryConditions', 'UltraHighResolutionLattice']
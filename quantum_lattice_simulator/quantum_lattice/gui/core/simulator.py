"""
Enhanced Simulation Engine with Boundary Detection and Ultra-High Resolution
Supports up to 1024³ lattices and automatic stopping when mass escapes.
"""
"""
Fixed Enhanced Simulation Engine - Smart Boundary Detection
Only checks mass escape AFTER nuclei enter collision region.
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
from .physics.nuclear import NuclearEquationTracker, NuclearReaction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SmartBoundaryConditions:
    """Smart boundary conditions that activate only after collision begins."""
    simulation_volume: Tuple[float, float, float]  # (x_size, y_size, z_size) in fm
    escape_threshold: float = 0.5  # Stop when 50% of mass escapes
    initial_total_mass: float = 0.0
    escaped_mass: float = 0.0
    escaped_particles: List[Dict] = field(default_factory=list)
    
    # Smart activation parameters
    collision_started: bool = False
    collision_start_time: float = 0.0
    min_activation_time: float = 5.0  # Don't check escapes for first 5 fm/c
    central_interaction_radius: float = 15.0  # fm - radius defining collision region
    particles_entered_collision: bool = False
    
    def check_collision_activation(self, particles: List[Dict], current_time: float):
        """Check if collision has started and we should begin boundary monitoring."""
        
        if self.collision_started:
            return  # Already activated
        
        # Method 1: Time-based activation (minimum time delay)
        if current_time >= self.min_activation_time:
            time_ready = True
        else:
            time_ready = False
        
        # Method 2: Check if particles have entered central collision region
        central_region_occupied = False
        if particles:
            for particle in particles:
                position = particle.get('position', np.zeros(3))
                distance_from_center = np.linalg.norm(position)
                
                if distance_from_center < self.central_interaction_radius:
                    central_region_occupied = True
                    break
        
        # Method 3: Check if nuclei are approaching each other
        nuclei_approaching = self._check_nuclei_proximity(particles)
        
        # Activate if time condition AND (central region occupied OR nuclei close)
        if time_ready and (central_region_occupied or nuclei_approaching):
            self.collision_started = True
            self.collision_start_time = current_time
            self.particles_entered_collision = central_region_occupied
            
            print(f"🎯 Collision started! Boundary monitoring activated at t = {current_time:.3f} fm/c")
            print(f"   Method: {'Central region occupied' if central_region_occupied else 'Nuclei proximity'}")
    
    def _check_nuclei_proximity(self, particles: List[Dict]) -> bool:
        """Check if nuclei are close enough to be considered colliding."""
        
        if len(particles) < 10:  # Need reasonable number of particles
            return False
        
        # Find center of mass positions for projectile and target
        projectile_particles = []
        target_particles = []
        
        for particle in particles:
            if particle.get('parent_nucleus') and 'A' in particle.get('parent_nucleus', ''):
                projectile_particles.append(particle)
            elif particle.get('parent_nucleus') and 'B' in particle.get('parent_nucleus', ''):
                target_particles.append(particle)
            else:
                # Guess based on initial x position
                x_pos = particle.get('position', [0, 0, 0])[0]
                if x_pos < 0:
                    projectile_particles.append(particle)
                else:
                    target_particles.append(particle)
        
        if not projectile_particles or not target_particles:
            return False
        
        # Calculate center of mass for each nucleus
        proj_com = np.mean([p['position'] for p in projectile_particles], axis=0)
        target_com = np.mean([p['position'] for p in target_particles], axis=0)
        
        # Distance between centers of mass
        separation = np.linalg.norm(proj_com - target_com)
        
        # Consider collision started when separation < 20 fm (roughly 2 nuclear radii)
        return separation < 20.0
    
    def check_particle_escape(self, particle: Dict) -> bool:
        """Check if particle has escaped simulation volume (only after collision started)."""
        
        if not self.collision_started:
            return False  # Don't check escapes before collision starts
        
        position = particle.get('position', np.zeros(3))
        x, y, z = position
        
        x_max, y_max, z_max = self.simulation_volume
        
        # Check if outside boundary (with buffer)
        buffer = 8.0  # fm - larger buffer to avoid premature detection
        escaped = (abs(x) > x_max/2 + buffer or 
                  abs(y) > y_max/2 + buffer or 
                  abs(z) > z_max/2 + buffer)
        
        # Additional check: don't consider it escaped if it's just outside the original positions
        # This prevents false positives from initial nuclear placement
        if escaped and not self.particles_entered_collision:
            # If particles haven't entered central region yet, be more lenient
            extended_buffer = 25.0  # Much larger buffer for initial phase
            really_escaped = (abs(x) > x_max/2 + extended_buffer or 
                            abs(y) > y_max/2 + extended_buffer or 
                            abs(z) > z_max/2 + extended_buffer)
            return really_escaped
        
        return escaped
    
    def update_escaped_mass(self, escaped_particle: Dict):
        """Update escaped mass tracking."""
        
        mass = escaped_particle.get('mass', 0.938)  # Default nucleon mass
        self.escaped_mass += mass
        self.escaped_particles.append(escaped_particle.copy())
        
        print(f"💨 Particle escaped: {escaped_particle.get('type', 'unknown')} "
              f"at t = {escaped_particle.get('escape_time', 0):.3f} fm/c")
    
    def get_escape_fraction(self) -> float:
        """Get fraction of mass that has escaped."""
        
        if self.initial_total_mass <= 0:
            return 0.0
        
        return self.escaped_mass / self.initial_total_mass
    
    def should_stop_simulation(self) -> bool:
        """Check if simulation should stop due to mass escape."""
        
        if not self.collision_started:
            return False  # Never stop before collision starts
        
        return self.get_escape_fraction() >= self.escape_threshold
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get comprehensive boundary status information."""
        
        return {
            'collision_started': self.collision_started,
            'collision_start_time': self.collision_start_time,
            'particles_in_central_region': self.particles_entered_collision,
            'monitoring_active': self.collision_started,
            'escaped_mass_fraction': self.get_escape_fraction(),
            'escaped_particle_count': len(self.escaped_particles),
            'should_stop': self.should_stop_simulation()
        }

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
        bytes_per_point = 10 * 16 * 4  # ~640 bytes per point
        self.memory_estimate_gb = (self.total_points * bytes_per_point) / (1024**3)
    
    def is_memory_feasible(self, available_gb: float = 16.0) -> bool:
        """Check if lattice fits in available memory."""
        return self.memory_estimate_gb <= available_gb
    
    def get_recommended_chunk_size(self) -> Tuple[int, int, int]:
        """Get recommended chunk size for distributed processing."""
        
        nx, ny, nz = self.size
        
        if max(self.size) <= 64:
            chunk_size = 16
        elif max(self.size) <= 256:
            chunk_size = 32  
        elif max(self.size) <= 512:
            chunk_size = 64
        else:
            chunk_size = 128
        
        chunk_x = min(chunk_size, nx)
        chunk_y = min(chunk_size, ny)
        chunk_z = min(chunk_size, nz)
        
        return (chunk_x, chunk_y, chunk_z)

class EnhancedSimulationEngine:
    """Enhanced simulation engine with smart boundary detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_workers = config.get('num_workers', mp.cpu_count())
        
        # Ultra-high resolution lattice support
        self.lattice_configs = []
        lattice_sizes = config.get('lattice_sizes', [(64, 64, 64)])
        spacings = config.get('spacings', [0.05])
        
        for size, spacing in zip(lattice_sizes, spacings):
            lattice = UltraHighResolutionLattice(size, spacing)
            
            if lattice.is_memory_feasible():
                self.lattice_configs.append(lattice)
                print(f"✅ Lattice {size} feasible: {lattice.memory_estimate_gb:.2f} GB")
            else:
                print(f"⚠️ Lattice {size} requires {lattice.memory_estimate_gb:.2f} GB")
                self.lattice_configs.append(lattice)
        
        # Nuclear equation tracking
        self.equation_tracker = NuclearEquationTracker()
        
        # Smart boundary conditions
        max_lattice = max(self.lattice_configs, key=lambda x: max(x.size))
        sim_volume = (max_lattice.size[0] * max_lattice.spacing,
                     max_lattice.size[1] * max_lattice.spacing,
                     max_lattice.size[2] * max_lattice.spacing)
        
        self.boundary_conditions = SmartBoundaryConditions(
            simulation_volume=sim_volume,
            escape_threshold=config.get('escape_threshold', 0.5),
            min_activation_time=config.get('min_collision_time', 5.0),  # Configurable
            central_interaction_radius=config.get('collision_radius', 15.0)  # Configurable
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
            'total_reactions': [],
            'collision_started': [],
            'boundary_monitoring_active': []
        }
        
        print(f"🚀 Enhanced simulation engine with smart boundary detection")
        print(f"   Lattice configurations: {len(self.lattice_configs)}")
        print(f"   Boundary volume: {sim_volume[0]:.1f} × {sim_volume[1]:.1f} × {sim_volume[2]:.1f} fm³")
        print(f"   Escape threshold: {self.boundary_conditions.escape_threshold:.1%}")
        print(f"   Min collision time: {self.boundary_conditions.min_activation_time:.1f} fm/c")
        print(f"   Collision radius: {self.boundary_conditions.central_interaction_radius:.1f} fm")
    
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
        print(f"   🎯 Boundary monitoring will activate after collision begins")
    
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
        
        # Calculate initial separation (place nuclei outside collision region)
        initial_separation = 25.0  # fm - start well separated
        
        # Create nucleus A (projectile) - starts on left
        self._create_nucleus_particles(
            data_a, f"{nucleus_a}_projectile",
            center=np.array([-initial_separation, impact_parameter/2, 0.0]),
            velocity=np.array([np.sqrt(1 - (0.938/(collision_energy_gev + 0.938))**2), 0, 0])
        )
        
        # Create nucleus B (target) - starts on right
        self._create_nucleus_particles(
            data_b, f"{nucleus_b}_target",
            center=np.array([initial_separation, -impact_parameter/2, 0.0]),
            velocity=np.array([0, 0, 0])  # Target at rest
        )
        
        print(f"   Nuclei placed at ±{initial_separation} fm separation")
        print(f"   Boundary monitoring will activate when they approach collision region")
    
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
            r = np.random.exponential(R/2)
            
            if r > 5 * R:
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
        
        # Approximate boost
        p_parallel_boosted = gamma * p_parallel + gamma * v_mag * 0.938
        
        return p_parallel_boosted * v_hat + p_perp
    
    def run_simulation(self, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run enhanced simulation with smart boundary detection."""
        
        print("🚀 Starting simulation with smart boundary detection")
        print("   📍 Boundary monitoring will activate after collision begins")
        
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
                
                # Smart boundary activation check
                self.boundary_conditions.check_collision_activation(self.particles, self.current_time)
                
                # Check for nuclear reactions
                self._check_nuclear_reactions()
                
                # Update boundary conditions (only if collision started)
                self._update_smart_boundary_conditions()
                
                # Store complete state
                self._store_time_step_state()
                
                # Compute observables
                self._compute_enhanced_observables()
                
                # Check boundary stopping condition (only if collision started)
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
            print(f"✅ Smart boundary simulation completed in {elapsed:.1f}s")
        
        return self._get_comprehensive_results()
    
    def _evolve_time_step(self):
        """Evolve system by one time step."""
        
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
        
        return momentum / total_energy
    
    def _compute_nuclear_forces(self, particle: Dict) -> np.ndarray:
        """Compute nuclear and electromagnetic forces."""
        
        force = np.zeros(3)
        
        for other in self.particles:
            if other is particle:
                continue
            
            r_vec = particle['position'] - other['position']
            r_mag = np.linalg.norm(r_vec)
            
            if r_mag < 0.1:
                continue
            
            r_hat = r_vec / r_mag
            
            # Nuclear force (Yukawa)
            if (particle['type'] in ['proton', 'neutron'] and 
                other['type'] in ['proton', 'neutron']):
                
                m_pion = 0.138  # GeV
                g_strong = 14.0
                
                yukawa_force = (g_strong / (4 * np.pi * r_mag**2) * 
                              np.exp(-m_pion * r_mag * 5.07) *
                              (1 + m_pion * r_mag * 5.07))
                
                force -= yukawa_force * r_hat
            
            # Electromagnetic force
            if particle['charge'] != 0 and other['charge'] != 0:
                alpha = 1.0 / 137.036
                coulomb_force = (alpha * particle['charge'] * other['charge'] / 
                               r_mag**2 * 0.197**2)
                
                force += coulomb_force * r_hat
        
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
                        
                        if products:
                            # Track the reaction
                            reaction = self.equation_tracker.track_reaction(
                                reactants, products,
                                (p1['position'] + p2['position']) / 2,
                                self.current_time
                            )
                            
                            reactions_this_step.append((i, j, products))
        
        # Apply reactions
        for i, j, products in sorted(reactions_this_step, reverse=True):
            if j < len(self.particles):
                del self.particles[j]
            if i < len(self.particles):
                del self.particles[i]
            
            for product in products:
                product['id'] = len(self.particles) + len(products)
                product['creation_time'] = self.current_time
                self.particles.append(product)
    
    def _calculate_reaction_probability(self, p1: Dict, p2: Dict, distance: float) -> float:
        """Calculate probability of nuclear reaction."""
        
        if not (p1['type'] in ['proton', 'neutron'] and p2['type'] in ['proton', 'neutron']):
            return 0.0
        
        v_rel = np.linalg.norm(p1['momentum']/p1['mass'] - p2['momentum']/p2['mass'])
        sigma = 50.0 * np.exp(-distance / 0.5)  # mb
        prob = sigma * v_rel * self.time_step * 1e-6
        
        return min(prob, 0.1)
    
    def _generate_reaction_products(self, p1: Dict, p2: Dict) -> List[Dict]:
        """Generate products of nuclear reaction."""
        
        total_momentum = p1['momentum'] + p2['momentum']
        total_energy = (p1['energy'] + p1['mass']) + (p2['energy'] + p2['mass'])
        
        invariant_mass_squared = total_energy**2 - np.sum(total_momentum**2)
        if invariant_mass_squared <= 0:
            return []
        
        products = []
        
        if p1['type'] == 'neutron' and p2['type'] == 'proton':
            # n + p → d + γ
            if np.sqrt(invariant_mass_squared) > 1.876:
                deuteron = {
                    'type': 'deuteron',
                    'A': 2, 'Z': 1,
                    'mass': 1.876,
                    'charge': 1,
                    'position': (p1['position'] + p2['position']) / 2,
                    'momentum': total_momentum * 0.9,
                    'energy': 0.0
                }
                deuteron['energy'] = np.sqrt(np.sum(deuteron['momentum']**2) + deuteron['mass']**2) - deuteron['mass']
                
                gamma = {
                    'type': 'gamma',
                    'A': 0, 'Z': 0,
                    'mass': 0.0,
                    'charge': 0,
                    'position': (p1['position'] + p2['position']) / 2,
                    'momentum': total_momentum * 0.1,
                    'energy': np.linalg.norm(total_momentum * 0.1)
                }
                
                products = [deuteron, gamma]
        
        return products
    
    def _update_smart_boundary_conditions(self):
        """Update smart boundary conditions."""
        
        escaped_particles = []
        
        for i, particle in enumerate(self.particles):
            if self.boundary_conditions.check_particle_escape(particle):
                # Mark escape time
                particle['escape_time'] = self.current_time
                self.boundary_conditions.update_escaped_mass(particle)
                escaped_particles.append(i)
        
        # Remove escaped particles
        for i in sorted(escaped_particles, reverse=True):
            del self.particles[i]
    
    def _store_time_step_state(self):
        """Store complete simulation state."""
        
        boundary_status = self.boundary_conditions.get_status_info()
        
        state = {
            'time': self.current_time,
            'particles': [p.copy() for p in self.particles],
            'boundary_status': boundary_status,
            'collision_started': boundary_status['collision_started'],
            'escaped_mass_fraction': boundary_status['escaped_mass_fraction'],
            'total_reactions': len(self.equation_tracker.reactions),
            'recent_reactions': self.equation_tracker.get_reactions_in_time_range(
                max(0, self.current_time - self.time_step * 5), 
                self.current_time
            )
        }
        
        self.time_history.append(state)
        
        # Compress history if too long
        if len(self.time_history) > self.max_history_length:
            compressed_history = []
            compressed_history.extend(self.time_history[-1000:])
            
            older_history = self.time_history[:-1000]
            for i in range(0, len(older_history), self.history_compression_factor):
                compressed_history.append(older_history[i])
            
            self.time_history = sorted(compressed_history, key=lambda x: x['time'])
    
    def _compute_enhanced_observables(self):
        """Compute enhanced global observables."""
        
        total_energy = sum(p['energy'] for p in self.particles)
        total_particles = len(self.particles)
        
        if total_particles > 0:
            avg_kinetic = total_energy / total_particles
            temperature = avg_kinetic * (2/3)
        else:
            temperature = 0.0
        
        # Store observables
        self.global_observables['time'].append(self.current_time)
        self.global_observables['temperature'].append(temperature)
        self.global_observables['energy_density'].append(total_energy / 1000)
        self.global_observables['pressure'].append(temperature / 3.0)
        self.global_observables['particle_count'].append(total_particles)
        self.global_observables['entropy_density'].append(temperature * total_particles / 1000)
        self.global_observables['escaped_mass_fraction'].append(self.boundary_conditions.get_escape_fraction())
        self.global_observables['collision_started'].append(self.boundary_conditions.collision_started)
        self.global_observables['boundary_monitoring_active'].append(self.boundary_conditions.collision_started)
        
        # Reaction rate
        recent_reactions = len(self.equation_tracker.get_reactions_in_time_range(
            max(0, self.current_time - 1.0), self.current_time
        ))
        self.global_observables['reaction_rate'].append(recent_reactions)
        self.global_observables['total_reactions'].append(len(self.equation_tracker.reactions))
    
    def _print_enhanced_progress(self, step: int, total_steps: int):
        """Print enhanced progress with smart boundary info."""
        
        progress = (step / total_steps) * 100
        temp = self.global_observables['temperature'][-1] if self.global_observables['temperature'] else 0
        particles = len(self.particles)
        reactions = len(self.equation_tracker.reactions)
        
        boundary_status = self.boundary_conditions.get_status_info()
        collision_status = "✅ Active" if boundary_status['collision_started'] else "⏳ Waiting"
        escaped_frac = boundary_status['escaped_mass_fraction']
        
        print(f"Step {step:5d}/{total_steps} ({progress:5.1f}%) | "
              f"t = {self.current_time:6.2f} fm/c | "
              f"T = {temp:6.1f} MeV | "
              f"N = {particles:4d} | "
              f"Reactions = {reactions:3d} | "
              f"Collision: {collision_status} | "
              f"Escaped = {escaped_frac:.1%}")
    
    def _get_comprehensive_results(self):
        """Get comprehensive simulation results."""
        
        reaction_summary = self.equation_tracker.get_reaction_summary()
        boundary_final_status = self.boundary_conditions.get_status_info()
        
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
            'smart_boundary_conditions': {
                'collision_started': boundary_final_status['collision_started'],
                'collision_start_time': boundary_final_status['collision_start_time'],
                'initial_mass': self.boundary_conditions.initial_total_mass,
                'escaped_mass': self.boundary_conditions.escaped_mass,
                'escape_fraction': boundary_final_status['escaped_mass_fraction'],
                'escaped_particles': len(self.boundary_conditions.escaped_particles),
                'monitoring_was_active': boundary_final_status['monitoring_active']
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

# Export main classes
SimulationEngine = EnhancedSimulationEngine  # Alias for backward compatibility
BoundaryConditions = SmartBoundaryConditions 
__all__ = ['EnhancedSimulationEngine', 'SmartBoundaryConditions', 'UltraHighResolutionLattice']
# Export main class
__all__ = ['EnhancedSimulationEngine', 'BoundaryConditions', 'UltraHighResolutionLattice']
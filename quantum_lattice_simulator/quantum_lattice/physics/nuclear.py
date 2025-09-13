"""
Nuclear structure and database.
"""
"""
Nuclear Reaction Equation Tracker
Tracks and displays all nuclear reactions as they occur during simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
from collections import defaultdict
import re

@dataclass
class NuclearReaction:
    """Complete nuclear reaction with conservation checks."""
    reaction_id: int
    time: float  # fm/c
    position: np.ndarray  # fm
    
    # Reactants
    reactants: List[Dict[str, any]]  # [{'type': 'proton', 'A': 1, 'Z': 1, 'energy': ...}, ...]
    
    # Products  
    products: List[Dict[str, any]]   # [{'type': 'neutron', 'A': 1, 'Z': 0, 'energy': ...}, ...]
    
    # Reaction details
    reaction_type: str  # 'fusion', 'fission', 'spallation', 'decay', 'production'
    q_value: float     # MeV (energy released/absorbed)
    threshold_energy: float  # MeV (minimum energy needed)
    cross_section: float     # barns
    
    # Conservation checks
    conserved_baryon_number: bool = True
    conserved_charge: bool = True
    conserved_energy: bool = True
    conserved_momentum: bool = True
    
    def __post_init__(self):
        """Verify conservation laws."""
        self._check_conservation_laws()
    
    def _check_conservation_laws(self):
        """Check all conservation laws for this reaction."""
        
        # Baryon number conservation
        initial_A = sum(r.get('A', 0) for r in self.reactants)
        final_A = sum(p.get('A', 0) for p in self.products)
        self.conserved_baryon_number = (initial_A == final_A)
        
        # Charge conservation
        initial_Z = sum(r.get('Z', 0) for r in self.reactants)
        final_Z = sum(p.get('Z', 0) for p in self.products)
        self.conserved_charge = (initial_Z == final_Z)
        
        # Energy conservation (within 1% tolerance)
        initial_E = sum(r.get('energy', 0) + r.get('mass', 0) for r in self.reactants)
        final_E = sum(p.get('energy', 0) + p.get('mass', 0) for p in self.products)
        energy_diff = abs(initial_E - final_E) / max(initial_E, 1e-6)
        self.conserved_energy = (energy_diff < 0.01)
        
        # Momentum conservation (within 1% tolerance)
        initial_p = np.sum([r.get('momentum', np.zeros(3)) for r in self.reactants], axis=0)
        final_p = np.sum([p.get('momentum', np.zeros(3)) for p in self.products], axis=0)
        momentum_diff = np.linalg.norm(initial_p - final_p) / max(np.linalg.norm(initial_p), 1e-6)
        self.conserved_momentum = (momentum_diff < 0.01)
    
    def to_equation_string(self) -> str:
        """Convert reaction to nuclear equation string."""
        
        # Format reactants
        reactant_strs = []
        for r in self.reactants:
            if r.get('type') == 'proton':
                reactant_strs.append('p')
            elif r.get('type') == 'neutron':
                reactant_strs.append('n')
            elif r.get('type') == 'alpha':
                reactant_strs.append('α')
            elif r.get('type') == 'deuteron':
                reactant_strs.append('d')
            else:
                A, Z = r.get('A', 1), r.get('Z', 0)
                if A > 1:
                    element = self._get_element_symbol(Z)
                    reactant_strs.append(f"^{A}{element}")
                else:
                    reactant_strs.append(r.get('type', 'X'))
        
        # Format products
        product_strs = []
        for p in self.products:
            if p.get('type') == 'proton':
                product_strs.append('p')
            elif p.get('type') == 'neutron':
                product_strs.append('n')
            elif p.get('type') == 'alpha':
                product_strs.append('α')
            elif p.get('type') == 'deuteron':
                product_strs.append('d')
            elif p.get('type') == 'gamma':
                product_strs.append('γ')
            elif 'pion' in p.get('type', ''):
                if 'plus' in p.get('type', ''):
                    product_strs.append('π⁺')
                elif 'minus' in p.get('type', ''):
                    product_strs.append('π⁻')
                else:
                    product_strs.append('π⁰')
            elif 'kaon' in p.get('type', ''):
                if 'plus' in p.get('type', ''):
                    product_strs.append('K⁺')
                elif 'minus' in p.get('type', ''):
                    product_strs.append('K⁻')
                else:
                    product_strs.append('K⁰')
            else:
                A, Z = p.get('A', 1), p.get('Z', 0)
                if A > 1:
                    element = self._get_element_symbol(Z)
                    product_strs.append(f"^{A}{element}")
                else:
                    product_strs.append(p.get('type', 'X'))
        
        # Create equation
        reactant_side = ' + '.join(reactant_strs)
        product_side = ' + '.join(product_strs)
        
        # Add Q-value
        q_str = f" (Q = {self.q_value:+.2f} MeV)" if abs(self.q_value) > 0.01 else ""
        
        return f"{reactant_side} → {product_side}{q_str}"
    
    def _get_element_symbol(self, Z: int) -> str:
        """Get element symbol from atomic number."""
        elements = {
            0: 'n', 1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 
            9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S',
            17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 26: 'Fe', 29: 'Cu', 47: 'Ag', 
            79: 'Au', 82: 'Pb', 92: 'U'
        }
        return elements.get(Z, f'Z{Z}')

class NuclearEquationTracker:
    """Tracks and analyzes all nuclear reactions during simulation."""
    
    def __init__(self):
        self.reactions: List[NuclearReaction] = []
        self.reaction_counter = 0
        self.reaction_types = defaultdict(int)
        self.conservation_violations = []
        
        # Nuclear data for Q-value calculations
        self.binding_energies = self._load_nuclear_data()
        
        print("🔬 Nuclear equation tracker initialized")
    
    def _load_nuclear_data(self) -> Dict[Tuple[int, int], float]:
        """Load nuclear binding energies for Q-value calculations."""
        
        # Simplified nuclear data (A, Z) -> Binding Energy (MeV)
        # In practice, this would load from a comprehensive nuclear database
        nuclear_data = {
            (1, 1): 0.0,           # Proton
            (1, 0): 0.0,           # Neutron  
            (2, 1): 2.225,         # Deuteron
            (3, 1): 8.482,         # Triton
            (3, 2): 7.718,         # He-3
            (4, 2): 28.296,        # Alpha
            (12, 6): 92.162,       # C-12
            (16, 8): 127.619,      # O-16
            (40, 20): 342.052,     # Ca-40
            (56, 26): 492.254,     # Fe-56
            (197, 79): 1559.4,     # Au-197
            (208, 82): 1636.4,     # Pb-208
            (238, 92): 1801.7      # U-238
        }
        
        return nuclear_data
    
    def track_reaction(self, reactants: List[Dict], products: List[Dict], 
                      position: np.ndarray, time: float) -> NuclearReaction:
        """Track a new nuclear reaction."""
        
        # Determine reaction type
        reaction_type = self._classify_reaction(reactants, products)
        
        # Calculate Q-value
        q_value = self._calculate_q_value(reactants, products)
        
        # Calculate threshold energy
        threshold_energy = self._calculate_threshold_energy(reactants, products, q_value)
        
        # Estimate cross-section
        cross_section = self._estimate_cross_section(reactants, products)
        
        # Create reaction object
        reaction = NuclearReaction(
            reaction_id=self.reaction_counter,
            time=time,
            position=position.copy(),
            reactants=reactants.copy(),
            products=products.copy(),
            reaction_type=reaction_type,
            q_value=q_value,
            threshold_energy=threshold_energy,
            cross_section=cross_section
        )
        
        # Store reaction
        self.reactions.append(reaction)
        self.reaction_types[reaction_type] += 1
        self.reaction_counter += 1
        
        # Check for conservation violations
        if not all([reaction.conserved_baryon_number, reaction.conserved_charge,
                   reaction.conserved_energy, reaction.conserved_momentum]):
            self.conservation_violations.append(reaction)
        
        return reaction
    
    def _classify_reaction(self, reactants: List[Dict], products: List[Dict]) -> str:
        """Classify the type of nuclear reaction."""
        
        n_reactants = len(reactants)
        n_products = len(products)
        
        # Get total mass numbers
        initial_A = sum(r.get('A', 1) for r in reactants)
        final_A = sum(p.get('A', 1) for p in products)
        
        # Count different particle types
        initial_heavy = len([r for r in reactants if r.get('A', 1) > 4])
        final_heavy = len([p for p in products if p.get('A', 1) > 4])
        
        initial_light = len([r for r in reactants if r.get('A', 1) <= 4])
        final_light = len([p for p in products if p.get('A', 1) <= 4])
        
        # Classification logic
        if n_reactants == 1 and n_products > 1:
            if final_A > 4:
                return 'spontaneous_fission'
            else:
                return 'radioactive_decay'
        
        elif n_reactants == 2 and n_products == 1:
            return 'fusion'
        
        elif n_reactants == 2 and final_heavy == 1 and final_light > initial_light:
            if final_light - initial_light >= 3:
                return 'spallation'
            else:
                return 'nuclear_reaction'
        
        elif n_reactants == 2 and final_heavy > initial_heavy:
            return 'fragmentation'
        
        elif any('pion' in p.get('type', '') for p in products):
            return 'pion_production'
        
        elif any('kaon' in p.get('type', '') for p in products):
            return 'strange_production'
        
        elif n_products > n_reactants + 2:
            return 'multifragmentation'
        
        else:
            return 'elastic_scattering'
    
    def _calculate_q_value(self, reactants: List[Dict], products: List[Dict]) -> float:
        """Calculate Q-value for the reaction."""
        
        initial_mass = 0.0
        final_mass = 0.0
        
        # Calculate initial mass
        for r in reactants:
            A, Z = r.get('A', 1), r.get('Z', 0)
            if (A, Z) in self.binding_energies:
                # Mass = A * u - B.E.
                mass = A * 931.494 - self.binding_energies[(A, Z)]  # MeV
            else:
                # Estimate using semi-empirical mass formula
                mass = self._estimate_nuclear_mass(A, Z)
            initial_mass += mass
        
        # Calculate final mass
        for p in products:
            A, Z = p.get('A', 1), p.get('Z', 0)
            if (A, Z) in self.binding_energies:
                mass = A * 931.494 - self.binding_energies[(A, Z)]
            else:
                mass = self._estimate_nuclear_mass(A, Z)
            final_mass += mass
        
        # Q = initial mass - final mass
        q_value = initial_mass - final_mass
        
        return q_value
    
    def _estimate_nuclear_mass(self, A: int, Z: int) -> float:
        """Estimate nuclear mass using semi-empirical mass formula."""
        
        N = A - Z
        
        # SEMF parameters (MeV)
        a_v = 15.75    # Volume
        a_s = -17.8    # Surface
        a_c = -0.711   # Coulomb
        a_a = -23.7    # Asymmetry
        
        # Pairing term
        if Z % 2 == 0 and N % 2 == 0:  # Even-even
            delta = 11.18 / np.sqrt(A)
        elif Z % 2 == 1 and N % 2 == 1:  # Odd-odd
            delta = -11.18 / np.sqrt(A)
        else:  # Even-odd
            delta = 0.0
        
        # Binding energy
        binding_energy = (a_v * A + 
                         a_s * A**(2/3) + 
                         a_c * Z**2 / A**(1/3) + 
                         a_a * (N - Z)**2 / A + 
                         delta)
        
        # Mass = A * atomic mass unit - binding energy
        mass = A * 931.494 - binding_energy
        
        return mass
    
    def _calculate_threshold_energy(self, reactants: List[Dict], products: List[Dict], 
                                   q_value: float) -> float:
        """Calculate threshold energy for endothermic reactions."""
        
        if q_value >= 0:
            return 0.0  # Exothermic reaction
        
        # For endothermic reaction: E_th = -Q * (1 + m_projectile/m_target)
        if len(reactants) >= 2:
            m1 = reactants[0].get('A', 1) * 931.494  # Projectile mass
            m2 = reactants[1].get('A', 1) * 931.494  # Target mass
            
            threshold = -q_value * (1 + m1/m2)
        else:
            threshold = -q_value
        
        return max(0.0, threshold)
    
    def _estimate_cross_section(self, reactants: List[Dict], products: List[Dict]) -> float:
        """Estimate reaction cross-section in barns."""
        
        if len(reactants) < 2:
            return 0.0
        
        # Get nuclear radii
        A1 = reactants[0].get('A', 1)
        A2 = reactants[1].get('A', 1)
        
        R1 = 1.2 * A1**(1/3)  # fm
        R2 = 1.2 * A2**(1/3)  # fm
        
        # Geometric cross-section
        R_interaction = R1 + R2
        sigma_geometric = np.pi * R_interaction**2 * 1e-24  # Convert fm² to barns
        
        # Adjust based on reaction type
        reaction_type = self._classify_reaction(reactants, products)
        
        if reaction_type == 'elastic_scattering':
            return sigma_geometric * 0.5
        elif reaction_type == 'fusion':
            return sigma_geometric * 0.01  # Fusion has low cross-section
        elif reaction_type in ['spallation', 'fragmentation']:
            return sigma_geometric * 0.3
        elif 'production' in reaction_type:
            return sigma_geometric * 0.05
        else:
            return sigma_geometric * 0.1
    
    def get_reaction_summary(self) -> Dict[str, any]:
        """Get summary of all tracked reactions."""
        
        total_reactions = len(self.reactions)
        
        if total_reactions == 0:
            return {'total_reactions': 0}
        
        # Time evolution
        times = [r.time for r in self.reactions]
        
        # Energy analysis
        q_values = [r.q_value for r in self.reactions]
        total_energy_released = sum(q for q in q_values if q > 0)
        total_energy_absorbed = sum(abs(q) for q in q_values if q < 0)
        
        # Conservation analysis
        conservation_success_rate = {
            'baryon_number': sum(r.conserved_baryon_number for r in self.reactions) / total_reactions,
            'charge': sum(r.conserved_charge for r in self.reactions) / total_reactions,
            'energy': sum(r.conserved_energy for r in self.reactions) / total_reactions,
            'momentum': sum(r.conserved_momentum for r in self.reactions) / total_reactions
        }
        
        return {
            'total_reactions': total_reactions,
            'reaction_types': dict(self.reaction_types),
            'time_range': (min(times), max(times)) if times else (0, 0),
            'total_energy_released': total_energy_released,
            'total_energy_absorbed': total_energy_absorbed,
            'net_energy_release': total_energy_released - total_energy_absorbed,
            'conservation_success_rate': conservation_success_rate,
            'conservation_violations': len(self.conservation_violations)
        }
    
    def get_reactions_in_time_range(self, t_start: float, t_end: float) -> List[NuclearReaction]:
        """Get reactions that occurred in a specific time range."""
        
        return [r for r in self.reactions if t_start <= r.time <= t_end]
    
    def get_most_significant_reactions(self, n: int = 10) -> List[NuclearReaction]:
        """Get the most significant reactions (by energy release)."""
        
        return sorted(self.reactions, key=lambda r: abs(r.q_value), reverse=True)[:n]
    
    def generate_reaction_equations_text(self) -> str:
        """Generate formatted text of all nuclear equations."""
        
        if not self.reactions:
            return "No nuclear reactions detected yet.\n"
        
        text = "🔬 NUCLEAR REACTIONS DETECTED:\n"
        text += "=" * 80 + "\n\n"
        
        # Group reactions by type
        reactions_by_type = defaultdict(list)
        for reaction in self.reactions:
            reactions_by_type[reaction.reaction_type].append(reaction)
        
        for reaction_type, reactions in reactions_by_type.items():
            text += f"📊 {reaction_type.upper().replace('_', ' ')} ({len(reactions)} reactions):\n"
            text += "-" * 60 + "\n"
            
            # Show up to 5 examples of each type
            for reaction in reactions[:5]:
                equation = reaction.to_equation_string()
                time_str = f"t = {reaction.time:.3f} fm/c"
                
                # Conservation status
                conservation_status = []
                if not reaction.conserved_baryon_number:
                    conservation_status.append("A")
                if not reaction.conserved_charge:
                    conservation_status.append("Z")
                if not reaction.conserved_energy:
                    conservation_status.append("E")
                if not reaction.conserved_momentum:
                    conservation_status.append("p")
                
                status_str = f" [VIOLATIONS: {','.join(conservation_status)}]" if conservation_status else " [OK]"
                
                text += f"  {equation}\n"
                text += f"    {time_str}, σ = {reaction.cross_section:.2e} barns{status_str}\n"
            
            if len(reactions) > 5:
                text += f"  ... and {len(reactions) - 5} more {reaction_type} reactions\n"
            
            text += "\n"
        
        # Summary statistics
        summary = self.get_reaction_summary()
        text += "📈 REACTION SUMMARY:\n"
        text += "-" * 40 + "\n"
        text += f"Total reactions: {summary['total_reactions']}\n"
        text += f"Energy released: {summary['total_energy_released']:.2f} MeV\n"
        text += f"Energy absorbed: {summary['total_energy_absorbed']:.2f} MeV\n"
        text += f"Net energy change: {summary['net_energy_release']:+.2f} MeV\n"
        text += f"Conservation violations: {summary['conservation_violations']}\n"
        
        return text

# Export main class
__all__ = ['NuclearEquationTracker', 'NuclearReaction']
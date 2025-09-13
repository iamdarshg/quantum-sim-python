
import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Dict, List, Optional, Any

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class AdvancedVisualizerWithMomentum:
    """Advanced visualizer with momentum vectors and time stepping."""
    
    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.current_time_index = 0
        self.time_history = []
        self.simulation_data = None
        
        # Enhanced particle type definitions with physics properties
        self.particle_properties = {
            'proton': {
                'color': '#FF4444', 'size': 100, 'marker': 'o', 'alpha': 0.8,
                'label': 'Protons (p)', 'mass': 0.938, 'charge': 1, 'baryon': 1
            },
            'neutron': {
                'color': '#4444FF', 'size': 100, 'marker': 'o', 'alpha': 0.8,
                'label': 'Neutrons (n)', 'mass': 0.939, 'charge': 0, 'baryon': 1
            },
            'pion_plus': {
                'color': '#FFFF44', 'size': 60, 'marker': '^', 'alpha': 0.7,
                'label': 'π⁺', 'mass': 0.140, 'charge': 1, 'baryon': 0
            },
            'pion_minus': {
                'color': '#FF8844', 'size': 60, 'marker': 'v', 'alpha': 0.7,
                'label': 'π⁻', 'mass': 0.140, 'charge': -1, 'baryon': 0
            },
            'pion_zero': {
                'color': '#44FFFF', 'size': 60, 'marker': 's', 'alpha': 0.7,
                'label': 'π⁰', 'mass': 0.135, 'charge': 0, 'baryon': 0
            },
            'kaon_plus': {
                'color': '#44FF44', 'size': 70, 'marker': 'D', 'alpha': 0.7,
                'label': 'K⁺', 'mass': 0.494, 'charge': 1, 'baryon': 0
            },
            'kaon_minus': {
                'color': '#88FF44', 'size': 70, 'marker': '>', 'alpha': 0.7,
                'label': 'K⁻', 'mass': 0.494, 'charge': -1, 'baryon': 0
            },
            'lambda': {
                'color': '#FF44FF', 'size': 80, 'marker': 'p', 'alpha': 0.7,
                'label': 'Λ', 'mass': 1.116, 'charge': 0, 'baryon': 1
            },
            'deuteron': {
                'color': '#8844FF', 'size': 120, 'marker': 'h', 'alpha': 0.8,
                'label': 'Deuteron (d)', 'mass': 1.876, 'charge': 1, 'baryon': 2
            },
            'alpha': {
                'color': '#FF8888', 'size': 140, 'marker': 'H', 'alpha': 0.8,
                'label': 'Alpha (α)', 'mass': 3.728, 'charge': 2, 'baryon': 4
            },
            'fragment': {
                'color': '#FFAAAA', 'size': 160, 'marker': '*', 'alpha': 0.6,
                'label': 'Heavy fragments', 'mass': 5.0, 'charge': 1, 'baryon': 3
            },
            'gamma': {
                'color': '#FFFFFF', 'size': 30, 'marker': '*', 'alpha': 0.9,
                'label': 'Gamma rays (γ)', 'mass': 0.0, 'charge': 0, 'baryon': 0
            }
        }
        
        if MATPLOTLIB_AVAILABLE:
            self.setup_advanced_matplotlib()
        else:
            self.setup_enhanced_text_display()
    
    def setup_advanced_matplotlib(self):
        """Setup advanced matplotlib with momentum vectors and real-time values."""
        
        self.fig = Figure(figsize=(20, 16))
        self.fig.patch.set_facecolor('#1e1e1e')
        
        # Advanced 3D collision view (larger, top-left)
        self.ax_3d = self.fig.add_subplot(2, 3, (1, 4), projection='3d')
        self.ax_3d.set_facecolor('#2e2e2e')
        
        # Physics plots with real-time value displays
        self.ax_energy = self.fig.add_subplot(2, 3, 2)
        self.ax_temp = self.fig.add_subplot(2, 3, 3)
        self.ax_pressure = self.fig.add_subplot(2, 3, 5)
        self.ax_entropy = self.fig.add_subplot(2, 3, 6)
        
        # Style all axes
        for ax in [self.ax_energy, self.ax_temp, self.ax_pressure, self.ax_entropy]:
            ax.set_facecolor('#2e2e2e')
            ax.grid(True, alpha=0.3, color='white')
            ax.tick_params(colors='white')
        
        # Current value text boxes
        self.value_texts = {}
        
        self.fig.tight_layout()
        
        print("✅ Advanced 3D visualization with momentum vectors initialized")
    
    def setup_enhanced_text_display(self):
        """Enhanced text display with complete physics information."""
        
        # Create main text frame with proper scrolling
        main_frame = tk.Frame(self.parent)
        main_frame.pack(fill='both', expand=True)
        
        # Text widget with better formatting
        self.text_widget = tk.Text(
            main_frame, 
            height=30, width=150,
            bg='#1a1a1a', fg='#00ff00', 
            font=('Courier New', 10),
            wrap='word',
            insertbackground='white'
        )
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Pack with proper expansion
        self.text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Enhanced header
        header = """
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                    🚀 SUPERCOMPUTER-LEVEL NUCLEAR PHYSICS SIMULATOR                         ║
║                          First Principles QCD + Nuclear Theory                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

🎯 COMPLETE NUCLEAR PRODUCT IDENTIFICATION SYSTEM:
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🔴 Protons (p)      - Spin 1/2, Charge +1, Mass 938.3 MeV                                 │
│ 🔵 Neutrons (n)     - Spin 1/2, Charge  0, Mass 939.6 MeV                                 │
│ 🟡 Pions (π±,π⁰)    - Spin 0, Pseudoscalar mesons, Mass 140 MeV                           │
│ 🟢 Kaons (K±,K⁰)    - Spin 0, Strange mesons, Mass 494 MeV                                │
│ 🟣 Deuterons (d)    - Spin 1, Bound np state, Mass 1876 MeV                               │
│ 🟠 Alpha particles  - Spin 0, 4He nucleus, Mass 3728 MeV                                  │
│ 🔮 Lambda (Λ)       - Spin 1/2, Strange baryon, Mass 1116 MeV                             │
│ 🌸 Heavy fragments  - Multi-nucleon clusters, A > 4                                        │
│ ⭐ Gamma rays (γ)    - Photons from nuclear de-excitation                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

🔬 ADVANCED PHYSICS FEATURES:
• First principles QCD field theory
• Nuclear shell model with magic numbers  
• Chiral effective field theory
• Momentum vector visualization
• Time stepping through collision evolution
• Distributed computing on multiple cores
• Real-time thermodynamic analysis
"""
        
        self.text_widget.insert('1.0', header)
        
        # Configure text colors
        self.text_widget.tag_configure("header", foreground="#00FFFF", font=('Courier New', 12, 'bold'))
        self.text_widget.tag_configure("physics", foreground="#FFFF00")
        self.text_widget.tag_configure("particles", foreground="#FF8888")
        self.text_widget.tag_configure("energy", foreground="#88FF88")
        
        print("⚠️  Using enhanced text display with complete physics analysis")
    
    def update_with_time_stepping(self, simulation_data: Dict, time_index: int = -1):
        """Update visualization with time stepping capability."""
        
        self.simulation_data = simulation_data
        
        if 'time_history' in simulation_data:
            self.time_history = simulation_data['time_history']
            
            # Use specified time index or latest
            if time_index >= 0 and time_index < len(self.time_history):
                self.current_time_index = time_index
            else:
                self.current_time_index = len(self.time_history) - 1
        
        try:
            if MATPLOTLIB_AVAILABLE and hasattr(self, 'fig'):
                self._update_advanced_matplotlib()
            else:
                self._update_enhanced_text()
        except Exception as e:
            print(f"Visualization update error: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_advanced_matplotlib(self):
        """Update matplotlib with full physics visualization."""
        
        # Clear 3D plot
        self.ax_3d.clear()
        self.ax_3d.set_facecolor('#2e2e2e')
        
        # Enhanced 3D plot title with current time
        current_time = 0.0
        if self.time_history and self.current_time_index < len(self.time_history):
            current_time = self.time_history[self.current_time_index]['time']
        
        self.ax_3d.set_title(
            f'3D Nuclear Collision with Momentum Vectors\n'
            f'Time: {current_time:.3f} fm/c', 
            fontsize=16, fontweight='bold', color='white', pad=20
        )
        
        # Set axis labels and limits
        self.ax_3d.set_xlabel('X (fm)', fontsize=12, color='white')
        self.ax_3d.set_ylabel('Y (fm)', fontsize=12, color='white')
        self.ax_3d.set_zlabel('Z (fm)', fontsize=12, color='white')
        
        # Dynamic limits based on particle positions
        if self.time_history and self.current_time_index < len(self.time_history):
            current_state = self.time_history[self.current_time_index]
            
            if 'particles' in current_state:
                particles = current_state['particles']
                
                if particles:
                    # Get position ranges
                    positions = np.array([p['position'] for p in particles])
                    
                    x_range = [positions[:, 0].min() - 5, positions[:, 0].max() + 5]
                    y_range = [positions[:, 1].min() - 5, positions[:, 1].max() + 5]
                    z_range = [positions[:, 2].min() - 5, positions[:, 2].max() + 5]
                    
                    # Ensure reasonable limits
                    x_range = [max(-30, x_range[0]), min(30, x_range[1])]
                    y_range = [max(-30, y_range[0]), min(30, y_range[1])]
                    z_range = [max(-30, z_range[0]), min(30, z_range[1])]
                    
                    self.ax_3d.set_xlim(x_range)
                    self.ax_3d.set_ylim(y_range)
                    self.ax_3d.set_zlim(z_range)
                    
                    # Plot particles with momentum vectors
                    self._plot_particles_with_momentum(particles)
                else:
                    # Default limits if no particles
                    self.ax_3d.set_xlim([-25, 25])
                    self.ax_3d.set_ylim([-25, 25])
                    self.ax_3d.set_zlim([-25, 25])
        
        # Style 3D plot
        self.ax_3d.xaxis.pane.fill = False
        self.ax_3d.yaxis.pane.fill = False
        self.ax_3d.zaxis.pane.fill = False
        self.ax_3d.tick_params(colors='white')
        
        # Update physics plots with real-time values
        self._update_physics_plots_with_values()
        
        # Add comprehensive legend
        self._add_comprehensive_legend()
    
    def _plot_particles_with_momentum(self, particles: List[Dict]):
        """Plot particles with momentum vectors and proper physics labels."""
        
        # Group particles by type for better visualization
        particle_groups = {}
        
        for particle in particles:
            ptype = particle.get('type', 'unknown')
            
            # Map particle types to our enhanced categories
            if ptype in ['proton', 'neutron']:
                group_key = ptype
            elif 'pion' in ptype:
                group_key = 'pion_plus'  # Simplified for now
            elif 'kaon' in ptype:
                group_key = 'kaon_plus'  # Simplified for now
            elif ptype in ['deuteron', 'alpha', 'lambda']:
                group_key = ptype
            else:
                group_key = 'fragment'
            
            if group_key not in particle_groups:
                particle_groups[group_key] = []
            
            particle_groups[group_key].append(particle)
        
        # Plot each particle group
        for group_key, group_particles in particle_groups.items():
            
            if group_key not in self.particle_properties:
                continue
            
            props = self.particle_properties[group_key]
            
            # Extract positions and momenta
            positions = np.array([p['position'] for p in group_particles])
            momenta = np.array([p.get('momentum', [0, 0, 0]) for p in group_particles])
            energies = np.array([p.get('energy', 0) for p in group_particles])
            
            if len(positions) == 0:
                continue
            
            # Plot particles
            scatter = self.ax_3d.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=props['color'],
                s=props['size'],
                marker=props['marker'],
                alpha=props['alpha'],
                label=f"{props['label']} (n={len(positions)})",
                edgecolors='white',
                linewidth=0.5
            )
            
            # Plot momentum vectors
            self._plot_momentum_vectors(positions, momenta, energies, props['color'])
        
        # Add collision axis and interaction region
        self._add_collision_geometry()
    
    def _plot_momentum_vectors(self, positions: np.ndarray, momenta: np.ndarray, 
                             energies: np.ndarray, color: str):
        """Plot momentum vectors with proper scaling."""
        
        if len(positions) == 0:
            return
        
        # Scale momentum vectors for visibility
        momentum_scale = 5.0  # fm per GeV/c
        
        for pos, mom, energy in zip(positions, momenta, energies):
            
            if np.linalg.norm(mom) < 0.01:  # Skip very small momenta
                continue
            
            # Scale momentum vector
            mom_scaled = mom * momentum_scale
            
            # Color based on kinetic energy
            if energy > 1.0:  # High energy
                arrow_color = '#FF4444'
                arrow_alpha = 0.8
            elif energy > 0.1:  # Medium energy
                arrow_color = '#FFFF44'
                arrow_alpha = 0.6
            else:  # Low energy
                arrow_color = '#4444FF'
                arrow_alpha = 0.4
            
            # Draw momentum vector as arrow
            self.ax_3d.quiver(
                pos[0], pos[1], pos[2],
                mom_scaled[0], mom_scaled[1], mom_scaled[2],
                color=arrow_color,
                alpha=arrow_alpha,
                arrow_length_ratio=0.1,
                linewidth=1.5
            )
    
    def _add_collision_geometry(self):
        """Add collision axis and interaction region indicators."""
        
        # Collision axis (z-axis)
        axis_points = np.array([[-25, 0, 0], [25, 0, 0]])
        self.ax_3d.plot(
            axis_points[:, 0], axis_points[:, 1], axis_points[:, 2],
            'w--', alpha=0.3, linewidth=2, label='Collision axis'
        )
    
    def _update_physics_plots_with_values(self):
        """Update physics plots with real-time value displays."""
        
        if not self.simulation_data or 'global_observables' not in self.simulation_data:
            return
        
        obs = self.simulation_data['global_observables']
        
        # Clear previous value text boxes
        for text_obj in self.value_texts.values():
            if hasattr(text_obj, 'remove'):
                text_obj.remove()
        self.value_texts.clear()
        
        # Energy density plot
        self._update_single_physics_plot(
            self.ax_energy, 
            obs.get('time', []), 
            obs.get('energy_density', []),
            'Energy Density Evolution',
            'Time (fm/c)',
            'Energy Density (GeV/fm³)',
            '#FF6B6B',
            'energy_density'
        )
        
        # Temperature plot with phase transitions
        self._update_temperature_plot(obs)
        
        # Pressure plot
        self._update_single_physics_plot(
            self.ax_pressure,
            obs.get('time', []),
            obs.get('pressure', []),
            'Pressure Evolution',
            'Time (fm/c)', 
            'Pressure (GeV/fm³)',
            '#4ECDC4',
            'pressure'
        )
        
        # Entropy plot
        self._update_single_physics_plot(
            self.ax_entropy,
            obs.get('time', []),
            obs.get('entropy_density', []),
            'Entropy Density',
            'Time (fm/c)',
            'Entropy Density',
            '#45B7D1',
            'entropy_density'
        )
    
    def _update_single_physics_plot(self, ax, times, values, title, xlabel, ylabel, 
                                   color, value_key):
        """Update a single physics plot with current value display."""
        
        ax.clear()
        ax.set_facecolor('#2e2e2e')
        ax.grid(True, alpha=0.3, color='white')
        ax.tick_params(colors='white')
        
        if not times or not values:
            ax.set_title(title, fontsize=12, fontweight='bold', color='white')
            ax.set_xlabel(xlabel, fontsize=10, color='white')
            ax.set_ylabel(ylabel, fontsize=10, color='white')
            return
        
        # Plot data
        ax.plot(times, values, color=color, linewidth=2, alpha=0.8)
        
        # Highlight current time point
        if self.current_time_index < len(times):
            current_time = times[self.current_time_index] if self.current_time_index < len(times) else times[-1]
            current_value = values[self.current_time_index] if self.current_time_index < len(values) else values[-1]
            
            ax.scatter([current_time], [current_value], 
                      color='white', s=100, zorder=5, edgecolor=color, linewidth=2)
            
            # Add current value text box
            value_text = f'{current_value:.3e}' if abs(current_value) < 0.01 else f'{current_value:.3f}'
            
            try:
                text_obj = ax.text(
                    0.02, 0.98, f'Current: {value_text}',
                    transform=ax.transAxes,
                    fontsize=11, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                    verticalalignment='top'
                )
                
                self.value_texts[value_key] = text_obj
            except:
                pass  # Ignore text creation errors
        
        # Enhanced titles and labels
        ax.set_title(title, fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel(xlabel, fontsize=10, color='white')
        ax.set_ylabel(ylabel, fontsize=10, color='white')
        
        # Style axes
        for spine in ax.spines.values():
            spine.set_color('white')
    
    def _update_temperature_plot(self, obs):
        """Update temperature plot with phase transition markers."""
        
        times = obs.get('time', [])
        temperatures = obs.get('temperature', [])
        
        self.ax_temp.clear()
        self.ax_temp.set_facecolor('#2e2e2e')
        self.ax_temp.grid(True, alpha=0.3, color='white')
        self.ax_temp.tick_params(colors='white')
        
        if not times or not temperatures:
            self.ax_temp.set_title('Temperature & Phase Transitions', fontsize=12, fontweight='bold', color='white')
            return
        
        # Plot temperature
        self.ax_temp.plot(times, temperatures, color='#FF4444', linewidth=2, alpha=0.8, label='Temperature')
        
        # Phase transition lines
        max_temp = max(temperatures) if temperatures else 200
        
        if max_temp > 170:
            self.ax_temp.axhline(y=170, color='orange', linestyle='--', alpha=0.7, 
                               label='QGP Transition (170 MeV)')
        if max_temp > 140:
            self.ax_temp.axhline(y=140, color='yellow', linestyle=':', alpha=0.7,
                               label='Chiral Transition (140 MeV)')
        if max_temp > 100:
            self.ax_temp.axhline(y=100, color='lightblue', linestyle='-.', alpha=0.7,
                               label='Hadron Gas (100 MeV)')
        
        # Current temperature point and value
        if self.current_time_index < len(times):
            current_time = times[self.current_time_index] if self.current_time_index < len(times) else times[-1]
            current_temp = temperatures[self.current_time_index] if self.current_time_index < len(temperatures) else temperatures[-1]
            
            self.ax_temp.scatter([current_time], [current_temp], 
                              color='white', s=100, zorder=5, edgecolor='#FF4444', linewidth=2)
            
            # Current temperature with phase identification
            phase = self._identify_phase(current_temp)
            
            try:
                text_obj = self.ax_temp.text(
                    0.02, 0.98, f'T: {current_temp:.1f} MeV\n{phase}',
                    transform=self.ax_temp.transAxes,
                    fontsize=11, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF4444', alpha=0.7),
                    verticalalignment='top'
                )
                
                self.value_texts['temperature'] = text_obj
            except:
                pass
        
        self.ax_temp.set_title('Temperature & Phase Transitions', fontsize=12, fontweight='bold', color='white')
        self.ax_temp.set_xlabel('Time (fm/c)', fontsize=10, color='white')
        self.ax_temp.set_ylabel('Temperature (MeV)', fontsize=10, color='white')
        
        try:
            self.ax_temp.legend(fontsize=8, facecolor='#2e2e2e', edgecolor='white')
        except:
            pass
        
        # Style axes
        for spine in self.ax_temp.spines.values():
            spine.set_color('white')
    
    def _identify_phase(self, temperature: float) -> str:
        """Identify thermodynamic phase based on temperature."""
        
        if temperature > 170:
            return "🔥 QGP Phase"
        elif temperature > 140:
            return "🌡️ Mixed Phase"
        elif temperature > 100:
            return "⚡ Hot Hadronic"
        elif temperature > 50:
            return "🎯 Warm Nuclear"
        elif temperature > 10:
            return "❄️ Cold Nuclear"
        else:
            return "🧊 Frozen Nuclear"
    
    def _add_comprehensive_legend(self):
        """Add comprehensive legend with physics information."""
        
        # Create legend with all particle types present
        try:
            handles, labels = self.ax_3d.get_legend_handles_labels()
            
            if handles:
                legend = self.ax_3d.legend(
                    handles, labels,
                    bbox_to_anchor=(1.05, 1), loc='upper left',
                    fontsize=10, facecolor='#2e2e2e', edgecolor='white',
                    framealpha=0.9
                )
                
                # Style legend text
                for text in legend.get_texts():
                    text.set_color('white')
                
                legend.set_title("Nuclear Products & Physics", 
                               prop={'weight': 'bold', 'size': 12, 'color': 'white'})
        except:
            pass  # Ignore legend errors
    
    def _update_enhanced_text(self):
        """Update text display with complete physics analysis."""
        
        if not self.simulation_data:
            return
        
        # Get current state
        current_state = None
        current_time = 0.0
        
        if self.time_history and self.current_time_index < len(self.time_history):
            current_state = self.time_history[self.current_time_index]
            current_time = current_state['time']
        
        # Get global observables
        obs = self.simulation_data.get('global_observables', {})
        
        # Build comprehensive status
        status = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                    🚀 FIRST PRINCIPLES NUCLEAR COLLISION ANALYSIS                           ║
║                          Time: {current_time:8.3f} fm/c                                         ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

🎯 COLLISION SYSTEM STATUS:
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Simulation Time: {current_time:8.3f} fm/c                                                    │
│ Time Step Index: {self.current_time_index:5d} / {len(self.time_history):5d}                           │
│ Physics Engine: First Principles QCD + Nuclear Theory                                      │
│ Computing Mode: Distributed Multi-Core Processing                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

"""
        
        # Current thermodynamics
        if obs.get('temperature') and self.current_time_index < len(obs['temperature']):
            temp = obs['temperature'][self.current_time_index] if self.current_time_index < len(obs['temperature']) else obs['temperature'][-1]
            energy = obs['energy_density'][self.current_time_index] if self.current_time_index < len(obs['energy_density']) else obs['energy_density'][-1]
            pressure = obs['pressure'][self.current_time_index] if self.current_time_index < len(obs['pressure']) else obs['pressure'][-1]
            
            phase = self._identify_phase(temp)
            
            status += f"""🌡️ THERMODYNAMIC STATE:
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Temperature:      {temp:8.3f} MeV                                                       │
│ Energy Density:   {energy:.3e} GeV/fm³                                                │
│ Pressure:         {pressure:.3e} GeV/fm³                                              │
│ Current Phase:    {phase}                                                     │
│                                                                                             │
│ Phase Transitions:                                                                          │
│   🔥 QGP Formation: {'✅ ACTIVE' if temp > 170 else '❌ INACTIVE'} (Tc = 170 MeV)                           │
│   🌡️ Chiral Restoration: {'✅ ACTIVE' if temp > 140 else '❌ INACTIVE'} (Tχ = 140 MeV)                    │
│   ⚡ Hadronization: {'✅ ACTIVE' if temp < 170 and temp > 100 else '❌ INACTIVE'} (100-170 MeV)               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

"""
        
        # Particle analysis
        if current_state and 'particles' in current_state:
            particles = current_state['particles']
            
            # Count particles by type
            particle_counts = {}
            total_momentum = np.zeros(3)
            total_energy = 0.0
            
            for particle in particles:
                ptype = particle.get('type', 'unknown')
                particle_counts[ptype] = particle_counts.get(ptype, 0) + 1
                
                momentum = np.array(particle.get('momentum', [0, 0, 0]))
                total_momentum += momentum
                total_energy += particle.get('energy', 0)
            
            status += f"""🎯 PARTICLE INVENTORY & MOMENTUM ANALYSIS:
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Total Particles: {len(particles):5d}                                                              │
│ Total Energy:    {total_energy:8.3f} GeV                                                    │
│ Total Momentum:  ({total_momentum[0]:6.3f}, {total_momentum[1]:6.3f}, {total_momentum[2]:6.3f}) GeV/c          │
│                                                                                             │
│ PARTICLE BREAKDOWN:                                                                         │
"""
            
            # Detailed particle breakdown
            for ptype, count in sorted(particle_counts.items()):
                if ptype in self.particle_properties:
                    props = self.particle_properties[ptype]
                    status += f"│   {props['label']:20s}: {count:4d} particles (Mass: {props['mass']:6.3f} GeV)   │\n"
                else:
                    status += f"│   {ptype:20s}: {count:4d} particles                                    │\n"
            
            status += """└─────────────────────────────────────────────────────────────────────────────────────────────┘

"""
        
        # Update text display
        self.text_widget.delete('1.0', tk.END)
        self.text_widget.insert('1.0', status)
        self.text_widget.see(tk.END)

class LowEnergyStatusDisplay:
    """Enhanced status display for low energy nuclear physics."""
    
    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.create_status_display()
    
    def create_status_display(self):
        """Create enhanced status display for low energy physics."""
        
        # Main status frame with proper scrolling
        main_frame = tk.Frame(self.parent)
        main_frame.pack(fill='both', expand=True)
        
        # Configure for proper scaling
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Text widget with enhanced formatting
        self.status_text = tk.Text(
            main_frame,
            bg='#1a1a2e', fg='#eee8d5',
            font=('Consolas', 11),
            wrap='word',
            insertbackground='white',
            state='normal'
        )
        
        # Scrollbar that properly scales
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # Grid with proper scaling - THIS IS THE KEY FIX
        self.status_text.grid(row=0, column=0, sticky='nsew', padx=(5,0), pady=5)
        scrollbar.grid(row=0, column=1, sticky='ns', padx=(0,5), pady=5)
        
        # Configure text tags for syntax highlighting
        self.status_text.tag_configure("header", foreground="#cb4b16", font=('Consolas', 12, 'bold'))
        self.status_text.tag_configure("energy", foreground="#859900", font=('Consolas', 11, 'bold'))
        self.status_text.tag_configure("particle", foreground="#268bd2")
        self.status_text.tag_configure("physics", foreground="#d33682")
        self.status_text.tag_configure("time", foreground="#2aa198", font=('Consolas', 11, 'bold'))
        
        # Initial display
        self.show_initial_status()
    
    def show_initial_status(self):
        """Show initial status with low energy physics information."""
        
        initial_text = """
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              🔬 ULTRA-LOW ENERGY NUCLEAR PHYSICS STATUS                              ║
║                                    Advanced Shell Model Analysis                                     ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝

🎯 LOW ENERGY NUCLEAR OUTPUTS & LABELS:

┌───────────────────── NUCLEAR STRUCTURE OBSERVABLES ─────────────────────┐
│                                                                         │
│ 🏗️ Shell Model Effects:                                                │
│   • Magic Number Enhancement: [N,Z] = [2,8,20,28,50,82,126,184]        │
│   • Shell Gap Energies: 2-10 MeV per shell closure                     │
│   • Pairing Correlations: δ = 11.18/√A MeV                            │
│   • Deformation Parameters: β₂, β₄ for non-spherical nuclei            │
│                                                                         │
│ 🔄 Nuclear Reactions (< 100 MeV):                                      │
│   • Elastic Scattering: σ_el vs angle and energy                      │
│   • Inelastic Excitation: E* → γ-ray cascades                         │
│   • Transfer Reactions: (d,p), (d,n), (³He,d), etc.                   │
│   • Compound Nucleus Formation: σ_compound vs E_excitation             │
│                                                                         │
│ ⚛️ Nuclear Decay Modes:                                                │
│   • Alpha Decay: Q_α, t₁/₂, penetration probability                    │
│   • Beta Decay: Q_β, ft values, neutrino spectrum                      │
│   • Gamma Transitions: E_γ, I_γ, multipolarity (E1,M1,E2,...)        │
│   • Internal Conversion: α_K, electron spectrum                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

🔬 START LOW ENERGY SIMULATION TO SEE DETAILED NUCLEAR ANALYSIS

"""
        
        self.status_text.insert('1.0', initial_text)
        
        # Apply syntax highlighting
        self._apply_text_highlighting()
    
    def update_low_energy_status(self, simulation_data, time_index=-1):
        """Update status with detailed low energy nuclear physics."""
        
        # Get current state
        current_state = None
        current_time = 0.0
        
        if 'time_history' in simulation_data and simulation_data['time_history']:
            time_history = simulation_data['time_history']
            if time_index >= 0 and time_index < len(time_history):
                current_state = time_history[time_index]
            else:
                current_state = time_history[-1]
            current_time = current_state['time']
        
        # Build comprehensive low energy status
        status = f"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              🔬 LOW ENERGY NUCLEAR COLLISION ANALYSIS                               ║
║                                 Time: {current_time:8.3f} fm/c                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝

"""
        
        # Extract observables
        obs = simulation_data.get('global_observables', {})
        
        if obs.get('temperature'):
            temp_index = min(time_index if time_index >= 0 else len(obs['temperature'])-1, len(obs['temperature'])-1)
            temp = obs['temperature'][temp_index]
            energy = obs['energy_density'][temp_index] if temp_index < len(obs['energy_density']) else 0
            
            status += f"""🌡️ NUCLEAR THERMODYNAMICS:
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Nuclear Temperature:    {temp:8.3f} MeV                                                            │
│ Excitation Energy:      {energy*1000:8.3f} MeV/nucleon                                             │
│ Thermal Motion:         √(3kT/m) = {np.sqrt(3*temp/938.3)*3e8*1e-6:6.2f} km/s                     │
│                                                                                                     │
│ Shell Model Status:                                                                                 │
"""
            
            # Shell model analysis
            if temp < 2.0:
                status += "│   🧊 GROUND STATE: Shell structure intact, discrete levels                                     │\n"
                status += "│   🏗️ Magic Numbers: Enhanced stability at closed shells                                        │\n"
                status += "│   ⚛️ Pairing: Strong correlations in even-even nuclei                                         │\n"
            elif temp < 5.0:
                status += "│   🌡️ LOW EXCITATION: Collective vibrations, rotational bands                                   │\n"
                status += "│   📊 Level Density: Discrete states → quasi-continuum                                         │\n"
                status += "│   🔄 Shape Changes: Quadrupole deformation effects                                            │\n"
            else:
                status += "│   🔥 HOT NUCLEAR MATTER: Shell structure melting                                               │\n"
                status += "│   💥 Compound Nucleus: Statistical decay modes dominate                                       │\n"
                status += "│   ⚡ Pre-equilibrium: Direct knockout processes                                               │\n"
            
            status += "└─────────────────────────────────────────────────────────────────────────────────────────────────────┘\n\n"
        
        # Update display
        self.status_text.delete('1.0', tk.END)
        self.status_text.insert('1.0', status)
        
        # Apply syntax highlighting
        self._apply_text_highlighting()
        
        # Auto-scroll to bottom
        self.status_text.see(tk.END)
    
    def _apply_text_highlighting(self):
        """Apply syntax highlighting to the text."""
        content = self.status_text.get('1.0', tk.END)
        
        # Clear existing tags
        for tag in ['header', 'energy', 'particle', 'physics', 'time']:
            self.status_text.tag_delete(tag)
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_start = f"{i+1}.0"
            line_end = f"{i+1}.end"
            
            # Header lines
            if '╔' in line or '║' in line and ('ANALYSIS' in line or 'STATUS' in line):
                self.status_text.tag_add("header", line_start, line_end)
            
            # Energy values
            elif 'MeV' in line or 'GeV' in line or 'Energy' in line:
                self.status_text.tag_add("energy", line_start, line_end)
            
            # Particle names
            elif any(particle in line for particle in ['proton', 'neutron', 'alpha', 'fragment']):
                self.status_text.tag_add("particle", line_start, line_end)
            
            # Physics terms
            elif any(term in line for term in ['Shell', 'Pairing', 'Magic', 'Transition']):
                self.status_text.tag_add("physics", line_start, line_end)
            
            # Time information
            elif 'Time:' in line or 'fm/c' in line:
                self.status_text.tag_add("time", line_start, line_end)

# Export classes
__all__ = ['AdvancedVisualizerWithMomentum', 'LowEnergyStatusDisplay']
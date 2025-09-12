"""
Enhanced GUI with comprehensive simulation controls including low-energy collisions.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    # Try to import modern 3D visualization libraries
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class ParticleVisualizer:
    """3D particle and nuclear collision visualization."""
    
    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.particles = []
        self.nuclei_positions = []
        
        if MATPLOTLIB_AVAILABLE:
            self.setup_matplotlib_3d()
        else:
            self.setup_text_display()
    
    def setup_matplotlib_3d(self):
        """Setup 3D visualization with matplotlib."""
        self.fig = Figure(figsize=(12, 8))
        
        # Create 3D subplot
        self.ax_3d = self.fig.add_subplot(221, projection='3d')
        self.ax_3d.set_title('3D Nuclear Collision')
        self.ax_3d.set_xlabel('X (fm)')
        self.ax_3d.set_ylabel('Y (fm)')
        self.ax_3d.set_zlabel('Z (fm)')
        
        # Physics plots
        self.ax_energy = self.fig.add_subplot(222)
        self.ax_energy.set_title('Energy Density')
        self.ax_energy.set_xlabel('Time (fm/c)')
        
        self.ax_temp = self.fig.add_subplot(223)
        self.ax_temp.set_title('Temperature')
        self.ax_temp.set_xlabel('Time (fm/c)')
        self.ax_temp.set_ylabel('T (MeV)')
        
        self.ax_particles = self.fig.add_subplot(224)
        self.ax_particles.set_title('Particle Production')
        self.ax_particles.set_xlabel('Time (fm/c)')
        self.ax_particles.set_ylabel('Multiplicity')
        
        self.fig.tight_layout()
        
        print("✅ Matplotlib 3D visualization initialized")
    
    def setup_text_display(self):
        """Fallback text display if no 3D libraries available."""
        self.text_widget = tk.Text(self.parent, height=20, width=80,
                                  bg='black', fg='green', font=('Courier', 10))
        self.text_widget.pack(fill='both', expand=True)
        self.text_widget.insert('1.0', "3D Visualization\n" + "="*50 + "\n\n")
        print("⚠️  Using text display (install matplotlib for 3D)")
    
    def update_collision_state(self, simulator):
        """Update visualization with current collision state."""
        try:
            if MATPLOTLIB_AVAILABLE:
                self._update_matplotlib(simulator)
            else:
                self._update_text(simulator)
        except Exception as e:
            print(f"Visualization update error: {e}")
    
    def _update_matplotlib(self, simulator):
        """Update matplotlib 3D visualization."""
        # Clear and update 3D plot
        self.ax_3d.clear()
        self.ax_3d.set_title('3D Nuclear Collision')
        self.ax_3d.set_xlabel('X (fm)')
        self.ax_3d.set_ylabel('Y (fm)')
        self.ax_3d.set_zlabel('Z (fm)')
        
        # Generate positions
        time_factor = simulator.current_time / 10.0
        
        nucleus_a_pos = self._generate_nucleus_positions(
            simulator.nucleus_A, offset_z=-10 + time_factor*5
        )
        nucleus_b_pos = self._generate_nucleus_positions(
            simulator.nucleus_B, offset_z=10 - time_factor*5
        )
        
        # Plot nuclei
        self.ax_3d.scatter(nucleus_a_pos[0], nucleus_a_pos[1], nucleus_a_pos[2],
                          c='red', s=50, alpha=0.7, label='Nucleus A')
        self.ax_3d.scatter(nucleus_b_pos[0], nucleus_b_pos[1], nucleus_b_pos[2], 
                          c='blue', s=50, alpha=0.7, label='Nucleus B')
        
        # Add produced particles
        if simulator.current_time > 2.0:
            particles = self._generate_produced_particles(simulator)
            self.ax_3d.scatter(particles[0], particles[1], particles[2],
                             c='yellow', s=10, alpha=0.5, label='Products')
        
        self.ax_3d.legend()
        self.ax_3d.set_xlim([-15, 15])
        self.ax_3d.set_ylim([-15, 15])  
        self.ax_3d.set_zlim([-15, 15])
        
        # Update physics plots
        obs = simulator.observables
        if obs['time']:
            self.ax_energy.clear()
            self.ax_energy.plot(obs['time'], obs['energy_density'], 'b-')
            self.ax_energy.set_title('Energy Density')
            self.ax_energy.set_xlabel('Time (fm/c)')
            
            self.ax_temp.clear()
            self.ax_temp.plot(obs['time'], obs['temperature'], 'r-')
            self.ax_temp.axhline(y=170, color='orange', linestyle='--', alpha=0.7, label='QGP threshold')
            self.ax_temp.axhline(y=140, color='yellow', linestyle=':', alpha=0.7, label='Pion threshold')
            self.ax_temp.set_title('Temperature')
            self.ax_temp.set_xlabel('Time (fm/c)')
            self.ax_temp.set_ylabel('T (MeV)')
            self.ax_temp.legend()
            
            self.ax_particles.clear()
            self.ax_particles.plot(obs['time'], obs['multiplicity'], 'g-')
            self.ax_particles.set_title('Particle Multiplicity')
            self.ax_particles.set_xlabel('Time (fm/c)')
    
    def _update_text(self, simulator):
        """Update text display."""
        status = f"""
3D Nuclear Collision Visualization
{'='*50}

Time: {simulator.current_time:.2f} fm/c
Iteration: {simulator.iteration}

Nuclear System: {simulator.params.nucleus_A} + {simulator.params.nucleus_B}
Collision Energy: {simulator.params.collision_energy_gev} GeV ({simulator.params.collision_energy_gev*1000:.0f} MeV)

Current State:
"""
        
        if simulator.observables['temperature']:
            temp = simulator.observables['temperature'][-1]
            energy = simulator.observables['energy_density'][-1]
            mult = simulator.observables['multiplicity'][-1]
            
            status += f"""  Temperature: {temp:.1f} MeV
  Energy Density: {energy:.2e}
  Particle Multiplicity: {mult:.0f}
  
Phase: {"🔥 QGP" if temp > 170 else "🌡️ Hadronic" if temp > 140 else "❄️ Nuclear"}

Physics Regime:
"""
            if simulator.params.collision_energy_gev < 1.0:
                status += "  🔬 Low-energy nuclear physics (fragmentation, transparency)\n"
            elif simulator.params.collision_energy_gev < 10.0:
                status += "  🎯 Medium-energy (pion production, resonances)\n"
            elif simulator.params.collision_energy_gev < 100.0:
                status += "  ⚡ High-energy (strangeness production)\n"
            else:
                status += "  🌟 Ultra-relativistic (QGP formation)\n"
            
            status += f"""
Nuclei Positions:
  Nucleus A: Moving from left (Z = {-10 + simulator.current_time/2:.1f} fm)
  Nucleus B: Moving from right (Z = {10 - simulator.current_time/2:.1f} fm)
"""
            
            if simulator.current_time > 2.0:
                status += f"\n🎆 Particle production active!"
                if simulator.params.collision_energy_gev < 1.0:
                    status += f"\n   Nucleons: ~{mult*0.9:.0f}"
                    status += f"\n   Light fragments: ~{mult*0.1:.0f}"
                else:
                    status += f"\n   Pions: ~{mult*0.8:.0f}"
                    status += f"\n   Kaons: ~{mult*0.15:.0f}" 
                    status += f"\n   Protons: ~{mult*0.05:.0f}"
        
        self.text_widget.delete('1.0', tk.END)
        self.text_widget.insert('1.0', status)
    
    def _generate_nucleus_positions(self, nucleus, offset_z=0):
        """Generate 3D positions for nucleons in nucleus."""
        positions = nucleus.generate_nucleon_positions(min(nucleus.A, 50))
        
        if not positions:
            return [0], [0], [offset_z]
        
        x_pos = [pos[0] for pos in positions]
        y_pos = [pos[1] for pos in positions] 
        z_pos = [pos[2] + offset_z for pos in positions]
        
        return x_pos, y_pos, z_pos
    
    def _generate_produced_particles(self, simulator):
        """Generate positions of produced particles."""
        if not simulator.observables['multiplicity']:
            return [], [], []
        
        mult = int(simulator.observables['multiplicity'][-1] / 10)  # Scale down for display
        expansion_radius = (simulator.current_time - 2.0) * 2  # Expanding fireball
        
        # Generate particles in expanding sphere
        x_particles = []
        y_particles = []
        z_particles = []
        
        for _ in range(min(mult, 200)):  # Limit for performance
            # Random position in expanding sphere
            r = expansion_radius * np.random.random()**(1/3)
            theta = np.arccos(2 * np.random.random() - 1)
            phi = 2 * np.pi * np.random.random()
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            x_particles.append(x)
            y_particles.append(y)
            z_particles.append(z)
        
        return x_particles, y_particles, z_particles

class SimulatorGUI:
    """Enhanced GUI for quantum lattice simulator with comprehensive controls including low energies."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Quantum Lattice Nuclear Collision Simulator v2.0")
        self.root.geometry("1500x1100")
        self.root.configure(bg='#2c3e50')
        
        # Simulation state
        self.simulator = None
        self.simulation_thread = None
        self.is_running = False
        
        self.create_interface()
        
    def create_interface(self):
        """Create the main interface."""
        
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.setup_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.setup_tab, text="🚀 Setup & Control")
        self.notebook.add(self.visualization_tab, text="🎆 3D Visualization")
        self.notebook.add(self.analysis_tab, text="📊 Analysis")
        
        self.create_setup_tab()
        self.create_visualization_tab()
        self.create_analysis_tab()
    
    def create_setup_tab(self):
        """Create enhanced simulation setup and control tab."""
        
        # Main container with scrollbar
        main_canvas = tk.Canvas(self.setup_tab, bg='#2c3e50')
        scrollbar = ttk.Scrollbar(self.setup_tab, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title
        title = tk.Label(scrollable_frame,
                        text="Quantum Lattice Nuclear Collision Simulator v2.0",
                        font=('Arial', 16, 'bold'), bg='#34495e', fg='white')
        title.pack(pady=10, fill='x')
        
        # Nuclear Parameters Frame
        nuclear_frame = ttk.LabelFrame(scrollable_frame, text="🔬 Nuclear Parameters")
        nuclear_frame.pack(fill='x', padx=20, pady=10)
        
        # Import here to avoid issues
        try:
            from ..physics.nuclear import NuclearDatabase
        except ImportError:
            # Fallback
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from physics.nuclear import NuclearDatabase
        
        # Nuclear selection
        tk.Label(nuclear_frame, text="Projectile:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.nucleus_a_var = tk.StringVar(value="Au197")
        nucleus_a_combo = ttk.Combobox(nuclear_frame, textvariable=self.nucleus_a_var,
                                      values=NuclearDatabase.get_available_nuclei(), width=15)
        nucleus_a_combo.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(nuclear_frame, text="Target:").grid(row=0, column=2, sticky='w', padx=5, pady=5)
        self.nucleus_b_var = tk.StringVar(value="Au197")
        nucleus_b_combo = ttk.Combobox(nuclear_frame, textvariable=self.nucleus_b_var,
                                      values=NuclearDatabase.get_available_nuclei(), width=15)
        nucleus_b_combo.grid(row=0, column=3, padx=5, pady=5)
        
        # Impact parameter
        tk.Label(nuclear_frame, text="Impact Parameter (fm):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.impact_var = tk.DoubleVar(value=5.0)
        impact_scale = tk.Scale(nuclear_frame, from_=0.0, to=15.0, orient='horizontal',
                               variable=self.impact_var, resolution=0.1, length=200)
        impact_scale.grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        
        # Collision Parameters Frame
        collision_frame = ttk.LabelFrame(scrollable_frame, text="⚡ Collision Parameters")
        collision_frame.pack(fill='x', padx=20, pady=10)
        
        # Energy with extended low-energy range
        tk.Label(collision_frame, text="Collision Energy:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        # Energy input methods
        energy_method_frame = tk.Frame(collision_frame)
        energy_method_frame.grid(row=0, column=1, columnspan=3, sticky='w', padx=5, pady=5)
        
        self.energy_unit_var = tk.StringVar(value="GeV")
        unit_menu = ttk.Combobox(energy_method_frame, textvariable=self.energy_unit_var,
                                values=["MeV", "GeV"], width=5, state="readonly")
        unit_menu.pack(side='right', padx=5)
        
        self.energy_var = tk.DoubleVar(value=2.0)  # Start at 2 GeV
        self.energy_entry = tk.Entry(energy_method_frame, textvariable=self.energy_var, width=10)
        self.energy_entry.pack(side='right', padx=5)
        
        # Bind unit change to update energy display
        unit_menu.bind('<<ComboboxSelected>>', self.on_unit_change)
        
        # Energy slider (in GeV, from 0.4 to 5000)
        tk.Label(collision_frame, text="Energy Slider:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.energy_scale = tk.Scale(collision_frame, from_=0.4, to=5000.0, orient='horizontal',
                                    variable=self.energy_var, resolution=0.1, length=350,
                                    command=self.update_energy_display)
        self.energy_scale.grid(row=1, column=1, columnspan=3, padx=5, pady=5)
        
        # Energy display
        self.energy_display_var = tk.StringVar()
        self.energy_display = tk.Label(collision_frame, textvariable=self.energy_display_var,
                                      font=('Arial', 10, 'bold'), fg='blue')
        self.energy_display.grid(row=2, column=1, columnspan=3, pady=5)
        self.update_energy_display()
        
        # Extended preset energies including low energies
        tk.Label(collision_frame, text="Energy Presets:").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        preset_frame = tk.Frame(collision_frame)
        preset_frame.grid(row=3, column=1, columnspan=3, sticky='w', padx=5, pady=5)
        
        # Low energy presets
        low_energy_frame = tk.Frame(preset_frame)
        low_energy_frame.pack(fill='x', pady=2)
        tk.Label(low_energy_frame, text="Low Energy:", font=('Arial', 9, 'bold')).pack(side='left')
        
        tk.Button(low_energy_frame, text="400 MeV", command=lambda: self.set_energy(0.4),
                 bg='#8e44ad', fg='white', width=8).pack(side='left', padx=2)
        tk.Button(low_energy_frame, text="800 MeV", command=lambda: self.set_energy(0.8),
                 bg='#8e44ad', fg='white', width=8).pack(side='left', padx=2)
        tk.Button(low_energy_frame, text="1.5 GeV", command=lambda: self.set_energy(1.5),
                 bg='#8e44ad', fg='white', width=8).pack(side='left', padx=2)
        tk.Button(low_energy_frame, text="5 GeV", command=lambda: self.set_energy(5.0),
                 bg='#8e44ad', fg='white', width=8).pack(side='left', padx=2)
        
        # Medium energy presets  
        med_energy_frame = tk.Frame(preset_frame)
        med_energy_frame.pack(fill='x', pady=2)
        tk.Label(med_energy_frame, text="Experiments:", font=('Arial', 9, 'bold')).pack(side='left')
        
        tk.Button(med_energy_frame, text="AGS (40 GeV)", command=lambda: self.set_energy(40),
                 bg='#27ae60', fg='white', width=10).pack(side='left', padx=2)
        tk.Button(med_energy_frame, text="RHIC (200 GeV)", command=lambda: self.set_energy(200),
                 bg='#3498db', fg='white', width=10).pack(side='left', padx=2)
        tk.Button(med_energy_frame, text="LHC (2760 GeV)", command=lambda: self.set_energy(2760),
                 bg='#e74c3c', fg='white', width=10).pack(side='left', padx=2)
        
        # Lattice Parameters Frame
        lattice_frame = ttk.LabelFrame(scrollable_frame, text="🔲 Lattice Parameters")
        lattice_frame.pack(fill='x', padx=20, pady=10)
        
        # Lattice size
        tk.Label(lattice_frame, text="Lattice Size:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.lattice_size_var = tk.StringVar(value="32")
        lattice_combo = ttk.Combobox(lattice_frame, textvariable=self.lattice_size_var,
                                    values=["16", "24", "32", "48", "64"], width=10)
        lattice_combo.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(lattice_frame, text="(NxNxN grid points)").grid(row=0, column=2, sticky='w', padx=5, pady=5)
        
        # Multi-scale toggle
        self.multi_scale_var = tk.BooleanVar(value=True)
        multi_check = tk.Checkbutton(lattice_frame, text="Multi-scale analysis (use multiple lattice sizes)",
                                    variable=self.multi_scale_var)
        multi_check.grid(row=1, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        
        # Lattice spacing
        tk.Label(lattice_frame, text="Lattice Spacing (fm):").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.spacing_var = tk.DoubleVar(value=0.10)
        spacing_scale = tk.Scale(lattice_frame, from_=0.05, to=0.25, orient='horizontal',
                                variable=self.spacing_var, resolution=0.01, length=200)
        spacing_scale.grid(row=2, column=1, columnspan=2, padx=5, pady=5)
        
        # Time Evolution Parameters Frame
        time_frame = ttk.LabelFrame(scrollable_frame, text="⏱️ Time Evolution Parameters")
        time_frame.pack(fill='x', padx=20, pady=10)
        
        # Max iterations
        tk.Label(time_frame, text="Max Iterations:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.iterations_var = tk.IntVar(value=1000)
        iterations_spin = tk.Spinbox(time_frame, from_=10, to=10000, textvariable=self.iterations_var, width=10)
        iterations_spin.grid(row=0, column=1, padx=5, pady=5)
        
        # Time step
        tk.Label(time_frame, text="Time Step (fm/c):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.timestep_var = tk.DoubleVar(value=0.01)
        timestep_scale = tk.Scale(time_frame, from_=0.001, to=0.1, orient='horizontal',
                                 variable=self.timestep_var, resolution=0.001, length=200)
        timestep_scale.grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        
        # Physics Parameters Frame
        physics_frame = ttk.LabelFrame(scrollable_frame, text="🔬 Physics Parameters")
        physics_frame.pack(fill='x', padx=20, pady=10)
        
        # QCD coupling
        tk.Label(physics_frame, text="QCD Coupling (αₛ):").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.qcd_coupling_var = tk.DoubleVar(value=0.118)
        qcd_scale = tk.Scale(physics_frame, from_=0.05, to=0.5, orient='horizontal',
                            variable=self.qcd_coupling_var, resolution=0.001, length=200)
        qcd_scale.grid(row=0, column=1, columnspan=2, padx=5, pady=5)
        
        # Include electroweak
        self.electroweak_var = tk.BooleanVar(value=True)
        ew_check = tk.Checkbutton(physics_frame, text="Include Electroweak interactions",
                                 variable=self.electroweak_var)
        ew_check.grid(row=1, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        
        # Performance Parameters Frame  
        perf_frame = ttk.LabelFrame(scrollable_frame, text="⚡ Performance Options")
        perf_frame.pack(fill='x', padx=20, pady=10)
        
        # Multithreading
        self.multithread_var = tk.BooleanVar(value=True)
        thread_check = tk.Checkbutton(perf_frame, text="Use multithreading",
                                     variable=self.multithread_var)
        thread_check.grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        # Thread count
        tk.Label(perf_frame, text="Thread Count:").grid(row=0, column=1, sticky='w', padx=5, pady=5)
        self.thread_count_var = tk.StringVar(value="Auto")
        thread_combo = ttk.Combobox(perf_frame, textvariable=self.thread_count_var,
                                   values=["Auto", "1", "2", "4", "8", "16"], width=8)
        thread_combo.grid(row=0, column=2, padx=5, pady=5)
        
        # Analysis Options Frame
        analysis_frame = ttk.LabelFrame(scrollable_frame, text="📊 Analysis Options")
        analysis_frame.pack(fill='x', padx=20, pady=10)
        
        # Calculate flow
        self.calc_flow_var = tk.BooleanVar(value=True)
        flow_check = tk.Checkbutton(analysis_frame, text="Calculate anisotropic flow",
                                   variable=self.calc_flow_var)
        flow_check.grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        # Calculate spectra
        self.calc_spectra_var = tk.BooleanVar(value=True)
        spectra_check = tk.Checkbutton(analysis_frame, text="Calculate particle spectra",
                                      variable=self.calc_spectra_var)
        spectra_check.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        # Save snapshots
        self.save_snapshots_var = tk.BooleanVar(value=False)
        snapshot_check = tk.Checkbutton(analysis_frame, text="Save field snapshots",
                                       variable=self.save_snapshots_var)
        snapshot_check.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        
        # Control buttons
        control_frame = ttk.Frame(scrollable_frame)
        control_frame.pack(pady=20)
        
        self.start_button = tk.Button(control_frame, text="🚀 Start Simulation",
                                     command=self.start_simulation,
                                     bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
                                     padx=20, pady=10)
        self.start_button.pack(side='left', padx=10)
        
        self.stop_button = tk.Button(control_frame, text="🛑 Stop",
                                    command=self.stop_simulation,
                                    bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
                                    padx=20, pady=10, state='disabled')
        self.stop_button.pack(side='left', padx=10)
        
        # Preset configurations including low-energy
        preset_config_frame = ttk.Frame(scrollable_frame)
        preset_config_frame.pack(pady=10)
        
        tk.Label(preset_config_frame, text="Complete Simulation Presets:", font=('Arial', 10, 'bold')).pack()
        
        preset_buttons_frame = tk.Frame(preset_config_frame)
        preset_buttons_frame.pack(pady=5)
        
        tk.Button(preset_buttons_frame, text="Low Energy (800 MeV)", command=self.preset_low_energy,
                 bg='#8e44ad', fg='white', width=15).pack(side='left', padx=3)
        tk.Button(preset_buttons_frame, text="RHIC Au+Au", command=self.preset_rhic_auau,
                 bg='#3498db', fg='white', width=12).pack(side='left', padx=3)
        tk.Button(preset_buttons_frame, text="LHC Pb+Pb", command=self.preset_lhc_pbpb,
                 bg='#e74c3c', fg='white', width=12).pack(side='left', padx=3)
        tk.Button(preset_buttons_frame, text="Test (Fast)", command=self.preset_test,
                 bg='#f39c12', fg='white', width=12).pack(side='left', padx=3)
        
        # Status
        status_frame = ttk.LabelFrame(scrollable_frame, text="Status")
        status_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.status_text = tk.Text(status_frame, height=10, bg='black', fg='lime',
                                  font=('Courier', 10))
        self.status_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Pack canvas and scrollbar
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add initial message
        self.log_message("🚀 Enhanced Quantum Lattice Simulator Ready!")
        self.log_message("⚡ Now supports low-energy collisions from 400 MeV!")
        self.log_message("Configure parameters above and click Start Simulation")
    
    def set_energy(self, energy_gev):
        """Set energy and update display."""
        self.energy_var.set(energy_gev)
        self.update_energy_display()
    
    def update_energy_display(self, *args):
        """Update energy display with physics context."""
        energy_gev = self.energy_var.get()
        energy_mev = energy_gev * 1000
        
        # Physics context
        if energy_gev < 0.5:
            context = "Nuclear fragmentation regime"
        elif energy_gev < 1.0:
            context = "Pion production threshold"
        elif energy_gev < 2.0:
            context = "Delta resonance production"
        elif energy_gev < 10.0:
            context = "Strange particle threshold"
        elif energy_gev < 50.0:
            context = "High-energy nuclear physics"
        elif energy_gev < 1000.0:
            context = "Relativistic heavy-ion collisions"
        else:
            context = "Ultra-relativistic (QGP formation)"
        
        display_text = f"{energy_gev:.1f} GeV ({energy_mev:.0f} MeV) - {context}"
        self.energy_display_var.set(display_text)
    
    def on_unit_change(self, event=None):
        """Handle unit conversion between MeV and GeV."""
        current_value = self.energy_var.get()
        unit = self.energy_unit_var.get()
        
        if unit == "MeV":
            # Convert GeV to MeV for display
            self.energy_entry.delete(0, tk.END)
            self.energy_entry.insert(0, f"{current_value * 1000:.0f}")
        else:
            # Keep in GeV
            self.energy_entry.delete(0, tk.END)
            self.energy_entry.insert(0, f"{current_value:.2f}")
    
    def preset_low_energy(self):
        """Set low-energy nuclear collision preset."""
        self.nucleus_a_var.set("Ca40")
        self.nucleus_b_var.set("Ca40")
        self.set_energy(0.8)  # 800 MeV
        self.impact_var.set(4.0)
        self.lattice_size_var.set("24")
        self.iterations_var.set(600)
        self.timestep_var.set(0.02)
        self.log_message("📋 Low-energy nuclear collision preset loaded (800 MeV)")
    
    def preset_rhic_auau(self):
        """Set RHIC Au+Au preset."""
        self.nucleus_a_var.set("Au197")
        self.nucleus_b_var.set("Au197")
        self.set_energy(200.0)
        self.impact_var.set(7.0)
        self.lattice_size_var.set("32")
        self.iterations_var.set(1000)
        self.timestep_var.set(0.01)
        self.log_message("📋 RHIC Au+Au preset loaded (200 GeV)")
    
    def preset_lhc_pbpb(self):
        """Set LHC Pb+Pb preset."""
        self.nucleus_a_var.set("Pb208")
        self.nucleus_b_var.set("Pb208")
        self.set_energy(2760.0)
        self.impact_var.set(5.0)
        self.lattice_size_var.set("48")
        self.iterations_var.set(1500)
        self.timestep_var.set(0.008)
        self.log_message("📋 LHC Pb+Pb preset loaded (2760 GeV)")
    
    def preset_test(self):
        """Set fast test preset."""
        self.nucleus_a_var.set("He4")
        self.nucleus_b_var.set("He4")
        self.set_energy(1.0)  # 1 GeV
        self.impact_var.set(2.0)
        self.lattice_size_var.set("16")
        self.iterations_var.set(100)
        self.timestep_var.set(0.02)
        self.log_message("📋 Fast test preset loaded (1 GeV)")
    
    def create_visualization_tab(self):
        """Create 3D visualization tab."""
        
        # Instructions
        info_label = tk.Label(self.visualization_tab,
                             text="3D Nuclear Collision Visualization\n"
                                  "Watch nuclei collide and particles form in real-time!\n"
                                  "Now supports low-energy fragmentation and high-energy QGP formation",
                             font=('Arial', 12), bg='#34495e', fg='white')
        info_label.pack(pady=10)
        
        # Create visualizer
        viz_frame = ttk.Frame(self.visualization_tab)
        viz_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.visualizer = ParticleVisualizer(viz_frame)
        
        if MATPLOTLIB_AVAILABLE and hasattr(self.visualizer, 'fig'):
            # Embed matplotlib figure
            self.canvas = FigureCanvasTkAgg(self.visualizer.fig, viz_frame)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_analysis_tab(self):
        """Create analysis and results tab."""
        
        analysis_frame = ttk.LabelFrame(self.analysis_tab, text="Physics Analysis")
        analysis_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.analysis_text = tk.Text(analysis_frame, height=25, width=80,
                                    bg='#2c3e50', fg='white', font=('Courier', 11))
        self.analysis_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Save button
        save_frame = ttk.Frame(self.analysis_tab)
        save_frame.pack(fill='x', padx=10, pady=5)
        
        save_button = tk.Button(save_frame, text="💾 Save Results",
                               command=self.save_results,
                               bg='#3498db', fg='white')
        save_button.pack(side='right', padx=10)
    
    def start_simulation(self):
        """Start the quantum lattice simulation with user parameters."""
        if self.is_running:
            return
        
        try:
            # Create parameters from GUI
            try:
                from ..core.parameters import SimulationParameters
                from ..core.simulator import QuantumLatticeSimulator
            except ImportError:
                # Fallback
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from core.parameters import SimulationParameters
                from core.simulator import QuantumLatticeSimulator
            
            params = SimulationParameters()
            
            # Nuclear parameters
            params.nucleus_A = self.nucleus_a_var.get()
            params.nucleus_B = self.nucleus_b_var.get()
            params.impact_parameter_fm = self.impact_var.get()
            
            # Collision parameters
            params.collision_energy_gev = self.energy_var.get()
            
            # Lattice parameters
            size = int(self.lattice_size_var.get())
            if self.multi_scale_var.get():
                # Multi-scale setup
                params.lattice_sizes = [(size//2, size//2, size//2), (size, size, size)]
                params.lattice_spacings_fm = [self.spacing_var.get() * 1.5, self.spacing_var.get()]
            else:
                # Single scale
                params.lattice_sizes = [(size, size, size)]
                params.lattice_spacings_fm = [self.spacing_var.get()]
            
            # Time evolution parameters
            params.max_iterations = self.iterations_var.get()
            params.time_step_fm = self.timestep_var.get()
            
            # Physics parameters
            params.qcd_coupling = self.qcd_coupling_var.get()
            
            # Performance parameters
            params.use_multithreading = self.multithread_var.get()
            if self.thread_count_var.get() != "Auto":
                params.num_threads = int(self.thread_count_var.get())
            
            # Analysis parameters
            params.calculate_flow = self.calc_flow_var.get()
            params.calculate_spectra = self.calc_spectra_var.get()
            params.save_snapshots = self.save_snapshots_var.get()
            
            # Create simulator
            self.simulator = QuantumLatticeSimulator(params)
            
            # Start simulation thread
            self.simulation_thread = threading.Thread(target=self.run_simulation_thread)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
            # Update GUI state
            self.is_running = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            
            energy_context = "low-energy" if params.collision_energy_gev < 2.0 else "high-energy"
            self.log_message(f"🚀 Starting {energy_context} {params.nucleus_A}+{params.nucleus_B}")
            self.log_message(f"   Energy: {params.collision_energy_gev} GeV ({params.collision_energy_gev*1000:.0f} MeV)")
            self.log_message(f"   Lattice: {params.lattice_sizes}")
            self.log_message(f"   Iterations: {params.max_iterations}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulation: {e}")
            import traceback
            traceback.print_exc()
    
    def run_simulation_thread(self):
        """Run simulation in background thread."""
        try:
            self.simulator.run_simulation(callback=self.simulation_callback)
            self.log_message("✅ Simulation completed successfully!")
            
        except Exception as e:
            self.log_message(f"❌ Simulation error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_button.config(state='normal'))
            self.root.after(0, lambda: self.stop_button.config(state='disabled'))
    
    def simulation_callback(self, simulator):
        """Callback for simulation updates."""
        # Schedule GUI updates on main thread
        self.root.after(0, lambda: self.update_visualization(simulator))
        self.root.after(0, lambda: self.update_analysis(simulator))
    
    def update_visualization(self, simulator):
        """Update 3D visualization."""
        try:
            self.visualizer.update_collision_state(simulator)
            if MATPLOTLIB_AVAILABLE and hasattr(self, 'canvas'):
                self.canvas.draw()
        except Exception as e:
            print(f"Visualization update error: {e}")
    
    def update_analysis(self, simulator):
        """Update analysis display with energy-specific information."""
        try:
            obs = simulator.observables
            
            # Determine energy regime
            energy_gev = simulator.params.collision_energy_gev
            if energy_gev < 1.0:
                regime = "Low-Energy Nuclear Physics"
                regime_icon = "🔬"
            elif energy_gev < 10.0:
                regime = "Medium-Energy Nuclear Physics"  
                regime_icon = "🎯"
            elif energy_gev < 100.0:
                regime = "High-Energy Nuclear Physics"
                regime_icon = "⚡"
            else:
                regime = "Ultra-Relativistic Heavy-Ion Physics"
                regime_icon = "🌟"
            
            analysis = f"""
{regime_icon} QUANTUM LATTICE COLLISION ANALYSIS
{'='*50}

Energy Regime: {regime}
Collision System: {simulator.params.nucleus_A} + {simulator.params.nucleus_B}
Energy: {energy_gev:.2f} GeV ({energy_gev*1000:.0f} MeV)
Impact Parameter: {simulator.params.impact_parameter_fm} fm
Time: {simulator.current_time:.3f} fm/c
Iteration: {simulator.iteration}/{simulator.params.max_iterations}

LATTICE CONFIGURATION:
Sizes: {simulator.params.lattice_sizes}
Spacings: {simulator.params.lattice_spacings_fm} fm
Time Step: {simulator.params.time_step_fm} fm/c

CURRENT STATE:
"""
            
            if obs['temperature']:
                temp = obs['temperature'][-1]
                energy = obs['energy_density'][-1]
                mult = obs['multiplicity'][-1]
                pressure = obs['pressure'][-1]
                
                analysis += f"""
Temperature: {temp:.1f} MeV
Energy Density: {energy:.2e} GeV/fm³
Pressure: {pressure:.2e} GeV/fm³
Particle Multiplicity: {mult:.0f}

PHASE INFORMATION:"""
                
                # Energy-dependent phase analysis
                if energy_gev < 1.0:
                    analysis += f"""
Phase: {"🔬 Nuclear Matter" if temp < 50 else "🌡️ Excited Nuclear Matter"}
Regime: Nuclear fragmentation and transparency effects
Expected: {"Nucleon knockout" if temp < 100 else "Light fragment production"}
"""
                else:
                    analysis += f"""
Phase: {"🔥 QGP" if temp > 170 else "🌡️ Hadronic" if temp > 140 else "❄️ Nuclear"}
Deconfinement: {"✅ Yes" if temp > 170 else "❌ No"} (Tc ≈ 170 MeV)
Chiral Restoration: {"✅ Yes" if temp > 155 else "❌ No"} (Tχ ≈ 155 MeV)
"""
                
                if len(obs['temperature']) > 10:
                    max_temp = max(obs['temperature'])
                    analysis += f"\nMaximum Temperature Reached: {max_temp:.1f} MeV"
                    avg_temp = np.mean(obs['temperature'][-10:])
                    analysis += f"\nRecent Average Temperature: {avg_temp:.1f} MeV"
                
                # Progress indicator
                progress = simulator.iteration / simulator.params.max_iterations * 100
                analysis += f"\n\nSIMULATION PROGRESS: {progress:.1f}%"
                analysis += f"\n{'█' * int(progress/5)}{'▒' * int((100-progress)/5)}"
                
                # Energy-dependent particle production
                analysis += f"\n\nPARTICLE PRODUCTION ESTIMATES:"
                
                if energy_gev < 0.5:
                    analysis += f"""
🔵 Nucleons: ~{mult*0.85:.0f}
🟡 Light fragments (d,t,α): ~{mult*0.10:.0f}
🟢 Heavy fragments: ~{mult*0.05:.0f}
Expected: Nuclear breakup and transparency
"""
                elif energy_gev < 2.0:
                    analysis += f"""
🔵 Nucleons: ~{mult*0.70:.0f}
🔴 Pions: ~{mult*0.25:.0f}
🟡 Light fragments: ~{mult*0.05:.0f}
Expected: Pion production threshold crossed
"""
                else:
                    analysis += f"""
🔴 Pions (π±, π⁰): ~{mult*0.80:.0f}
🟡 Kaons (K±, K⁰): ~{mult*0.15:.0f} 
🔵 Protons/Neutrons: ~{mult*0.05:.0f}
🟢 Strange Baryons: ~{mult*0.02:.0f}
"""
                
                analysis += f"""

COLLISION DYNAMICS:
Initial Contact: t = 0 fm/c
{"Nuclear overlap" if energy_gev < 1.0 else "Thermalization"}: t ≈ 0.5 fm/c  
{"Fragment formation" if energy_gev < 1.0 else "QGP formation" if energy_gev > 100 else "Hadronization"}: t ≈ 1.0 fm/c
Expansion: t > 2.0 fm/c
{"Nuclear breakup" if energy_gev < 1.0 else "Freeze-out"}: t ≈ 10 fm/c
"""
            
            # Add physics insights
            if obs['temperature'] and len(obs['temperature']) > 5:
                temp_trend = np.diff(obs['temperature'][-5:])
                if np.mean(temp_trend) > 0:
                    analysis += "\n📈 Temperature rising (heating phase)"
                elif np.mean(temp_trend) < -1:
                    analysis += "\n📉 Temperature falling (cooling/expansion)"
                else:
                    analysis += "\n⚖️ Temperature stable (equilibrium)"
            
            self.analysis_text.delete('1.0', tk.END)
            self.analysis_text.insert('1.0', analysis)
            
        except Exception as e:
            print(f"Analysis update error: {e}")
    
    def stop_simulation(self):
        """Stop the running simulation."""
        if self.simulator:
            self.simulator.stop()
        
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        self.log_message("🛑 Simulation stopped by user")
    
    def save_results(self):
        """Save simulation results."""
        if not self.simulator:
            messagebox.showwarning("Warning", "No simulation results to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.simulator.save_results(filename)
                self.log_message(f"💾 Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")
    
    def log_message(self, message):
        """Add message to status log."""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def run(self):
        """Start the GUI."""
        self.root.mainloop()
"""
Updated GUI with Smart Boundary Detection - FINAL VERSION
Only checks mass escape AFTER nuclei actually collide.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import numpy as np
import multiprocessing as mp
from typing import Dict, List, Optional, Any
import json
import sys
import os

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import enhanced components with smart boundary detection
try:
    from core.simulator import EnhancedSimulationEngine, SmartBoundaryConditions, UltraHighResolutionLattice
    from core.physics.nuclear import NuclearEquationTracker
    from core.time_stepping import BidirectionalTimeSteppingControls
    from components import AdvancedVisualizerWithMomentum, LowEnergyStatusDisplay
    ENHANCED_COMPONENTS_AVAILABLE = True
    print("✅ All enhanced components with SMART BOUNDARY DETECTION imported!")
except ImportError as e:
    ENHANCED_COMPONENTS_AVAILABLE = False
    print(f"⚠️ Enhanced components import error: {e}")

class SmartBoundaryNuclearGUI:
    """Nuclear physics simulator with smart boundary detection that only activates after collision starts."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚀 Nuclear Physics Simulator v4.1 - Smart Boundary Detection")
        self.root.geometry("2200x1400")
        self.root.configure(bg='#0d1117')
        
        # Try to maximize window
        try:
            self.root.state('zoomed')  # Windows
        except:
            try:
                self.root.attributes('-zoomed', True)  # Linux
            except:
                pass  # Mac or other
        
        # Configure for proper scaling
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        
        # Simulation state
        self.simulation_engine = None
        self.simulation_thread = None
        self.is_running = False
        self.simulation_results = None
        
        # Smart boundary features
        self.smart_features = {
            'matplotlib': MATPLOTLIB_AVAILABLE,
            'enhanced_components': ENHANCED_COMPONENTS_AVAILABLE,
            'smart_boundary_detection': True,  # New feature!
            'collision_activation': True,     # New feature!
            'ultra_high_resolution': True,
            'nuclear_equations': True,
            'bidirectional_playback': True,
            'distributed_computing': mp.cpu_count() > 1
        }
        
        # Initialize GUI
        self.create_smart_boundary_interface()
        
        print("🚀 Nuclear Physics Simulator v4.1 - Smart Boundary Detection")
        print("=" * 70)
        print("🎯 NEW SMART FEATURES:")
        print("   ✅ Smart boundary detection - only active AFTER collision starts")
        print("   ✅ Collision activation detection - multiple methods")
        print("   ✅ No premature stopping due to initial nuclear placement")
        print("   ✅ Configurable collision detection radius and timing")
        print("=" * 70)
        print(f"✅ CPU Cores: {mp.cpu_count()}")
        print(f"✅ Enhanced Components: {ENHANCED_COMPONENTS_AVAILABLE}")
        print(f"✅ Matplotlib: {MATPLOTLIB_AVAILABLE}")
    
    def create_smart_boundary_interface(self):
        """Create interface with smart boundary controls."""
        
        # Enhanced styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#0d1117', borderwidth=0)
        style.configure('TNotebook.Tab', padding=[25, 12], font=('Arial', 12, 'bold'))
        
        # Main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=8, pady=8)
        
        # Create all tabs
        self.setup_tab = ttk.Frame(self.notebook)
        self.equations_tab = ttk.Frame(self.notebook) 
        self.visualization_tab = ttk.Frame(self.notebook)
        self.time_stepping_tab = ttk.Frame(self.notebook)
        self.boundary_tab = ttk.Frame(self.notebook)
        
        # Add tabs with updated labels
        self.notebook.add(self.setup_tab, text="🚀 Smart Boundary Setup")
        self.notebook.add(self.equations_tab, text="⚛️ Nuclear Equations (n+p→d+γ)")
        self.notebook.add(self.visualization_tab, text="🎆 3D Visualization")
        self.notebook.add(self.time_stepping_tab, text="⏱️ Time Stepping")
        self.notebook.add(self.boundary_tab, text="🎯 Smart Boundary Analysis")
        
        # Create each tab
        self.create_smart_setup_tab()
        self.create_equations_tab()
        self.create_visualization_tab()
        self.create_time_stepping_tab()
        self.create_smart_boundary_tab()
    
    def create_smart_setup_tab(self):
        """Create setup with smart boundary configuration."""
        
        # Main scrollable frame
        canvas = tk.Canvas(self.setup_tab, bg='#0d1117')
        scrollbar = ttk.Scrollbar(self.setup_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enhanced header with smart boundary info
        header_frame = tk.Frame(scrollable_frame, bg='#161b22', pady=25)
        header_frame.pack(fill='x', padx=25, pady=15)
        
        title_label = tk.Label(
            header_frame,
            text="🚀 NUCLEAR PHYSICS SIMULATOR v4.1",
            font=('Arial', 20, 'bold'),
            bg='#161b22', fg='#58a6ff'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="🎯 NEW: Smart Boundary Detection - Only Active After Collision Begins",
            font=('Arial', 13, 'bold'),
            bg='#161b22', fg='#39d353'
        )
        subtitle_label.pack()
        
        feature_label = tk.Label(
            header_frame,
            text="Nuclear Equations • Ultra-High Lattices • Smart Boundaries • Bidirectional Playback",
            font=('Arial', 12),
            bg='#161b22', fg='#7d8590'
        )
        feature_label.pack()
        
        # Smart Boundary Feature Highlight
        smart_frame = ttk.LabelFrame(scrollable_frame, text="🎯 Smart Boundary Detection Features")
        smart_frame.pack(fill='x', padx=25, pady=15)
        
        smart_info = tk.Text(
            smart_frame,
            height=6, width=100,
            bg='#1a1a2e', fg='#39d353',
            font=('Consolas', 11),
            wrap='word'
        )
        smart_info.pack(padx=15, pady=15)
        
        smart_text = """🎯 SMART BOUNDARY DETECTION v4.1:
✅ NO MORE PREMATURE STOPPING: Boundary detection only activates AFTER nuclei collide
✅ COLLISION DETECTION: Multiple methods detect when collision actually begins
✅ TIME-BASED ACTIVATION: Configurable delay before boundary monitoring starts
✅ PROXIMITY DETECTION: Monitors nuclear center-of-mass separation
✅ CENTRAL REGION MONITORING: Detects when particles enter collision zone
✅ INTELLIGENT THRESHOLDS: Different escape criteria before/after collision"""
        
        smart_info.insert('1.0', smart_text)
        smart_info.config(state='disabled')
        
        # Nuclear System Configuration
        nuclear_frame = ttk.LabelFrame(scrollable_frame, text="🔬 Nuclear System Configuration")
        nuclear_frame.pack(fill='x', padx=25, pady=15)
        
        nucl_controls = tk.Frame(nuclear_frame)
        nucl_controls.pack(padx=15, pady=15)
        
        tk.Label(nucl_controls, text="Projectile:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky='w', padx=8)
        
        nucleus_options = ["H", "D", "He4", "C12", "O16", "Ca40", "Fe56", "Au197", "Pb208", "U238"]
        
        self.nucleus_a_var = tk.StringVar(value="Au197")
        nucleus_a_combo = ttk.Combobox(nucl_controls, textvariable=self.nucleus_a_var, 
                                       values=nucleus_options, width=10)
        nucleus_a_combo.grid(row=0, column=1, padx=8)
        
        tk.Label(nucl_controls, text="Target:", font=('Arial', 12, 'bold')).grid(row=0, column=2, sticky='w', padx=(25,8))
        self.nucleus_b_var = tk.StringVar(value="Au197")
        nucleus_b_combo = ttk.Combobox(nucl_controls, textvariable=self.nucleus_b_var,
                                       values=nucleus_options, width=10)
        nucleus_b_combo.grid(row=0, column=3, padx=8)
        
        # Energy configuration
        tk.Label(nucl_controls, text="Energy:", font=('Arial', 12, 'bold')).grid(row=1, column=0, sticky='w', padx=8, pady=(10,0))
        
        self.energy_var = tk.DoubleVar(value=200.0)
        energy_entry = tk.Entry(nucl_controls, textvariable=self.energy_var, width=12)
        energy_entry.grid(row=1, column=1, padx=8, pady=(10,0))
        
        self.energy_unit_var = tk.StringVar(value="GeV")
        unit_combo = ttk.Combobox(nucl_controls, textvariable=self.energy_unit_var,
                                 values=["MeV", "GeV", "TeV"], width=8)
        unit_combo.grid(row=1, column=2, padx=8, pady=(10,0))
        
        tk.Label(nucl_controls, text="Impact (fm):", font=('Arial', 12, 'bold')).grid(row=1, column=3, sticky='w', padx=(25,8), pady=(10,0))
        
        self.impact_var = tk.DoubleVar(value=5.0)
        impact_spin = tk.Spinbox(nucl_controls, from_=0.0, to=20.0, textvariable=self.impact_var, 
                                width=10, increment=0.5)
        impact_spin.grid(row=1, column=4, padx=8, pady=(10,0))
        
        # Smart Boundary Configuration
        smart_boundary_frame = ttk.LabelFrame(scrollable_frame, text="🎯 Smart Boundary Configuration")
        smart_boundary_frame.pack(fill='x', padx=25, pady=15)
        
        boundary_controls = tk.Frame(smart_boundary_frame)
        boundary_controls.pack(padx=15, pady=15)
        
        # Activation delay
        tk.Label(boundary_controls, text="Activation Delay:", font=('Arial', 11, 'bold')).grid(row=0, column=0, sticky='w')
        
        self.activation_delay_var = tk.DoubleVar(value=5.0)
        delay_spin = tk.Spinbox(boundary_controls, from_=1.0, to=20.0, textvariable=self.activation_delay_var, 
                               width=8, increment=1.0)
        delay_spin.grid(row=0, column=1, padx=5)
        tk.Label(boundary_controls, text="fm/c", font=('Arial', 10)).grid(row=0, column=2, sticky='w')
        
        # Collision radius
        tk.Label(boundary_controls, text="Collision Radius:", font=('Arial', 11, 'bold')).grid(row=0, column=3, sticky='w', padx=(30,5))
        
        self.collision_radius_var = tk.DoubleVar(value=15.0)
        radius_spin = tk.Spinbox(boundary_controls, from_=5.0, to=30.0, textvariable=self.collision_radius_var,
                                width=8, increment=2.5)
        radius_spin.grid(row=0, column=4, padx=5)
        tk.Label(boundary_controls, text="fm", font=('Arial', 10)).grid(row=0, column=5, sticky='w')
        
        # Escape threshold
        tk.Label(boundary_controls, text="Escape Threshold:", font=('Arial', 11, 'bold')).grid(row=1, column=0, sticky='w', pady=(10,0))
        
        self.escape_threshold_var = tk.DoubleVar(value=50.0)
        threshold_spin = tk.Spinbox(boundary_controls, from_=10.0, to=90.0, 
                                   textvariable=self.escape_threshold_var, width=8, increment=10.0)
        threshold_spin.grid(row=1, column=1, padx=5, pady=(10,0))
        tk.Label(boundary_controls, text="% mass", font=('Arial', 10)).grid(row=1, column=2, sticky='w', pady=(10,0))
        
        # Lattice configuration (simplified)
        lattice_frame = ttk.LabelFrame(scrollable_frame, text="🎯 Lattice Resolution")
        lattice_frame.pack(fill='x', padx=25, pady=15)
        
        lattice_controls = tk.Frame(lattice_frame)
        lattice_controls.pack(padx=15, pady=15)
        
        tk.Label(lattice_controls, text="Resolution:", font=('Arial', 11, 'bold')).pack(side='left')
        
        self.resolution_var = tk.StringVar(value="Medium")
        resolution_combo = ttk.Combobox(lattice_controls, textvariable=self.resolution_var,
                                       values=["Low", "Medium", "High", "Ultra"], width=15)
        resolution_combo.pack(side='left', padx=10)
        
        self.memory_var = tk.StringVar(value="~2 GB")
        tk.Label(lattice_controls, textvariable=self.memory_var, font=('Arial', 11), fg='#f9e2af').pack(side='left', padx=20)
        
        # Computing configuration
        computing_frame = ttk.LabelFrame(scrollable_frame, text="🖥️ Computing Configuration")
        computing_frame.pack(fill='x', padx=25, pady=15)
        
        comp_controls = tk.Frame(computing_frame)
        comp_controls.pack(padx=15, pady=15)
        
        tk.Label(comp_controls, text="CPU Cores:", font=('Arial', 11, 'bold')).pack(side='left')
        self.num_cores_var = tk.IntVar(value=min(8, mp.cpu_count()))
        cores_spin = tk.Spinbox(comp_controls, from_=1, to=mp.cpu_count(),
                               textvariable=self.num_cores_var, width=8)
        cores_spin.pack(side='left', padx=10)
        
        tk.Label(comp_controls, text="Max Time:", font=('Arial', 11, 'bold')).pack(side='left', padx=(30,5))
        self.max_time_var = tk.DoubleVar(value=100.0)
        time_spin = tk.Spinbox(comp_controls, from_=20.0, to=500.0,
                              textvariable=self.max_time_var, width=8, increment=10.0)
        time_spin.pack(side='left', padx=5)
        tk.Label(comp_controls, text="fm/c", font=('Arial', 10)).pack(side='left')
        
        # Main controls
        control_frame = tk.Frame(scrollable_frame, bg='#21262d', pady=25)
        control_frame.pack(fill='x', padx=25, pady=25)
        
        self.start_button = tk.Button(
            control_frame,
            text="🚀 START SMART BOUNDARY SIMULATION",
            command=self.start_smart_simulation,
            bg='#238636', fg='white',
            font=('Arial', 18, 'bold'),
            padx=50, pady=25,
            relief='raised', bd=4
        )
        self.start_button.pack(side='left', padx=25)
        
        self.stop_button = tk.Button(
            control_frame,
            text="🛑 STOP",
            command=self.stop_simulation,
            bg='#da3633', fg='white',
            font=('Arial', 18, 'bold'),
            padx=50, pady=25,
            state='disabled',
            relief='raised', bd=4
        )
        self.stop_button.pack(side='left', padx=15)
        
        # Status display
        self.create_status_display(scrollable_frame)
        
        # Pack canvas
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_status_display(self, parent):
        """Create enhanced status display with smart boundary info."""
        
        status_frame = ttk.LabelFrame(parent, text="📊 Smart Boundary Simulation Status")
        status_frame.pack(fill='both', expand=True, padx=25, pady=15)
        
        status_frame.rowconfigure(0, weight=1)
        status_frame.columnconfigure(0, weight=1)
        
        self.status_text = tk.Text(
            status_frame,
            bg='#0d1117', fg='#58a6ff',
            font=('Consolas', 11),
            wrap='word',
            height=12
        )
        
        status_scrollbar = ttk.Scrollbar(status_frame, orient='vertical', command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        
        self.status_text.grid(row=0, column=0, sticky='nsew', padx=(8,0), pady=8)
        status_scrollbar.grid(row=0, column=1, sticky='ns', padx=(0,8), pady=8)
        
        # Initial status
        initial_status = f"""
🚀 NUCLEAR PHYSICS SIMULATOR v4.1 - SMART BOUNDARY DETECTION

🎯 SMART BOUNDARY FEATURES ACTIVE:
✅ Intelligent collision detection - no premature stopping
✅ Multiple activation methods (time, proximity, central region)
✅ Configurable activation delay: {self.activation_delay_var.get():.1f} fm/c
✅ Collision detection radius: {self.collision_radius_var.get():.1f} fm
✅ Escape threshold: {self.escape_threshold_var.get():.0f}% of mass

🔬 NUCLEAR EQUATIONS:
Real-time tracking of reactions like n + p → d + γ with complete conservation laws

🎆 3D VISUALIZATION:
Momentum vectors, particle trajectories, and collision evolution

⏱️ TIME STEPPING:
Bidirectional playback through entire collision history

🚀 Ready for intelligent simulation - boundary monitoring will activate automatically
when nuclei begin to collide, preventing false stops from initial placement.

System Status: {'✅ All components ready' if ENHANCED_COMPONENTS_AVAILABLE else '⚠️ Some components limited'}
"""
        
        self.status_text.insert('1.0', initial_status)
        self.log_status("🎯 Smart Boundary Nuclear Physics Simulator ready")
    
    def create_equations_tab(self):
        """Create nuclear equations tab."""
        
        self.equations_tab.rowconfigure(0, weight=1)
        self.equations_tab.columnconfigure(0, weight=1)
        
        equations_frame = ttk.LabelFrame(self.equations_tab, text="⚛️ Real-Time Nuclear Equations")
        equations_frame.grid(row=0, column=0, sticky='nsew', padx=15, pady=15)
        
        equations_frame.rowconfigure(0, weight=1)
        equations_frame.columnconfigure(0, weight=1)
        
        self.equations_text = tk.Text(
            equations_frame,
            bg='#0d1117', fg='#a6e3a1',
            font=('Courier New', 12),
            wrap='word'
        )
        
        equations_scrollbar = ttk.Scrollbar(equations_frame, orient='vertical', command=self.equations_text.yview)
        self.equations_text.configure(yscrollcommand=equations_scrollbar.set)
        
        self.equations_text.grid(row=0, column=0, sticky='nsew', padx=(8,0), pady=8)
        equations_scrollbar.grid(row=0, column=1, sticky='ns', padx=(0,8), pady=8)
        
        # Initial content
        initial_equations = """
⚛️ NUCLEAR EQUATIONS WILL APPEAR HERE DURING SIMULATION

Examples of reactions that will be tracked in real-time:

🎯 FUSION REACTIONS:
n + p → d + γ         (Q = +2.225 MeV, deuteron formation)
d + d → ³He + n       (Q = +3.269 MeV, helium-3 production)

💥 NUCLEAR BREAKUP:
²H → p + n            (Q = -2.225 MeV, deuteron breakup)

🎯 MESON PRODUCTION:
p + p → p + p + π⁰     (Q = -134.9 MeV, pion production)

🚀 Start simulation to see real nuclear equations with complete physics analysis!
"""
        
        self.equations_text.insert('1.0', initial_equations)
    
    def create_visualization_tab(self):
        """Create visualization tab."""
        
        if ENHANCED_COMPONENTS_AVAILABLE and MATPLOTLIB_AVAILABLE:
            try:
                self.visualizer = AdvancedVisualizerWithMomentum(self.visualization_tab)
                if hasattr(self.visualizer, 'fig'):
                    self.visualization_canvas = FigureCanvasTkAgg(self.visualizer.fig, self.visualization_tab)
                    self.visualization_canvas.get_tk_widget().pack(fill='both', expand=True)
            except Exception as e:
                print(f"Visualization error: {e}")
                self._create_fallback_viz()
        else:
            self._create_fallback_viz()
    
    def _create_fallback_viz(self):
        """Create fallback visualization."""
        fallback_frame = tk.Frame(self.visualization_tab, bg='#0d1117')
        fallback_frame.pack(fill='both', expand=True)
        
        tk.Label(
            fallback_frame,
            text="🎆 3D VISUALIZATION\n\nVisualization will appear here during simulation.\nInstall matplotlib for advanced 3D graphics.",
            font=('Arial', 14), fg='#58a6ff', bg='#0d1117',
            justify='center'
        ).pack(expand=True)
    
    def create_time_stepping_tab(self):
        """Create time stepping tab."""
        
        if ENHANCED_COMPONENTS_AVAILABLE:
            try:
                self.time_controls = BidirectionalTimeSteppingControls(
                    self.time_stepping_tab,
                    self.on_time_step_changed
                )
            except Exception as e:
                print(f"Time stepping error: {e}")
                self._create_fallback_time()
        else:
            self._create_fallback_time()
    
    def _create_fallback_time(self):
        """Create fallback time stepping."""
        tk.Label(
            self.time_stepping_tab,
            text="⏱️ TIME STEPPING CONTROLS\n\nBidirectional time stepping will appear here after simulation.",
            font=('Arial', 14), fg='#cba6f7', bg='#0d1117',
            justify='center'
        ).pack(expand=True)
    
    def create_smart_boundary_tab(self):
        """Create smart boundary analysis tab."""
        
        self.boundary_tab.rowconfigure(0, weight=1)
        self.boundary_tab.columnconfigure(0, weight=1)
        
        boundary_frame = ttk.LabelFrame(self.boundary_tab, text="🎯 Smart Boundary Analysis")
        boundary_frame.grid(row=0, column=0, sticky='nsew', padx=15, pady=15)
        
        boundary_frame.rowconfigure(0, weight=1)
        boundary_frame.columnconfigure(0, weight=1)
        
        self.boundary_text = tk.Text(
            boundary_frame,
            bg='#0d1117', fg='#f9e2af',
            font=('Consolas', 11),
            wrap='word'
        )
        
        boundary_scrollbar = ttk.Scrollbar(boundary_frame, orient='vertical', command=self.boundary_text.yview)
        self.boundary_text.configure(yscrollcommand=boundary_scrollbar.set)
        
        self.boundary_text.grid(row=0, column=0, sticky='nsew', padx=(8,0), pady=8)
        boundary_scrollbar.grid(row=0, column=1, sticky='ns', padx=(0,8), pady=8)
        
        # Initial content
        initial_boundary = """
🎯 SMART BOUNDARY DETECTION v4.1
════════════════════════════════════════════════════════════════════════════════════════════════════════════

🚀 INTELLIGENT COLLISION DETECTION:

⏳ BEFORE COLLISION:
• Nuclei placed at safe distance (±25 fm separation)
• Boundary monitoring DISABLED to prevent false positives
• System waits for actual collision to begin

🎯 COLLISION ACTIVATION DETECTION:
Method 1: Time-based activation (minimum delay: 5.0 fm/c)
Method 2: Proximity detection (nuclei separation < 20 fm)  
Method 3: Central region monitoring (particles within 15 fm of center)

✅ AFTER COLLISION STARTS:
• Boundary monitoring ACTIVATED
• Real-time escape fraction calculation
• Automatic stop at configured threshold (50% mass)

📊 REAL-TIME MONITORING:
Collision Status: ⏳ Waiting for simulation to start
Boundary Monitoring: ❌ Inactive (will activate after collision)
Escape Fraction: 0.0%
Escaped Particles: 0

🎯 This smart system prevents premature simulation stopping due to initial nuclear placement!

🚀 Start simulation to see intelligent boundary detection in action.
"""
        
        self.boundary_text.insert('1.0', initial_boundary)
    
    def start_smart_simulation(self):
        """Start simulation with smart boundary detection."""
        
        if self.is_running:
            return
        
        try:
            # Enhanced configuration with smart boundary parameters
            config = {
                'lattice_sizes': [(128, 128, 128), (256, 256, 256)],
                'spacings': [0.1, 0.05],
                'num_workers': self.num_cores_var.get(),
                'escape_threshold': self.escape_threshold_var.get() / 100.0,
                'min_collision_time': self.activation_delay_var.get(),      # Smart boundary config
                'collision_radius': self.collision_radius_var.get(),       # Smart boundary config
                'time_step': 0.005,
                'max_time': self.max_time_var.get(),
                'max_history_steps': 10000
            }
            
            nucleus_a = self.nucleus_a_var.get()
            nucleus_b = self.nucleus_b_var.get()
            
            energy_val = self.energy_var.get()
            unit = self.energy_unit_var.get()
            
            # Convert to GeV
            energy_gev = energy_val
            if unit == "MeV":
                energy_gev = energy_val / 1000.0
            elif unit == "TeV":
                energy_gev = energy_val * 1000.0
            
            if ENHANCED_COMPONENTS_AVAILABLE:
                # Create smart boundary simulation engine
                self.simulation_engine = EnhancedSimulationEngine(config)
                
                # Initialize simulation
                self.simulation_engine.initialize_simulation(
                    nucleus_a, nucleus_b, energy_gev, self.impact_var.get()
                )
                
                # Start simulation thread
                self.simulation_thread = threading.Thread(target=self._run_smart_simulation_thread)
                self.simulation_thread.daemon = True
                self.simulation_thread.start()
                
                # Update UI
                self.is_running = True
                self.start_button.config(state='disabled')
                self.stop_button.config(state='normal')
                
                self.log_status("🚀 SMART BOUNDARY SIMULATION STARTED")
                self.log_status(f"   System: {nucleus_a} + {nucleus_b}")
                self.log_status(f"   Energy: {energy_val} {unit}")
                self.log_status(f"   🎯 Smart boundary: Delay={config['min_collision_time']}fm/c, Radius={config['collision_radius']}fm")
                
            else:
                messagebox.showerror("Components Missing", 
                                   "Enhanced simulation components not available.")
                
        except Exception as e:
            messagebox.showerror("Simulation Error", f"Failed to start simulation:\n{str(e)}")
            self.log_status(f"❌ Simulation startup failed: {str(e)}")
    
    def _run_smart_simulation_thread(self):
        """Run smart boundary simulation."""
        
        try:
            self.log_status("🎯 Running simulation with smart boundary detection...")
            
            # Run simulation
            self.simulation_results = self.simulation_engine.run_simulation(
                callback=self._simulation_progress_callback
            )
            
            # Update time controls
            if hasattr(self, 'time_controls'):
                self.root.after(0, lambda: self.time_controls.set_simulation_data(self.simulation_results))
            
            # Show smart boundary results
            boundary_info = self.simulation_results.get('smart_boundary_conditions', {})
            if boundary_info.get('collision_started'):
                self.log_status(f"✅ Collision detected at t = {boundary_info.get('collision_start_time', 0):.3f} fm/c")
                self.log_status(f"   Final escape fraction: {boundary_info.get('escape_fraction', 0):.1%}")
            else:
                self.log_status("⚠️ Collision not detected - check energy and impact parameter")
            
            self.log_status("✅ SMART BOUNDARY SIMULATION COMPLETED")
            
        except Exception as e:
            self.log_status(f"❌ Simulation error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_button.config(state='normal'))
            self.root.after(0, lambda: self.stop_button.config(state='disabled'))
    
    def _simulation_progress_callback(self, simulation_engine):
        """Handle simulation progress updates."""
        
        try:
            # Update equations display
            if hasattr(simulation_engine, 'equation_tracker'):
                equations_text = simulation_engine.equation_tracker.generate_reaction_equations_text()
                if equations_text and hasattr(self, 'equations_text'):
                    self.root.after(0, lambda: self._update_equations_display(equations_text))
            
            # Update boundary status
            if hasattr(simulation_engine, 'boundary_conditions'):
                boundary_status = simulation_engine.boundary_conditions.get_status_info()
                if boundary_status.get('collision_started') and hasattr(self, 'boundary_text'):
                    self.root.after(0, lambda: self._update_boundary_display(boundary_status))
            
        except Exception as e:
            print(f"Progress callback error: {e}")
    
    def _update_equations_display(self, equations_text):
        """Update equations display."""
        try:
            self.equations_text.delete('1.0', tk.END)
            self.equations_text.insert('1.0', equations_text)
            self.equations_text.see(tk.END)
        except:
            pass
    
    def _update_boundary_display(self, boundary_status):
        """Update boundary display with smart status."""
        try:
            current_time = getattr(self.simulation_engine, 'current_time', 0)
            
            update_text = f"""
🎯 SMART BOUNDARY UPDATE - Time: {current_time:.3f} fm/c
════════════════════════════════════════════════════════════════════════════════════════════════════════════

✅ COLLISION DETECTED at t = {boundary_status.get('collision_start_time', 0):.3f} fm/c
🎯 Boundary monitoring: ACTIVE
📊 Escape fraction: {boundary_status.get('escaped_mass_fraction', 0):.2%}
💨 Escaped particles: {boundary_status.get('escaped_particle_count', 0)}
🛑 Auto-stop threshold: {self.escape_threshold_var.get():.0f}%

Status: {'🟢 Simulation continuing' if not boundary_status.get('should_stop', False) else '🔴 Will stop soon'}
"""
            
            self.boundary_text.insert(tk.END, update_text)
            self.boundary_text.see(tk.END)
        except:
            pass
    
    def stop_simulation(self):
        """Stop simulation."""
        if self.simulation_engine:
            self.simulation_engine.stop_simulation()
        
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        self.log_status("🛑 Simulation stopped by user")
    
    def on_time_step_changed(self, simulation_data, time_index):
        """Handle time step changes."""
        
        if hasattr(self, 'visualizer'):
            try:
                self.visualizer.update_with_time_stepping(simulation_data, time_index)
            except:
                pass
    
    def log_status(self, message):
        """Log status message."""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.status_text.insert(tk.END, formatted_message)
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def run(self):
        """Start the GUI."""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nGUI interrupted by user")
        finally:
            if self.simulation_engine:
                try:
                    self.simulation_engine.stop_simulation()
                except:
                    pass

# Main execution
SimulatorGUI = SmartBoundaryNuclearGUI
if __name__ == "__main__":
    print("🚀 Launching Nuclear Physics Simulator v4.1 - Smart Boundary Detection...")
    
    app = SmartBoundaryNuclearGUI()
    app.run()
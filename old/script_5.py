class EnhancedQuantumLatticeGUI:
    """Enhanced GUI with nuclear selection and systematic accuracy controls."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced Quantum Lattice Nuclear Collision Simulator v2.0")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e1e1e')  # Dark theme
        
        # Apply dark theme styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background='#2d2d2d', foreground='white')
        style.configure('TNotebook', background='#2d2d2d')
        style.configure('TNotebook.Tab', background='#404040', foreground='white')
        
        # Enhanced simulation parameters
        self.params = EnhancedSimulationParameters()
        self.simulator = None
        self.simulation_thread = None
        self.is_simulation_running = False
        
        # Performance monitoring
        self.performance_thread = None
        self.monitor_performance = False
        
        self.create_enhanced_interface()
        
    def create_enhanced_interface(self):
        """Create the enhanced user interface."""
        
        # Main notebook with multiple tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.collision_tab = ttk.Frame(self.notebook)
        self.nuclear_tab = ttk.Frame(self.notebook)
        self.physics_tab = ttk.Frame(self.notebook)
        self.accuracy_tab = ttk.Frame(self.notebook)
        self.performance_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.collision_tab, text="🚀 Collision Setup")
        self.notebook.add(self.nuclear_tab, text="⚛️  Nuclear Physics")
        self.notebook.add(self.physics_tab, text="🔬 Field Theory")
        self.notebook.add(self.accuracy_tab, text="📊 Systematic Accuracy")
        self.notebook.add(self.performance_tab, text="⚡ Performance")
        self.notebook.add(self.analysis_tab, text="📈 Real-time Analysis")
        
        self.create_collision_controls()
        self.create_nuclear_selection()
        self.create_physics_controls()
        self.create_accuracy_controls()
        self.create_performance_controls()
        self.create_analysis_interface()
        
    def create_collision_controls(self):
        """Create collision parameter controls."""
        
        # Header
        header = tk.Label(self.collision_tab, 
                         text="Enhanced Quantum Lattice Nuclear Collision Simulator v2.0",
                         font=('Arial', 18, 'bold'), fg='#4CAF50', bg='#1e1e1e')
        header.pack(pady=10)
        
        subtitle = tk.Label(self.collision_tab,
                           text="Systematic accuracy improvements • Full nuclear support • Multithreaded performance",
                           font=('Arial', 12), fg='#888888', bg='#1e1e1e')
        subtitle.pack(pady=5)
        
        # Collision energy frame
        energy_frame = ttk.LabelFrame(self.collision_tab, text="Collision Parameters")
        energy_frame.pack(fill='x', padx=20, pady=10)
        
        # Energy with presets
        tk.Label(energy_frame, text="Collision Energy:").grid(row=0, column=0, sticky='w', padx=5)
        self.energy_var = tk.DoubleVar(value=self.params.collision_energy_gev)
        self.energy_scale = tk.Scale(energy_frame, from_=1.0, to=5500.0, 
                                   orient='horizontal', variable=self.energy_var,
                                   resolution=1.0, length=400)
        self.energy_scale.grid(row=0, column=1, padx=5)
        
        # Energy presets
        preset_frame = ttk.Frame(energy_frame)
        preset_frame.grid(row=1, column=1, pady=5)
        
        presets = [
            ("SPS: 17 GeV", 17.3),
            ("AGS: 40 GeV", 40.0), 
            ("RHIC: 200 GeV", 200.0),
            ("LHC: 2760 GeV", 2760.0),
            ("LHC: 5020 GeV", 5020.0)
        ]
        
        for i, (label, energy) in enumerate(presets):
            btn = tk.Button(preset_frame, text=label, 
                           command=lambda e=energy: self.energy_var.set(e),
                           bg='#404040', fg='white', font=('Arial', 8))
            btn.grid(row=0, column=i, padx=2)
        
        # Impact parameter
        tk.Label(energy_frame, text="Impact Parameter (fm):").grid(row=2, column=0, sticky='w', padx=5)
        self.impact_var = tk.DoubleVar(value=self.params.impact_parameter_fm)
        self.impact_scale = tk.Scale(energy_frame, from_=0.0, to=15.0,
                                   orient='horizontal', variable=self.impact_var,
                                   resolution=0.1, length=400)
        self.impact_scale.grid(row=2, column=1, padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(self.collision_tab)
        control_frame.pack(pady=20)
        
        self.start_btn = tk.Button(control_frame, text="🚀 Start Enhanced Simulation",
                                  command=self.start_enhanced_simulation,
                                  bg='#4CAF50', fg='white', font=('Arial', 14, 'bold'),
                                  padx=30, pady=15)
        self.start_btn.pack(side='left', padx=10)
        
        self.stop_btn = tk.Button(control_frame, text="🛑 Stop Simulation",
                                 command=self.stop_simulation,
                                 bg='#f44336', fg='white', font=('Arial', 14, 'bold'),
                                 padx=30, pady=15, state='disabled')
        self.stop_btn.pack(side='left', padx=10)
        
        # Status display
        status_frame = ttk.LabelFrame(self.collision_tab, text="Simulation Status")
        status_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.status_text = tk.Text(status_frame, height=12, width=100,
                                  bg='#000000', fg='#00ff00', font=('Consolas', 10))
        self.status_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(status_frame, command=self.status_text.yview)
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
    
    def create_nuclear_selection(self):
        """Create nuclear selection interface."""
        
        # Nuclear database display
        db_frame = ttk.LabelFrame(self.nuclear_tab, text="Nuclear Database")
        db_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for nuclear data
        columns = ("Nucleus", "A", "Z", "Radius (fm)", "BE/A (MeV)", "Spin")
        self.nuclear_tree = ttk.Treeview(db_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.nuclear_tree.heading(col, text=col)
            self.nuclear_tree.column(col, width=100)
        
        # Populate nuclear database
        for name, data in NUCLEAR_DATABASE.items():
            be_per_a = data["binding_energy"] / data["A"] if data["A"] > 0 else 0
            self.nuclear_tree.insert("", "end", values=(
                name, data["A"], data["Z"], f"{data['radius_fm']:.2f}", 
                f"{be_per_a:.2f}", data["spin"]
            ))
        
        self.nuclear_tree.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Selection controls
        select_frame = ttk.Frame(self.nuclear_tab)
        select_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(select_frame, text="Projectile:").grid(row=0, column=0, padx=5)
        self.nucleus_a_var = tk.StringVar(value=self.params.nucleus_A)
        nucleus_a_combo = ttk.Combobox(select_frame, textvariable=self.nucleus_a_var,
                                      values=list(NUCLEAR_DATABASE.keys()), state='readonly')
        nucleus_a_combo.grid(row=0, column=1, padx=5)
        
        tk.Label(select_frame, text="Target:").grid(row=0, column=2, padx=5)
        self.nucleus_b_var = tk.StringVar(value=self.params.nucleus_B)
        nucleus_b_combo = ttk.Combobox(select_frame, textvariable=self.nucleus_b_var,
                                      values=list(NUCLEAR_DATABASE.keys()), state='readonly')
        nucleus_b_combo.grid(row=0, column=3, padx=5)
        
        # Nuclear structure visualization
        self.nuclear_info_text = tk.Text(self.nuclear_tab, height=8, width=80)
        self.nuclear_info_text.pack(fill='x', padx=10, pady=5)
        
        # Bind selection events
        nucleus_a_combo.bind('<<ComboboxSelected>>', self.update_nuclear_info)
        nucleus_b_combo.bind('<<ComboboxSelected>>', self.update_nuclear_info)
        
        self.update_nuclear_info()
    
    def create_physics_controls(self):
        """Create quantum field theory parameter controls."""
        
        # QED parameters
        qed_frame = ttk.LabelFrame(self.physics_tab, text="QED (Quantum Electrodynamics)")
        qed_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(qed_frame, text="Fine Structure Constant α:").grid(row=0, column=0, sticky='w')
        self.alpha_var = tk.DoubleVar(value=1/137.036)
        alpha_entry = tk.Entry(qed_frame, textvariable=self.alpha_var, width=15)
        alpha_entry.grid(row=0, column=1, padx=5)
        
        self.radiative_var = tk.BooleanVar(value=True)
        tk.Checkbutton(qed_frame, text="Include radiative corrections", 
                      variable=self.radiative_var).grid(row=0, column=2, padx=10)
        
        # QCD parameters
        qcd_frame = ttk.LabelFrame(self.physics_tab, text="QCD (Quantum Chromodynamics)")
        qcd_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(qcd_frame, text="Strong coupling αs:").grid(row=0, column=0, sticky='w')
        self.qcd_coupling_var = tk.DoubleVar(value=0.118)
        qcd_entry = tk.Entry(qcd_frame, textvariable=self.qcd_coupling_var, width=15)
        qcd_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(qcd_frame, text="Fermion action:").grid(row=1, column=0, sticky='w')
        self.fermion_action_var = tk.StringVar(value="wilson_improved")
        fermion_combo = ttk.Combobox(qcd_frame, textvariable=self.fermion_action_var,
                                   values=["wilson", "wilson_improved", "staggered", "domain_wall"])
        fermion_combo.grid(row=1, column=1, padx=5)
        
        # Electroweak parameters
        ew_frame = ttk.LabelFrame(self.physics_tab, text="Electroweak Theory")
        ew_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(ew_frame, text="Weak coupling gw:").grid(row=0, column=0, sticky='w')
        self.weak_coupling_var = tk.DoubleVar(value=0.65379)
        weak_entry = tk.Entry(ew_frame, textvariable=self.weak_coupling_var, width=15)
        weak_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(ew_frame, text="Higgs VEV (GeV):").grid(row=0, column=2, sticky='w')
        self.higgs_vev_var = tk.DoubleVar(value=246.22)
        higgs_entry = tk.Entry(ew_frame, textvariable=self.higgs_vev_var, width=15)
        higgs_entry.grid(row=0, column=3, padx=5)
    
    def create_accuracy_controls(self):
        """Create systematic accuracy improvement controls."""
        
        # Lattice parameters
        lattice_frame = ttk.LabelFrame(self.accuracy_tab, text="Multi-Scale Lattice Analysis")
        lattice_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(lattice_frame, text="Lattice spacings (fm):").grid(row=0, column=0, sticky='w')
        self.spacings_var = tk.StringVar(value="0.15, 0.10, 0.07")
        spacings_entry = tk.Entry(lattice_frame, textvariable=self.spacings_var, width=20)
        spacings_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(lattice_frame, text="Lattice sizes:").grid(row=1, column=0, sticky='w')
        self.sizes_var = tk.StringVar(value="24³, 32³, 48³")
        sizes_entry = tk.Entry(lattice_frame, textvariable=self.sizes_var, width=20)
        sizes_entry.grid(row=1, column=1, padx=5)
        
        # Time evolution improvements
        evolution_frame = ttk.LabelFrame(self.accuracy_tab, text="Time Evolution Improvements")
        evolution_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(evolution_frame, text="Trotter order:").grid(row=0, column=0, sticky='w')
        self.trotter_order_var = tk.IntVar(value=4)
        trotter_combo = ttk.Combobox(evolution_frame, textvariable=self.trotter_order_var,
                                   values=[2, 4, 6], state='readonly')
        trotter_combo.grid(row=0, column=1, padx=5)
        
        self.adaptive_timestep_var = tk.BooleanVar(value=True)
        tk.Checkbutton(evolution_frame, text="Adaptive time stepping",
                      variable=self.adaptive_timestep_var).grid(row=0, column=2, padx=10)
        
        # Systematic error analysis
        analysis_frame = ttk.LabelFrame(self.accuracy_tab, text="Systematic Error Analysis")
        analysis_frame.pack(fill='x', padx=10, pady=5)
        
        self.continuum_extrap_var = tk.BooleanVar(value=True)
        tk.Checkbutton(analysis_frame, text="Continuum extrapolation",
                      variable=self.continuum_extrap_var).grid(row=0, column=0, sticky='w')
        
        self.finite_volume_var = tk.BooleanVar(value=True)
        tk.Checkbutton(analysis_frame, text="Finite volume corrections",
                      variable=self.finite_volume_var).grid(row=0, column=1, sticky='w')
        
        # HMC parameters
        hmc_frame = ttk.LabelFrame(self.accuracy_tab, text="Hybrid Monte Carlo")
        hmc_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(hmc_frame, text="Trajectory length:").grid(row=0, column=0, sticky='w')
        self.hmc_traj_var = tk.DoubleVar(value=1.0)
        hmc_traj_entry = tk.Entry(hmc_frame, textvariable=self.hmc_traj_var, width=10)
        hmc_traj_entry.grid(row=0, column=1, padx=5)
        
        tk.Label(hmc_frame, text="Step size:").grid(row=0, column=2, sticky='w')
        self.hmc_step_var = tk.DoubleVar(value=0.02)
        hmc_step_entry = tk.Entry(hmc_frame, textvariable=self.hmc_step_var, width=10)
        hmc_step_entry.grid(row=0, column=3, padx=5)
    
    def create_performance_controls(self):
        """Create performance monitoring and optimization controls."""
        
        # Threading controls
        thread_frame = ttk.LabelFrame(self.performance_tab, text="Parallel Computing")
        thread_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(thread_frame, text=f"CPU cores available: {mp.cpu_count()}").grid(row=0, column=0, sticky='w')
        
        tk.Label(thread_frame, text="Threads to use:").grid(row=1, column=0, sticky='w')
        self.num_threads_var = tk.IntVar(value=mp.cpu_count())
        thread_scale = tk.Scale(thread_frame, from_=1, to=mp.cpu_count(),
                              orient='horizontal', variable=self.num_threads_var)
        thread_scale.grid(row=1, column=1, padx=5)
        
        self.use_c_extensions_var = tk.BooleanVar(value=True)
        tk.Checkbutton(thread_frame, text="Use C extensions",
                      variable=self.use_c_extensions_var).grid(row=2, column=0, sticky='w')
        
        self.use_gpu_var = tk.BooleanVar(value=True)
        tk.Checkbutton(thread_frame, text="GPU acceleration",
                      variable=self.use_gpu_var).grid(row=2, column=1, sticky='w')
        
        # Performance monitoring
        monitor_frame = ttk.LabelFrame(self.performance_tab, text="Performance Metrics")
        monitor_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.perf_text = tk.Text(monitor_frame, height=15, bg='#000000', fg='#00ffff',
                                font=('Consolas', 10))
        self.perf_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_analysis_interface(self):
        """Create real-time analysis interface."""
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Create matplotlib figure
            self.analysis_fig = Figure(figsize=(16, 10), facecolor='#2d2d2d')
            
            # Create subplots for multi-lattice analysis
            self.ax_temp_multi = self.analysis_fig.add_subplot(2, 3, 1)
            self.ax_energy_multi = self.analysis_fig.add_subplot(2, 3, 2)
            self.ax_chiral = self.analysis_fig.add_subplot(2, 3, 3)
            self.ax_topology = self.analysis_fig.add_subplot(2, 3, 4)
            self.ax_polyakov = self.analysis_fig.add_subplot(2, 3, 5)
            self.ax_extrapolation = self.analysis_fig.add_subplot(2, 3, 6)
            
            # Configure dark theme for plots
            for ax in [self.ax_temp_multi, self.ax_energy_multi, self.ax_chiral, 
                      self.ax_topology, self.ax_polyakov, self.ax_extrapolation]:
                ax.set_facecolor('#1e1e1e')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
            
            self.ax_temp_multi.set_title('Temperature vs Time (Multi-Scale)')
            self.ax_energy_multi.set_title('Energy Density (All Lattices)')
            self.ax_chiral.set_title('Chiral Condensate')
            self.ax_topology.set_title('Topological Charge')
            self.ax_polyakov.set_title('Polyakov Loop')
            self.ax_extrapolation.set_title('Continuum Extrapolation')
            
            self.analysis_fig.tight_layout()
            
            # Embed in tkinter
            self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, self.analysis_tab)
            self.analysis_canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except ImportError:
            tk.Label(self.analysis_tab, text="Matplotlib not available for real-time analysis",
                    fg='red', font=('Arial', 14)).pack(pady=50)
    
    def update_nuclear_info(self, event=None):
        """Update nuclear structure information display."""
        nucleus_a = self.nucleus_a_var.get()
        nucleus_b = self.nucleus_b_var.get()
        
        info_text = f"Collision System: {nucleus_a} + {nucleus_b}\\n\\n"
        
        for name, nucleus in [(nucleus_a, "Projectile"), (nucleus_b, "Target")]:
            if name in NUCLEAR_DATABASE:
                data = NUCLEAR_DATABASE[name]
                info_text += f"{nucleus} ({name}):\\n"
                info_text += f"  Mass number A = {data['A']}\\n"
                info_text += f"  Charge Z = {data['Z']}\\n"
                info_text += f"  Neutron number N = {data['A'] - data['Z']}\\n"
                info_text += f"  Nuclear radius = {data['radius_fm']:.2f} fm\\n"
                info_text += f"  Binding energy = {data['binding_energy']:.1f} MeV\\n"
                info_text += f"  BE/A = {data['binding_energy']/data['A']:.2f} MeV/nucleon\\n"
                info_text += f"  Nuclear spin = {data['spin']}\\n\\n"
        
        self.nuclear_info_text.delete(1.0, tk.END)
        self.nuclear_info_text.insert(1.0, info_text)
    
    def start_enhanced_simulation(self):
        """Start the enhanced quantum lattice simulation."""
        if self.is_simulation_running:
            return
        
        # Update parameters from GUI
        self.update_enhanced_parameters()
        
        # Create enhanced simulator
        self.simulator = EnhancedQuantumLatticeSimulator(self.params)
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self.run_enhanced_simulation_thread)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Start performance monitoring
        self.monitor_performance = True
        self.performance_thread = threading.Thread(target=self.performance_monitoring_thread)
        self.performance_thread.daemon = True
        self.performance_thread.start()
        
        # Update GUI state
        self.is_simulation_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        self.log_message("🚀 Enhanced quantum lattice simulation started!")
        self.log_message(f"Nuclear system: {self.params.nucleus_A} + {self.params.nucleus_B}")
        self.log_message(f"Collision energy: {self.params.collision_energy_gev} GeV")
        self.log_message(f"Systematic improvements: {len(self.params.lattice_sizes)} lattice scales")
    
    def update_enhanced_parameters(self):
        """Update enhanced simulation parameters from GUI."""
        self.params.collision_energy_gev = self.energy_var.get()
        self.params.impact_parameter_fm = self.impact_var.get()
        self.params.nucleus_A = self.nucleus_a_var.get()
        self.params.nucleus_B = self.nucleus_b_var.get()
        
        # Parse lattice spacings
        spacings_str = self.spacings_var.get().replace(" ", "")
        self.params.lattice_spacings_fm = [float(x) for x in spacings_str.split(",")]
        
        # Parse lattice sizes (simplified parsing)
        sizes_str = self.sizes_var.get().replace("³", "").replace(" ", "")
        sizes = [int(x) for x in sizes_str.split(",")]
        self.params.lattice_sizes = [(s, s, s) for s in sizes]
        
        # Physics parameters
        self.params.qcd_coupling = self.qcd_coupling_var.get()
        self.params.fermion_action = self.fermion_action_var.get()
        self.params.weak_coupling = self.weak_coupling_var.get()
        self.params.higgs_vev_gev = self.higgs_vev_var.get()
        
        # Accuracy parameters
        self.params.trotter_order = self.trotter_order_var.get()
        self.params.adaptive_time_stepping = self.adaptive_timestep_var.get()
        self.params.continuum_extrapolation = self.continuum_extrap_var.get()
        self.params.finite_volume_extrapolation = self.finite_volume_var.get()
        
        # HMC parameters
        self.params.hmc_trajectory_length = self.hmc_traj_var.get()
        self.params.hmc_step_size = self.hmc_step_var.get()
        
        # Performance parameters
        self.params.num_threads = self.num_threads_var.get()
        self.params.use_c_extensions = self.use_c_extensions_var.get()
        self.params.use_gpu = self.use_gpu_var.get()
    
    def run_enhanced_simulation_thread(self):
        """Run enhanced simulation in background thread."""
        try:
            results = self.simulator.run_enhanced_simulation(gui_callback=self.update_analysis_plots)
            self.log_message("✅ Simulation completed successfully!")
            
            # Display final results
            for observable, result in results.items():
                if isinstance(result, dict) and 'continuum_value' in result:
                    self.log_message(f"📊 Final {observable}: {result['continuum_value']:.6f} ± {result['continuum_error']:.6f}")
                    
        except Exception as e:
            self.log_message(f"❌ Simulation error: {str(e)}")
        finally:
            self.monitor_performance = False
            self.is_simulation_running = False
            self.root.after(0, lambda: self.start_btn.config(state='normal'))
            self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
    
    def performance_monitoring_thread(self):
        """Monitor and display performance metrics."""
        import psutil
        
        while self.monitor_performance and self.is_simulation_running:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                perf_info = f"""
🖥️  SYSTEM PERFORMANCE MONITORING
{'='*50}
CPU Usage: {cpu_percent:.1f}%
Memory Usage: {memory.percent:.1f}% ({memory.used/1024**3:.1f}/{memory.total/1024**3:.1f} GB)
Available Cores: {mp.cpu_count()}
Threads Configured: {self.params.num_threads}

⚡ SIMULATION PERFORMANCE
{'='*50}
"""
                
                if self.simulator:
                    metrics = self.simulator.performance_metrics
                    perf_info += f"""
Iterations/Second: {metrics['iterations_per_second']:.2f}
Current Iteration: {self.simulator.iteration}/{self.params.max_iterations}
Simulation Time: {self.simulator.current_time:.3f} fm/c
Progress: {100*self.simulator.iteration/self.params.max_iterations:.1f}%

🔬 PHYSICS STATUS
{'='*50}
"""
                    
                    # Add physics information if available
                    if len(self.simulator.observables["temperature"][0]) > 0:
                        latest_temp = self.simulator.observables["temperature"][0][-1]
                        latest_energy = self.simulator.observables["energy_density"][0][-1]
                        
                        perf_info += f"""
Temperature: {latest_temp:.1f} MeV
Energy Density: {latest_energy:.2e}
QGP Status: {'🔥 FORMED' if latest_temp > 170 else '❄️  HADRONIC'}
Chiral Status: {'⚖️  RESTORED' if abs(self.simulator.observables["chiral_condensate"][0][-1]) < 0.01 else '🔒 BROKEN'}
"""
                
                # Update performance display
                self.root.after(0, lambda: self.update_performance_display(perf_info))
                
            except Exception as e:
                perf_info = f"Performance monitoring error: {str(e)}"
                self.root.after(0, lambda: self.update_performance_display(perf_info))
            
            time.sleep(2)  # Update every 2 seconds
    
    def update_performance_display(self, info: str):
        """Update performance display in GUI."""
        self.perf_text.delete(1.0, tk.END)
        self.perf_text.insert(1.0, info)
    
    def stop_simulation(self):
        """Stop the running simulation."""
        if self.simulator:
            self.simulator.stop_simulation()
        
        self.monitor_performance = False
        self.is_simulation_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        self.log_message("🛑 Simulation stopped by user")
    
    def log_message(self, message: str):
        """Add timestamped message to status log."""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_analysis_plots(self, simulator):
        """Update real-time analysis plots."""
        if not hasattr(self, 'analysis_canvas'):
            return
        
        try:
            obs = simulator.observables
            
            if len(obs["time"]) < 2:
                return
            
            # Clear all subplots
            for ax in [self.ax_temp_multi, self.ax_energy_multi, self.ax_chiral,
                      self.ax_topology, self.ax_polyakov, self.ax_extrapolation]:
                ax.clear()
                ax.set_facecolor('#1e1e1e')
                ax.tick_params(colors='white')
            
            # Multi-scale temperature plot
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
            for scale in range(len(simulator.params.lattice_sizes)):
                if len(obs["temperature"][scale]) > 0:
                    spacing = simulator.params.lattice_spacings_fm[scale]
                    self.ax_temp_multi.plot(obs["time"], obs["temperature"][scale], 
                                          color=colors[scale % len(colors)], linewidth=2,
                                          label=f'a = {spacing:.3f} fm')
            
            self.ax_temp_multi.axhline(y=170, color='orange', linestyle='--', alpha=0.7)
            self.ax_temp_multi.set_title('Temperature Evolution (Multi-Scale)', color='white')
            self.ax_temp_multi.set_xlabel('Time (fm/c)', color='white')
            self.ax_temp_multi.set_ylabel('Temperature (MeV)', color='white')
            self.ax_temp_multi.legend()
            self.ax_temp_multi.grid(True, alpha=0.3)
            
            # Multi-scale energy density
            for scale in range(len(simulator.params.lattice_sizes)):
                if len(obs["energy_density"][scale]) > 0:
                    spacing = simulator.params.lattice_spacings_fm[scale]
                    self.ax_energy_multi.plot(obs["time"], obs["energy_density"][scale],
                                            color=colors[scale % len(colors)], linewidth=2,
                                            label=f'a = {spacing:.3f} fm')
            
            self.ax_energy_multi.set_title('Energy Density (Multi-Scale)', color='white')
            self.ax_energy_multi.set_xlabel('Time (fm/c)', color='white')
            self.ax_energy_multi.set_ylabel('Energy Density', color='white')
            self.ax_energy_multi.legend()
            self.ax_energy_multi.grid(True, alpha=0.3)
            
            # Chiral condensate
            if len(obs["chiral_condensate"][0]) > 0:
                self.ax_chiral.plot(obs["time"], obs["chiral_condensate"][0], 'purple', linewidth=2)
                self.ax_chiral.axhline(y=0, color='white', linestyle='-', alpha=0.5)
                self.ax_chiral.set_title('Chiral Condensate', color='white')
                self.ax_chiral.set_xlabel('Time (fm/c)', color='white')
                self.ax_chiral.set_ylabel('⟨ψ̄ψ⟩', color='white')
                self.ax_chiral.grid(True, alpha=0.3)
            
            # Topological charge
            if len(obs["topological_charge"][0]) > 0:
                self.ax_topology.plot(obs["time"], obs["topological_charge"][0], 'cyan', linewidth=2)
                self.ax_topology.set_title('Topological Charge', color='white')
                self.ax_topology.set_xlabel('Time (fm/c)', color='white')
                self.ax_topology.set_ylabel('Q', color='white')
                self.ax_topology.grid(True, alpha=0.3)
            
            # Polyakov loop (magnitude)
            if len(obs["polyakov_loop"][0]) > 0:
                polyakov_mag = [abs(p) for p in obs["polyakov_loop"][0]]
                self.ax_polyakov.plot(obs["time"], polyakov_mag, 'gold', linewidth=2)
                self.ax_polyakov.set_title('|Polyakov Loop|', color='white')
                self.ax_polyakov.set_xlabel('Time (fm/c)', color='white')
                self.ax_polyakov.set_ylabel('|⟨P⟩|', color='white')
                self.ax_polyakov.grid(True, alpha=0.3)
            
            # Continuum extrapolation (if enough data)
            if len(obs["temperature"][0]) > 10:
                # Show continuum extrapolation for temperature
                temp_values = [obs["temperature"][scale][-1] for scale in range(len(simulator.params.lattice_sizes)) if obs["temperature"][scale]]
                if len(temp_values) == len(simulator.params.lattice_spacings_fm):
                    a_values = np.array(simulator.params.lattice_spacings_fm)
                    a_sq = a_values**2
                    
                    self.ax_extrapolation.plot(a_sq, temp_values, 'ro', markersize=8, label='Data')
                    
                    # Simple linear fit for demonstration
                    if len(a_sq) >= 2:
                        fit = np.polyfit(a_sq, temp_values, 1)
                        fit_line = np.poly1d(fit)
                        a_fit = np.linspace(0, max(a_sq), 100)
                        self.ax_extrapolation.plot(a_fit, fit_line(a_fit), 'r--', alpha=0.7, label='Linear fit')
                        
                        # Continuum value
                        continuum_temp = fit_line(0)
                        self.ax_extrapolation.axhline(y=continuum_temp, color='lime', linestyle='-', 
                                                    label=f'Continuum: {continuum_temp:.1f} MeV')
                    
                    self.ax_extrapolation.set_title('Temperature Continuum Extrapolation', color='white')
                    self.ax_extrapolation.set_xlabel('a² (fm²)', color='white')
                    self.ax_extrapolation.set_ylabel('Temperature (MeV)', color='white')
                    self.ax_extrapolation.legend()
                    self.ax_extrapolation.grid(True, alpha=0.3)
            
            self.analysis_fig.tight_layout()
            self.analysis_canvas.draw()
            
        except Exception as e:
            print(f"Plot update error: {e}")
    
    def run(self):
        """Start the enhanced GUI."""
        self.log_message("🚀 Enhanced Quantum Lattice Nuclear Collision Simulator v2.0")
        self.log_message("✨ Systematic accuracy improvements enabled")
        self.log_message("⚛️  Full nuclear database available")
        self.log_message("⚡ Multithreaded high-performance computing ready")
        self.log_message(f"🖥️  System: {mp.cpu_count()} CPU cores detected")
        self.root.mainloop()

print("🎯 Enhanced GUI completed with full nuclear support and systematic accuracy!")
print("🚀 Ready to launch advanced quantum lattice nuclear collision simulations!")
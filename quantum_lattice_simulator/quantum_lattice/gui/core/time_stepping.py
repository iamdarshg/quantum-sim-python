"""
Enhanced Time Stepping Controls with Bidirectional Navigation
Complete playback controls for nuclear collision analysis.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Dict, List, Optional, Any, Callable

class BidirectionalTimeSteppingControls:
    """Enhanced time stepping with full bidirectional navigation and playback."""
    
    def __init__(self, parent_frame, visualizer_callback):
        self.parent = parent_frame
        self.visualizer_callback = visualizer_callback
        self.simulation_data = None
        self.time_history = []
        self.current_index = 0
        self.max_time_index = 0
        
        # Playback state
        self.is_playing_forward = False
        self.is_playing_backward = False
        self.play_speed = 1.0
        self.animation_job = None
        
        # Bookmarks for interesting time points
        self.bookmarks = {}
        
        self.create_enhanced_controls()
    
    def create_enhanced_controls(self):
        """Create comprehensive time stepping interface."""
        
        # Main control frame with better styling
        control_frame = tk.Frame(self.parent, bg='#1e1e2e', pady=15)
        control_frame.pack(fill='x', padx=15, pady=10)
        
        # Enhanced title with physics info
        title_frame = tk.Frame(control_frame, bg='#1e1e2e')
        title_frame.pack(fill='x', pady=(0, 15))
        
        title_label = tk.Label(
            title_frame,
            text="⏱️ BIDIRECTIONAL TIME STEPPING & NUCLEAR EQUATION ANALYSIS",
            font=('Arial', 16, 'bold'),
            bg='#1e1e2e', fg='#cba6f7'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Navigate through collision evolution • View nuclear reactions • Analyze momentum transfer",
            font=('Arial', 11),
            bg='#1e1e2e', fg='#9399b2'
        )
        subtitle_label.pack()
        
        # Current time and physics display
        info_frame = tk.Frame(control_frame, bg='#313244', relief='sunken', bd=2)
        info_frame.pack(fill='x', pady=10, padx=20, ipady=10)
        
        # Time display
        time_frame = tk.Frame(info_frame, bg='#313244')
        time_frame.pack(fill='x')
        
        tk.Label(time_frame, text="Current Time:", font=('Arial', 12, 'bold'),
                bg='#313244', fg='#f38ba8').pack(side='left')
        
        self.time_display_var = tk.StringVar(value="0.000 fm/c")
        time_display = tk.Label(time_frame, textvariable=self.time_display_var,
                               font=('Courier New', 14, 'bold'),
                               bg='#313244', fg='#a6e3a1')
        time_display.pack(side='left', padx=(10, 0))
        
        # Physics display
        physics_frame = tk.Frame(info_frame, bg='#313244')
        physics_frame.pack(fill='x', pady=(5, 0))
        
        self.physics_info_var = tk.StringVar(value="Particles: 0 | Reactions: 0 | Escaped: 0%")
        physics_info = tk.Label(physics_frame, textvariable=self.physics_info_var,
                               font=('Arial', 11),
                               bg='#313244', fg='#fab387')
        physics_info.pack()
        
        # Enhanced control buttons
        button_frame = tk.Frame(control_frame, bg='#1e1e2e')
        button_frame.pack(pady=15)
        
        # Navigation buttons - First row
        nav_frame1 = tk.Frame(button_frame, bg='#1e1e2e')
        nav_frame1.pack(pady=5)
        
        # First/Previous controls
        tk.Button(nav_frame1, text="⏮️ First", command=self.go_to_first,
                 bg='#89b4fa', fg='white', font=('Arial', 10, 'bold'), 
                 width=10, height=2).pack(side='left', padx=3)
        
        tk.Button(nav_frame1, text="⏪ -100", command=self.step_back_100,
                 bg='#f38ba8', fg='white', font=('Arial', 10, 'bold'),
                 width=10, height=2).pack(side='left', padx=3)
        
        tk.Button(nav_frame1, text="◀️ -10", command=self.step_back_10,
                 bg='#eba0ac', fg='white', font=('Arial', 10, 'bold'),
                 width=10, height=2).pack(side='left', padx=3)
        
        tk.Button(nav_frame1, text="◀ -1", command=self.step_back,
                 bg='#fab387', fg='white', font=('Arial', 10, 'bold'),
                 width=10, height=2).pack(side='left', padx=3)
        
        # Play controls
        self.play_backward_button = tk.Button(nav_frame1, text="◀️◀️ Play Back",
                                            command=self.toggle_play_backward,
                                            bg='#f9e2af', fg='black', 
                                            font=('Arial', 10, 'bold'),
                                            width=12, height=2)
        self.play_backward_button.pack(side='left', padx=5)
        
        self.pause_button = tk.Button(nav_frame1, text="⏸️ Pause",
                                    command=self.pause_all,
                                    bg='#6c7086', fg='white',
                                    font=('Arial', 10, 'bold'),
                                    width=10, height=2, state='disabled')
        self.pause_button.pack(side='left', padx=5)
        
        self.play_forward_button = tk.Button(nav_frame1, text="Play ▶️▶️",
                                           command=self.toggle_play_forward,
                                           bg='#a6e3a1', fg='black',
                                           font=('Arial', 10, 'bold'),
                                           width=12, height=2)
        self.play_forward_button.pack(side='left', padx=5)
        
        # Navigation buttons - Second row
        nav_frame2 = tk.Frame(button_frame, bg='#1e1e2e')
        nav_frame2.pack(pady=5)
        
        tk.Button(nav_frame2, text="+1 ▶", command=self.step_forward,
                 bg='#fab387', fg='white', font=('Arial', 10, 'bold'),
                 width=10, height=2).pack(side='left', padx=3)
        
        tk.Button(nav_frame2, text="+10 ⏩", command=self.step_forward_10,
                 bg='#eba0ac', fg='white', font=('Arial', 10, 'bold'),
                 width=10, height=2).pack(side='left', padx=3)
        
        tk.Button(nav_frame2, text="+100 ⏭️", command=self.step_forward_100,
                 bg='#f38ba8', fg='white', font=('Arial', 10, 'bold'),
                 width=10, height=2).pack(side='left', padx=3)
        
        tk.Button(nav_frame2, text="Last ⏭️", command=self.go_to_last,
                 bg='#89b4fa', fg='white', font=('Arial', 10, 'bold'),
                 width=10, height=2).pack(side='left', padx=3)
        
        # Enhanced time slider with markers
        slider_frame = tk.Frame(control_frame, bg='#1e1e2e')
        slider_frame.pack(fill='x', pady=15, padx=20)
        
        slider_label_frame = tk.Frame(slider_frame, bg='#1e1e2e')
        slider_label_frame.pack(fill='x', pady=(0, 5))
        
        tk.Label(slider_label_frame, text="Time Navigation:", 
                font=('Arial', 12, 'bold'), bg='#1e1e2e', fg='#cba6f7').pack(side='left')
        
        # Current step info
        self.step_info_var = tk.StringVar(value="Step 0 / 0")
        tk.Label(slider_label_frame, textvariable=self.step_info_var,
                font=('Arial', 10), bg='#1e1e2e', fg='#9399b2').pack(side='right')
        
        # Time slider
        self.time_slider = tk.Scale(
            slider_frame, from_=0, to=100, orient='horizontal',
            command=self.on_slider_changed,
            bg='#313244', fg='#cba6f7', 
            highlightbackground='#1e1e2e',
            troughcolor='#585b70',
            activebackground='#89b4fa',
            length=800, width=20,
            font=('Arial', 10)
        )
        self.time_slider.pack(fill='x', pady=5)
        
        # Bookmark controls
        bookmark_frame = tk.Frame(control_frame, bg='#1e1e2e')
        bookmark_frame.pack(fill='x', pady=10, padx=20)
        
        tk.Label(bookmark_frame, text="📌 Bookmarks:", font=('Arial', 11, 'bold'),
                bg='#1e1e2e', fg='#f2cdcd').pack(side='left')
        
        tk.Button(bookmark_frame, text="📌 Bookmark Current", 
                 command=self.add_bookmark,
                 bg='#f2cdcd', fg='black', font=('Arial', 9, 'bold')).pack(side='left', padx=5)
        
        self.bookmark_listbox = tk.Listbox(bookmark_frame, height=1, bg='#313244', 
                                         fg='#cdd6f4', selectbackground='#89b4fa',
                                         font=('Arial', 9), width=30)
        self.bookmark_listbox.pack(side='left', padx=10)
        self.bookmark_listbox.bind('<Double-Button-1>', self.go_to_bookmark)
        
        tk.Button(bookmark_frame, text="🗑️ Clear", command=self.clear_bookmarks,
                 bg='#f38ba8', fg='white', font=('Arial', 9, 'bold')).pack(side='left', padx=5)
        
        # Animation and analysis controls
        analysis_frame = tk.Frame(control_frame, bg='#1e1e2e')
        analysis_frame.pack(fill='x', pady=15, padx=20)
        
        # Speed control
        speed_frame = tk.Frame(analysis_frame, bg='#1e1e2e')
        speed_frame.pack(side='left')
        
        tk.Label(speed_frame, text="🎬 Playback Speed:", font=('Arial', 10, 'bold'),
                bg='#1e1e2e', fg='#94e2d5').pack()
        
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = tk.Scale(
            speed_frame, from_=0.1, to=10.0, resolution=0.1,
            orient='horizontal', variable=self.speed_var,
            bg='#313244', fg='#94e2d5', length=200,
            command=self.on_speed_changed
        )
        speed_scale.pack()
        
        # Analysis controls
        analysis_controls = tk.Frame(analysis_frame, bg='#1e1e2e')
        analysis_controls.pack(side='right')
        
        tk.Label(analysis_controls, text="🔬 Analysis:", font=('Arial', 10, 'bold'),
                bg='#1e1e2e', fg='#b4befe').pack()
        
        controls_row = tk.Frame(analysis_controls, bg='#1e1e2e')
        controls_row.pack()
        
        tk.Button(controls_row, text="⚛️ Show Reactions", command=self.show_reactions,
                 bg='#b4befe', fg='black', font=('Arial', 9, 'bold')).pack(side='left', padx=2)
        
        tk.Button(controls_row, text="📊 Momentum Analysis", command=self.analyze_momentum,
                 bg='#cba6f7', fg='black', font=('Arial', 9, 'bold')).pack(side='left', padx=2)
        
        tk.Button(controls_row, text="💾 Export Frame", command=self.export_current_frame,
                 bg='#89dceb', fg='black', font=('Arial', 9, 'bold')).pack(side='left', padx=2)
        
        # Status display
        status_frame = tk.Frame(control_frame, bg='#313244', relief='sunken', bd=2)
        status_frame.pack(fill='x', pady=(10, 0), padx=20, ipady=5)
        
        self.status_var = tk.StringVar(value="⚡ Ready for time stepping navigation")
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                               bg='#313244', fg='#a6adc8', font=('Arial', 10))
        status_label.pack()
        
        # Nuclear equation display (initially hidden)
        self.equation_frame = None
    
    def set_simulation_data(self, simulation_data):
        """Set simulation data for enhanced time stepping."""
        
        self.simulation_data = simulation_data
        
        if 'time_history' in simulation_data:
            self.time_history = simulation_data['time_history']
            self.max_time_index = len(self.time_history) - 1
            self.time_slider.configure(to=self.max_time_index)
            
            # Update status
            self.status_var.set(f"🎬 Loaded {self.max_time_index + 1} time steps for analysis")
            
            # Initialize display
            self.go_to_first()
            
            # Auto-bookmark interesting events
            self._create_auto_bookmarks()
    
    def _create_auto_bookmarks(self):
        """Automatically create bookmarks for interesting events."""
        
        if not self.time_history:
            return
        
        # Bookmark first collision
        if len(self.time_history) > 10:
            self.bookmarks["🎯 First Contact"] = 10
        
        # Bookmark maximum particle count
        max_particles = 0
        max_particle_time = 0
        
        for i, state in enumerate(self.time_history):
            particle_count = len(state.get('particles', []))
            if particle_count > max_particles:
                max_particles = particle_count
                max_particle_time = i
        
        if max_particle_time > 0:
            self.bookmarks["🌟 Max Particles"] = max_particle_time
        
        # Bookmark when reactions start
        for i, state in enumerate(self.time_history):
            if state.get('total_reactions', 0) > 0:
                self.bookmarks["⚛️ First Reaction"] = i
                break
        
        # Bookmark significant escape events
        for i, state in enumerate(self.time_history):
            escaped_frac = state.get('escaped_mass_fraction', 0)
            if escaped_frac > 0.1:  # 10% escaped
                self.bookmarks["🚀 10% Escaped"] = i
                break
        
        # Update bookmark display
        self._update_bookmark_display()
    
    def _update_bookmark_display(self):
        """Update the bookmark listbox."""
        
        self.bookmark_listbox.delete(0, tk.END)
        
        for name, index in sorted(self.bookmarks.items(), key=lambda x: x[1]):
            time_val = self.time_history[index]['time'] if index < len(self.time_history) else 0
            display_text = f"{name} (t={time_val:.2f} fm/c)"
            self.bookmark_listbox.insert(tk.END, display_text)
    
    # Navigation methods
    def go_to_first(self):
        """Go to first time step."""
        self.current_index = 0
        self._update_display()
    
    def go_to_last(self):
        """Go to last time step."""
        self.current_index = self.max_time_index
        self._update_display()
    
    def step_forward(self):
        """Step forward one time step."""
        if self.current_index < self.max_time_index:
            self.current_index += 1
            self._update_display()
    
    def step_back(self):
        """Step back one time step."""
        if self.current_index > 0:
            self.current_index -= 1
            self._update_display()
    
    def step_forward_10(self):
        """Step forward 10 time steps."""
        self.current_index = min(self.current_index + 10, self.max_time_index)
        self._update_display()
    
    def step_back_10(self):
        """Step back 10 time steps."""
        self.current_index = max(self.current_index - 10, 0)
        self._update_display()
    
    def step_forward_100(self):
        """Step forward 100 time steps."""
        self.current_index = min(self.current_index + 100, self.max_time_index)
        self._update_display()
    
    def step_back_100(self):
        """Step back 100 time steps."""
        self.current_index = max(self.current_index - 100, 0)
        self._update_display()
    
    def toggle_play_forward(self):
        """Toggle forward playback."""
        
        if self.is_playing_forward:
            self._stop_all_playback()
        else:
            self._stop_all_playback()
            self.is_playing_forward = True
            self.play_forward_button.configure(text="⏸️ Pause Fwd", bg='#f38ba8')
            self.pause_button.configure(state='normal')
            self._animate_forward()
    
    def toggle_play_backward(self):
        """Toggle backward playback."""
        
        if self.is_playing_backward:
            self._stop_all_playback()
        else:
            self._stop_all_playback()
            self.is_playing_backward = True
            self.play_backward_button.configure(text="⏸️ Pause Back", bg='#f38ba8')
            self.pause_button.configure(state='normal')
            self._animate_backward()
    
    def pause_all(self):
        """Pause all playback."""
        self._stop_all_playback()
    
    def _stop_all_playback(self):
        """Stop all playback animations."""
        
        self.is_playing_forward = False
        self.is_playing_backward = False
        
        self.play_forward_button.configure(text="Play ▶️▶️", bg='#a6e3a1')
        self.play_backward_button.configure(text="◀️◀️ Play Back", bg='#f9e2af')
        self.pause_button.configure(state='disabled')
        
        if self.animation_job:
            self.parent.after_cancel(self.animation_job)
            self.animation_job = None
    
    def _animate_forward(self):
        """Animate forward through time steps."""
        
        if not self.is_playing_forward:
            return
        
        if self.current_index < self.max_time_index:
            self.current_index += 1
            self._update_display()
            
            # Schedule next frame
            delay = max(10, int(100 / self.play_speed))
            self.animation_job = self.parent.after(delay, self._animate_forward)
        else:
            # Reached end, stop
            self._stop_all_playback()
    
    def _animate_backward(self):
        """Animate backward through time steps."""
        
        if not self.is_playing_backward:
            return
        
        if self.current_index > 0:
            self.current_index -= 1
            self._update_display()
            
            # Schedule next frame
            delay = max(10, int(100 / self.play_speed))
            self.animation_job = self.parent.after(delay, self._animate_backward)
        else:
            # Reached beginning, stop
            self._stop_all_playback()
    
    def on_slider_changed(self, value):
        """Handle slider position change."""
        
        new_index = int(float(value))
        if new_index != self.current_index:
            self.current_index = new_index
            self._update_display()
    
    def on_speed_changed(self, value):
        """Handle speed change."""
        self.play_speed = float(value)
        self.status_var.set(f"🎬 Playback speed: {self.play_speed:.1f}x")
    
    def add_bookmark(self):
        """Add bookmark at current time."""
        
        if not self.time_history or self.current_index >= len(self.time_history):
            return
        
        current_time = self.time_history[self.current_index]['time']
        bookmark_name = f"📍 User t={current_time:.2f}"
        
        self.bookmarks[bookmark_name] = self.current_index
        self._update_bookmark_display()
        self.status_var.set(f"📌 Bookmarked time {current_time:.3f} fm/c")
    
    def go_to_bookmark(self, event):
        """Go to selected bookmark."""
        
        selection = self.bookmark_listbox.curselection()
        if not selection:
            return
        
        bookmark_text = self.bookmark_listbox.get(selection[0])
        
        # Find bookmark by text
        for name, index in self.bookmarks.items():
            if name in bookmark_text:
                self.current_index = index
                self._update_display()
                break
    
    def clear_bookmarks(self):
        """Clear all bookmarks."""
        self.bookmarks.clear()
        self._update_bookmark_display()
        self.status_var.set("🗑️ All bookmarks cleared")
    
    def show_reactions(self):
        """Show nuclear reactions for current time step."""
        
        if not self.simulation_data or 'nuclear_reactions' not in self.simulation_data:
            self.status_var.set("⚠️ No reaction data available")
            return
        
        # Create reaction display window
        self._show_reaction_window()
    
    def _show_reaction_window(self):
        """Display nuclear reactions in popup window."""
        
        reaction_window = tk.Toplevel(self.parent)
        reaction_window.title("⚛️ Nuclear Reactions")
        reaction_window.geometry("800x600")
        reaction_window.configure(bg='#1e1e2e')
        
        # Reaction text display
        text_frame = tk.Frame(reaction_window)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        reaction_text = tk.Text(text_frame, bg='#313244', fg='#cdd6f4',
                               font=('Courier New', 11), wrap='word')
        
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=reaction_text.yview)
        reaction_text.configure(yscrollcommand=scrollbar.set)
        
        reaction_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Get reaction equations
        reactions = self.simulation_data.get('nuclear_reactions', {})
        equations_text = reactions.get('equations', 'No nuclear reactions detected.')
        
        reaction_text.insert('1.0', equations_text)
        reaction_text.configure(state='disabled')
    
    def analyze_momentum(self):
        """Analyze momentum transfer at current time."""
        
        if not self.time_history or self.current_index >= len(self.time_history):
            self.status_var.set("⚠️ No time step data available")
            return
        
        current_state = self.time_history[self.current_index]
        particles = current_state.get('particles', [])
        
        # Calculate momentum statistics
        total_momentum = np.zeros(3)
        total_energy = 0.0
        
        for particle in particles:
            momentum = np.array(particle.get('momentum', [0, 0, 0]))
            energy = particle.get('energy', 0)
            
            total_momentum += momentum
            total_energy += energy
        
        momentum_mag = np.linalg.norm(total_momentum)
        
        # Display results
        self.status_var.set(f"📊 Total momentum: {momentum_mag:.3f} GeV/c, Energy: {total_energy:.3f} GeV")
    
    def export_current_frame(self):
        """Export current frame data."""
        
        if not self.time_history or self.current_index >= len(self.time_history):
            self.status_var.set("⚠️ No frame to export")
            return
        
        current_state = self.time_history[self.current_index]
        current_time = current_state['time']
        
        # Simple export (could be enhanced to save to file)
        print(f"📁 Exporting frame at t = {current_time:.3f} fm/c")
        print(f"Particles: {len(current_state.get('particles', []))}")
        print(f"Reactions: {current_state.get('total_reactions', 0)}")
        
        self.status_var.set(f"💾 Exported frame at t = {current_time:.3f} fm/c")
    
    def _update_display(self):
        """Update all display elements."""
        
        # Update slider position
        self.time_slider.set(self.current_index)
        
        # Update step info
        self.step_info_var.set(f"Step {self.current_index + 1} / {self.max_time_index + 1}")
        
        # Update time and physics info
        if (self.time_history and self.current_index < len(self.time_history)):
            current_state = self.time_history[self.current_index]
            current_time = current_state['time']
            
            self.time_display_var.set(f"{current_time:.3f} fm/c")
            
            # Physics info
            particles = len(current_state.get('particles', []))
            reactions = current_state.get('total_reactions', 0)
            escaped_frac = current_state.get('escaped_mass_fraction', 0)
            
            self.physics_info_var.set(
                f"Particles: {particles} | Reactions: {reactions} | Escaped: {escaped_frac:.1%}"
            )
            
            # Update visualizer
            if self.visualizer_callback:
                self.visualizer_callback(self.simulation_data, self.current_index)
            
            # Update status
            if not (self.is_playing_forward or self.is_playing_backward):
                self.status_var.set(f"⚡ Viewing t = {current_time:.3f} fm/c")

# Export main class
__all__ = ['BidirectionalTimeSteppingControls']
#!/usr/bin/env python3
"""
Quantum Lattice Nuclear Collision Simulator v2.0 - Demo

This script demonstrates the quantum lattice simulator with various collision scenarios.
"""

import sys
import os
import time

def main():
    """Main demo function."""
    print("🎯 QUANTUM LATTICE NUCLEAR COLLISION SIMULATOR v2.0")
    print("Advanced quantum field theory simulation with 3D visualization")
    print("="*70)
    
    try:
        # Try to import the package
        from quantum_lattice import (
            create_simulator, 
            launch_gui, 
            list_nuclei, 
            version_info,
            SimulationParameters,
            NuclearDatabase
        )
        print("✅ Successfully imported quantum_lattice package")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTry installing the package first:")
        print("   pip install -e .")
        return
    
    # Show system info
    version_info()
    
    # Check if command line arguments provided
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ["gui", "interface"]:
            demo_gui_launch()
        elif arg in ["basic", "simple"]:
            demo_basic_collision()
        elif arg in ["database", "nuclei"]:
            demo_nuclear_database()
        elif arg in ["help", "-h", "--help"]:
            print("\nUsage:")
            print("   python demo.py              # Interactive menu")
            print("   python demo.py gui          # Launch GUI directly")
            print("   python demo.py basic        # Run basic collision")
            print("   python demo.py database     # Show nuclear database")
        else:
            print(f"Unknown argument: {arg}")
            print("Use 'python demo.py help' for usage info")
    else:
        # Interactive mode
        interactive_menu()

def demo_basic_collision():
    """Demonstrate basic Au+Au collision."""
    print("\n🔬 DEMO 1: Basic Au+Au Collision at RHIC Energy")
    print("=" * 60)
    
    from quantum_lattice import create_simulator
    
    # Create simulator with defaults
    simulator = create_simulator("Au197", "Au197", 200.0)
    
    # Run short simulation for demo
    simulator.params.max_iterations = 100  # Short for demo
    
    print("🚀 Starting simulation...")
    results = simulator.run_simulation()
    
    print("\n📊 Results Summary:")
    obs = results['observables']
    if obs['temperature']:
        max_temp = max(obs['temperature'])
        final_mult = obs['multiplicity'][-1] if obs['multiplicity'] else 0
        
        print(f"   Maximum temperature: {max_temp:.1f} MeV")
        print(f"   Final multiplicity: {final_mult:.0f} particles")
        print(f"   QGP formation: {'✅ Yes' if max_temp > 170 else '❌ No'}")
    
    return results

def demo_light_heavy_collision():
    """Demonstrate light-heavy collision."""
    print("\n🔬 DEMO 2: Light-Heavy Collision (O16 + Au197)")
    print("=" * 60)
    
    from quantum_lattice import SimulationParameters
    from quantum_lattice.core.simulator import QuantumLatticeSimulator
    
    # Create parameters for asymmetric collision
    params = SimulationParameters()
    params.nucleus_A = "O16"
    params.nucleus_B = "Au197"
    params.collision_energy_gev = 0.01
    params.max_iterations = 1000
    
    print(f"Collision system: {params.nucleus_A} + {params.nucleus_B}")
    print(f"Energy: {params.collision_energy_gev} GeV")
    
    # Create and run simulator
    simulator = QuantumLatticeSimulator(params)
    
    results = simulator.run_simulation()
    
    print("\n📊 Asymmetric collision completed!")
    return results

def demo_nuclear_database():
    """Demonstrate nuclear database features."""
    print("\n🔬 DEMO 3: Nuclear Database Exploration")
    print("=" * 60)
    
    from quantum_lattice import list_nuclei, NuclearDatabase
    
    # List available nuclei
    nuclei = list_nuclei()
    print(f"📚 Available nuclei ({len(nuclei)} total):")
    
    # Group by mass
    light = [n for n in nuclei if NuclearDatabase.get_nucleus_info(n)['A'] <= 20]
    medium = [n for n in nuclei if 20 < NuclearDatabase.get_nucleus_info(n)['A'] <= 100]
    heavy = [n for n in nuclei if NuclearDatabase.get_nucleus_info(n)['A'] > 100]
    
    print(f"   Light nuclei (A ≤ 20): {', '.join(light)}")
    print(f"   Medium nuclei (20 < A ≤ 100): {', '.join(medium)}")  
    print(f"   Heavy nuclei (A > 100): {', '.join(heavy)}")
    
    # Show details for interesting nuclei
    interesting = ["He4", "C12", "Fe56", "Au197", "U238"]
    print("\n🔍 Nuclear Properties:")
    
    for nucleus in interesting:
        if nucleus in nuclei:
            info = NuclearDatabase.get_nucleus_info(nucleus)
            be_per_a = info['binding_energy'] / info['A'] if info['A'] > 0 else 0
            print(f"   {nucleus:>5}: A={info['A']:3d}, Z={info['Z']:2d}, "
                  f"R={info['radius_fm']:5.2f} fm, BE/A={be_per_a:6.2f} MeV")
    
    # Show common collision systems
    print("\n🎯 Common Experimental Collision Systems:")
    systems = NuclearDatabase.get_collision_systems()
    for i, (proj, targ) in enumerate(systems, 1):
        print(f"   {i}. {proj} + {targ}")

def demo_gui_launch():
    """Demonstrate GUI launch."""
    print("\n🔬 DEMO 4: GUI Launch")
    print("=" * 60)
    
    try:
        from quantum_lattice import launch_gui
        
        print("🖼️  Launching graphical interface...")
        print("   - Select nuclei from dropdown menus")
        print("   - Adjust collision energy with slider")
        print("   - Watch 3D visualization in real-time")
        print("   - View physics analysis")
        print("\n💡 Close the GUI window to continue...")
        
        # Launch GUI (this will block until GUI is closed)
        launch_gui()
        
        print("✅ GUI demo completed!")
        
    except Exception as e:
        print(f"⚠️  GUI demo failed: {e}")
        print("   (This is normal if running in headless environment)")

def interactive_menu():
    """Interactive demo menu."""
    while True:
        print("\n" + "="*60)
        print("🚀 QUANTUM LATTICE SIMULATOR - INTERACTIVE DEMO")
        print("="*60)
        
        options = [
            "1. Basic Au+Au collision (RHIC energy)",
            "2. Light-heavy collision (O16 + Au197)", 
            "3. Explore nuclear database",
            "4. Launch 3D GUI (if available)",
            "5. Show version info",
            "6. Exit"
        ]
        
        for option in options:
            print(f"   {option}")
        
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "1":
                demo_basic_collision()
            elif choice == "2":
                demo_light_heavy_collision()
            elif choice == "3":
                demo_nuclear_database()
            elif choice == "4":
                demo_gui_launch()
            elif choice == "5":
                from quantum_lattice import version_info
                version_info()
            elif choice == "6":
                print("\n👋 Thanks for trying the Quantum Lattice Simulator!")
                break
            else:
                print("❌ Invalid option. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()


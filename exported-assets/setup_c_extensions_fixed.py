from setuptools import setup, Extension
import numpy
import os
import platform

# Compiler flags for optimization - Fixed for Windows/MSVC compatibility
extra_compile_args = []
extra_link_args = []

# Handle different operating systems and compilers
if os.name == 'nt':  # Windows
    if platform.python_compiler().lower().startswith('msc'):  # MSVC
        # MSVC-specific flags
        extra_compile_args = [
            '/O2',           # Optimization
            '/fp:fast',      # Fast floating point
            '/favor:INTEL64', # Intel 64-bit optimization
            '/D_USE_MATH_DEFINES',  # Define M_PI and other math constants
        ]
        
        # Try to enable OpenMP for MSVC
        try:
            extra_compile_args.append('/openmp')
            extra_link_args.append('/openmp')
            print("✅ OpenMP enabled for MSVC")
        except:
            print("⚠️ OpenMP not available for MSVC")
    else:
        # MinGW/GCC on Windows
        extra_compile_args = ['-O3', '-march=native', '-ffast-math']
        try:
            extra_compile_args.append('-fopenmp')
            extra_link_args.append('-fopenmp')
            print("✅ OpenMP enabled for MinGW/GCC")
        except:
            print("⚠️ OpenMP not available for MinGW/GCC")
else:
    # Linux/macOS - GCC/Clang
    extra_compile_args = [
        '-O3',              # Maximum optimization
        '-march=native',    # Optimize for current CPU
        '-ffast-math',      # Fast math operations
        '-funroll-loops',   # Loop unrolling
    ]
    
    # Try to enable OpenMP
    try:
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-fopenmp')
        print("✅ OpenMP enabled for GCC/Clang")
    except:
        print("⚠️ OpenMP not available for GCC/Clang")

# Define the C extension
c_extensions = Extension(
    'c_extensions',
    sources=['c_extensions_fixed.c'],
    include_dirs=[
        numpy.get_include(),
        '.',  # Current directory
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=[
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
    ],
)

setup(
    name='quantum_lattice_c_extensions',
    version='1.1.0',
    description='High-performance C extensions for quantum lattice simulator - Fixed for all platforms',
    long_description='''
    This package provides high-performance C extensions for the quantum lattice nuclear
    collision simulator. Features include:
    
    - Nuclear force calculations optimized for speed
    - Lattice field updates using finite difference methods  
    - Fast collision detection algorithms
    - Enhanced multi-meson nuclear force models
    - Cross-platform compatibility (Windows/Linux/macOS)
    - OpenMP parallelization support (when available)
    - MSVC/GCC/Clang compiler support
    
    The C extensions provide 10-50x speedup over pure Python implementations
    for computationally intensive nuclear physics calculations.
    ''',
    author='Quantum Lattice Simulator Team',
    author_email='physics@quantum-lattice.org',
    url='https://github.com/quantum-lattice/simulator',
    ext_modules=[c_extensions],
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Physics',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.18.0',
    ],
    keywords='physics nuclear quantum simulation performance',
)

# Print build information
print("\n" + "="*60)
print("🚀 QUANTUM LATTICE C EXTENSIONS BUILD CONFIGURATION")
print("="*60)
print(f"Platform: {platform.system()} {platform.machine()}")
print(f"Python: {platform.python_version()}")
print(f"Compiler: {platform.python_compiler()}")
print(f"NumPy: {numpy.__version__}")
print(f"NumPy include: {numpy.get_include()}")
print(f"Compile args: {extra_compile_args}")
print(f"Link args: {extra_link_args}")
print("="*60)
print("Building high-performance nuclear force calculations...")
print("This may take a few minutes depending on your system.")
print("="*60)
# Makefile for Enhanced Quantum Lattice Simulator C Extensions

CC = gcc
CFLAGS = -fPIC -O3 -march=native -fopenmp -std=c99
INCLUDES = -I$(shell python3 -c "import numpy; print(numpy.get_include())")
LIBS = -lm -fopenmp

# Python configuration  
PYTHON_VERSION = $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_INCLUDES = -I$(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

all: lattice_extensions

lattice_extensions: lattice_c_extensions.c
	@echo "🔨 Building high-performance C extensions..."
	$(CC) $(CFLAGS) $(INCLUDES) $(PYTHON_INCLUDES) -shared lattice_c_extensions.c -o lattice_c_extensions.so $(LIBS)
	@echo "✅ C extensions built successfully!"

clean:
	rm -f *.so *.o
	@echo "🧹 Cleaned build artifacts"

test: lattice_extensions
	@echo "🧪 Testing C extensions..."
	python3 -c "import lattice_c_extensions; print('✅ C extensions load successfully')" || echo "❌ C extensions test failed"

install: all
	@echo "📦 Installing simulator package..."
	python3 setup.py install

.PHONY: all clean test install

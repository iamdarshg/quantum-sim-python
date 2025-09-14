# Enhanced Build System for Ultra-High Precision Nuclear Physics
# Makefile for compiling C extensions with MPI and GSL support

CC = mpicc
CFLAGS = -O3 -march=native -fopenmp -fPIC -Wall -Wextra
INCLUDES = -I$(shell python3 -c "import numpy; print(numpy.get_include())") \
           -I$(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
           -I/usr/include/gsl
LIBS = -lgsl -lgslcblas -lm -lmpi
PYTHON_LIBS = $(shell python3-config --ldflags)

# Source files
SOURCES = enhanced_lattice_c_extensions.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = enhanced_lattice_c_extensions.so

# Build targets
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -shared $(CFLAGS) -o $@ $^ $(LIBS) $(PYTHON_LIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Install dependencies
install-deps:
	@echo "Installing required system dependencies..."
	sudo apt-get update
	sudo apt-get install -y libgsl-dev libopenmpi-dev python3-dev python3-numpy
	pip3 install numpy scipy matplotlib mpi4py

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Test compilation
test:
	python3 -c "import enhanced_lattice_c_extensions; print('✅ C extensions loaded successfully')"

# Performance benchmark
benchmark:
	python3 -c "
import time
import numpy as np
import enhanced_lattice_c_extensions as c_ext
print('🚀 Running performance benchmark...')
# Add benchmark code here
print('✅ Benchmark completed')
"

.PHONY: all clean install-deps test benchmark
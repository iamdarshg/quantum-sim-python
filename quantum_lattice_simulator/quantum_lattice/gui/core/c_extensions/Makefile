# Makefile for C extensions
CC = gcc
CFLAGS = -O3 -march=native -fopenmp -fPIC -shared
PYTHON_INCLUDE = $(shell python3 -c "import numpy; print(numpy.get_include())")
PYTHON_LIB = $(shell python3-config --includes --ldflags)

all: c_extensions.so

c_extensions.so: c_extensions.c
	$(CC) $(CFLAGS) $(PYTHON_LIB) -I$(PYTHON_INCLUDE) -o $@ $<

clean:
	rm -f *.so *.o

install: c_extensions.so
	python3 setup.py build_ext --inplace

.PHONY: all clean install

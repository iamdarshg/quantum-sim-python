
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <stdio.h>

// Define M_PI for Windows/MSVC compatibility
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// OpenMP support (optional)
#ifdef _OPENMP
#include <omp.h>
#else
// Define empty macros if OpenMP not available
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#pragma message("OpenMP not available, using single-threaded version")
#endif

// Nuclear force calculation in C - much faster than Python
static PyObject* calculate_nuclear_forces_c(PyObject* self, PyObject* args) {
    PyArrayObject *positions, *momenta, *forces;
    int n_particles;
    double g_strong = 14.0;
    double m_pion = 0.138;  // GeV
    double alpha = 1.0/137.036;
    double hbar_c = 0.197;  // GeV·fm
    
    // Parse arguments
    if (!PyArg_ParseTuple(args, "O!O!O!i", 
                          &PyArray_Type, &positions,
                          &PyArray_Type, &momenta, 
                          &PyArray_Type, &forces,
                          &n_particles)) {
        return NULL;
    }
    
    // Get data pointers
    double *pos_data = (double*)PyArray_DATA(positions);
    double *mom_data = (double*)PyArray_DATA(momenta);
    double *force_data = (double*)PyArray_DATA(forces);
    
    // Clear forces
    memset(force_data, 0, n_particles * 3 * sizeof(double));
    
    // Calculate forces - Fixed for Windows/MSVC
    int i, j;
    #ifdef _OPENMP
    #pragma omp parallel for private(j)
    #endif
    for (i = 0; i < n_particles; i++) {
        for (j = i + 1; j < n_particles; j++) {
            // Position difference
            double dx = pos_data[i*3] - pos_data[j*3];
            double dy = pos_data[i*3+1] - pos_data[j*3+1];
            double dz = pos_data[i*3+2] - pos_data[j*3+2];
            
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            
            if (r < 0.1) continue;  // Avoid singularity
            
            double r_inv = 1.0 / r;
            double r_hat_x = dx * r_inv;
            double r_hat_y = dy * r_inv;
            double r_hat_z = dz * r_inv;
            
            // Nuclear force (Yukawa potential)
            double yukawa_exp = exp(-m_pion * r * 5.07);  // Convert to natural units
            double yukawa_force = (g_strong / (4.0 * M_PI * r * r)) * 
                                  yukawa_exp * (1.0 + m_pion * r * 5.07);
            
            // Coulomb force (assuming charges = +1 for simplicity)
            double coulomb_force = alpha * hbar_c * hbar_c / (r * r);
            
            double total_force = -yukawa_force + coulomb_force;  // Attractive nuclear + repulsive Coulomb
            
            // Apply forces (Newton's 3rd law) - Use atomic operations for thread safety
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_data[i*3] += total_force * r_hat_x;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_data[i*3+1] += total_force * r_hat_y;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_data[i*3+2] += total_force * r_hat_z;
            
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_data[j*3] -= total_force * r_hat_x;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_data[j*3+1] -= total_force * r_hat_y;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_data[j*3+2] -= total_force * r_hat_z;
        }
    }
    
    Py_RETURN_NONE;
}

// Fast lattice field updates - Fixed for Windows/MSVC
static PyObject* update_lattice_fields_c(PyObject* self, PyObject* args) {
    PyArrayObject *field_real, *field_imag, *source_term;
    double dt, dx, dy, dz;
    int nx, ny, nz;
    
    if (!PyArg_ParseTuple(args, "O!O!O!dddiii",
                          &PyArray_Type, &field_real,
                          &PyArray_Type, &field_imag,
                          &PyArray_Type, &source_term,
                          &dt, &dx, &dy, &dz, &nx, &ny, &nz)) {
        return NULL;
    }
    
    double *real_data = (double*)PyArray_DATA(field_real);
    double *imag_data = (double*)PyArray_DATA(field_imag);
    double *source_data = (double*)PyArray_DATA(source_term);
    
    // Laplacian coefficients
    double dx2_inv = 1.0 / (dx * dx);
    double dy2_inv = 1.0 / (dy * dy);
    double dz2_inv = 1.0 / (dz * dz);
    
    // Update fields using finite difference scheme - Fixed loop variables
    int i, j, k;
    #ifdef _OPENMP
    #pragma omp parallel for private(j, k)
    #endif
    for (i = 1; i < nx - 1; i++) {
        for (j = 1; j < ny - 1; j++) {
            for (k = 1; k < nz - 1; k++) {
                int idx = i * ny * nz + j * nz + k;
                
                // Calculate Laplacian
                double laplacian_real = 
                    (real_data[(i+1)*ny*nz + j*nz + k] + real_data[(i-1)*ny*nz + j*nz + k] - 2*real_data[idx]) * dx2_inv +
                    (real_data[i*ny*nz + (j+1)*nz + k] + real_data[i*ny*nz + (j-1)*nz + k] - 2*real_data[idx]) * dy2_inv +
                    (real_data[i*ny*nz + j*nz + (k+1)] + real_data[i*ny*nz + j*nz + (k-1)] - 2*real_data[idx]) * dz2_inv;
                
                double laplacian_imag = 
                    (imag_data[(i+1)*ny*nz + j*nz + k] + imag_data[(i-1)*ny*nz + j*nz + k] - 2*imag_data[idx]) * dx2_inv +
                    (imag_data[i*ny*nz + (j+1)*nz + k] + imag_data[i*ny*nz + (j-1)*nz + k] - 2*imag_data[idx]) * dy2_inv +
                    (imag_data[i*ny*nz + j*nz + (k+1)] + imag_data[i*ny*nz + j*nz + (k-1)] - 2*imag_data[idx]) * dz2_inv;
                
                // Schrödinger equation update
                real_data[idx] += dt * (0.5 * laplacian_imag + source_data[idx] * imag_data[idx]);
                imag_data[idx] -= dt * (0.5 * laplacian_real + source_data[idx] * real_data[idx]);
            }
        }
    }
    
    Py_RETURN_NONE;
}

// Fast particle collision detection
static PyObject* detect_collisions_c(PyObject* self, PyObject* args) {
    PyArrayObject *positions, *collision_pairs;
    double collision_radius;
    int n_particles;
    
    if (!PyArg_ParseTuple(args, "O!O!di",
                          &PyArray_Type, &positions,
                          &PyArray_Type, &collision_pairs,
                          &collision_radius, &n_particles)) {
        return NULL;
    }
    
    double *pos_data = (double*)PyArray_DATA(positions);
    int *pair_data = (int*)PyArray_DATA(collision_pairs);
    
    int collision_count = 0;
    double radius_sq = collision_radius * collision_radius;
    
    // Find all collision pairs
    int i, j;
    for (i = 0; i < n_particles; i++) {
        for (j = i + 1; j < n_particles; j++) {
            double dx = pos_data[i*3] - pos_data[j*3];
            double dy = pos_data[i*3+1] - pos_data[j*3+1];
            double dz = pos_data[i*3+2] - pos_data[j*3+2];
            
            double dist_sq = dx*dx + dy*dy + dz*dz;
            
            if (dist_sq < radius_sq) {
                if (collision_count < 10000) {  // Max pairs limit
                    pair_data[collision_count*2] = i;
                    pair_data[collision_count*2+1] = j;
                    collision_count++;
                }
            }
        }
    }
    
    return PyLong_FromLong(collision_count);
}

// Enhanced nuclear force calculation with multiple meson exchange
static PyObject* calculate_enhanced_nuclear_forces_c(PyObject* self, PyObject* args) {
    PyArrayObject *positions, *charges, *forces;
    int n_particles;
    
    // Parse arguments
    if (!PyArg_ParseTuple(args, "O!O!O!i", 
                          &PyArray_Type, &positions,
                          &PyArray_Type, &charges,
                          &PyArray_Type, &forces,
                          &n_particles)) {
        return NULL;
    }
    
    // Get data pointers
    double *pos_data = (double*)PyArray_DATA(positions);
    int *charge_data = (int*)PyArray_DATA(charges);
    double *force_data = (double*)PyArray_DATA(forces);
    
    // Physical constants
    double g_strong = 14.0;
    double m_pion = 0.138;  // GeV
    double m_eta = 0.548;   // GeV
    double m_rho = 0.775;   // GeV
    double alpha = 1.0/137.036;
    double hbar_c = 0.197;  // GeV·fm
    
    // Clear forces
    memset(force_data, 0, n_particles * 3 * sizeof(double));
    
    // Calculate enhanced nuclear forces
    int i, j;
    #ifdef _OPENMP
    #pragma omp parallel for private(j)
    #endif
    for (i = 0; i < n_particles; i++) {
        for (j = i + 1; j < n_particles; j++) {
            // Position difference
            double dx = pos_data[i*3] - pos_data[j*3];
            double dy = pos_data[i*3+1] - pos_data[j*3+1];
            double dz = pos_data[i*3+2] - pos_data[j*3+2];
            
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            
            if (r < 0.05) continue;  // Avoid singularity
            
            double r_inv = 1.0 / r;
            double r_hat_x = dx * r_inv;
            double r_hat_y = dy * r_inv;
            double r_hat_z = dz * r_inv;
            
            double total_force = 0.0;
            
            // 1. Pion exchange (attractive, long range)
            double x_pi = m_pion * r / hbar_c;
            if (x_pi > 0.1) {
                double yukawa_pi = (g_strong * g_strong * m_pion * m_pion) / (4.0 * M_PI);
                double force_pi = -yukawa_pi * (1.0 + 1.0/x_pi) * exp(-x_pi) / (r * r / (hbar_c * hbar_c));
                total_force += force_pi;
            }
            
            // 2. Eta meson exchange (repulsive at medium range)
            double x_eta = m_eta * r / hbar_c;
            if (x_eta > 0.1) {
                double yukawa_eta = (g_strong * g_strong * m_eta * m_eta) / (8.0 * M_PI);
                double force_eta = yukawa_eta * (1.0 + 1.0/x_eta) * exp(-x_eta) / (r * r / (hbar_c * hbar_c));
                total_force += force_eta * 0.5;  // Reduced coupling
            }
            
            // 3. Rho meson exchange (attractive, shorter range)
            double x_rho = m_rho * r / hbar_c;
            if (x_rho > 0.1) {
                double yukawa_rho = (g_strong * g_strong * m_rho * m_rho) / (6.0 * M_PI);
                double force_rho = -yukawa_rho * (1.0 + 1.0/x_rho) * exp(-x_rho) / (r * r / (hbar_c * hbar_c));
                total_force += force_rho * 0.3;  // Reduced coupling
            }
            
            // 4. Electromagnetic force
            if (charge_data[i] != 0 && charge_data[j] != 0) {
                double coulomb_force = alpha * hbar_c * charge_data[i] * charge_data[j] / (r * r);
                total_force += coulomb_force;
            }
            
            // 5. Short-range repulsion (phenomenological)
            double r_fm = r / hbar_c;
            if (r_fm < 0.5) {
                double repulsion_strength = 1000.0;  // MeV
                double force_repulsion = repulsion_strength * exp(-4.0 * r_fm) / (r_fm * r_fm);
                total_force += force_repulsion;
            }
            
            // Apply forces (Newton's 3rd law) with thread safety
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_data[i*3] += total_force * r_hat_x;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_data[i*3+1] += total_force * r_hat_y;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_data[i*3+2] += total_force * r_hat_z;
            
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_data[j*3] -= total_force * r_hat_x;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_data[j*3+1] -= total_force * r_hat_y;
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            force_data[j*3+2] -= total_force * r_hat_z;
        }
    }
    
    Py_RETURN_NONE;
}

// Get system information
static PyObject* get_system_info_c(PyObject* self, PyObject* args) {
    PyObject *info_dict = PyDict_New();
    
    // OpenMP information
    #ifdef _OPENMP
    PyDict_SetItemString(info_dict, "openmp_available", PyBool_FromLong(1));
    PyDict_SetItemString(info_dict, "max_threads", PyLong_FromLong(omp_get_max_threads()));
    #else
    PyDict_SetItemString(info_dict, "openmp_available", PyBool_FromLong(0));
    PyDict_SetItemString(info_dict, "max_threads", PyLong_FromLong(1));
    #endif
    
    // Compiler information
    #ifdef _MSC_VER
    PyDict_SetItemString(info_dict, "compiler", PyUnicode_FromString("MSVC"));
    PyDict_SetItemString(info_dict, "compiler_version", PyLong_FromLong(_MSC_VER));
    #elif defined(__GNUC__)
    PyDict_SetItemString(info_dict, "compiler", PyUnicode_FromString("GCC"));
    PyDict_SetItemString(info_dict, "compiler_version", PyLong_FromLong(__GNUC__ * 10000 + __GNUC_MINOR__ * 100));
    #else
    PyDict_SetItemString(info_dict, "compiler", PyUnicode_FromString("Unknown"));
    PyDict_SetItemString(info_dict, "compiler_version", PyLong_FromLong(0));
    #endif
    
    return info_dict;
}

// Method definitions
static PyMethodDef CExtensionMethods[] = {
    {"calculate_nuclear_forces", calculate_nuclear_forces_c, METH_VARARGS,
     "Calculate nuclear forces between particles using C"},
    {"update_lattice_fields", update_lattice_fields_c, METH_VARARGS,
     "Update quantum field lattice using C"},
    {"detect_collisions", detect_collisions_c, METH_VARARGS,
     "Detect particle collisions using C"},
    {"calculate_enhanced_nuclear_forces", calculate_enhanced_nuclear_forces_c, METH_VARARGS,
     "Calculate enhanced nuclear forces with multiple meson exchange"},
    {"get_system_info", get_system_info_c, METH_VARARGS,
     "Get system and compiler information"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef cextensionmodule = {
    PyModuleDef_HEAD_INIT,
    "c_extensions",
    "High-performance C extensions for quantum simulator - Fixed for Windows/MSVC",
    -1,
    CExtensionMethods
};

PyMODINIT_FUNC PyInit_c_extensions(void) {
    PyObject *module = PyModule_Create(&cextensionmodule);
    import_array();  // Initialize NumPy C API
    
    if (module == NULL) {
        return NULL;
    }
    
    // Add version information
    PyModule_AddStringConstant(module, "__version__", "1.1.0");
    PyModule_AddStringConstant(module, "__author__", "Quantum Lattice Simulator Team");
    
    return module;
}
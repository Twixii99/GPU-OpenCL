/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "kernel_funcs.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[2] = {
      { NULL, NULL },
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 2, initBlocks, pair_map };

// Block function
void (^simple_multiply_kernel)(const cl_ndrange *ndrange, cl_float* matrix_C_buffer, cl_float* matrix_A_buffer, cl_float* matrix_B_buffer, cl_int matrix_dimension) =
^(const cl_ndrange *ndrange, cl_float* matrix_C_buffer, cl_float* matrix_A_buffer, cl_float* matrix_B_buffer, cl_int matrix_dimension) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel simple_multiply does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, matrix_C_buffer, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, matrix_A_buffer, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, matrix_B_buffer, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(matrix_dimension), &matrix_dimension, &kargs);
  gcl_log_cl_fatal(err, "setting argument for simple_multiply failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing simple_multiply failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^simple_addition_kernel)(const cl_ndrange *ndrange, cl_float* matrix_A_buffer, cl_float* matrix_B_buffer, cl_int matrix_dimension) =
^(const cl_ndrange *ndrange, cl_float* matrix_A_buffer, cl_float* matrix_B_buffer, cl_int matrix_dimension) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel simple_addition does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, matrix_A_buffer, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, matrix_B_buffer, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(matrix_dimension), &matrix_dimension, &kargs);
  gcl_log_cl_fatal(err, "setting argument for simple_addition failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing simple_addition failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("kernel_funcs.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == simple_multiply_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "simple_multiply", &err);
          assert(bmap.map[1].block_ptr == simple_addition_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "simple_addition", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = simple_multiply_kernel;
  bmap.map[1].block_ptr = simple_addition_kernel;
}


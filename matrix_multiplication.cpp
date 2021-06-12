/**
 * @author Mahmoud Kamal Mahmoud
 * @date 12 Jun 2021
 * @link https://www.youtube.com/watch?v=tduqROrpHTg&list=PLZ9YeF_1_vF8gozOgYaW2660Na1bFhEIR
 * @result Matrix_A = Matrix_A * (Matrix_A + Matrix_B)
 *      then all matrices must be of the same size
 */ 

/**
 * 1) command queue will be used later for buffer operations and for kernel launches
 * 2) buffers are used to copy data from the host memory to the device memory and after the device is making the results
 * the buffer is used to copy the resault back to the computer
 * 3) compile kernel_function stored in .cl file (device specific)
 * 4) set the proper arguments to the kernel function even before running it.
 * 5) after the kernel execution is completed copy back the results from the device to the host through the buffer
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <chrono>

#ifdef __LINUX__
    #include <CL/cl.h>
#else
    #include <OpenCL/opencl.h>
#endif

#define PLATFORM_NAME_SIZE 150
#define DEVICE_NAME_SIZE 150

#define MAX_KERNEL_FILE_SIZE 0x100000

#define MAX_VALUE 10.0F

float * matrix_A, * matrix_B, * matrix_C;
int matrix_dimension;

float * initialize_matrix(int = ::matrix_dimension, int = ::matrix_dimension);
float * initialize_matrix_to_zero(int = ::matrix_dimension, int = ::matrix_dimension);

void print_matrix(float *, int = ::matrix_dimension, int = ::matrix_dimension);

int main(int argc, char const *argv[]) {
    // Getting the matrices dimentions
    std::cout << "Enter #Matrices' dimension: ";
    std::cin >> ::matrix_dimension;

    /* intializing the buffer arrays on the host side */
    ::matrix_A = ::initialize_matrix();
    ::matrix_B = ::initialize_matrix();
    ::matrix_C = ::initialize_matrix_to_zero();

    // std::cout << "The created Matrices are: \n" << std::endl;
    // ::print_matrix(::matrix_A);
    // std::cout << std::endl;
    // ::print_matrix(::matrix_B);
    // std::cout << std::endl;

    /* Getting the number of platforms */
    cl_uint platform_count = 0;
    cl_int ret = clGetPlatformIDs(1, NULL, &platform_count);
    if(ret == CL_SUCCESS) 
        std::cout << "no of platforms: " << platform_count << std::endl;
    else
        std::cerr <<  "Can't find any platform in platform_layer!! no vendors..." << std::endl,
        exit(-1);

    /* get an array of platformIDs */
    cl_platform_id *platforms_id = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platform_count);
    ret = clGetPlatformIDs(platform_count, platforms_id, NULL);
    if(ret == CL_SUCCESS) 
        std::cout << "platformID: " << *platforms_id << std::endl;
    else
        std::cerr <<  "Can't access the platformID in platform_layer!!!" << std::endl,
        exit(-1);
    
    /* Getting the vendor name */
    char* platform_name = (char*) malloc(sizeof(char) * PLATFORM_NAME_SIZE);
    ret = clGetPlatformInfo(*platforms_id, CL_PLATFORM_NAME, PLATFORM_NAME_SIZE, platform_name, NULL);
    if(ret == CL_SUCCESS) 
        std::cout << "platform name(vendor): " << platform_name << std::endl;
    else
        std::cerr <<  "Can't get the platform_Name in platform_layer!!!" << std::endl,
        exit(-1);

    /* devices within the platform(vendor) */
    cl_device_id device_id = 0;
    cl_uint num_of_devices = 0;
    ret = clGetDeviceIDs(*platforms_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_of_devices);
    if(ret == CL_SUCCESS)
        std::cout << "Number of devices within the platform = "<< num_of_devices << "\n" 
            << "First Device ID: " << device_id << std::endl;
    else
        std::cerr <<  "Can't get the platform_devices within platform_layer!!!" << std::endl,
        exit(-1);

    /* get the devices names */
    char *device_name = (char*) malloc(sizeof(char) * DEVICE_NAME_SIZE);
    ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, DEVICE_NAME_SIZE, device_name, NULL);
    if(ret == CL_SUCCESS)
        std::cout << "Device name: " << device_name << std::endl;
    else
        std::cerr <<  "Can't get the platform_devices names within platform_layer!!!" << std::endl,
        exit(-1);

    // making a context: which is an enclosure of all the resources like "command queues", "buffers" all of them tied into this context.
    // so we can make a command_queue withinn this context for the corresponding device(device_id).
    // the "ret" is highly recommending you do that to make sure that every API("command_queue", "buffer" -> within the context)
    // call succedded.

   /* return value is a context object. */
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if(ret != CL_SUCCESS)
        std::cerr << "can't make the Enviroment for the future operations!!" << std::endl,
        exit(-1);

    /* declaring the command queue that will be within the created context for the device with device_id ID <GPU> */
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if(ret != CL_SUCCESS)
        std::cerr << "can't make the Command queue model for the future operations!!" << std::endl,
        exit(-1);


    // remember the linear nature of storing the data on the physical memory.
    // because we need three arrays INPUT(A, B) OUTPUT(C) then we need to make three buffers.
    // the device has only to read data from the A, B buffers so the types of those buffers should be READ_ONLY.
    // there maybe buffers of types (READ_ONLY) (WRITE_ONLY) (READ_AND_WRITE).

    /* Allocate space for Matrix_A on the device */
    cl_mem matrix_A_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, ::matrix_dimension * ::matrix_dimension * sizeof(float), NULL, &ret);
    if(ret != CL_SUCCESS)
        std::cerr << "Can't create buffer within the context for the selected device for Matrix_A!" << std::endl,
        exit(-1);
    
    /* now we will move the matrix_A to the buffers which is inside the device <GPU>. */
    ret = clEnqueueWriteBuffer(command_queue, matrix_A_buffer, CL_TRUE, 0, ::matrix_dimension * ::matrix_dimension * sizeof(float),
        (void *) ::matrix_A, 0, NULL, NULL);
    if(ret != CL_SUCCESS)
        std::cerr << "Can't push matrix_A from the host to the device!" << std::endl,
        exit(-1);

    // Allocate space for Matrix_B on the device (creating READ_WRITE buffer for matrix_B)
    cl_mem matrix_B_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, ::matrix_dimension * ::matrix_dimension * sizeof(float), NULL, &ret);
    if(ret != CL_SUCCESS)
        std::cerr << "Can't create buffer within the context for the selected device for Matrix_B!" << std::endl,
        exit(-1);

    // now we will move the matrix_B to the buffers which is inshide the device <GPU>.
    ret = clEnqueueWriteBuffer(command_queue, matrix_B_buffer, CL_TRUE, 0, ::matrix_dimension * ::matrix_dimension * sizeof(float),
        (void *) ::matrix_B, 0, NULL, NULL);
    if(ret != CL_SUCCESS)
        std::cerr << "Can't push matrix_B from the host to the device!" << std::endl,
        exit(-1);
    
    // Eventhough declaring Buffer_C as WRITE_ONLY it doesn't prevent us from reading it from the host side.
    // the host has to retrieve the results from the device and copy the resulting matrix from BufferCL to S.W in the host 
    // physical memory (Not pushing the buffer into the device it expects to write to it not read).
    cl_mem matrix_C_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ::matrix_dimension  * ::matrix_dimension * sizeof(float), NULL, &ret); 
    if(ret != CL_SUCCESS)
        std::cerr << "Can't create buffer within the context for the selected device for Matrix_C!" << std::endl,
        exit(-1);

    
    /* starting making our kernel program */
    std::FILE *kernel_program_file;
    const char *kernel_program_file_name = "./kernel_funcs.cl";
    kernel_program_file = fopen(kernel_program_file_name, "r");
    if(!kernel_program_file) {
        fprintf(stderr, "Failed to find the kernel source file!\n");
        exit(-1);
    }
    char *source_program = (char *)malloc(MAX_KERNEL_FILE_SIZE);
    size_t source_program_size = fread(source_program, 1, MAX_KERNEL_FILE_SIZE, kernel_program_file);
    fclose(kernel_program_file);

    /* creating the program */
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_program, (const size_t*)&source_program_size, &ret);
    if(ret != CL_SUCCESS) {
        std::cerr << "Can't make a program for the entered Kernel file!" << std::endl;
        exit(-1);
    }

    /* building the program */
    /* takes the program we'd already creadted */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if(ret != CL_SUCCESS) {
         std::cerr << "Can't build the program for the entered Kernel file!" << std::endl;
        exit(-1);
    }


    /* setting the local_work_group(work items) and the global_work_group(work group) */
    size_t globalws[2] = {(size_t)::matrix_dimension, (size_t)::matrix_dimension};
    size_t localws[2] = {2, 2}; /* if you use FPGA it should have low number of work items */
 
    // the next section is about dividing the compilation into its basic.
    // kernel functions (simple_addition & simple_multiply) and execute them due to the task.

    /* create the kernel program pointing to a specific kernel function in the .cl file */
    cl_kernel addition_kernel_func = clCreateKernel(program, "simple_addition", &ret);
    if(ret != CL_SUCCESS) {
        std::cerr << "Can't make the kernel for the entered Kernel file! or the function name isnot correct!!" << std::endl;
        exit(-1);
    }

    /* filling the kernel function arguments */
    clSetKernelArg(addition_kernel_func, 0, sizeof(cl_mem), (void *)&matrix_A_buffer);
    clSetKernelArg(addition_kernel_func, 1, sizeof(cl_mem), (void *)&matrix_B_buffer);
    clSetKernelArg(addition_kernel_func, 2, sizeof(int), (void *)&::matrix_dimension);


    /* Initiatiating the kernel function execution */
    // it will make bufferB = bufferA + bufferB
    auto start_time_add = std::chrono::high_resolution_clock::now();
    ret = clEnqueueNDRangeKernel(command_queue, addition_kernel_func, 2, NULL, globalws, localws, 0, NULL, NULL);
    if(ret != CL_SUCCESS) {
        std::cerr << "The kernel execution failed, check the number of work items and work group!" << std::endl;
        exit(-1);
    }
    auto finish_time_add = std::chrono::high_resolution_clock::now();

    /* create the kernel program pointing to a specific kernel function in the .cl file */
    cl_kernel multiplication_kernel_func = clCreateKernel(program, "simple_multiply", &ret);
    if(ret != CL_SUCCESS) {
         std::cerr << "Can't make the kernel for the entered Kernel file! or the function name isnot correct!!" << std::endl;
        exit(-1);
    }

    /* filling the kernel function arguments */
    clSetKernelArg(multiplication_kernel_func, 0, sizeof(cl_mem), (void *)&matrix_C_buffer);
    clSetKernelArg(multiplication_kernel_func, 1, sizeof(cl_mem), (void *)&matrix_A_buffer);
    clSetKernelArg(multiplication_kernel_func, 2, sizeof(cl_mem), (void *)&matrix_B_buffer);
    clSetKernelArg(multiplication_kernel_func, 3, sizeof(int), (void *)&::matrix_dimension);

    /* Initiatiating the kernel function execution */
    // it will make bufferC = bufferA + bufferB
    auto start_time = std::chrono::high_resolution_clock::now();
    ret = clEnqueueNDRangeKernel(command_queue, multiplication_kernel_func, 2, NULL, globalws, localws, 0, NULL, NULL);
    if(ret != CL_SUCCESS) {
        std::cerr << "The kernel execution failed, check the number of work items and work group!" << std::endl;
        exit(-1);
    }
    auto finish_time = std::chrono::high_resolution_clock::now();

    /* Getting the data results back to the host memory */
    // storing bufferC into matrixA
    clEnqueueReadBuffer(command_queue, matrix_C_buffer, CL_TRUE, 0, ::matrix_dimension  * ::matrix_dimension * sizeof(float), 
                (void *)::matrix_A, 0, NULL, NULL);
    
    /* verification */
    // std::cout << "\nVerification Phase\n" << std::endl;
    // ::print_matrix(::matrix_A);
    // std::cout << std::endl;

    std::cout << "For matrices of size: " << ::matrix_dimension 
            << " the execution time is " 
            << std::chrono::duration_cast<std::chrono::microseconds>(finish_time - start_time).count() +
               std::chrono::duration_cast<std::chrono::microseconds>(finish_time_add - start_time_add).count()
            << " microseconds\n" << std::endl;

    /* free the allocated resources */
    free(platforms_id);
    free(device_name);
    free(source_program);
    free(::matrix_A);
    free(::matrix_B);
    free(::matrix_C);
    ::matrix_A = ::matrix_B = ::matrix_C = NULL;

    clReleaseMemObject(matrix_A_buffer);
    clReleaseMemObject(matrix_B_buffer);
    clReleaseMemObject(matrix_C_buffer);
    clReleaseCommandQueue(command_queue);
    clReleaseKernel(addition_kernel_func);
    clReleaseKernel(multiplication_kernel_func);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 0;
}

/**
 * @param mat
 *      The matrix need to be printed
 * @param rows
 *      the number of rows in the created matrix
 * @param cols
 *      the number of columns in the created matrix
 * printing the input matrix
 */
void print_matrix(float *mat, int rows, int cols) {
    if(mat == NULL) {
        std::cout << "The passing matrix is NULL!" << std::endl;
        return;
    }

    for(int i = 0; i < rows; ++i)  {
        for(int j = 0; j < cols; ++j)
            std::cout << mat[i * rows + j] << " "; 
        std::cout << std::endl;
    }
}

/**
 * @param rows
 *      the number of rows in the created matrix
 * @param cols
 *      the number of columns in the created matrix
 * @return
 *      the created matrix
 * initialize a matrix all to random variables
 */
float * initialize_matrix(int rows, int cols) {
    float *mat = (float *) malloc(sizeof(float) * rows * cols);
    srand(random() * time(NULL)); /* to make some random real number */
    for(int i = 0; i < rows * cols; ++i) 
        *(mat + i) = (float)(rand()) / (float)RAND_MAX * MAX_VALUE;
    return mat;
}

/**
 * @param rows
 *      the number of rows in the created matrix
 * @param cols
 *      the number of columns in the created matrix
 * @return
 *      the created matrix
 * initialize a matrix all to zeros
 */
float * initialize_matrix_to_zero(int rows, int cols) {
    float *mat = (float *) calloc(rows * cols, sizeof(float)); 
    return mat;
}
__kernel void simple_multiply(
    __global float *matrix_C_buffer,
     __global float *matrix_A_buffer, 
     __global float *matrix_B_buffer,
     int matrix_dimension
) {
    int row_idx = get_global_id(1);
    int col_idx = get_global_id(0);

    float sum = 0.0f;

    for(int i = 0; i < matrix_dimension; ++i)
        sum += matrix_A_buffer[row_idx * matrix_dimension + i] * matrix_B_buffer[i * matrix_dimension + col_idx];
    
    matrix_C_buffer[row_idx * matrix_dimension + col_idx] = sum;
}

__kernel void simple_addition(
     __global float *matrix_A_buffer, 
     __global float *matrix_B_buffer,
     int matrix_dimension
) {
    int row_idx = get_global_id(1);
    int col_idx = get_global_id(0);

    float sum = 0.0f;

    for(int i = 0; i < matrix_dimension; ++i)
        matrix_B_buffer[row_idx * matrix_dimension + i] = matrix_A_buffer[row_idx * matrix_dimension + i] + matrix_B_buffer[row_idx * matrix_dimension + i];
}
import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array
from numpy import inf
from numpy.testing import assert_array_equal, assert_array_almost_equal

# def test_opencl():
#     context = cl.create_some_context()  # Initialize the Context
#     queue = cl.CommandQueue(context)  # Instantiate a Queue
#     n = 5
#     m = 5
#     thr = 3
#
#     part_c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]).astype(np.float32)
#     probe_c = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18], [20, 22, 24], [16, 17, 18]]).astype(np.float32)
#     e_c = np.zeros((m, 3)).astype(np.float32)
#     charge_n = np.array([5, 5, 5, 5, 5]).astype(np.float32)
#
#     part = pycl_array.to_device(queue, part_c)
#     probe = pycl_array.to_device(queue, probe_c)
#     e = pycl_array.to_device(queue, e_c)
#
#     charge = pycl_array.to_device(queue, charge_n)
#
#     program = cl.Program(context, """
#     __kernel void calc_field(__global const float *part, __global const float *probe, __global const float *charge, __global float *e, const int thr, const int n)
#     {
#       int j = get_global_id(1);
#       int i = get_global_id(0);
#       for(int k = 0; k<n; k++){
#         e[i*thr + j] += charge[k]/(probe[i*thr + j] - part[k*thr + j]);
#       }
#       //float k_0 = pow(10.0,9.0);
#       //float k = 9*k_0;
#     }""").build()  # Create the OpenCL program
#
#     program.calc_field(queue, probe.shape, None, part.data, probe.data, charge.data, e.data, np.int32(thr), np.int32(n))
#     # Enqueue the program for execution and store the result in c
#
#     for j in range(probe_c.shape[0]):  # m2
#         for i in range(probe_c.shape[1]):  # m3
#             for k in range(part_c.shape[0]):  # n5
#                 e_c[j][i] += charge_n[k] / (probe_c[j][i] - part_c[k][i])
#
#     assert_array_equal(part, [[1, 2, 3],
#                               [4, 5, 6],
#                               [7, 8, 9],
#                               [10, 11, 12],
#                               [13, 14, 15]])
#     assert_array_equal(probe, [[2, 4, 6, ],
#                                [8, 10, 12, ],
#                                [14, 16, 18, ],
#                                [20, 22, 24, ],
#                                [16, 17, 18]])
#     assert_array_equal(charge, [5, 5, 5, 5, 5, ])
#     assert_array_almost_equal(e.get(), [[0.42045453, -4.964286, inf],
#                                         [3.4642859, -2.125, inf],
#                                         [7.8489013, 4.9366884, 3.8055556],
#                                         [2.174559, 1.980806, 1.8214287],
#                                         [3.8055556, 3.8055556, 3.8055556]])
#     assert_array_almost_equal(e_c, [[0.42045453, -4.964286, inf],
#                                     [3.4642859, -2.125, inf],
#                                     [7.8489013, 4.9366884, 3.8055553],
#                                     [2.174559, 1.980806, 1.8214285],
#                                     [3.8055553, 3.8055553, 3.8055553]])


def test_opencl():
    context = cl.create_some_context()  # Initialize the Context
    queue = cl.CommandQueue(context)  # Instantiate a Queue
    n = 1
    m = 1
    thr = 3

    part_c = np.array([[0, 0, 0]]).astype(np.float32)
    probe_c = np.array([[1, 0, 0]]).astype(np.float32)
    e_c = np.zeros((m, 3)).astype(np.float32)
    charge_n = np.array([5]).astype(np.float32)

    part = pycl_array.to_device(queue, part_c)
    probe = pycl_array.to_device(queue, probe_c)
    e = pycl_array.to_device(queue, e_c)

    charge = pycl_array.to_device(queue, charge_n)

    program = cl.Program(context, """
    __kernel void calc_field(__global const float *part, __global const float *probe, __global const float *charge, __global float *e, const int thr, const int n)
    {
      int j = get_global_id(1);
      int i = get_global_id(0);
      for(int k = 0; k<n; k++){
        e[i*thr + j] += charge[k]/(probe[i*thr + j] - part[k*thr + j]);
      }      
      //float k_0 = pow(10.0,9.0);
      //float k = 9*k_0;  
    }""").build()  # Create the OpenCL program

    program.calc_field(queue, probe.shape, None, part.data, probe.data, charge.data, e.data, np.int32(thr), np.int32(n))
    # Enqueue the program for execution and store the result in c

    for j in range(probe_c.shape[0]):  # m2
        for i in range(probe_c.shape[1]):  # m3
            for k in range(part_c.shape[0]):  # n5
                e_c[j][i] += charge_n[k] / (probe_c[j][i] - part_c[k][i])

    assert_array_equal(part, [[0, 0, 0]])
    assert_array_equal(probe, [[1, 0, 0]])
    assert_array_equal(charge, [5])
    assert_array_almost_equal(e.get(), [[5, 0, 0]])
    assert_array_almost_equal(e_c, [[5, 0, 0]])

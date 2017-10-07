This repo is intended to illustrate the power of the register cache mechanism for CUDA applications.
Using the register cache one can maintain an intra-warp cache as part of the cache hierarchy in CUDA GPUs.
The implementation of the register cache is based on the use of shuffle (_shfl) instruction.


Prerequisits:
1) Cuda GPU.

2) Linux system (ensure you have git, make and nvidia drivers installed).

How to use this code:
1) Clone this repository.

2) Please check your GPU compute capabilities, if they are different from 3.5 please modify the simple makefile accordingly. You may refer to this link: https://developer.nvidia.com/cuda-gpus

3) Run get_times script, you may need to run "chmod u+x ./get_times" first.

The output:
A subdirectory named "results" will be created with different files in it.
Each file, except for "full_times" contains the results of a single implementation.
For each implementation we record the time of k-stencil for k from 1 to 20.
Each result is averaged over 5 executions, with min and max left out.

The following implementations are considered:
shmem_times - Will contain the shared memory based implementation results.
rc_times_X - Will contain the register cache based implementation with X outputs per thread. (For 1<=X<=8)

For convenience, we also create full_times csv file, which contains at the first column the shmem_times and next rc_times_1 up to rc_times_8.

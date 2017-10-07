CC=nvcc
OPTIMIZE=3
arch=35
FLAGS=-arch=sm_${arch} -gencode arch=compute_${arch},code=compute_${arch} -O${OPTIMIZE} -lineinfo ${AF} --ptxas-options -v  --ptxas-options -warn-spills -g
test_stencil : final_code.cu
	${CC} ${FLAGS} final_code.cu -o test_stencil

clean:
	rm -f test_stencil

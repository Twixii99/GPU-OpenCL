#!/user/bin/bash

all:
	/System/Library/Frameworks/OpenCL.framework/Libraries/openclc -x cl -cl-std=CL1.1 -cl-auto-vectorize-enable -emit-gcl kernel_funcs.cl
	mkdir -p ./build
	clang -c -Os -Wall -arch x86_64 -o build/kernel_funcs.cl.o -c kernel_funcs.cl.c
	mkdir -p ./build
	clang -c -Os -Wall -arch x86_64 -std=c++11 -I./Utils -I./Images -o build/vendors.o -c matrix_multiplication.cpp
	/System/Library/Frameworks/OpenCL.framework/Libraries/openclc -x cl -cl-std=CL1.1 -Os -arch i386 -emit-llvm -o kernel_funcs.cl.i386.bc -c kernel_funcs.cl
	/System/Library/Frameworks/OpenCL.framework/Libraries/openclc -x cl -cl-std=CL1.1 -Os -arch x86_64 -emit-llvm -o kernel_funcs.cl.x86_64.bc -c kernel_funcs.cl
	/System/Library/Frameworks/OpenCL.framework/Libraries/openclc -x cl -cl-std=CL1.1 -Os -arch gpu_32 -emit-llvm -o kernel_funcs.cl.gpu_32.bc -c kernel_funcs.cl
	/System/Library/Frameworks/OpenCL.framework/Libraries/openclc -x cl -cl-std=CL1.1 -Os -arch gpu_64 -emit-llvm -o kernel_funcs.cl.gpu_64.bc -c kernel_funcs.cl
	g++ -framework OpenCL -std=c++11 -I./Utils -I./Images -o ./vendors ./build/kernel_funcs.cl.o ./build/vendors.o
	
run: 
	./vendors

clean:
	rm -rf build kernel_funcs.cl.c kernel_funcs.cl.gpu_32.bc kernel_funcs.cl.gpu_64.bc kernel_funcs.cl.i386.bc kernel_funcs.cl.x86_64.bc kernel_funcs.cl.h


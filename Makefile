PROJ_NAME=cuda_example
BUILD_DIR=build
INCLUDES=/usr/local/cuda/include
CXX=/usr/local/cuda/bin/nvcc

$(PROJ_NAME): main.cu
	$(CXX) --cudart static --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50 -link -o $(BUILD_DIR)/$@ $^
	
clean:
	rm -fR $(BUILD_DIR)/*


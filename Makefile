NVCC=nvcc
NVCCFLAGS=-std=c++17
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-I./src/cutf/include
NVCCFLAGS+=-I./src/mateval/include
NVCCFLAGS+=-I./src/wmma_extension/include
NVCCFLAGS+=-lcublas -lcurand

TARGET=gemmec-tcec.test

$(TARGET):src/main.cu src/mateval/src/comparison_cuda.cu
	$(NVCC) $+ -o $@ $(NVCCFLAGS)
  
clean:
	rm -f $(TARGET)

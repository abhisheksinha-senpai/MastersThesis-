SHELL = /bin/sh

APP = sim

.SUFFIXES:
.SUFFIXES: .o .cpp .hpp .cu .cuh

IDIR = include
LDIR = lib
ODIR = obj
SDIR = src


RENDER = $(IDIR)/renderer
EXT = $(IDIR)/ext
CPPCC = g++
CPP_FLAGS = -I$(RENDER) -I$(EXT) -I. -std=c++11 -w -lcudart -O3

_CPPDEPS = Definitions.hpp Shader.hpp ResourceManager.hpp Mesh.hpp Model.hpp ParticleSystem.hpp Particle.hpp
CPPDEPS = $(patsubst %,$(RENDER)/%,$(_CPPDEPS))

_CPPOBJECTS = ResourceManager.o Shader.o Mesh.o Model.o ParticleSystem.o
CPPOBJECTS = $(patsubst %,$(ODIR)/%,$(_CPPOBJECTS))

$(ODIR)/%.o: $(SDIR)/renderer/%.cpp $(CPPDEPS)
	$(CPPCC) -c -o $@ $< $(CPP_FLAGS)

_EXTOBJECTS = glad.o stbi_image.o 
EXTOBJECTS = $(patsubst %,$(ODIR)/%,$(_EXTOBJECTS))

$(ODIR)/%.o: $(SDIR)/renderer/ext/%.c
	$(CPPCC) -c -o $@ $< $(CPP_FLAGS)

SIM = $(IDIR)/simulator
CUDACC = nvcc
CUDA_FLAGS = -I$(SIM) -I$(EXT) -I$(RENDER) -O3 -w -m64 --gpu-architecture=sm_75 -x cu -std=c++11 -lcudart --default-stream per-thread

_CUDADEPS = ImmersedBoundary.cuh LatticeBoltzmann.cuh utilities.cuh Helper.cuh
CUDADEPS = $(patsubst %,$(SIM)/%,$(_CUDADEPS))

_CUDAOBJECTS = LatticeBoltzmann.o ImmersedBoundary.o Helper.o utilities.o 
CUDAOBJECTS = $(patsubst %,$(ODIR)/%,$(_CUDAOBJECTS))

$(ODIR)/%.o:  $(SDIR)/simulator/%.cu $(CUDADEPS)
	$(CUDACC) -x cu -c -o $@ $< $(CUDA_FLAGS)

$(ODIR)/%.o: %.cu
	$(CUDACC) -x cu -c -o $@ $< $(CUDA_FLAGS)

LIBS = -lassimp -lglfw -lGL -lX11 -lpthread -lXrandr -lXi -ldl
CUDA_LD_FLAGS = --gpu-architecture=sm_75 -lcudart --default-stream per-thread
OBJECTS = $(EXTOBJECTS) $(CPPOBJECTS) $(CUDAOBJECTS) $(ODIR)/main.o 
$(APP): $(OBJECTS)
	$(CUDACC) $(OBJECTS) -o $(APP) $(LIBS) $(CUDA_LD_FLAGS)
	rm -f *.csv out*

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o $(APP) *.csv
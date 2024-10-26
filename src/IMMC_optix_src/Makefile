## gets the directory 1 level up
ROOTDIR ?= ..
## sets the directory for mcx to one level up
MCXDIR  ?= $(ROOTDIR)
## sets the output directory for the binary
OUTPUT_DIR=$(MCXDIR)/bin
## names the binary mcxoptix by default
BINARY=mcxoptix

## gnu flag to say we want to make a binary
OUTPUTFLAG:=-o

## Identifies where cuda is located 
CUDA_HOME ?=/usr/local/cuda
## Identifies where optix is located
# for 7.7 
#OPTIX_HOME?=/home/users/aidenlewis/OptiX-7.7.0
#OPTIX_HOME?=/pub/optix-7.5
OPTIX_HOME?=/home/users/aidenlewis/OptiX-7.6.0

## INCLUDEDIRS is a list of directories to search for other included makefiles
## Probably from other codebases like cuda??
INCLUDEDIRS=-Izmat -Izmat/easylzma -Iubj -I$(CUDA_HOME)/include -I$(OPTIX_HOME)/include -I$(OPTIX_HOME)/SDK -I$(OPTIX_HOME)/SDK/support 
## a symbolic link to cuda files?
LINKOPT   ?=-L/usr/local/cuda/lib64 -lcudart -ldl

## GPU Architecture flag for generation on CUDA 11.0 for compatibility with V100 and T4 Turing cards
CUGENCODE ?=-arch=sm_52
## Sets CUDACC to NVIDIA cuda compiler
CUDACC=nvcc
## cuda compiler directives
CUCCOPT=-g -lineinfo -cubin -O3 -Xptxas -O3 -Xptxas -allow-expensive-optimizations -ptx --expt-relaxed-constexpr $(CUGENCODE) -use_fast_math -maxrregcount=0
## --use-local-env

## Something for linking matlab libraries?
MKDIR=mkdir

## linking cuda ray tracing libraries
CUDART=-lcudart
## c++ compiler and compiler options
CXX ?=g++
##CPPOPT=-g -Wall -O3 -DNDEBUG
## for debug version
CPPOPT=-g -Wall -O3

AR=g++

OBJSUFFIX=.o
PTXSUFFIX=.ptx

## Lines involving cuda and c++ files to compile
CUDAFILES=mcx_core
CXXFILES=mcx_optix_host mcx_context shader_pipeline\
tetrahedral_mesh
## removed util and ubj/ubjw from cxxfiles mcx_bench and mcx_tictoc

CXXOBJS   := $(addsuffix $(OBJSUFFIX), $(CXXFILES))
OBJS      := $(addsuffix $(PTXSUFFIX), $(CUDAFILES)) $(CXXOBJS)

ARCH = $(shell uname -m)
PLATFORM = $(shell uname -s)
CUDA_HOME ?= /usr/local/cuda

## Directives for cygwin, a linux emulator used on windows
ifeq ($(findstring CYGWIN,$(PLATFORM)), CYGWIN)
  CC=nvcc
  CXX=nvcc
  CUOMPLINK=-Xcompiler
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
      LINKOPT=-L"$(CUDA_PATH)/lib/x64" $(CUDART)
  else
      LINKOPT=-L"$(CUDA_PATH)/lib/Win32" $(CUDART)
  endif
  INCLUDEDIRS +=-I"$(CUDA_PATH)/lib/include"
  CPPOPT =-c -D_CRT_SECURE_NO_DEPRECATE -DWIN32 #-DNO_LZMA
  OBJSUFFIX=.obj
  EXESUFFIX=.exe
  DLLFLAG=
  OMP=/openmp
  MEX=cmd /c mex
  CUCCOPT+=-Xcompiler "$(OMP) /W0"
  CUDA_STATIC=--cudart static -Xcompiler $(OMP) -Xcompiler /MT
  CPPOPT+=-Xcompiler $(OMP)
else ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
  CUDA_STATIC=--cudart static
  CPPOPT+=$(OMP)
else
  CPPOPT+=$(OMP)
  CUCCOPT+=-Xcompiler $(OMP)
  ifeq ($(findstring x86_64,$(ARCH)), x86_64)
     CPPOPT +=-m64
     CUCCOPT +=-m64
     ifeq "$(wildcard $(CUDA_HOME)/lib64)" "$(CUDA_HOME)/lib64"
        ifeq ($(BACKEND),cuda)
           LINKOPT=-L$(CUDA_HOME)/lib64 $(CUDART) -lm -lstdc++
        else ifeq ($(BACKEND),cudastatic)
           LINKOPT=-L$(CUDA_HOME)/lib64 $(CUDART) -lm -static-libgcc -static-libstdc++
        endif
     endif
  endif
endif


all: cudasdk $(OUTPUT_DIR)/$(BINARY)

%$(PTXSUFFIX): %.cu
	$(CUDACC) -c $(CUCCOPT) $(INCLUDEDIRS) -o $@  $<

%$(OBJSUFFIX): %.cpp
	$(CXX) $(INCLUDEDIRS) $(CPPOPT) -c -o $@  $<

$(OUTPUT_DIR)/$(BINARY): $(OBJS)
	$(AR) $(CXXOBJS) $(OUTPUTFLAG) $(OUTPUT_DIR)/$(BINARY) $(LINKOPT) $(USERLINKOPT)

%$(OBJSUFFIX): %.c
	$(CC) $(INCLUDEDIRS) $(CPPOPT) -c -o $@  $<

mcx_bench$(OBJSUFFIX): mcx_bench.c
	$(CC) $(INCLUDEDIRS) -c -o mcx_bench$(OBJSUFFIX)  $<

cudasdk:
	@if [ -z `which ${CUDACC}` ]; then \
	   echo "Please first install CUDA SDK and add the path to nvcc to your PATH environment variable."; exit 1;\
	fi
makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) $(OUTPUT_DIR); fi
clean:
	-rm -f $(OBJS) $(CXXOBJS) $(OUTPUT_DIR)/$(BINARY)$(EXESUFFIX)

# derived the astyle settings from https://github.com/nlohmann/json
pretty:
	astyle \
	    --style=attach \
            --indent=spaces=4 \
            --indent-modifiers \
            --indent=spaces=4 \
            --indent-modifiers \
            --indent-switches \
            --indent-preproc-block \
            --indent-preproc-define \
            --indent-col1-comments \
            --pad-oper \
            --pad-header \
            --align-pointer=type \
            --align-reference=type \
            --add-brackets \
            --convert-tabs \
            --close-templates \
            --lineend=linux \
            --preserve-date \
            --suffix=none \
            --formatted \

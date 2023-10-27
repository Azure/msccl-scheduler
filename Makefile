# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
PLATFORM ?= "NCCL"
BIN_HOME ?= ""
SRC_HOME ?= ""
CXXFLAGS := --compiler-options -fPIC,-shared -DNCCL
LDFLAGS := --linker-options -soname,$(SCHEDULER_SO)
INC := -I$(BIN_HOME)/include -I$(SRC_HOME)/src/include
SCHEDULER_SO := libmsccl-scheduler.so

ifeq ($(PLATFORM), RCCL)
	CXXFLAGS := -fPIC -shared -DRCCL
	LDFLAGS := -Wl,-soname,$(SCHEDULER_SO)
endif

default: $(SCHEDULER_SO)

$(SCHEDULER_SO): scheduler.cc parser.cc
	$(CXX) $(INC) $(CXXFLAGS) -o $@ $(LDFLAGS) $^ $(LNK)

clean:
	rm -f $(SCHEDULER_SO)

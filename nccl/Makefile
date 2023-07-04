# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

NCCL_BIN_HOME ?= ""
NCCL_SRC_HOME ?= ""
INC := -I$(NCCL_BIN_HOME)/include -I$(NCCL_SRC_HOME)/src/include
SCHEDULER_SO := libmsccl-scheduler.so

default: $(SCHEDULER_SO)

$(SCHEDULER_SO): scheduler.cc parser.cc
	$(CXX) $(INC) --compiler-options -fPIC,-shared -o $@ --linker-options -soname,$(SCHEDULER_SO) $^ $(LNK)

clean:
	rm -f $(SCHEDULER_SO)

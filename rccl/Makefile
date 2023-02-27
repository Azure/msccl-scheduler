# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

RCCL_BIN_HOME ?= ""
RCCL_SRC_HOME ?= ""
INC := -I$(RCCL_BIN_HOME)/include -I$(RCCL_SRC_HOME)/src/include
SCHEDULER_SO := libmsccl-scheduler.so

default: $(SCHEDULER_SO)

$(SCHEDULER_SO): scheduler.cc parser.cc
	$(CXX) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(SCHEDULER_SO) $^ $(LNK)

clean:
	rm -f $(SCHEDULER_SO)

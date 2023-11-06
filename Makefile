# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
##### version
include makefiles/version.mk
PREFIX ?= /usr/local

PLATFORM ?= "NCCL"
BIN_HOME ?= ""
SRC_HOME ?= ""

BUILDDIR ?= $(abspath build)
ABSBUILDDIR := $(abspath $(BUILDDIR))
LIBDIR := $(BUILDDIR)/lib
MSCCLALGORITHMDIR := $(LIBDIR)/msccl-algorithms
LIBNAME := libmsccl-scheduler.so
LIBSONAME  := $(LIBNAME:%=%.$(SCHEDULER_MAJOR))
LIBTARGET  := $(LIBNAME:%=%.$(SCHEDULER_MAJOR).$(SCHEDULER_MINOR).$(SCHEDULER_PATCH))
LICENSE_FILES := LICENSE.txt
LICENSE_TARGETS := $(LICENSE_FILES:%=$(BUILDDIR)/%)

CXXFLAGS := --compiler-options -fPIC,-shared,-g -DNCCL
LDFLAGS := --linker-options -soname,$(LIBSONAME)
INC := -I$(BIN_HOME)/include -I$(SRC_HOME)/src/include

ifeq ($(PLATFORM), RCCL)
	CXXFLAGS := -fPIC -shared -DRCCL
	LDFLAGS := -Wl,-soname,$(LIBSONAME)
endif

default: build
build : $(LIBDIR)/$(LIBTARGET) $(MSCCLALGORITHMDIR) $(LICENSE_TARGETS)

lic: $(LICENSE_TARGETS)

${BUILDDIR}/%.txt: %.txt
	@printf "Copying    %-35s > %s\n" $< $@
	mkdir -p ${BUILDDIR}
	cp $< $@

$(LIBDIR)/$(LIBTARGET): src/scheduler.cc src/parser.cc
	@printf "Compiling & Linking    %-35s > %s\n" $(LIBTARGET) $@ $^
	mkdir -p $(LIBDIR)
	$(CXX) $(INC) $(CXXFLAGS) -o $@ $(LDFLAGS) -lcurl $^ $(LNK)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(MSCCLALGORITHMDIR):
	mkdir -p $(MSCCLALGORITHMDIR)
	cp -r tools/msccl-algorithms/* $(MSCCLALGORITHMDIR)

clean:
	rm -f $(LIBNAME)

install : build
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/share/msccl-scheduler/msccl-algorithms
	cp -P -v $(BUILDDIR)/lib/lib* $(PREFIX)/lib/
	cp -P -r -v $(LIBDIR)/msccl-algorithms/* $(PREFIX)/share/msccl-scheduler/msccl-algorithms

pkg.%:
	${MAKE} -C pkg $* BUILDDIR=${ABSBUILDDIR}

pkg.debian.prep: lic
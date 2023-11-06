#
# Copyright (c) 2015-2023, Microsoft CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include ../../makefiles/version.mk
BUILDDIR ?= $(abspath ../../build)
DEBPREPDIR := $(BUILDDIR)/debian
PKGDIR  := $(BUILDDIR)/pkg/deb/

DEBGEN_IN  := $(wildcard *.in)
DEBGEN     := $(DEBGEN_IN:.in=)
DEBFILES   := compat copyright libmsccl-scheduler-dev.install rules $(DEBGEN)
DEBTARGETS := $(patsubst %, $(DEBPREPDIR)/%, $(DEBFILES))

PKG_TIMESTAMP  := $(shell date -R)
PKG_ARCH       ?= $(shell dpkg-architecture -qDEB_HOST_ARCH)
PKG_MULTIARCH  ?= $(shell dpkg-architecture -qDEB_HOST_MULTIARCH)

prep : $(DEBTARGETS)
	$(MAKE) -C ../.. lic BUILDDIR=$(BUILDDIR)

build : prep
	$(MAKE) -C ../.. build BUILDDIR=$(BUILDDIR)
	@printf "Building Debian package\n"
	(cd $(BUILDDIR); debuild -eLD_LIBRARY_PATH -uc -us -d -b)
	mkdir -p $(PKGDIR)
	mv $(BUILDDIR)/../libmsccl-scheduler*.deb $(PKGDIR)/

clean:
	rm -Rf $(DEBPREPDIR) $(PKGDIR)

$(DEBPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(DEBPREPDIR)
	sed -e "s/\$${scheduler:Major}/$(SCHEDULER_MAJOR)/g" \
	    -e "s/\$${scheduler:Minor}/$(SCHEDULER_MINOR)/g" \
	    -e "s/\$${scheduler:Patch}/$(SCHEDULER_PATCH)/g" \
	    -e "s/\$${scheduler:Suffix}/$(SCHEDULER_SUFFIX)/g" \
	    -e "s/\$${cuda:Major}/$(CUDA_MAJOR)/g" \
	    -e "s/\$${cuda:Minor}/$(CUDA_MINOR)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    -e "s/\$${pkg:Timestamp}/$(PKG_TIMESTAMP)/g" \
	    -e "s/\$${pkg:Arch}/$(PKG_ARCH)/g" \
	    -e "s/\$${pkg:MultiArch}/$(PKG_MULTIARCH)/g" \
	    $< > $@

$(DEBPREPDIR)/% : %
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(DEBPREPDIR)
	cp -f $< $@

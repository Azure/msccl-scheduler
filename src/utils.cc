/*************************************************************************
 * Copyright (c) 2019-2023 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt for license information
 ************************************************************************/
#include <dlfcn.h>
#include <iostream>
#include <set>
#include <string>
#include <unistd.h>
#include <unordered_set>

#include "comm.h"
#include "bootstrap.h"

#include "include/comm.h"
#include "include/utils.h"

#ifdef RCCL
  static const char* mscclExecutorDefaultPath = "librccl.so";
#else 
  static const char* mscclExecutorDefaultPath = "libnccl.so";
#endif

ncclResult_t GetRunningHostNames(ncclComm_t comm, std::vector<std::string> &hostNames){
    void* mscclExecutorLib = dlopen(mscclExecutorDefaultPath, RTLD_NOW | RTLD_LOCAL);
    if (mscclExecutorLib == nullptr) {
        fprintf(stdout, "%s: %s No ExecutorLib found\n", MSCCL_SCHEDULER_NAME, LOG_ERROR);
        return ncclInvalidUsage;
    }   
  
    ncclBootstrapInterface *ncclBootstrapPtr = (ncclBootstrapInterface *)dlsym(mscclExecutorLib, "ncclBootstrap");
    if (ncclBootstrapPtr == nullptr) {
        fprintf(stdout, "%s: %s Failed to find msccl Executor symbol ncclBootstrap\n", MSCCL_SCHEDULER_NAME, LOG_ERROR);
        return ncclInvalidUsage;
    }

    char hostname[1024];
    gethostname(hostname, 1024);

    char** fullHostNames = NULL;
    fullHostNames = new char*[comm->nRanks];
    for (int i = 0; i < comm->nRanks; ++i) {
        fullHostNames[i] = new char[1024];
        memset(fullHostNames[i], 0, 1024);
    }
    strcpy(fullHostNames[comm->rank], hostname);
    ncclBootstrapPtr->allgather(comm->bootstrap, fullHostNames, comm->nRanks * sizeof(char*));
    
    std::unordered_set<std::string> s;
    for (int i = 0; i < comm->nRanks; ++i) {
        if ("" != fullHostNames[i]) {
            s.insert(std::string(fullHostNames[i]));
        }
    }
    fprintf(stdout, "%s: %s after merge, rank:%d, ranks:%d \n", MSCCL_SCHEDULER_NAME, LOG_INFO, comm->rank, comm->nRanks);
    hostNames.assign(s.begin(), s.end());
    for (int i = 0; i < comm->nRanks; ++i) {
        fprintf(stdout, "%s: %s fullHostNames:%d, %s\n", MSCCL_SCHEDULER_NAME, LOG_INFO, i, fullHostNames[i]);
        delete[] fullHostNames[i];
    }
    delete[] fullHostNames;
    return ncclSuccess;
}
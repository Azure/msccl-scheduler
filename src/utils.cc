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

#include "include/comm.h"
#include "include/utils.h"

typedef ncclResult_t (*bootstrapAllGather_t)(void* commState, void* allData, int size);

#ifdef RCCL
  static const char* mscclExecutorDefaultPath = "librccl.so";
#else 
  static const char* mscclExecutorDefaultPath = "libnccl.so";
#endif

ncclResult_t GetRunningHostNames(ncclComm_t comm, std::vector<std::string> &hostNames){
    void* mscclExecutorLib = dlopen(mscclExecutorDefaultPath, RTLD_NOW | RTLD_LOCAL);
    if (mscclExecutorLib == nullptr) {
        fprintf(stdout, "%s: %s No ExecutorLib found, error %d\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, errno);
        return ncclInvalidUsage;
    }   
  
    bootstrapAllGather_t allGatherPtr = (bootstrapAllGather_t)dlsym(mscclExecutorLib, "bootstrapAllGather");
    if (allGatherPtr == nullptr) {
        fprintf(stdout, "%s: %s Failed to find mscclScheduler symbol, error %d\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, errno);
        return ncclInvalidUsage;
    }

    char hostname[1024];
    gethostname(hostname, 1024);

    char** fullHostNames = NULL;
    fullHostNames = new char*[comm->nRanks];
    fullHostNames[comm->rank] = new char[strlen(hostname) + 1];
    strcpy(fullHostNames[comm->rank], hostname);
    allGatherPtr(comm->bootstrap, fullHostNames, comm->nRanks * sizeof(char*));
    
    std::unordered_set<std::string> s;
    for (int i = 0; i < comm->nRanks; ++i) {
        s.insert(std::string(fullHostNames[i]));
    }
    hostNames.assign(s.begin(), s.end());
    for (int i = 0; i < comm->nRanks; ++i) {
        delete[] fullHostNames[i];
    }
    delete[] fullHostNames;

    return ncclSuccess;
}
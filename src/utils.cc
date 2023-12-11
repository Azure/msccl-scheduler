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

extern ncclBootstrapInterface *ncclBootstrapPtr;

ncclResult_t GetRunningHostNames(ncclComm_t comm, std::vector<std::string> &hostNames){
    char hostname[BUFFER_SIZE];
    gethostname(hostname, BUFFER_SIZE);

    if (0 == comm->rank)
    {
        char** fullHostNames = NULL;
        fullHostNames = new char*[comm->nRanks];
        for (int i = 0; i < comm->nRanks; ++i) {
            fullHostNames[i] = new char[BUFFER_SIZE];
            memset(fullHostNames[i], 0, BUFFER_SIZE);
        }
        strcpy(fullHostNames[comm->rank], hostname);
        for (int i=1;i<comm->nRanks;i++){
            ncclBootstrapPtr->receive(comm->bootstrap, i, TAG_HOSTINFO, fullHostNames[i], BUFFER_SIZE);
        }
        std::unordered_set<std::string> s;
        for (int i = 0; i < comm->nRanks; ++i) {
            if ('\0' != fullHostNames[i][0]) {
                s.insert(std::string(fullHostNames[i]));
            }
        }
        
        hostNames.assign(s.begin(), s.end());
        for (int i = 0; i < comm->nRanks; ++i) {
            fprintf(stdout, "%s: %s on rank %d hostname: %s\n", MSCCL_SCHEDULER_NAME, LOG_INFO, i, fullHostNames[i]);
            delete[] fullHostNames[i];
        }
        delete[] fullHostNames;
    }
    else{
        ncclBootstrapPtr->send(comm->bootstrap, 0, TAG_HOSTINFO, hostname, BUFFER_SIZE);
    }
    return ncclSuccess;
}
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
#include <vector>

#include "common.h"
#include "utils.h"
#include "msccl/msccl_scheduler.h"

ncclResult_t GetRunningHostNames(mscclSchedulerInitParam *initParam, std::vector<std::string> &hostNames){
    char hostname[BUFFER_SIZE];
    gethostname(hostname, BUFFER_SIZE);

    if (0 == initParam->rank)
    {
        char** fullHostNames = NULL;
        fullHostNames = new char*[initParam->nRanks];
        for (int i = 0; i < initParam->nRanks; ++i) {
            fullHostNames[i] = new char[BUFFER_SIZE];
            memset(fullHostNames[i], 0, BUFFER_SIZE);
        }
        strcpy(fullHostNames[initParam->rank], hostname);
        for (int i=1;i<initParam->nRanks;i++){
            initParam->receive(initParam->bootstrap, i, TAG_HOSTINFO, fullHostNames[i], BUFFER_SIZE);
        }
        std::unordered_set<std::string> s;
        for (int i = 0; i < initParam->nRanks; ++i) {
            if ('\0' != fullHostNames[i][0]) {
                s.insert(std::string(fullHostNames[i]));
            }
        }
        
        hostNames.assign(s.begin(), s.end());
        for (int i = 0; i < initParam->nRanks; ++i) {
            fprintf(stdout, "%s: %s on rank %d hostname: %s\n", MSCCL_SCHEDULER_NAME, LOG_INFO, i, fullHostNames[i]);
            delete[] fullHostNames[i];
        }
        delete[] fullHostNames;
    }
    else{
        initParam->send(initParam->bootstrap, 0, TAG_HOSTINFO, hostname, BUFFER_SIZE);
    }
    return ncclSuccess;
}
/*************************************************************************
 * Copyright (c) 2019-2023 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt for license information
 ************************************************************************/
#ifndef MSCCL_UTILS_H_
#define MSCCL_UTILS_H_

#include <nccl.h>
#include <vector>
#include <unistd.h>
#include <fcntl.h>

ncclResult_t GetRunningHostNames(ncclComm_t comm, std::vector<std::string> &hostNames);

#endif
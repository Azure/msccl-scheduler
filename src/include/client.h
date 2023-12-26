/*************************************************************************
 * Copyright (c) 2019-2023 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt for license information
 ************************************************************************/
#ifndef MSCCL_CLIENT_H_
#define MSCCL_CLIENT_H_

#include "common.h"

typedef struct
{
    char buffer[BUFFER_SIZE];
    int returncode;
} response;

std::string getOptimizedAlgoFile();
int shutDownServer();
#endif
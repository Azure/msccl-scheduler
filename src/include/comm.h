/*************************************************************************
 * Copyright (c) 2019-2023 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt for license information
 ************************************************************************/
#ifndef MSCCL_COMMON_H_
#define MSCCL_COMMON_H_

#define __hidden __attribute__ ((visibility("hidden")))
#define MSCCL_SCHEDULER_NAME "github.com/Azure/msccl-scheduler"

static const int TAG_HOSTINFO = 0;
static const int TAG_ALGOINFO = 1;

static const char* LOG_INFO = "INFO";
static const char* LOG_WARN = "WARN";
static const char* LOG_ERROR = "ERROR";

static const int BUFFER_SIZE = 1024;
static const int SERVER_PORT = 12345;
static const char* HOST_ADDR = "127.0.0.1";

#endif
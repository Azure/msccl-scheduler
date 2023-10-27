/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include <cstdio>
#include <vector>
#include <map>
#include <dirent.h>
#include <dlfcn.h>
#include <link.h>
#ifdef NCCL
  #include "nccl.h"
#elif defined(RCCL)
  #include "rccl/rccl.h"
#endif
#include "parser.h"

#define __hidden __attribute__ ((visibility("hidden")))

#define MSCCL_SCHEDULER_NAME "github.com/microsoft/msccl-scheduler"

static const char* mscclAlgoDirEnv = "MSCCL_ALGO_DIR";
static const char* mscclAlgoDefaultDir = "msccl-algorithms";
extern "C" bool mscclUnitTestMode() __attribute__((__weak__));
static const char* mscclUnitTestAlgoDefaultDir = "msccl-unit-test-algorithms";
#ifdef NCCL
  static const char* mscclAlgoShareDirPath = "share/msccl-algorithms/nccl";
#elif defined(RCCL)
  static const char* mscclAlgoShareDirPath = "share/msccl-algorithms/rccl";
#endif
static const char* mscclUnitTestAlgoShareDirPath = "share/msccl-unit-test-algorithms";

static std::vector<mscclAlgoMeta> mscclAlgoMetas;
static std::vector<std::map<int, mscclAlgoHandle_t>> rankToAlgoHandles;

// Load meta information of algorithms
__hidden ncclResult_t mscclSchedulerInit() {
  ncclResult_t ret = ncclSuccess, tmpRet = ncclSuccess;
  const char* mscclAlgoDir = getenv(mscclAlgoDirEnv);
  const char* mscclAlgoShareDir = nullptr;
  std::string mscclAlgoDirStr;
  std::string mscclAlgoShareDirStr;
  const char *fullDirPath = nullptr;
  if (mscclAlgoDir == nullptr) {
    // Try to find default algorithm directory based on libnccl.so or librccl.so path
    Dl_info dl_info;
    struct link_map *link_map_ptr = nullptr;
    if (!dladdr1((void *)ncclAllReduce, &dl_info, (void **)&link_map_ptr, RTLD_DL_LINKMAP)) {
      fprintf(stderr, "%s: dladdr1 failed", MSCCL_SCHEDULER_NAME);
      return ncclInvalidUsage;
    }
    std::string selfLibPath = link_map_ptr->l_name;
    mscclAlgoDirStr = selfLibPath.substr(0, selfLibPath.find_last_of("/\\") + 1);
    mscclAlgoDirStr += (mscclUnitTestMode && mscclUnitTestMode()) ? mscclUnitTestAlgoDefaultDir : mscclAlgoDefaultDir;
    mscclAlgoDir = mscclAlgoDirStr.c_str();
    // Get share Directory Paths
    mscclAlgoShareDirStr = selfLibPath.substr(0, selfLibPath.find("lib"));
    mscclAlgoShareDirStr += (mscclUnitTestMode && mscclUnitTestMode()) ? mscclUnitTestAlgoShareDirPath : mscclAlgoShareDirPath;
    mscclAlgoShareDir = mscclAlgoShareDirStr.c_str();
  }
  struct dirent *entry = nullptr;
  DIR *dp = nullptr;
  dp = opendir(mscclAlgoDir);
  if (dp == nullptr) {
    // Try to find the algorithm directory under share folder based on libnccl.so or librccl.so path
    dp = opendir(mscclAlgoShareDir);
    if (dp == nullptr) {
      fprintf(stderr, "%s: open algorithm in share directory %s failed", MSCCL_SCHEDULER_NAME, mscclAlgoShareDir);
      return ncclInvalidUsage;
    }
    fullDirPath = mscclAlgoShareDir;
  } else {
    fullDirPath = mscclAlgoDir;
  }
  while ((entry = readdir(dp))) {
    if (entry->d_type != DT_LNK && entry->d_type != DT_REG) {
      continue;
    }
    mscclAlgoMetas.emplace_back();
    std::string fullPath = fullDirPath;
    fullPath += "/";
    fullPath += entry->d_name;
    tmpRet = mscclGetAlgoMetaFromXmlFile(fullPath.c_str(), &(mscclAlgoMetas.back()));
    if (ret == ncclSuccess) {
      ret = tmpRet;
    }
  }
  if (closedir(dp)) {
    fprintf(stderr, "%s: closedir failed, error %d", MSCCL_SCHEDULER_NAME, errno);
    return ncclInvalidUsage;
  }
  rankToAlgoHandles.resize(mscclAlgoMetas.size());
  return ret;
}

static __inline__ int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
#if defined(RCCL_BFLOAT16)
    case ncclBfloat16:
#endif
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return -1;
  }
}

// Select algorithm, load if necessary
__hidden ncclResult_t mscclSchedulerSelectAlgo(struct mscclSchedulerParam* param) {
  ncclResult_t ret = ncclSuccess;

  param->scheduled = false;

  // Whether the algorithm is in-place
  bool isInPlace = false;
  if (param->func == mscclFuncReduce ||
      param->func == mscclFuncBroadcast ||
      param->func == mscclFuncAllReduce ||
      param->func == mscclFuncAllToAll ||
      param->func == mscclFuncAllToAllv) {
    isInPlace = param->sendBuff == param->recvBuff;
  } else if (param->func == mscclFuncAllGather ||
             param->func == mscclFuncGather) {
    isInPlace = (char*)param->sendBuff == (char*)param->recvBuff + param->rank * param->count * ncclTypeSize(param->dataType);
  } else if (param->func == mscclFuncReduceScatter ||
             param->func == mscclFuncScatter) {
    isInPlace = (char*)param->recvBuff == (char*)param->sendBuff + param->rank * param->count * ncclTypeSize(param->dataType);
  }

  // Search suitable algorithms
  for (size_t i = 0; i < mscclAlgoMetas.size(); i++) {
    auto &m = mscclAlgoMetas[i];
    size_t nBytes = param->count * ncclTypeSize(param->dataType) * m.sizeMultiplier;
    bool msgSizeIsValid =
      param->count > 0 && (param->count % m.nChunksPerLoop) == 0 &&
      nBytes >= m.minBytes && (m.maxBytes == 0 || nBytes <= m.maxBytes);
    if (msgSizeIsValid &&
        m.nRanks == param->nRanks &&
        m.func == param->func &&
        (isInPlace ? m.inPlace : m.outOfPlace)) {
      // If not loaded for current rank, load it
      if (rankToAlgoHandles[i].find(param->rank) == rankToAlgoHandles[i].end()) {
        mscclAlgoHandle_t algoHandle;
        ret = mscclLoadAlgo(m.filePath.c_str(), &algoHandle, param->rank);
        if (ret != ncclSuccess) {
          return ret;
        }
        rankToAlgoHandles[i][param->rank] = algoHandle;
      }
      param->handle = rankToAlgoHandles[i][param->rank];
      param->scheduled = true;
      return ncclSuccess;
    }
  }

  return ncclSuccess;
}

__hidden ncclResult_t mscclSchedulerTearDown() {
  ncclResult_t ret = ncclSuccess, tmpRet = ncclSuccess;
  for (auto &m : rankToAlgoHandles) {
    for (auto &p : m) {
      tmpRet = mscclUnloadAlgo(p.second);
      if (ret == ncclSuccess) {
        ret = tmpRet;
      }
    }
  }
  mscclAlgoMetas.clear();
  rankToAlgoHandles.clear();
  return ret;
}

mscclSchedulerInterface mscclScheduler = {
  .name = MSCCL_SCHEDULER_NAME,
  .init = mscclSchedulerInit,
  .selectAlgo = mscclSchedulerSelectAlgo,
  .teardown = mscclSchedulerTearDown,
};

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
#include <string>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

#ifdef RCCL
  #include "rccl/rccl.h"
#else 
  #include "nccl.h"
#endif
#include "parser.h"

#define __hidden __attribute__ ((visibility("hidden")))

#define MSCCL_SCHEDULER_NAME "github.com/Azure/msccl-scheduler"

static const char* mscclAlgoDirEnv = "MSCCL_ALGO_DIR";
static const char* mscclAlgoDefaultDir = "msccl-algorithms";
extern "C" bool mscclUnitTestMode() __attribute__((__weak__));
static const char* mscclUnitTestAlgoDefaultDir = "msccl-unit-test-algorithms";
static const char* mscclAlgoShareDirPath = "../share/msccl-scheduler/msccl-algorithms";
static const char* mscclUnitTestAlgoShareDirPath = "../share/msccl-scheduler/msccl-unit-test-algorithms";
static const char* mscclAzureVMDetectionAgent = "http://169.254.169.254/metadata/instance?api-version=2019-06-04";

static const char* LOG_INFO = "INFO";
static const char* LOG_WARN = "WARN";
static const char* LOG_ERROR = "ERROR";

static std::vector<mscclAlgoMeta> mscclAlgoMetas;
static std::vector<std::map<int, mscclAlgoHandle_t>> rankToAlgoHandles;

static size_t writeCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

static std::string updateAlgoDirByVMSize(std::string algoDir){
    CURL* curl;
    CURLcode res;
    std::string readBuffer;
    std::string vmSize;
    std::string updatedAlgoDir = algoDir;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, mscclAzureVMDetectionAgent);
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Metadata:true");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 1L);
        res = curl_easy_perform(curl);
        if(res != CURLE_OK) {
            fprintf(stdout, "%s: %s Get Azure VM Size failed: %s \n", MSCCL_SCHEDULER_NAME, LOG_WARN, curl_easy_strerror(res));
        }
        else {
            auto json = nlohmann::json::parse(readBuffer);
            vmSize = json["compute"]["vmSize"].get<std::string>();
        }
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
    if (vmSize.find("ND") != std::string::npos && vmSize.find("A100") != std::string::npos) {
      updatedAlgoDir.append("/ndv4");
    }
    else{
      fprintf(stdout, "%s: %s There is no related algo file for the detected Azure VM SKU:%s been finded, MSCCL will use nccl as default communication channel \n", MSCCL_SCHEDULER_NAME, LOG_WARN, vmSize.c_str());
    }
    return updatedAlgoDir;
}

// Load meta information of algorithms
__hidden ncclResult_t mscclSchedulerInit() {
  ncclResult_t ret = ncclSuccess, tmpRet = ncclSuccess;
  const char* mscclAlgoDir = getenv(mscclAlgoDirEnv);
  const char* mscclAlgoShareDir = nullptr;
  std::string mscclAlgoDirStr;
  std::string mscclAlgoShareDirStr;
  const char *fullDirPath = nullptr;
  if (mscclAlgoDir == nullptr) {
    // Try to find default algorithm directory based on scheduler.so and shcheduler algo installtion path.
    Dl_info dl_info;
    struct link_map *link_map_ptr = nullptr;
    if (!dladdr1((void *)mscclSchedulerInit, &dl_info, (void **)&link_map_ptr, RTLD_DL_LINKMAP)) {
      fprintf(stdout, "%s: %s Get dladdr1 failed \n", MSCCL_SCHEDULER_NAME, LOG_ERROR);
      return ncclInvalidUsage;
    }
    std::string selfLibPath = link_map_ptr->l_name;
    mscclAlgoDirStr = selfLibPath.substr(0, selfLibPath.find_last_of("/\\") + 1);
    mscclAlgoDirStr += (mscclUnitTestMode && mscclUnitTestMode()) ? mscclUnitTestAlgoDefaultDir : updateAlgoDirByVMSize(std::string(mscclAlgoDefaultDir));
    mscclAlgoDir = mscclAlgoDirStr.c_str();
    // Get share Directory Paths
    mscclAlgoShareDirStr = selfLibPath.substr(0, selfLibPath.find_last_of("/\\") + 1);
    mscclAlgoShareDirStr += (mscclUnitTestMode && mscclUnitTestMode()) ? mscclUnitTestAlgoShareDirPath : updateAlgoDirByVMSize(std::string(mscclAlgoShareDirPath));
    mscclAlgoShareDir = mscclAlgoShareDirStr.c_str();
  }
  fprintf(stdout, "%s: %s External Scheduler will use %s as algorithm directory and %s as share algorithm directory \n", MSCCL_SCHEDULER_NAME, LOG_INFO, mscclAlgoDir, mscclAlgoShareDir);
  struct dirent *entry = nullptr;
  DIR *dp = nullptr;
  dp = opendir(mscclAlgoDir);
  if (dp == nullptr) {
    // Try to find the algorithm directory under share folder based on libmsccl-scheduler.so path
    dp = opendir(mscclAlgoShareDir);
    if (dp == nullptr) {
      fprintf(stdout, "%s: %s Open algorithm in share directory %s failed \n", MSCCL_SCHEDULER_NAME, LOG_ERROR, mscclAlgoShareDir);
      return ncclInvalidUsage;
    }
    fullDirPath = mscclAlgoShareDir;
  } else {
    fullDirPath = mscclAlgoDir;
  }
  fprintf(stdout, "%s: %s Using MSCCL Algo files from %s \n", MSCCL_SCHEDULER_NAME, LOG_INFO, fullDirPath);
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
    fprintf(stdout, "%s: %s Closedir failed, error %d \n", MSCCL_SCHEDULER_NAME, LOG_ERROR, errno);
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

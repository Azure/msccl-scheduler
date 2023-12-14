/*************************************************************************
 * Copyright (c) 2019-2023 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt for license information
 ************************************************************************/

#include <cstdio>
#include <vector>
#include <map>
#include <dirent.h>
#include <dlfcn.h>
#include <link.h>
#include <string>
#include <cstring>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <pthread.h>
#include <iostream>

#ifdef RCCL
  #include "rccl/rccl.h"
#else 
  #include "nccl.h"
#endif

#include "common.h"
#include "client.h"
#include "parser.h"
#include "server.h"
#include "include/utils.h"

static const char* mscclAlgoDirEnv = "MSCCL_ALGO_DIR";
static const char* mscclAlgoDefaultDir = "msccl-algorithms";
extern "C" bool mscclUnitTestMode() __attribute__((__weak__));
static const char* mscclUnitTestAlgoDefaultDir = "msccl-unit-test-algorithms";
static const char* mscclAlgoShareDirPath = "../share/msccl-scheduler/msccl-algorithms";
static const char* mscclUnitTestAlgoShareDirPath = "../share/msccl-scheduler/msccl-unit-test-algorithms";
static const char* mscclPackageInstalledAlgoShareDirPath = "/usr/share/msccl-scheduler/msccl-algorithms";
static const char* mscclUnitTestPackageInstalledAlgoShareDirPath = "/usr/share/msccl-scheduler/msccl-unit-test-algorithms";
static const char* mscclAzureVMDetectionAgent = "http://169.254.169.254/metadata/instance?api-version=2019-06-04";

static pthread_t detectionServerThread;  
int world_rank;
int detectionServerExit;
std::vector<std::string> runningHostNames;
std::string fullDirPathStr;

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
            fprintf(stdout, "%s: %s Get Azure VM Size failed: %s\n", MSCCL_SCHEDULER_NAME, LOG_WARN, curl_easy_strerror(res));
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
      fprintf(stdout, "%s: %s There is no related algo file for the detected Azure VM SKU:%s been finded, MSCCL will use nccl as default communication channel\n", MSCCL_SCHEDULER_NAME, LOG_WARN, vmSize.c_str());
    }
    return updatedAlgoDir;
}

// Load meta information of algorithmsmscclSchedulerParam
__hidden ncclResult_t mscclSchedulerInit(mscclSchedulerInitParam *initParam) {
  ncclResult_t ret = ncclSuccess, tmpRet = ncclSuccess;
  const char* mscclAlgoDir = getenv(mscclAlgoDirEnv);
  const char* mscclAlgoShareDir = nullptr;
  const char* mscclPackageInstalledAlgoShareDir = nullptr;
  const char *fullDirPath = nullptr;
  std::string mscclAlgoDirStr;
  std::string mscclAlgoShareDirStr;
  std::string mscclPackageInstalledAlgoShareDirStr;
  world_rank = initParam->rank;
  
  if (mscclAlgoDir == nullptr) {
    // Try to find default algorithm directory based on scheduler.so and shcheduler algo installtion path.
    Dl_info dl_info;
    struct link_map *link_map_ptr = nullptr;
    if (!dladdr1((void *)mscclSchedulerInit, &dl_info, (void **)&link_map_ptr, RTLD_DL_LINKMAP)) {
      fprintf(stdout, "%s: %s Get dladdr1 failed\n", MSCCL_SCHEDULER_NAME, LOG_ERROR);
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
    // Get Package Installed share Directory Paths
    mscclPackageInstalledAlgoShareDirStr = (mscclUnitTestMode && mscclUnitTestMode()) ? mscclUnitTestPackageInstalledAlgoShareDirPath : mscclPackageInstalledAlgoShareDirPath;
    mscclPackageInstalledAlgoShareDir = mscclPackageInstalledAlgoShareDirStr.c_str();
  }
  fprintf(stdout, "%s: %s External Scheduler will use %s as algorithm directory and %s as share algorithm directory and %s as package installed share algorithm directory\n", MSCCL_SCHEDULER_NAME, LOG_INFO, mscclAlgoDir, mscclAlgoShareDir, mscclPackageInstalledAlgoShareDir);
  
  struct dirent *entry = nullptr;
  DIR *dp = nullptr;
  dp = opendir(mscclAlgoDir);
  if (dp == nullptr) {
    // Try to find the algorithm directory under share folder based on libmsccl-scheduler.so path
    dp = opendir(mscclAlgoShareDir);
    if (dp == nullptr) {
      //Try to find the algorithm directory under /usr/share folder which is package installed share algorithm directory
      dp = opendir(mscclPackageInstalledAlgoShareDir);
      if (dp == nullptr) {
        fprintf(stdout, "%s: %s Open algorithm in share directory %s failed\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, mscclPackageInstalledAlgoShareDir);
        return ncclInvalidUsage;
      }
      fullDirPath = mscclPackageInstalledAlgoShareDir;
    } 
    else{
      fullDirPath = mscclAlgoShareDir;
    }
  } else {
    fullDirPath = mscclAlgoDir;
  }
  fprintf(stdout, "%s: %s Using MSCCL Algo files from %s\n", MSCCL_SCHEDULER_NAME, LOG_INFO, fullDirPath);
  fullDirPathStr = std::string(fullDirPath);

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
    fprintf(stdout, "%s: %s Closedir failed, error %d\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, errno);
    return ncclInvalidUsage;
  }
  rankToAlgoHandles.resize(mscclAlgoMetas.size());

  fprintf(stdout, "%s: %s Start to get running HostNames, rank:%d, nrank:%d\n", MSCCL_SCHEDULER_NAME, LOG_INFO, initParam->rank, initParam->nRanks);

  if (GetRunningHostNames(initParam, runningHostNames) != ncclSuccess)
  {
    return ncclInvalidUsage;
  }

  if (0 == world_rank)
  {  
    if (pthread_create(&detectionServerThread, NULL, detectionServer, &initParam->nNodes))
    {
      fprintf(stdout, "%s: %s Create detection server failed, error %d\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, errno);
      return ncclInvalidUsage;
    }
  }
  detectionServerExit=false;
  
  return ret;
}

static __inline__ int ncclTypeSize(ncclDataType_t type) {
   switch (type) {
    case ncclInt8:
    case ncclUint8:
#if defined(__CUDA_FP8_TYPES_EXIST__)
    case ncclFp8E4M3:
    case ncclFp8E5M2:
#endif
      return 1;
    case ncclFloat16:
#if defined(__CUDA_BF16_TYPES_EXIST__)
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

__hidden ncclResult_t mscclScheduleAlternative(std::string xmlPath)
{
  ncclResult_t ret = ncclSuccess, tmpRet = ncclSuccess;

  // append a new algorithm into the algorithmmetas by opening a new xml file.
  fprintf(stdout, "%s: %s append a new algorithm into the algorithmmetas, xml: %s\n", MSCCL_SCHEDULER_NAME, LOG_INFO, xmlPath.c_str());
  mscclAlgoMetas.emplace_back();
  tmpRet = mscclGetAlgoMetaFromXmlFile(xmlPath.c_str(), &(mscclAlgoMetas.back()));
  rankToAlgoHandles.resize(mscclAlgoMetas.size());

  return ncclSuccess;
}

// Select algorithm, load if necessary
__hidden ncclResult_t mscclSchedulerSelectAlgo(struct mscclSchedulerParam* param) {
  ncclResult_t ret = ncclSuccess;

  param->scheduled = false;

  if (param->repair)
  {
    std::vector<std::string> xmlPaths;
    int nRet = 0;
    char* arr = new char[BUFFER_SIZE];
    std::memset(arr, 0, BUFFER_SIZE);
    arr[0] = '\0'; 

    if (0 == param->rank)
    {
      nRet = getOptimizedAlgoFiles(xmlPaths);
      fprintf(stdout, "%s: %s start to prepare send new algorithm to peers\n", MSCCL_SCHEDULER_NAME, LOG_INFO);
      if (0 == nRet)
      {
        size_t pos = 0;
        for (const auto& str : xmlPaths) {
          if (pos < BUFFER_SIZE)
          {
            std::copy(str.begin(), str.end(), arr + pos);
            pos += str.size();
            arr[pos] = '\0'; 
            ++pos;
          }
        }
      }
      fprintf(stdout, "%s: %s start to send new algorithm to peers\n", MSCCL_SCHEDULER_NAME, LOG_INFO);
      for (int i =1;i<param->nRanks;i++){
          param->send(param->bootstrap, i, TAG_ALGOINFO, arr, BUFFER_SIZE);
      }
      fprintf(stdout, "%s: %s finish send new algorithm to peers\n", MSCCL_SCHEDULER_NAME, LOG_INFO);
    }
    else{
      fprintf(stdout, "%s: %s start to receive new algorithm from rank 0\n", MSCCL_SCHEDULER_NAME, LOG_INFO);
      param->receive(param->bootstrap, 0, TAG_ALGOINFO, arr, BUFFER_SIZE);
      fprintf(stdout, "%s: %s finish receive new algorithm from rank 0\n", MSCCL_SCHEDULER_NAME, LOG_INFO);
      char* token = strtok(arr, "\0");
      while (token != nullptr) {
        xmlPaths.push_back(token);
        token = strtok(nullptr, "\0");
      }
    }
    for (const auto &xmlPath : xmlPaths) {
        mscclScheduleAlternative(xmlPath);
    }
    delete[] arr;
  }

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
    // fprintf(stdout, "%s: %s select algorithm isInPlace %d, AlgoMeta Size: %ld, msgSizeIsValid: %d, m.nRanks == param->nRanks: %d, m.func == param->func:%d, isInPlace ? m.inPlace : m.outOfPlace:%d\n", MSCCL_SCHEDULER_NAME, LOG_INFO, isInPlace, mscclAlgoMetas.size(), msgSizeIsValid, m.nRanks == param->nRanks, m.func == param->func, isInPlace ? m.inPlace : m.outOfPlace);  
    if (msgSizeIsValid &&
        m.nRanks == param->nRanks &&
        m.func == param->func &&
        (isInPlace ? m.inPlace : m.outOfPlace)) {
      fprintf(stdout, "%s: %s not loaded for current rank, load it\n", MSCCL_SCHEDULER_NAME, LOG_INFO);  
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

  if (0 == world_rank)
  {
    detectionServerExit = true;
    pthread_join(detectionServerThread, NULL);
  }

  return ret;
}

mscclSchedulerInterface mscclScheduler = {
  .name = MSCCL_SCHEDULER_NAME,
  .init = mscclSchedulerInit,
  .selectAlgo = mscclSchedulerSelectAlgo,
  .teardown = mscclSchedulerTearDown,
};

/*************************************************************************
 * Copyright (c) 2019-2023 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt for license information
 ************************************************************************/
#include <arpa/inet.h>
#include <iostream>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <future>
#include <set>
#include <thread>
#include <unistd.h>
#include "include/comm.h"
#include "include/server.h"
#include "include/utils.h"

#define PORT 12345

const int num_processes = 8;
std::string mscclShareDirPath;
extern int detectionServerExit;
extern int world_rank;
extern std::vector<std::string> mpiRunningHosts;
extern std::string fullDirPathStr;

std::string testNicPairs(std::pair<int, int> nicPair)
{
    char bufferPairKey[50];
    std::sprintf(bufferPairKey, "%d-%d", nicPair.first, nicPair.second);
    std::string pair_key = bufferPairKey;

    std::string hostList;
    for (size_t i = 0; i < mpiRunningHosts.size(); ++i)
    {
        if (i != 0)
            hostList += ",";
        hostList += mpiRunningHosts[i] + ":8";
    }

    std::string command = "python " + 
                            mscclShareDirPath + 
                            "scripts/test_nicpair.py " + 
                            std::to_string(nicPair.first) + " " + 
                            std::to_string(nicPair.second) +  " " + 
                            std::to_string(mpiRunningHosts.size()) + " " + 
                            hostList;
    
    std::string output, line;
    FILE* stream = popen(command.c_str(), "r");

    if (stream) {
        const int max_buffer = 256;
        char buffer[max_buffer];
        while (!feof(stream) && !ferror(stream)) {
            if (fgets(buffer, max_buffer, stream) != NULL) {
                line = buffer;
                if (line.find("No device found") != std::string::npos)
                {
                    fprintf(stdout, "%s: %s No device found detected on pair %s!\n", MSCCL_SCHEDULER_NAME, LOG_WARN, pair_key.c_str());
                    pclose(stream);
                    return pair_key;
                }
                output += line;
                fprintf(stdout, "%s: %s nic detection logs: %s\n", MSCCL_SCHEDULER_NAME, LOG_INFO, line.c_str()); // Print output dynamically
            }
        }
        if (pclose(stream) != 0) {
            fprintf(stdout, "%s: %s Subprocess returned non-zero exit code from key pair: %s!\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, pair_key.c_str());
        }
    }
    return std::string();
}

int detectNicFailure()
{
    std::vector<std::pair<int, int>> nic_pair;
    for (int i = 0; i < 8; ++i)
    {
        nic_pair.push_back(std::make_pair(i, i));
    }

    std::vector<std::future<std::string>> pool;
    std::set<std::string> failedPairsSet;

    for (std::pair<int, int> pair : nic_pair)
    {
        pool.push_back(std::async(testNicPairs, pair));
    }

    for (auto &t : pool)
    {
        std::string result = t.get();
        if (!result.empty())
        {
            failedPairsSet.insert(result);
        }
    }

    for (const std::string &pair : failedPairsSet)
    {
        fprintf(stdout, "%s: %s Failed nic pairs %s.\n", MSCCL_SCHEDULER_NAME, LOG_INFO, pair.c_str());
    }

    return 0;
}

std::string genNewSchedule(int nics) 
{
    std::string algoFileName = "new_algo.xml";
    std::string algoFileFullPath = fullDirPathStr + "/" + algoFileName;
    std::string command = "python " + 
                            mscclShareDirPath + 
                            "scripts/generate_newschedule.py > " +
                            algoFileFullPath;
    fprintf(stdout, "%s: %s generateNewSchedule command: %s\n", MSCCL_SCHEDULER_NAME, LOG_INFO, command.c_str());
    std:system(command.c_str());
    return algoFileFullPath;
}

void applyNewSchedule(std::string algoFileFullPath) 
{
    char localHostName[1024];
    gethostname(localHostName, 1024);

    for (size_t i = 0; i < mpiRunningHosts.size(); ++i)
    {
        if (mpiRunningHosts[i] != localHostName)
        {       
            std::string command = "scp " +
                                    algoFileFullPath +
                                    mpiRunningHosts[i] + ":" +
                                    algoFileFullPath; 
            fprintf(stdout, "%s: %s applyNewSchedule command: %s\n", MSCCL_SCHEDULER_NAME, LOG_INFO, command.c_str());
            std:system(command.c_str());    
        }
    }
}

void *detectionServer(void *args)
{
    fprintf(stdout, "%s: %s Starting Server thread.\n", MSCCL_SCHEDULER_NAME, LOG_INFO);

    mscclShareDirPath = fullDirPathStr.substr(0, fullDirPathStr.rfind("msccl-algorithms"));
    fprintf(stdout, "%s: %s MSCCL Share file path is %s\n", MSCCL_SCHEDULER_NAME, LOG_INFO, mscclShareDirPath.c_str());

    int server_socket, client_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(client_addr);
    fd_set set;
    struct timeval timeout;
    char buffer[1024] = {0};

    // Create a socket object
    if ((server_socket = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        fprintf(stdout, "%s: %s Detection Server Socket creation failed.\n", MSCCL_SCHEDULER_NAME, LOG_ERROR);
        return (void *)(intptr_t)-1;
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    // Bind the socket to the host and port
    if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        fprintf(stdout, "%s: %s Detection Server Socket bind failed.\n", MSCCL_SCHEDULER_NAME, LOG_ERROR);
        return (void *)(intptr_t)-1;
    }

    // Listen for incoming connections
    if (listen(server_socket, 3) < 0)
    {
        fprintf(stdout, "%s: %s Detection Server Socket listen failed.\n", MSCCL_SCHEDULER_NAME, LOG_ERROR);
        return (void *)(intptr_t)-1;
    }

    fprintf(stdout, "%s: %s Detection Server listening on: %s:%d.\n", MSCCL_SCHEDULER_NAME, LOG_INFO, inet_ntoa(server_addr.sin_addr), ntohs(server_addr.sin_port));

    FD_ZERO(&set);
    FD_SET(server_socket, &set);
    timeout.tv_sec = 10; // 10 second timeout
    timeout.tv_usec = 0;

    while (!detectionServerExit)
    {
        // Accept a client connection
        int rv = select(server_socket + 1, &set, NULL, NULL, &timeout);
        if (0 == rv)
        {
            continue;
        }

        if ((client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &addr_len)) < 0)
        {
            fprintf(stdout, "%s: %s Detection Server Client connection accept failed.\n", MSCCL_SCHEDULER_NAME, LOG_ERROR);
            return (void *)(intptr_t)-1;
        }

        fprintf(stdout, "%s: %s Detection Server Connected to: %s:%d.\n", MSCCL_SCHEDULER_NAME, LOG_INFO, inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));

        // Receive data from the client
        read(client_socket, buffer, 1024);
        std::string msg(buffer);

        if (msg == "detect_nic")
        {
            fprintf(stdout, "%s: %s Detection Server Received: %s and will start the detect NIC failure routine now.\n", MSCCL_SCHEDULER_NAME, LOG_INFO, msg.c_str());
            // detect NIC failure routine start.
            auto failed_nic = detectNicFailure();
            std::string algoFileFullPath = genNewSchedule(failed_nic);
            applyNewSchedule(algoFileFullPath);
            std::string response = algoFileFullPath;
            send(client_socket, response.c_str(), response.size(), 0);
        }

        // Close the client socket
        close(client_socket);
    }

    // Close the server socket
    close(server_socket);

    fprintf(stdout, "%s: %s Exit Server thread.\n", MSCCL_SCHEDULER_NAME, LOG_INFO);
    return 0;
}
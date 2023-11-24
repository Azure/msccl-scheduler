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
//#define BUFFER_SIZE 256

const int num_processes = 8;
std::string mscclShareDirPath;
extern int detectionServerExit;
extern int world_rank;
extern std::vector<std::string> mpiRunningHosts;
extern std::string fullDirPathStr;

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
                std::cout << line;  // Print output dynamically
            }
        }
        if (pclose(stream) != 0) {
            fprintf(stdout, "%s: %s Subprocess returned non-zero exit code from key pair: %s!\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, pair_key.c_str());
        }
    }

    // std::string command = "mpirun";
    // std::string args = "--mca btl_tcp_if_include eth0 "
    //                    "--mca pml ob1 "
    //                    "--mca btl ^openib "
    //                    "--allow-run-as-root "
    //                    "--bind-to numa "
    //                    "--tag-output "
    //                    "-np " + std::to_string(mpiRunningHosts.size()) + " "
    //                    "-npernode 1 -H " + hostList + " "
    //     "-x CUDA_VISIBLE_DEVICES=" + std::to_string(nicPair.first) +
    //     " echo helloworld! > ";
    //     // " python /root/msccl/scheduler/msccl-scheduler/tools/scripts/NIC_mpi4py.py " + std::to_string(nicPair.first) + " " +
    //     // std::to_string(nicPair.second);
    // std::string whole_command = command + " " + args;
    // // whole_command = "python /root/msccl/scheduler/msccl-scheduler/tools/scripts/NIC_mpi4py.py " + std::to_string(nicPair.first) + " " + std::to_string(nicPair.second);

    // fprintf(stdout, "%s: %s will execute the mpi command now: %s.\n", MSCCL_SCHEDULER_NAME, LOG_INFO, whole_command.c_str());
    
    // FILE* pipe = popen(whole_command.c_str(), "r");
    // if (!pipe) {
    //     return pair_key;
    // }
    // char buffer[128];
    // std::string result = "";
    // while (!feof(pipe)) {
    //     if (fgets(buffer, 128, pipe) != NULL)
    //         result += buffer;
    // }
    // pclose(pipe);
    // std::cout << result;


    // int pipe_fd[2];
    // pid_t pid;
    // char buffer[BUFFER_SIZE];
 
    // if (pipe(pipe_fd) < 0) {
    //     printf("Failed to create pipe\n");
    //      return pair_key;
    // }
 
    // pid = fork();
 
    // if (pid < 0) {
    //     printf("Failed to fork\n");
    //      return pair_key;
    // }
    // else if (pid == 0) {
    //     // child process
    //     close(pipe_fd[0]); // close the read end of the pipe
    //     const char *msg = "Hello, parent process!";
    //     write(pipe_fd[1], msg, strlen(msg)+1);
    //     exit(0);
    // }
    // else {
    //     // parent process
    //     close(pipe_fd[1]); // close the write end of the pipe
    //     int nbytes = 0;
    //     while ((nbytes = read(pipe_fd[0], buffer, BUFFER_SIZE)) > 0) {
    //         printf("Received message from child process: %s\n", buffer);
    //     }
    //     printf("ok\n");
    // }

    
    // int outfp;
    // pid_t pid = popen2(whole_command.c_str(), NULL, &outfp);
    // if (pid <= 0)
    // {
    //     fprintf(stdout, "%s: %s popen2() failed! %d\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, pid);
    //     return pair_key;
    // }

    // fprintf(stdout, "%s: %s popen2() returned: %d\n", MSCCL_SCHEDULER_NAME, LOG_INFO, pid);


    // // FILE *outstream = fdopen(outfp, "r");
    // // int fd = fileno(outstream);
    // // int flags = fcntl(fd, F_GETFL, 0);
    // // flags |= O_NONBLOCK;
    // // fcntl(fd, F_SETFL, flags);


    // char buffer[128];
    // buffer[128] = 0;
    // int n =0;
    // std::string result = "";
    // while((n = read(outfp, buffer, 128)) > 0){
    //     buffer[n] = 0;
    //     fprintf(stdout, "%s: %s read n:%d results in buffer: %s.\n", MSCCL_SCHEDULER_NAME, LOG_INFO, n, buffer);
    // }

    // fprintf(stdout, "%s: %s testNicPairs Result: %s.\n", MSCCL_SCHEDULER_NAME, LOG_INFO, result.c_str());

    
    // while (fgets(buffer, 128, outstream) != NULL){
    //     fprintf(stdout, "%s: %s testNicPairs result in buffer: %s.\n", MSCCL_SCHEDULER_NAME, LOG_INFO, buffer);
    //     result += buffer;}

    // fprintf(stdout, "%s: %s testNicPairs Result: %s.\n", MSCCL_SCHEDULER_NAME, LOG_INFO, result.c_str());

    // fprintf(stdout, "%s: %s waitpid() started\n", MSCCL_SCHEDULER_NAME, LOG_ERROR);
    // int status;
    // pid_t result_pid = waitpid(pid, &status, WNOHANG);
    // fprintf(stdout, "%s: %s waitpid() returned: %d\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, result_pid);
    // if (result_pid == 0)
    // {
    //     // timeout
    //     kill(pid, SIGKILL);
    //     return pair_key;
    // }
    // else if (result_pid == -1)
    // {
    //     // exception
    //     fprintf(stdout, "%s: %s waitpid() failed!\n", MSCCL_SCHEDULER_NAME, LOG_ERROR);
    //     return pair_key;
    // }

    // int return_code = pclose(outstream);

    // if (return_code != 0)
    // {
    //     fprintf(stdout, "%s: %s Subprocess returned non-zero exit code: %d \n", MSCCL_SCHEDULER_NAME, LOG_ERROR, return_code);
    //     return pair_key;
    // }

    // fprintf(stdout, "%s: %s testNicPairs Result: %s.\n", MSCCL_SCHEDULER_NAME, LOG_INFO, result.c_str());

    // if (result.find("No device found") != std::string::npos)
    // {
    //     fprintf(stdout, "%s: %s No device found detected!\n", MSCCL_SCHEDULER_NAME, LOG_WARN);
    //     return pair_key;
    // }
    return std::string();
}

void genNewSchedule(int nics) 
{
    std::string command = "python " + 
                            mscclShareDirPath + 
                            "scripts/generate_newschedule.py > " +
                            fullDirPathStr + 
                            "/new_algo.xml";
    fprintf(stdout, "%s: %s generateNewSchedule command: %s\n", MSCCL_SCHEDULER_NAME, LOG_INFO, command.c_str());
    std:system(command.c_str());
}

void applyNewSchedule() 
{
    char localHostName[1024];
    gethostname(localHostName, 1024);

    for (size_t i = 0; i < mpiRunningHosts.size(); ++i)
    {
        if (mpiRunningHosts[i] != localHostName)
        {       
            std::string command = "scp " +
                                    fullDirPathStr + 
                                    "/new_algo.xml " +
                                    mpiRunningHosts[i] + ":" +
                                    fullDirPathStr + 
                                    "/new_algo.xml"; 
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
            std::string response = std::to_string(failed_nic);
            send(client_socket, response.c_str(), response.size(), 0);
            genNewSchedule(failed_nic);
            applyNewSchedule();
        }

        // Close the client socket
        close(client_socket);
    }

    // Close the server socket
    close(server_socket);

    fprintf(stdout, "%s: %s Exit Server thread.\n", MSCCL_SCHEDULER_NAME, LOG_INFO);
    return 0;
}
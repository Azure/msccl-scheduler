/*************************************************************************
 * Copyright (c) 2019-2023 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt for license information
 ************************************************************************/
#include <iostream>
#include <set>
#include <string>
#include <unistd.h>
#include <unordered_set>

#include "include/comm.h"
#include "include/utils.h"

int testNicPair(ncclUniqueId id, int nic1, int nic2){
    // int rank, size;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);

    // // Set environment variables
    // std::string nic;
    // if (rank < 8) {
    //     nic = "mlx5_ib" + std::to_string(nic1);
    // } else {
    //     nic = "mlx5_ib" + std::to_string(nic2);
    // }
    // setenv("NCCL_IB_HCA", nic.c_str(), 1);
    // setenv("NCCL_DEBUG", "INFO", 1);
    // setenv("LD_LIBRARY_PATH", "/opt/openmpi-4.1.5/lib/:" + std::getenv("LD_LIBRARY_PATH"), 1);

    // // Initialize NCCL
    // ncclComm_t comm;
    // ncclCommInitRank(&comm, size, id, rank);

    // // Prepare data for all-gather operation
    // std::vector<float> sendbuf(100, rank);
    // std::vector<float> recvbuf(100 * size);

    // // Perform all-gather operation
    // ncclAllGather(sendbuf.data(), recvbuf.data(), 100, ncclFloat, comm, NULL);

    // // Finalize NCCL and MPI
    // ncclCommDestroy(comm);

    return 0;
}

pid_t popen2(const char *command, int *infp, int *outfp)
{
    int p_stdin[2], p_stdout[2];
    pid_t pid;

    if (pipe(p_stdin) != 0 || pipe(p_stdout) != 0)
        return -1;

    pid = fork();

    if (pid < 0)
        return pid;
    else if (pid == 0)
    {
        // close(p_stdin[1]);
        // dup2(p_stdin[0], STDIN_FILENO);



        close(p_stdout[0]);
        // dup2(p_stdout[1], STDOUT_FILENO);

        extern char** environ;
        std::string env_vars;
        for (char** env = environ; *env; ++env) {
            std::string env_var(*env);
            // Filter out environment variables containing "MPI"
            if (env_var.find("MPI") == std::string::npos) {
                env_vars += env_var + " ";
            }
        }

        // Clear all environment variables
        clearenv();

        // Set filtered environment variables
        char* env_var = strtok(&env_vars[0], " ");
        while (env_var != NULL) {
            char* equal_sign = strchr(env_var, '=');
            if (equal_sign != NULL) {
                *equal_sign = '\0';
                setenv(env_var, equal_sign + 1, 1);
            }
            env_var = strtok(NULL, " ");
        }

        command = "echo helloworld!";

        // std:system(command);
        // system("exit 0");
        // execl("/bin/sh", "sh", "-c", command, NULL);
        // execl("/bin/echo", "echo", "hello world!", NULL);
        // perror("execl");
        // close(p_stdin[0]);
        // close(p_stdout[1]);

        write(p_stdout[1], "hello world!!!", 14);
        exit(0);
    }

    if (infp == NULL)
        close(p_stdin[1]);
    else
        *infp = p_stdin[1];

    if (outfp == NULL)
        close(p_stdout[0]);
    else
        *outfp = p_stdout[0];

    return pid;
}

void* mpiNodeExecutor(void* args){
    std::vector<std::string> strVec = mpiGetHostNames();
    std::cout << "MPI_Gather result0: ";
    for (const auto& str : strVec) {
        std::cout << str << ' ';
    }
    std::cout << '\n';

    // MPI_Bcast(&number, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return NULL;
}

std::vector<std::string> mpiGetHostNames(){
    std::vector<std::string> strVec;
    
    char hostname[1024];
    gethostname(hostname, 1024);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string sendbuf(hostname);

    if (rank == 0) {
        std::vector<char> recvbuf(size * sendbuf.size());
        MPI_Gather(sendbuf.data(), sendbuf.size(), MPI_CHAR, recvbuf.data(), sendbuf.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
        
        for (int i = 0; i < size; ++i) {
            strVec.push_back(std::string(recvbuf.begin() + i * sendbuf.size(), recvbuf.begin() + (i + 1) * sendbuf.size()));
        }

        std::unordered_set<std::string> s(strVec.begin(), strVec.end());
        strVec.assign(s.begin(), s.end());
    } else {
        MPI_Gather(sendbuf.data(), sendbuf.size(), MPI_CHAR, nullptr, 0, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    return strVec;
}


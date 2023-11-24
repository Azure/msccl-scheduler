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


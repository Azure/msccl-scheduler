/*************************************************************************
 * Copyright (c) 2019-2023 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt for license information
 ************************************************************************/
#include <iostream>
#include <string>
#include <cstring> 
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sstream>
#include <unistd.h>
#include <vector>

#include "include/comm.h"

int getOptimizedAlgoFiles(std::vector<std::string> &xmlPaths)
{
    // Create a socket
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1) {
        fprintf(stdout, "%s: %s Failed to create socket: %s\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, std::strerror(errno));
        return 1;
    }

    // Set up server details
    sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = inet_addr(HOST_ADDR);
    server_address.sin_port = htons(SERVER_PORT);
    if (inet_addr(HOST_ADDR) == -1) {
        fprintf(stdout, "%s: %s Invalid IP address: %s\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, HOST_ADDR);
        return 1;
    }
    
    fprintf(stdout, "%s: %s start to connect to server\n", MSCCL_SCHEDULER_NAME, LOG_INFO);
    // Connect to the server
    if (connect(client_socket, (struct sockaddr*)&server_address, sizeof(server_address)) < 0) {
        fprintf(stdout, "%s: %s Failed to connect to server: %s\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, std::strerror(errno));
        return 1;
    }

    // Send data to the server
    std::string message = "detect_nic";
    if (send(client_socket, message.c_str(), message.size(), 0) < 0) {
        fprintf(stdout, "%s: %s Failed to send message: %s\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, std::strerror(errno));
        return 1;
    }

    // Receive the response from the server
    char buffer[BUFFER_SIZE] = {0};
    if (recv(client_socket, buffer, sizeof(buffer) - 1, 0) < 0) {
        fprintf(stdout, "%s: %s Failed to receive response: %s\n", MSCCL_SCHEDULER_NAME, LOG_ERROR, std::strerror(errno));
        return 1;
    }

    std::istringstream iss(buffer);
    std::string temp;

    while (std::getline(iss, temp, ';')) {
        xmlPaths.push_back(temp);
    }
    
    // Close the socket
    close(client_socket);
    return 0;
}
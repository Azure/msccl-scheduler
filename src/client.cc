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
#include <vector>

int sendDetectInfo(std::vector<std::string> &xmlPaths)
{
    const char* HOST = "127.0.0.1";
    int PORT = 12345;

    // Create a socket
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1) {
        std::cerr << "Failed to create socket: " << std::strerror(errno) << std::endl;
        return 1;
    }

    // Set up server details
    sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(PORT);
    if (inet_addr(HOST) == -1) {
        std::cerr << "Invalid IP address: " << HOST << std::endl;
        return 1;
    }
    server_address.sin_addr.s_addr = inet_addr(HOST);

    // Connect to the server
    if (connect(client_socket, (struct sockaddr*)&server_address, sizeof(server_address)) < 0) {
        std::cerr << "Failed to connect to server: " << std::strerror(errno) << std::endl;
        return 1;
    }

    // Send data to the server
    std::string message = "detect_nic";
    if (send(client_socket, message.c_str(), message.size(), 0) < 0) {
        std::cerr << "Failed to send message: " << std::strerror(errno) << std::endl;
        return 1;
    }

    // Receive the response from the server
    char buffer[1024] = {0};
    if (recv(client_socket, buffer, sizeof(buffer) - 1, 0) < 0) {
        std::cerr << "Failed to receive response: " << std::strerror(errno) << std::endl;
        return 1;
    }
    std::cout << buffer << std::endl;

    std::istringstream iss(buffer);
    std::string temp;

    while (std::getline(iss, temp, ';')) {
        xmlPaths.push_back(temp);
    }
    
    // Close the socket
    close(client_socket);
}
/*************************************************************************
 *
 * RENAISSANCE ROBOT LLC CONFIDENTIAL
 * __________________
 *
 *  [2017] RENAISSANCE ROBOT LLC
 *  All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains the property of
 * Renaissance Robot LLC and its suppliers, if any. The intellectual and
 * technical concepts contained herein are proprietary to Renaissance Robot LLC
 * and its suppliers and may be covered by U.S. and Foreign Patents, patents in
 * process, and are protected by trade secret or copyright law.
 *
 * Dissemination of this information or reproduction of this material is strictly
 * forbidden unless prior written permission is obtained from Renaissance Robot LLC.
 */

#include <iostream>
#include <cstdio>

#include "clutils.h"

#define TERMINAL_GREEN  "\033[0;32m"
#define TERMINAL_RED    "\033[0;31m"
#define TERMINAL_RESET  "\033[0;39;49m"

cl::Device selectDevice(int devid = -1)
{
    std::vector<cl::Platform> plats = CLUtils::getPlatforms();

    std::vector<cl::Device> devices;

    for (cl::Platform plat : plats) {
        std::cout << "Platform: " << plat.getInfo<CL_PLATFORM_NAME>()
                  << ", by " << plat.getInfo<CL_PLATFORM_VENDOR>()
                  << ", version " << plat.getInfo<CL_PLATFORM_VERSION>()
                  << std::endl;


        std::vector<cl::Device> devs = CLUtils::getDevices(plat, CL_DEVICE_TYPE_ALL);

        if (devs.size() < 1)
            break;

        for (cl::Device dev : devs) {
            std::cout << "\t[" << devices.size() << "] Device: " << dev.getInfo<CL_DEVICE_NAME>()
                      << ", by " << dev.getInfo<CL_DEVICE_VENDOR>()
                      << ", type " << dev.getInfo<CL_DEVICE_TYPE>()
                      << ", version " << dev.getInfo<CL_DEVICE_VERSION>()
                      << std::endl;
            devices.push_back(dev);
        }
    }

    if (devices.empty()) {
        std::cout << "No platform/device available." << std::endl;
        exit(1);
    }

    int devnum;

    do {
        if (devid != -1) {
            devnum = devid;
            devid = -1;
        } else {
            std::cout << "Select a device [0";
            if (devices.size() != 1)
                std::cout << "-" << devices.size() - 1;
            std::cout << "]: " << std::endl;

            std::scanf("%d", &devnum);
        }

        if (devnum < 0 || devnum >= devices.size()) {
            std::cout << "Invalid device" << std::endl;
            continue;
        } else {
            std::cout << "Selected " << devnum << std::endl;
            break;
        }
    } while (true);

    return devices.at(devnum);
}


int main(int argc, char *argv[])
{
    std::string src =
            "__kernel void vecadd(__global int* restrict A, __global int* restrict B, __global int* restrict C)"
            "{"
            "   int i = get_global_id(0);"
            "   C[i] = A[i] + B[i];"
            "}";



    int devid;
    if (argc == 2)
        devid = std::atoi(argv[1]);

    cl::Device dev = selectDevice(devid);

    cl::Context ctx = CLUtils::createContext(dev);

    cl_int ok;


    cl::Buffer bufferA(ctx, (cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR), 128 * 4, nullptr, &ok);
    cl::Buffer bufferB(ctx, (cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR), 128 * 4, nullptr, &ok);
    cl::Buffer bufferC(ctx, (cl_mem_flags)(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR), 128 * 4, nullptr, &ok);

    bool bool_ok;
    std::string err;
    cl::Program program = CLUtils::buildProgram(ctx, dev, {src}, std::string(), &bool_ok, &err);
    cl::Kernel kernel(program, "vecadd");

    cl::CommandQueue queue(ctx, dev);

    cl_int *A = (cl_int *)queue.enqueueMapBuffer(bufferA, CL_TRUE, CL_MAP_WRITE, 0, 128 * 4, nullptr, nullptr, &ok);
    if (ok != CL_SUCCESS)
        return -ok;

    cl_int *B = (cl_int *)queue.enqueueMapBuffer(bufferB, CL_TRUE, CL_MAP_WRITE, 0, 128 * 4, nullptr, nullptr, &ok);
    if (ok != CL_SUCCESS)
        return -ok;


    for (int i = 0; i < 128; ++i) {
        A[i] = i;
        B[i] = 10;
    }

    queue.enqueueUnmapMemObject(bufferA, A);
    queue.enqueueUnmapMemObject(bufferB, B);

    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);

    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(128));

//CUDA: kernel<<<128>>>(bufferA, bufferB, bufferC);

    queue.finish();

    cl_int *C = (cl_int *)queue.enqueueMapBuffer(bufferC, CL_TRUE, CL_MAP_READ, 0, 128 * 4);
    for (int i = 0; i < 128; ++i) {
        if (i % 8 == 0)
            std::cout << std::endl;

        if (C[i] == 10 + i)
            std::cout << TERMINAL_GREEN <<  "OK     " << TERMINAL_RESET;
        else
            std::cout << TERMINAL_RED <<    "FAILED " << TERMINAL_RESET;
        std::cout << C[i] << "\t";
    }
    std::cout << std::endl;

    queue.enqueueUnmapMemObject(bufferC, C);

    return 0;
}














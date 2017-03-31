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

#include <cstdio>
#include <iostream>
#include <unistd.h>
#include <malloc.h>

#include "lodepng.h"
#include "clutils.h"

#define TERMINAL_GREEN  "\033[0;32m"
#define TERMINAL_RED    "\033[0;31m"
#define TERMINAL_RESET  "\033[0;39;49m"

const int MEM_ALIGNMENT = 4096; // 4k aligned

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


int arithmetic_test(const cl::Context &ctx, const cl::Device &dev, const cl::CommandQueue &queue)
{
    cl_int ok;

    cl::Buffer bufferA(ctx, (cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR), 128 * 4, nullptr, &ok);
    cl::Buffer bufferB(ctx, (cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR), 128 * 4, nullptr, &ok);
    cl::Buffer bufferC(ctx, (cl_mem_flags)(CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR), 128 * 4, nullptr, &ok);

    bool bool_ok;
    std::string err;
    std::string src =
            "__kernel void vecadd(__global int* restrict A, __global int* restrict B, __global int* restrict C)"
            "{"
            "   int i = get_global_id(0);"
            "   C[i] = A[i] + B[i];"
            "}";
    cl::Program program = CLUtils::buildProgram(ctx, dev, {src}, std::string(), &bool_ok, &err);
    cl::Kernel kernel(program, "vecadd");

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

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(128));

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

int texture_test(const cl::Context &ctx, const cl::Device &dev, const cl::CommandQueue &queue,
                 const size_t w, const size_t h, const size_t watermarkW, const size_t watermarkH,
                 const std::vector<unsigned char> &imagea,
                 const std::vector<unsigned char> &imageb,
                 std::vector<unsigned char> &imagec)
{
    bool bool_ok;
    std::string err;
    std::string src =
            "__constant sampler_t inputSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;"
            "__constant sampler_t watermarkSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;"
            "__kernel void imgblend(__read_only image2d_t input, __read_only image2d_t watermark, __write_only image2d_t output, float2 stepInput, float2 stepWatermark)"
            "{"
                "const int2 posi = { get_global_id(0), get_global_id(1) };"
                "const float2 pos = convert_float2(posi) * stepInput;"
                "const float2 wmpos = convert_float2(posi) * stepWatermark;"
                "float4 ptA = read_imagef(input, inputSampler, pos);"
                "float4 ptB = read_imagef(watermark, watermarkSampler, wmpos);"
                "float4 ptC = ptA * (float4)(0.8f, 0.8f, 0.8f, 1.f) + ptB * (float4)(0.2f, 0.2f, 0.2f, 1.f);"
                "write_imagef(output, posi, ptC);"
            "}"
            ;

    cl::Program program = CLUtils::buildProgram(ctx, dev, {src}, std::string(), &bool_ok, &err);
    if (!bool_ok) {
        std::cerr << err << std::endl;
        return 1;
    }

    cl::Kernel kernel(program, "imgblend");

    const cl::ImageFormat imageFormat(CL_RGBA, CL_UNORM_INT8);

    size_t size = w * h * 4;
    size_t watermarkSize = watermarkW * watermarkH * 4;

    unsigned char *inputImg     = (unsigned char *)memalign(MEM_ALIGNMENT, size);
    unsigned char *watermarkImg = (unsigned char *)memalign(MEM_ALIGNMENT, watermarkSize);
    unsigned char *outputImg    = (unsigned char *)memalign(MEM_ALIGNMENT, size);


    cl_int ok;
    cl::Image2D clImgInput      = cl::Image2D(ctx, (cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR),
                                              imageFormat, w, h, 0, inputImg, &ok);
    cl::Image2D clImgWatermark  = cl::Image2D(ctx, (cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR),
                                              imageFormat, watermarkW, watermarkH, 0, watermarkImg, &ok);
    cl::Image2D clImgOutput     = cl::Image2D(ctx, (cl_mem_flags)(CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR),
                                              imageFormat, w, h, 0, outputImg, &ok);

    size_t mappedImgInputPitch, mappedImgWatermarkPitch, mappedImgOutputPitch;
    void *mappedImgInput        = queue.enqueueMapImage(clImgInput, CL_TRUE, CL_MAP_WRITE,
                                                        { 0, 0, 0 }, { w, h, 1 },
                                                        &mappedImgInputPitch, nullptr, nullptr, nullptr, &ok);

    void *mappedImgWatermark    = queue.enqueueMapImage(clImgWatermark, CL_TRUE, CL_MAP_WRITE,
                                                        { 0, 0, 0 }, { watermarkW, watermarkH, 1 },
                                                        &mappedImgWatermarkPitch, nullptr, nullptr, nullptr, &ok);

    // Put memcpy above enqueueMapImage is undefined in OpenCL Specificiation!
    // Even it works!
    std::memcpy(inputImg, imagea.data(), size);
    std::memcpy(watermarkImg, imageb.data(), watermarkSize);


    queue.enqueueUnmapMemObject(clImgInput, mappedImgInput);
    queue.enqueueUnmapMemObject(clImgWatermark, mappedImgWatermark);

    mappedImgInput = 0; mappedImgWatermark = 0;

    cl_float2 stepInput = { 1.f / w, 1.f / h };
    cl_float2 stepWatermark = { 1.f / watermarkW, 1.f / watermarkH };

    kernel.setArg(0, clImgInput);
    kernel.setArg(1, clImgWatermark);
    kernel.setArg(2, clImgOutput);
    kernel.setArg(3, stepInput);
    kernel.setArg(4, stepWatermark);
    ok = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(w, h));

    queue.finish();

    void *mappedC = queue.enqueueMapImage(clImgOutput, CL_TRUE, CL_MAP_READ, { 0, 0, 0 }, { w, h, 1 },
                                          &mappedImgOutputPitch, nullptr, nullptr, nullptr, &ok);

    imagec.resize(size);
    std::memcpy(imagec.data(), mappedC, size);
    queue.enqueueUnmapMemObject(clImgOutput, mappedC);

    free(inputImg);
    free(watermarkImg);
    free(outputImg);
}

int main(int argc, char *argv[])
{
    int devid;
    if (argc == 2)
        devid = std::atoi(argv[1]);

    cl::Device dev = selectDevice(devid);

    cl::Context ctx = CLUtils::createContext(dev);

    cl::CommandQueue queue(ctx, dev);

    int res;

    res = arithmetic_test(ctx, dev, queue);
    if (res)
        return res;

    std::vector<unsigned char> imagea;
    std::vector<unsigned char> imageb;
    unsigned int widtha, widthb, heighta, heightb;

    unsigned int err;
    err = lodepng::decode(imagea, widtha, heighta, "a.png");
    if (err) {
        std::cerr << lodepng_error_text(err);
        return -2;
    }

    err = lodepng::decode(imageb, widthb, heightb, "b.png");
    if (err) {
        std::cerr << lodepng_error_text(err);
        return -3;
    }


    std::vector<unsigned char> imagec;

    int oriW = widtha;
    int oriH = heighta;
    texture_test(ctx, dev, queue, oriW, oriH, widthb, heightb, imagea, imageb, imagec);

    std::cout << "Writing..." << std::endl;

    lodepng::encode("output.png", imagec, oriW, oriH);

    return 0;
}














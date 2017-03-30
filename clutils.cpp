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

#include "clutils.h"

std::vector<cl::Platform> CLUtils::getPlatforms()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    return platforms;
}

std::vector<cl::Device> CLUtils::getDevices(const cl::Platform &platform, cl_device_type type)
{
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    return devices;
}

cl::Context CLUtils::createContext(const cl::Device &device)
{
    return cl::Context(device);
}

cl::Program CLUtils::buildProgram(const cl::Context &context, const cl::Device &device,
                                  const std::vector<std::string> &src, const std::string &buildOption,
                                  bool *ok, std::string *log)
{
    cl::Program::Sources sources;

    for (std::string s : src) {
        sources.push_back( { s.c_str(), s.length() } );
    }

    cl::Program program(context, sources);

    auto res = program.build({ device }, buildOption.c_str());
    if (ok)
        *ok = res == CL_SUCCESS;
    if (log)
        *log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);

    return program;
}

cl::Program CLUtils::buildProgram(const cl::Context &context, const cl::Device &device,
                                  const std::vector<std::pair<void *, size_t>> &bin, bool *ok, std::string *error)
{
    throw ("Not implemented");
}












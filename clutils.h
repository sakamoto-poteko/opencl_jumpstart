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

#ifndef CLUTILS_H
#define CLUTILS_H

#include <CL/cl.hpp>

#include <vector>

class CLUtils
{
public:
    CLUtils() {}

    static std::vector<cl::Platform> getPlatforms();
    static std::vector<cl::Device> getDevices(const cl::Platform &platform, cl_device_type type = CL_DEVICE_TYPE_ALL);

    static cl::Context createContext(const cl::Device &device);
    static cl::Program buildProgram(const cl::Context &context, const cl::Device &device,
                                    const std::vector<std::string> &src, const std::string &buildOption = std::string(),
                                    bool *ok = 0, std::string *log = 0);
    static cl::Program buildProgram(const cl::Context &context, const cl::Device &device,
                                    const std::vector<std::pair<void *, size_t>> &bin,
                                    bool *ok = 0, std::string *error = 0);

private:

};

#endif

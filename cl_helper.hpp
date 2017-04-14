#ifndef _CL_HELPER_H
#define _CL_HELPER_H

#ifndef __OPENCL_VERSION__


#define __kernel
#define __global
#define __local
#define __constant
#define __read_only
#define __write_only
#define __read_write
#define unsigned

enum {
    CLK_NORMALIZED_COORDS_TRUE      = 0b00000000,
    CLK_NORMALIZED_COORDS_FALSE     = 0b00000001,
    CLK_ADDRESS_NONE                = 0b00000000,
    CLK_ADDRESS_CLAMP               = 0b00000010,
    CLK_ADDRESS_CLAMP_TO_EDGE       = 0b00000100,
    CLK_ADDRESS_REPEAT              = 0b00000110,
    CLK_FILTER_LINEAR               = 0b00000000,
    CLK_FILTER_NEAREST              = 0b00001000,
};

typedef int sampler_t;
typedef int image2d_t;

int get_global_id(int dim);

struct int2
{
    int x;
    int y;
};

struct int4
{
    int x;
    int y;
    int z;
    int w;
};

struct float2
{
    float x;
    float y;
};

struct float4
{
    float x;
    float y;
    float z;
    float w;
};

float4 read_imagef(image2d_t image, sampler_t sampler, int2 coord);
float4 read_imagef(image2d_t image, sampler_t sampler, float2 coord);

void write_imagef(image2d_t image, int2 coord, float4 color);
void write_imagei(image2d_t image, int2 coord, int4 color);
void write_imageui(image2d_t image, int2 coord, unsigned int4 color);

float2 convert_float2(int2 val);

#endif

#endif

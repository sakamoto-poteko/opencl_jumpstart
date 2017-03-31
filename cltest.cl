__kernel void vecadd(__global int* restrict A, __global int* restrict B, __global int* restrict C)
{
    int i = get_global_id(0);
    C[i] = A[i] + B[i];
}

__constant sampler_t inputSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
__constant sampler_t watermarkSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;
__kernel void imgblend(__read_only image2d_t input, __read_only image2d_t watermark, __write_only image2d_t output,
                       float2 stepInput, float2 stepWatermark)
{
    const int2 posi = { get_global_id(0), get_global_id(1) };
    const float2 pos = convert_float2(posi) * stepInput;
    const float2 wmpos = convert_float2(posi) * stepWatermark;
    float4 ptA = read_imagef(input, inputSampler, pos);
    float4 ptB = read_imagef(watermark, watermarkSampler, wmpos);
    float4 ptC = ptA * (float4)(0.8f, 0.8f, 0.8f, 1.f) + ptB * (float4)(0.2f, 0.2f, 0.2f, 1.f);
    write_imagef(output, posi, ptC);
}

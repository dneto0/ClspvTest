//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_COPYIMAGETOBUFFER_KERNEL_HPP
#define CLSPVTEST_COPYIMAGETOBUFFER_KERNEL_HPP

#include "clspv_utils.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

namespace copyimagetobuffer_kernel {

    clspv_utils::execution_time_t
    invoke(clspv_utils::kernel&             kernel,
           vulkan_utils::image&             src_image,
           vulkan_utils::storage_buffer&    dst_buffer,
           int                              dst_offset,
           int                              dst_pitch,
           cl_channel_order                 dst_channel_order,
           cl_channel_type                  dst_channel_type,
           bool                             swap_components,
           int                              width,
           int                              height);

    void test_matrix(clspv_utils::kernel&               kernel,
                     const std::vector<std::string>&    args,
                     bool                               verbose,
                     test_utils::InvocationResultSet&   resultSet);

    template <typename BufferPixelType, typename ImagePixelType>
    void test(clspv_utils::kernel&              kernel,
              const std::vector<std::string>&   args,
              bool                              verbose,
              test_utils::InvocationResultSet&  resultSet)
    {
        test_utils::InvocationResult invocationResult;
        invocationResult.mVariation = "<src:";
        invocationResult.mVariation += pixels::traits<ImagePixelType>::type_name;
        invocationResult.mVariation += " dst:";
        invocationResult.mVariation += pixels::traits<BufferPixelType>::type_name;
        invocationResult.mVariation += ">";

        auto& device = kernel.getDevice();

        const int buffer_height = 64;
        const int buffer_width = 64;

        const std::size_t buffer_length = buffer_width * buffer_height;
        const std::size_t buffer_size = buffer_length * sizeof(BufferPixelType);

        // allocate buffers and images
        vulkan_utils::storage_buffer    dst_buffer(device.mDevice, device.mMemoryProperties, buffer_size);
        vulkan_utils::image             srcImage(device.mDevice, device.mMemoryProperties, buffer_width, buffer_height, vk::Format(pixels::traits<ImagePixelType>::vk_pixel_type));

        // initialize source memory with random data
        test_utils::fill_random_pixels<ImagePixelType>(srcImage.mem, buffer_length);

        // initialize destination memory (copy source and invert, thereby forcing the kernel to make the change back to the source value)
        test_utils::copy_pixel_buffer<ImagePixelType, BufferPixelType>(srcImage.mem, dst_buffer.mem, buffer_length);
        test_utils::invert_pixel_buffer<BufferPixelType>(dst_buffer.mem, buffer_length);

        invocationResult.mExecutionTime = invoke(kernel,
                                                 srcImage,
                                                 dst_buffer,
                                                 0,
                                                 buffer_width,
                                                 pixels::traits<BufferPixelType>::cl_pixel_order,
                                                 pixels::traits<BufferPixelType>::cl_pixel_type,
                                                 false,
                                                 buffer_width,
                                                 buffer_height);

        test_utils::check_results<ImagePixelType, BufferPixelType>(srcImage.mem, dst_buffer.mem,
                                                                   buffer_width, buffer_height,
                                                                   buffer_height,
                                                                   verbose,
                                                                   invocationResult);

        resultSet.push_back(std::move(invocationResult));
    }

    template <typename ImagePixelType>
    void test_series(clspv_utils::kernel&               kernel,
                     const std::vector<std::string>&    args,
                     bool                               verbose,
                     test_utils::InvocationResultSet&   resultSet)
    {
        const test_utils::test_kernel_fn tests[] = {
                test<gpu_types::uchar, ImagePixelType>,
                test<gpu_types::uchar4, ImagePixelType>,
                test<gpu_types::half, ImagePixelType>,
                test<gpu_types::half4, ImagePixelType>,
                test<float, ImagePixelType>,
                test<gpu_types::float2, ImagePixelType>,
                test<gpu_types::float4, ImagePixelType>,
        };

        test_utils::test_kernel_invocations(kernel,
                                            std::begin(tests), std::end(tests),
                                            args,
                                            verbose,
                                            resultSet);
    }
}

#endif //CLSPVTEST_COPYIMAGETOBUFFER_KERNEL_HPP

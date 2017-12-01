//
// Created by Eric Berdahl on 10/31/17.
//

#ifndef CLSPVTEST_COPYIMAGETOBUFFER_KERNEL_HPP
#define CLSPVTEST_COPYIMAGETOBUFFER_KERNEL_HPP

#include "clspv_utils.hpp"
#include "gpu_types.hpp"
#include "test_utils.hpp"
#include "util.hpp"
#include "vulkan_utils.hpp"

#include <vulkan/vulkan.hpp>

namespace copyimagetobuffer_kernel {

    void invoke(const clspv_utils::kernel_module&   module,
                const clspv_utils::kernel&          kernel,
                const sample_info&                  info,
                vk::ArrayProxy<const vk::Sampler>   samplers,
                vk::ImageView                       src_image,
                vk::Buffer                          dst_buffer,
                int                                 dst_offset,
                int                                 dst_pitch,
                cl_channel_order                    dst_channel_order,
                cl_channel_type                     dst_channel_type,
                bool                                swap_components,
                int                                 width,
                int                                 height);

    test_utils::Results test_matrix(const clspv_utils::kernel_module&     module,
                                    const clspv_utils::kernel&            kernel,
                                    const sample_info&                    info,
                                    vk::ArrayProxy<const vk::Sampler>     samplers,
                                    const test_utils::options&            opts);

    template <typename BufferPixelType, typename ImagePixelType>
    test_utils::Results test(const clspv_utils::kernel_module&    module,
                             const clspv_utils::kernel&           kernel,
                             const sample_info&                   info,
                             vk::ArrayProxy<const vk::Sampler>    samplers,
                             const test_utils::options&           opts)
    {
        std::string typeLabel = pixels::traits<BufferPixelType>::type_name;
        typeLabel += '-';
        typeLabel += pixels::traits<ImagePixelType>::type_name;

        std::string testLabel = "memory.spv/CopyImageToBufferKernel/";
        testLabel += typeLabel;

        const int buffer_height = 64;
        const int buffer_width = 64;

        const std::size_t buffer_size = buffer_width * buffer_height * sizeof(BufferPixelType);

        // allocate buffers and images
        vulkan_utils::buffer  dst_buffer(info, buffer_size);
        vulkan_utils::image   srcImage(info, buffer_width, buffer_height, vk::Format(pixels::traits<ImagePixelType>::vk_pixel_type));

        // initialize source and destination buffers
        {
            auto src_value = pixels::traits<ImagePixelType>::translate((gpu_types::float4){ 0.2f, 0.4f, 0.8f, 1.0f });
            vulkan_utils::memory_map src_map(srcImage);
            auto src_data = static_cast<decltype(src_value)*>(src_map.map());
            std::fill(src_data, src_data + (buffer_width * buffer_height), src_value);
        }

        {
            auto dst_value = pixels::traits<BufferPixelType>::translate((gpu_types::float4){ 0.1f, 0.3f, 0.5f, 0.7f });
            vulkan_utils::memory_map dst_map(dst_buffer);
            auto dst_data = static_cast<decltype(dst_value)*>(dst_map.map());
            std::fill(dst_data, dst_data + (buffer_width * buffer_height), dst_value);
        }

        invoke(module, kernel,
               info,
               samplers,
               *srcImage.view,
               *dst_buffer.buf,
               0,
               buffer_width,
               pixels::traits<BufferPixelType>::cl_pixel_order,
               pixels::traits<BufferPixelType>::cl_pixel_type,
               false,
               buffer_width,
               buffer_height);

        const bool success = test_utils::check_results<ImagePixelType, BufferPixelType>(srcImage.mem, dst_buffer.mem,
                                                                                        buffer_width, buffer_height,
                                                                                        buffer_height,
                                                                                        testLabel.c_str(),
                                                                                        opts);

        return (success ? test_utils::Results::sTestSuccess : test_utils::Results::sTestFailure);
    }

    template <typename ImagePixelType>
    test_utils::Results test_series(const clspv_utils::kernel_module&     module,
                                    const clspv_utils::kernel&            kernel,
                                    const sample_info&                    info,
                                    vk::ArrayProxy<const vk::Sampler>     samplers,
                                    const test_utils::options&            opts)
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

        return test_utils::test_kernel_invocation(module, kernel,
                                                  std::begin(tests), std::end(tests),
                                                  info,
                                                  samplers,
                                                  opts);
    }
}

#endif //CLSPVTEST_COPYIMAGETOBUFFER_KERNEL_HPP
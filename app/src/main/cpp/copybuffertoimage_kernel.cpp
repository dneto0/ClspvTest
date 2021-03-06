//
// Created by Eric Berdahl on 10/31/17.
//

#include "copybuffertoimage_kernel.hpp"

namespace copybuffertoimage_kernel {

    void invoke(const clspv_utils::kernel_module&   module,
                const clspv_utils::kernel&          kernel,
                const sample_info&                  info,
                vk::ArrayProxy<const vk::Sampler>   samplers,
                vk::Buffer                          src_buffer,
                vk::ImageView                       dst_image,
                int                                 src_offset,
                int                                 src_pitch,
                cl_channel_order                    src_channel_order,
                cl_channel_type                     src_channel_type,
                bool                                swap_components,
                bool                                premultiply,
                int                                 width,
                int                                 height)
    {
        struct scalar_args {
            int inSrcOffset;        // offset 0
            int inSrcPitch;         // offset 4
            int inSrcChannelOrder;  // offset 8 -- cl_channel_order
            int inSrcChannelType;   // offset 12 -- cl_channel_type
            int inSwapComponents;   // offset 16 -- bool
            int inPremultiply;      // offset 20 -- bool
            int inWidth;            // offset 24
            int inHeight;           // offset 28
        };
        static_assert(0 == offsetof(scalar_args, inSrcOffset), "inSrcOffset offset incorrect");
        static_assert(4 == offsetof(scalar_args, inSrcPitch), "inSrcPitch offset incorrect");
        static_assert(8 == offsetof(scalar_args, inSrcChannelOrder),
                      "inSrcChannelOrder offset incorrect");
        static_assert(12 == offsetof(scalar_args, inSrcChannelType),
                      "inSrcChannelType offset incorrect");
        static_assert(16 == offsetof(scalar_args, inSwapComponents),
                      "inSwapComponents offset incorrect");
        static_assert(20 == offsetof(scalar_args, inPremultiply), "inPremultiply offset incorrect");
        static_assert(24 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");
        static_assert(28 == offsetof(scalar_args, inHeight), "inHeight offset incorrect");

        const scalar_args scalars = {
                src_offset,
                src_pitch,
                src_channel_order,
                src_channel_type,
                (swap_components ? 1 : 0),
                (premultiply ? 1 : 0),
                width,
                height
        };

        const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
        const clspv_utils::WorkgroupDimensions num_workgroups(
                (width + workgroup_sizes.x - 1) / workgroup_sizes.x,
                (height + workgroup_sizes.y - 1) / workgroup_sizes.y);

        clspv_utils::kernel_invocation invocation(*info.device, *info.cmd_pool,
                                                  info.memory_properties);

        invocation.addLiteralSamplers(samplers);
        invocation.addBufferArgument(src_buffer);
        invocation.addWriteOnlyImageArgument(dst_image);
        invocation.addPodArgument(scalars);

        invocation.run(info.graphics_queue, kernel, num_workgroups);
    }

    test_utils::Results test_matrix(const clspv_utils::kernel_module&   module,
                                    const clspv_utils::kernel&          kernel,
                                    const sample_info&                  info,
                                    vk::ArrayProxy<const vk::Sampler>   samplers,
                                    const test_utils::options&          opts)
    {
        const test_utils::test_kernel_fn tests[] = {
                test_series<gpu_types::float4>,
                test_series<gpu_types::half4>,
                test_series<gpu_types::uchar4>,
                test_series<gpu_types::float2>,
                test_series<gpu_types::half2>,
                test_series<gpu_types::uchar2>,
                test_series<float>,
                test_series<gpu_types::half>,
                test_series<gpu_types::uchar>,
        };

        return test_utils::test_kernel_invocation(module, kernel,
                                                  std::begin(tests), std::end(tests),
                                                  info,
                                                  samplers,
                                                  opts);
    }
}

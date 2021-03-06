//
// Created by Eric Berdahl on 10/31/17.
//

#include "readconstantdata_kernel.hpp"

namespace readconstantdata_kernel {

    void invoke(const clspv_utils::kernel_module&   module,
                const clspv_utils::kernel&          kernel,
                const sample_info&                  info,
                vk::ArrayProxy<const vk::Sampler>   samplers,
                vk::Buffer                          dst_buffer,
                int                                 width)
    {
        struct scalar_args {
            int inWidth;            // offset 0
        };
        static_assert(0 == offsetof(scalar_args, inWidth), "inWidth offset incorrect");

        const scalar_args scalars = {
                width
        };

        const clspv_utils::WorkgroupDimensions workgroup_sizes = kernel.getWorkgroupSize();
        const clspv_utils::WorkgroupDimensions num_workgroups(
                (width + workgroup_sizes.x - 1) / workgroup_sizes.x);

        clspv_utils::kernel_invocation invocation(*info.device, *info.cmd_pool,
                                                  info.memory_properties);

        invocation.addLiteralSamplers(samplers);
        invocation.addBufferArgument(dst_buffer);
        invocation.addPodArgument(scalars);

        invocation.run(info.graphics_queue, kernel, num_workgroups);
    }

    test_utils::Results test_all(const clspv_utils::kernel_module&    module,
                                 const clspv_utils::kernel&           kernel,
                                 const sample_info&                   info,
                                 vk::ArrayProxy<const vk::Sampler>    samplers,
                                 const test_utils::options&           opts)
    {
        const std::string testLabel = "readconstantdata.spv/ReadConstantData";

        const int buffer_height = 1;
        const int buffer_width = 64;

        const std::size_t buffer_length = buffer_width * buffer_height;
        const std::size_t buffer_size = buffer_length * sizeof(float);

        // number of elements in the constant data array (in the kernel itself)
        const std::size_t constant_data_length = 10;

        // allocate buffers and images
        vulkan_utils::buffer  dstBuffer(info, buffer_size);

        // initialize destination memory with random data
        test_utils::fill_random_pixels<float>(dstBuffer.mem, buffer_length);

        // set up expected results of the destination buffer
        int index = 0;
        std::vector<float> expectedResults(buffer_length);
        std::generate(expectedResults.begin(), expectedResults.end(), [&index, buffer_length, constant_data_length]() {
            float result = std::pow(2.0f, index);
            if (index >= std::min(buffer_length, constant_data_length)) {
                result = -1.0f;
            }

            ++index;

            return result;
        });

        invoke(module, kernel,
               info,
               samplers,
               *dstBuffer.buf,
               buffer_width);

        const bool success = test_utils::check_results<float, float>(expectedResults.data(), dstBuffer.mem,
                                                                     buffer_width, buffer_height,
                                                                     buffer_height,
                                                                     testLabel.c_str(),
                                                                     opts);

        return (success ? test_utils::Results::sTestSuccess : test_utils::Results::sTestFailure);
    }
}

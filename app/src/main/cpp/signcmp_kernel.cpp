//
// dneto@ derived from work ...
// Created by Eric Berdahl on 10/31/17.
//

#include "signcmp_kernel.hpp"

namespace signcmp_kernel {

clspv_utils::execution_time_t invoke(clspv_utils::kernel& kernel,
                                     vulkan_utils::storage_buffer& dst_buffer,
                                     vk::Extent3D extent, int offset) {
  if (1 != extent.depth) {
    throw std::runtime_error("Depth must be 1");
  }

  struct scalar_args {
    int inWidth;   // offset 0
    int inHeight;  // offset 4
    int offset;    // offset 8
  };
  static_assert(0 == offsetof(scalar_args, inWidth),
                "inWidth offset incorrect");
  static_assert(4 == offsetof(scalar_args, inHeight),
                "inHeight offset incorrect");
  static_assert(8 == offsetof(scalar_args, offset), "offset offset incorrect");

  vulkan_utils::uniform_buffer scalarBuffer(
      kernel.getDevice().mDevice, kernel.getDevice().mMemoryProperties,
      sizeof(scalar_args));
  auto scalars = scalarBuffer.map<scalar_args>();
  scalars->inWidth = extent.width;
  scalars->inHeight = extent.height;
  scalars->offset = offset;
  scalars.reset();

  const vk::Extent3D workgroup_sizes = kernel.getWorkgroupSize();
  const vk::Extent3D num_workgroups(
      (extent.width + workgroup_sizes.width - 1) / workgroup_sizes.width,
      (extent.height + workgroup_sizes.height - 1) / workgroup_sizes.height, 1);

  clspv_utils::kernel_invocation invocation = kernel.createInvocation();

  invocation.addStorageBufferArgument(dst_buffer);
  invocation.addUniformBufferArgument(scalarBuffer);

  return invocation.run(num_workgroups);
}

test_utils::InvocationResult test(clspv_utils::kernel& kernel,
                                  const std::vector<std::string>& args,
                                  bool verbose) {
  test_utils::InvocationResult invocationResult;
  auto& device = kernel.getDevice();

  // We'll test a 16x16 grid, so we need 256 entries
  const vk::Extent3D bufferExtent(256, 1, 1);
  const std::size_t buffer_length =
      bufferExtent.width * bufferExtent.height * bufferExtent.depth;
  const std::size_t buffer_size = buffer_length * sizeof(float);

  // allocate buffers and images
  vulkan_utils::storage_buffer dstBuffer(device.mDevice,
                                         device.mMemoryProperties, buffer_size);

  // initialize destination memory with unexpected value. the kernel should
  // write either 0 or
  // 1. so, initialize thedestination with 2.
  auto dstBufferMap = dstBuffer.map<float>();
  std::fill(dstBufferMap.get(), dstBufferMap.get() + buffer_length, 2.0f);
  dstBufferMap.reset();

  // set up expected results of the destination buffer
  int index = 0;
  const int offset = -8;
  std::vector<float> expectedResults(buffer_length);
  std::generate(expectedResults.begin(), expectedResults.end(),
                [&index, bufferExtent, offset]() {
                  const int inWidth = bufferExtent.width;
                  int x = index % inWidth;
                  int y = index / inWidth;

                  int x_cmp = x + offset;
                  int y_cmp = y + offset;
                  int ind = (y * bufferExtent.width) + x;
                  float result = 0.0f;
                  if (x < inWidth && y < inWidth) {
                    result = (x_cmp > y_cmp) ? 1.0 : -1.0f;
                  }

                  ++index;

                  return (x < bufferExtent.width && y < bufferExtent.height
                              ? result
                              : 0.0f);
                });

  invocationResult.mExecutionTime =
      invoke(kernel, dstBuffer, bufferExtent, offset);

  dstBufferMap = dstBuffer.map<float>();
  test_utils::check_results(expectedResults.data(), dstBufferMap.get(),
                            bufferExtent, bufferExtent.width, verbose,
                            invocationResult);

  return invocationResult;
}

test_utils::KernelTest::invocation_tests getAllTestVariants() {
  test_utils::InvocationTest t({"", test});
  return test_utils::KernelTest::invocation_tests({t});
}

}  // namespace signcmp_kernel

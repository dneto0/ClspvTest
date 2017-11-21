//
// Created by Eric Berdahl on 10/22/17.
//

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>

#include "clspv_utils.hpp"
#include "util.hpp"

namespace clspv_utils {

    namespace {

        details::spv_map create_spv_map(const char *spvmapFilename) {
            // Read the spvmap file into a string buffer
            std::unique_ptr<std::FILE, decltype(&std::fclose)> spvmap_file(AndroidFopen(spvmapFilename, "rb"),
                                                                           &std::fclose);
            assert(spvmap_file);

            std::fseek(spvmap_file.get(), 0, SEEK_END);
            std::string buffer(std::ftell(spvmap_file.get()), ' ');
            std::fseek(spvmap_file.get(), 0, SEEK_SET);
            std::fread(&buffer.front(), 1, buffer.length(), spvmap_file.get());

            spvmap_file.reset();

            // parse the spvmap file contents
            std::istringstream in(buffer);
            return details::spv_map::parse(in);
        }

        std::string read_csv_field(std::istream& in) {
            std::string result;

            if (in.good()) {
                const bool is_quoted = (in.peek() == '"');

                if (is_quoted) {
                    in.ignore(std::numeric_limits<std::streamsize>::max(), '"');
                }

                std::getline(in, result, is_quoted ? '"' : ',');

                if (is_quoted) {
                    in.ignore(std::numeric_limits<std::streamsize>::max(), ',');
                }
            }

            return result;
        }

        vk::UniqueShaderModule create_shader(const vk::Device& device, const std::string& spvFilename) {
            std::unique_ptr<std::FILE, decltype(&std::fclose)> spv_file(AndroidFopen(spvFilename.c_str(), "rb"),
                                                                        &std::fclose);
            if (!spv_file) {
                throw std::runtime_error("can't open file: " + spvFilename);
            }

            std::fseek(spv_file.get(), 0, SEEK_END);
            // Use vector of uint32_t to ensure alignment is satisfied.
            const auto num_bytes = std::ftell(spv_file.get());
            if (0 != (num_bytes % sizeof(uint32_t))) {
                throw std::runtime_error("file size of " + spvFilename + " inappropriate for spv file");
            }
            const auto num_words = (num_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
            std::vector<uint32_t> spvModule(num_words);
            assert(num_bytes == (spvModule.size() * sizeof(uint32_t)));

            std::fseek(spv_file.get(), 0, SEEK_SET);
            std::fread(spvModule.data(), 1, num_bytes, spv_file.get());

            spv_file.reset();

            vk::ShaderModuleCreateInfo shaderModuleCreateInfo;
            shaderModuleCreateInfo.setCodeSize(num_bytes)
                    .setPCode(spvModule.data());

            return device.createShaderModuleUnique(shaderModuleCreateInfo);
        }

        VkCommandBuffer allocate_command_buffer(VkDevice device, VkCommandPool cmd_pool) {
            VkCommandBufferAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = cmd_pool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;

            VkCommandBuffer result = VK_NULL_HANDLE;
            vulkan_utils::throwIfNotSuccess(vkAllocateCommandBuffers(device, &allocInfo, &result),
                                            "vkAllocateCommandBuffers");

            return result;
        }

        std::vector<VkDescriptorSet> allocate_descriptor_sets(
                VkDevice                                    device,
                VkDescriptorPool                            pool,
                const std::vector<VkDescriptorSetLayout>&   layouts) {
            std::vector<VkDescriptorSet> result;

            VkDescriptorSetAllocateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            createInfo.descriptorPool = pool;
            createInfo.descriptorSetCount = layouts.size();
            createInfo.pSetLayouts = layouts.data();

            result.resize(createInfo.descriptorSetCount, VK_NULL_HANDLE);
            vulkan_utils::throwIfNotSuccess(vkAllocateDescriptorSets(device,
                                                                     &createInfo,
                                                                     result.data()),
                                            "vkAllocateDescriptorSets");

            return result;
        }

        VkDescriptorSetLayout create_descriptor_set_layout(
                VkDevice                                device,
                const std::vector<VkDescriptorType>&    descriptorTypes) {
            std::vector<VkDescriptorSetLayoutBinding> bindingSet;

            VkDescriptorSetLayoutBinding binding = {};
            binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            binding.descriptorCount = 1;
            binding.binding = 0;

            for (auto type : descriptorTypes) {
                binding.descriptorType = type;
                bindingSet.push_back(binding);

                ++binding.binding;
            }

            VkDescriptorSetLayoutCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            createInfo.bindingCount = bindingSet.size();
            createInfo.pBindings = createInfo.bindingCount ? bindingSet.data() : NULL;

            VkDescriptorSetLayout result = VK_NULL_HANDLE;
            vulkan_utils::throwIfNotSuccess(vkCreateDescriptorSetLayout(device,
                                                                        &createInfo,
                                                                        NULL,
                                                                        &result),
                                            "vkCreateDescriptorSetLayout");

            return result;
        }

        details::pipeline_layout create_pipeline_layout(VkDevice                device,
                                                        const details::spv_map& spvMap,
                                                        const std::string&      entryPoint) {
            assert(!entryPoint.empty());

            details::pipeline_layout result;
            result.device = device;

            std::vector<VkDescriptorType> descriptorTypes;

            if (!spvMap.samplers.empty()) {
                assert(0 == spvMap.samplers_desc_set);

                descriptorTypes.clear();
                descriptorTypes.resize(spvMap.samplers.size(), VK_DESCRIPTOR_TYPE_SAMPLER);
                result.descriptors.push_back(create_descriptor_set_layout(device, descriptorTypes));
            }

            const auto kernel = spvMap.findKernel(entryPoint);
            if (kernel) {
                assert(kernel->descriptor_set == (spvMap.samplers.empty() ? 0 : 1));

                descriptorTypes.clear();

                // If the caller has asked only for a pipeline layout for a single entry point,
                // create empty descriptor layouts for all argument descriptors other than the
                // one used by the requested entry point.
                for (auto &ka : kernel->args) {
                    // ignore any argument not in offset 0
                    if (0 != ka.offset) continue;

                    VkDescriptorType argType;

                    switch (ka.kind) {
                        case details::spv_map::arg::kind_pod:
                        case details::spv_map::arg::kind_buffer:
                            argType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                            break;

                        case details::spv_map::arg::kind_ro_image:
                            argType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                            break;

                        case details::spv_map::arg::kind_wo_image:
                            argType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                            break;

                        case details::spv_map::arg::kind_sampler:
                            argType = VK_DESCRIPTOR_TYPE_SAMPLER;
                            break;

                        default:
                            assert(0 && "unkown argument type");
                    }

                    descriptorTypes.push_back(argType);
                }

                result.descriptors.push_back(create_descriptor_set_layout(device, descriptorTypes));
            };

            VkPipelineLayoutCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            createInfo.setLayoutCount = result.descriptors.size();
            createInfo.pSetLayouts = createInfo.setLayoutCount ? result.descriptors.data() : NULL;

            vulkan_utils::throwIfNotSuccess(vkCreatePipelineLayout(device,
                                                                   &createInfo,
                                                                   NULL,
                                                                   &result.pipeline),
                                            "vkCreatePipelineLayout");

            return result;
        }

    } // anonymous namespace

    namespace details {

        spv_map::arg::kind_t spv_map::parse_argType(const std::string &argType) {
            arg::kind_t result = arg::kind_unknown;

            if (argType == "pod") {
                result = arg::kind_pod;
            } else if (argType == "buffer") {
                result = arg::kind_buffer;
            } else if (argType == "ro_image") {
                result = arg::kind_ro_image;
            } else if (argType == "wo_image") {
                result = arg::kind_wo_image;
            } else if (argType == "sampler") {
                result = arg::kind_sampler;
            } else {
                assert(0 && "unknown spvmap arg type");
            }

            return result;
        }

        spv_map spv_map::parse(std::istream &in) {
            spv_map result;

            while (!in.eof()) {
                // read one line
                std::string line;
                std::getline(in, line);

                std::istringstream in_line(line);
                std::string key = read_csv_field(in_line);
                std::string value = read_csv_field(in_line);
                if ("sampler" == key) {
                    auto s = result.samplers.insert(result.samplers.end(), spv_map::sampler());
                    assert(s != result.samplers.end());

                    s->opencl_flags = std::atoi(value.c_str());

                    while (!in_line.eof()) {
                        key = read_csv_field(in_line);
                        value = read_csv_field(in_line);

                        if ("descriptorSet" == key) {
                            // all samplers, if any, are documented to share descriptor set 0
                            const int ds = std::atoi(value.c_str());
                            assert(ds == 0);

                            if (-1 == result.samplers_desc_set) {
                                result.samplers_desc_set = ds;
                            }
                        } else if ("binding" == key) {
                            s->binding = std::atoi(value.c_str());
                        }
                    }
                } else if ("kernel" == key) {
                    auto kernel = result.findKernel(value);
                    if (!kernel) {
                        result.kernels.push_back(spv_map::kernel());
                        kernel = &result.kernels.back();
                        kernel->name = value;
                    }
                    assert(kernel);

                    auto ka = kernel->args.end();

                    while (!in_line.eof()) {
                        key = read_csv_field(in_line);
                        value = read_csv_field(in_line);

                        if ("argOrdinal" == key) {
                            assert(ka == kernel->args.end());

                            const int arg_index = std::atoi(value.c_str());

                            if (kernel->args.size() <= arg_index) {
                                kernel->args.resize(arg_index + 1, spv_map::arg());
                            }
                            ka = std::next(kernel->args.begin(), arg_index);

                            assert(ka != kernel->args.end());
                        } else if ("descriptorSet" == key) {
                            const int ds = std::atoi(value.c_str());
                            if (-1 == kernel->descriptor_set) {
                                kernel->descriptor_set = ds;
                            }

                            // all args for a kernel are documented to share the same descriptor set
                            assert(ds == kernel->descriptor_set);
                        } else if ("binding" == key) {
                            ka->binding = std::atoi(value.c_str());
                        } else if ("offset" == key) {
                            ka->offset = std::atoi(value.c_str());
                        } else if ("argKind" == key) {
                            ka->kind = parse_argType(value);
                        }
                    }
                }
            }

            return result;
        }

        spv_map::kernel* spv_map::findKernel(const std::string& name) {
            return const_cast<kernel*>(const_cast<const spv_map*>(this)->findKernel(name));
        }

        const spv_map::kernel* spv_map::findKernel(const std::string& name) const {
            auto kernel = std::find_if(kernels.begin(), kernels.end(),
                                       [&name](const spv_map::kernel &iter) {
                                           return iter.name == name;
                                       });

            return (kernel == kernels.end() ? nullptr : &(*kernel));
        }

        pipeline_layout::pipeline_layout(pipeline_layout&& other) :
                pipeline_layout()
        {
            swap(other);
        }

        pipeline_layout::~pipeline_layout() {
            reset();
        }

        pipeline_layout& pipeline_layout::operator=(pipeline_layout&& other)
        {
            swap(other);
            return *this;
        }


        void pipeline_layout::reset() {
            std::for_each(descriptors.begin(), descriptors.end(), [this] (VkDescriptorSetLayout dsl) {
                vkDestroyDescriptorSetLayout(this->device, dsl, nullptr);
            });
            descriptors.clear();

            if (VK_NULL_HANDLE != device && VK_NULL_HANDLE != pipeline) {
                vkDestroyPipelineLayout(device, pipeline, NULL);
            }

            device = VK_NULL_HANDLE;
            pipeline = VK_NULL_HANDLE;
        }

        void pipeline_layout::swap(pipeline_layout& other) {
            using std::swap;

            swap(device, other.device);
            swap(descriptors, other.descriptors);
            swap(pipeline, other.pipeline);
        }

        void pipeline::reset() {
            if (mPipeline) {
                assert(VK_NULL_HANDLE != mPipelineLayout.device);

                vkDestroyPipeline(mPipelineLayout.device, mPipeline, NULL);
                mPipeline = VK_NULL_HANDLE;
            }

            // NULL out cached descriptors
            mArgumentsDescriptor = VK_NULL_HANDLE;
            mLiteralSamplerDescriptor = VK_NULL_HANDLE;

            if (!mDescriptors.empty()) {
                assert(VK_NULL_HANDLE != mPipelineLayout.device);

                VkResult U_ASSERT_ONLY res = vkFreeDescriptorSets(mPipelineLayout.device,
                                                                  mDescriptorPool,
                                                                  mDescriptors.size(),
                                                                  mDescriptors.data());
                assert(res == VK_SUCCESS);

                mDescriptors.clear();
            }

            mDescriptorPool = VK_NULL_HANDLE;
            mPipelineLayout.reset();
        }

    } // namespace details

    kernel_module::kernel_module(VkDevice           device,
                                 VkDescriptorPool   pool,
                                 const std::string& moduleName) :
            mName(moduleName),
            mDevice(device),
            mDescriptorPool(pool),
            mShaderModule(),
            mSpvMap() {
        const std::string spvFilename = moduleName + ".spv";
        mShaderModule = create_shader(mDevice, spvFilename.c_str());

        const std::string mapFilename = moduleName + ".spvmap";
        mSpvMap = create_spv_map(mapFilename.c_str());
    }

    kernel_module::~kernel_module() {
    }

    std::vector<std::string> kernel_module::getEntryPoints() const {
        std::vector<std::string> result;

        std::transform(mSpvMap.kernels.begin(), mSpvMap.kernels.end(),
                       std::back_inserter(result),
                       [](const details::spv_map::kernel& k) { return k.name; });

        return result;
    }

    details::pipeline kernel_module::createPipeline(const std::string&         entryPoint,
                                                    const WorkgroupDimensions& work_group_sizes) const {
        details::pipeline result;
        try {
            result.mPipelineLayout = create_pipeline_layout((VkDevice) mDevice, mSpvMap, entryPoint);
            result.mDescriptorPool = (VkDescriptorPool) mDescriptorPool;
            result.mDescriptors = allocate_descriptor_sets((VkDevice) mDevice, (VkDescriptorPool) mDescriptorPool,
                                                           result.mPipelineLayout.descriptors);

            if (-1 != mSpvMap.samplers_desc_set) {
                result.mLiteralSamplerDescriptor = result.mDescriptors[mSpvMap.samplers_desc_set];
            }

            const auto kernel_arg_map = mSpvMap.findKernel(entryPoint);
            if (kernel_arg_map && -1 != kernel_arg_map->descriptor_set) {
                result.mArgumentsDescriptor = result.mDescriptors[kernel_arg_map->descriptor_set];
            }

            const unsigned int num_workgroup_sizes = 3;
            const int32_t workGroupSizes[num_workgroup_sizes] = {
                    work_group_sizes.x,
                    work_group_sizes.y,
                    1
            };
            const vk::SpecializationMapEntry specializationEntries[num_workgroup_sizes] = {
                    {
                            0,                          // specialization constant 0 - workgroup size X
                            0 * sizeof(int32_t),          // offset - start of workGroupSizes array
                            sizeof(workGroupSizes[0])   // sizeof the first element
                    },
                    {
                            1,                          // specialization constant 1 - workgroup size Y
                            1 * sizeof(int32_t),            // offset - one element into the array
                            sizeof(workGroupSizes[1])   // sizeof the second element
                    },
                    {
                            2,                          // specialization constant 2 - workgroup size Z
                            2 * sizeof(int32_t),          // offset - two elements into the array
                            sizeof(workGroupSizes[2])   // sizeof the second element
                    }
            };
            vk::SpecializationInfo specializationInfo;
            specializationInfo.setMapEntryCount(num_workgroup_sizes)
                    .setPMapEntries(specializationEntries)
                    .setDataSize(sizeof(workGroupSizes))
                    .setPData(workGroupSizes);

            vk::ComputePipelineCreateInfo createInfo;
            createInfo.setLayout(vk::PipelineLayout(result.mPipelineLayout.pipeline))
                    .stage.setStage(vk::ShaderStageFlagBits::eCompute)
                    .setModule(*mShaderModule)
                    .setPName(entryPoint.c_str())
                    .setPSpecializationInfo(&specializationInfo);

            auto pipelines = mDevice.createComputePipelines(vk::PipelineCache(), createInfo);
            result.mPipeline = (VkPipeline) pipelines[0];
        }
        catch(...)
        {
            result.reset();
            throw;
        }

        return result;
    }

    kernel::kernel(VkDevice                     device,
                   const kernel_module&         module,
                   std::string                  entryPoint,
                   const WorkgroupDimensions&   workgroup_sizes) :
            mEntryPoint(entryPoint),
            mWorkgroupSizes(workgroup_sizes),
            mPipeline() {
        mPipeline = module.createPipeline(entryPoint, workgroup_sizes);
    }

    kernel::~kernel() {
        mPipeline.reset();
    }

    void kernel::bindCommand(VkCommandBuffer command) const {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, mPipeline.mPipeline);

        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                mPipeline.mPipelineLayout.pipeline,
                                0,
                                mPipeline.mDescriptors.size(), mPipeline.mDescriptors.data(),
                                0, NULL);
    }

    kernel_invocation::kernel_invocation(VkDevice           device,
                                         VkCommandPool      cmdPool,
                                         const VkPhysicalDeviceMemoryProperties&    memoryProperties) :
            mDevice(device),
            mCmdPool(cmdPool),
            mCommand(VK_NULL_HANDLE),
            mMemoryProperties(memoryProperties),
            mLiteralSamplers(),
            mArguments() {
        mCommand = allocate_command_buffer(mDevice, mCmdPool);
    }

    kernel_invocation::~kernel_invocation() {
        std::for_each(mPodBuffers.begin(), mPodBuffers.end(), std::mem_fun_ref(&vulkan_utils::buffer::reset));

        if (mDevice) {
            if (mCmdPool && mCommand) {
                vkFreeCommandBuffers(mDevice, mCmdPool, 1, &mCommand);
            }
        }
    }

    void kernel_invocation::addBufferArgument(VkBuffer buf) {
        arg item;

        item.type = vk::DescriptorType::eStorageBuffer;
        item.buffer = vk::Buffer(buf);

        mArguments.push_back(item);
    }

    void kernel_invocation::addSamplerArgument(VkSampler samp) {
        arg item;

        item.type = vk::DescriptorType::eSampler;
        item.sampler = vk::Sampler(samp);

        mArguments.push_back(item);
    }

    void kernel_invocation::addReadOnlyImageArgument(VkImageView im) {
        arg item;

        item.type = vk::DescriptorType::eSampledImage;
        item.image = vk::ImageView(im);

        mArguments.push_back(item);
    }

    void kernel_invocation::addWriteOnlyImageArgument(VkImageView im) {
        arg item;

        item.type = vk::DescriptorType::eStorageImage;
        item.image = vk::ImageView(im);

        mArguments.push_back(item);
    }

    void kernel_invocation::updateDescriptorSets(VkDescriptorSet literalSamplerDescSet,
                                                 VkDescriptorSet argumentDescSet) {
        std::vector<VkDescriptorImageInfo>  imageList;
        std::vector<VkDescriptorBufferInfo> bufferList;

        //
        // Collect information about the literal samplers
        //
        for (auto s : mLiteralSamplers) {
            VkDescriptorImageInfo samplerInfo = {};
            samplerInfo.sampler = s;

            imageList.push_back(samplerInfo);
        }

        //
        // Collect information about the arguments
        //
        for (auto& a : mArguments) {
            switch (a.type) {
                case vk::DescriptorType::eStorageImage:
                case vk::DescriptorType::eSampledImage: {
                    VkDescriptorImageInfo imageInfo = {};
                    imageInfo.imageView = (VkImageView) a.image;
                    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                    imageList.push_back(imageInfo);
                    break;
                }

                case vk::DescriptorType::eStorageBuffer: {
                    VkDescriptorBufferInfo bufferInfo = {};
                    bufferInfo.range = VK_WHOLE_SIZE;
                    bufferInfo.buffer = (VkBuffer) a.buffer;

                    bufferList.push_back(bufferInfo);
                    break;
                }

                case vk::DescriptorType::eSampler: {
                    VkDescriptorImageInfo samplerInfo = {};
                    samplerInfo.sampler = (VkSampler) a.sampler;

                    imageList.push_back(samplerInfo);
                    break;
                }

                default:
                    assert(0 && "unkown argument type");
            }
        }

        //
        // Set up to create the descriptor set write structures
        // We will iterate the param lists in the same order,
        // picking up image and buffer infos in order.
        //

        std::vector<VkWriteDescriptorSet> writeSets;
        auto nextImage = imageList.begin();
        auto nextBuffer = bufferList.begin();


        //
        // Update the literal samplers' descriptor set
        //

        VkWriteDescriptorSet literalSamplerSet = {};
        literalSamplerSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        literalSamplerSet.dstSet = literalSamplerDescSet;
        literalSamplerSet.dstBinding = 0;
        literalSamplerSet.descriptorCount = 1;
        literalSamplerSet.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;

        assert(mLiteralSamplers.empty() || literalSamplerDescSet);

        for (auto s : mLiteralSamplers) {
            literalSamplerSet.pImageInfo = &(*nextImage);
            ++nextImage;

            writeSets.push_back(literalSamplerSet);

            ++literalSamplerSet.dstBinding;
        }

        //
        // Update the kernel's argument descriptor set
        //

        VkWriteDescriptorSet argSet = {};
        argSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        argSet.dstSet = argumentDescSet;
        argSet.dstBinding = 0;
        argSet.descriptorCount = 1;

        assert(mArguments.empty() || argumentDescSet);

        for (auto& a : mArguments) {
            switch (a.type) {
                case vk::DescriptorType::eStorageImage:
                case vk::DescriptorType::eSampledImage:
                    argSet.descriptorType = (VkDescriptorType) a.type;
                    argSet.pImageInfo = &(*nextImage);
                    ++nextImage;
                    break;

                case vk::DescriptorType::eStorageBuffer:
                    argSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    argSet.pBufferInfo = &(*nextBuffer);
                    ++nextBuffer;
                    break;

                case vk::DescriptorType::eSampler:
                    argSet.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                    argSet.pImageInfo = &(*nextImage);
                    ++nextImage;
                    break;

                default:
                    assert(0 && "unkown argument type");
            }

            writeSets.push_back(argSet);

            ++argSet.dstBinding;
        }

        //
        // Do the actual descriptor set updates
        //
        vkUpdateDescriptorSets(mDevice, writeSets.size(), writeSets.data(), 0, nullptr);
    }

    void kernel_invocation::fillCommandBuffer(const kernel&                 inKernel,
                                              const WorkgroupDimensions&    num_workgroups) {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vulkan_utils::throwIfNotSuccess(vkBeginCommandBuffer(mCommand, &beginInfo),
                                        "vkBeginCommandBuffer");

        inKernel.bindCommand(mCommand);

        vkCmdDispatch(mCommand, num_workgroups.x, num_workgroups.y, 1);

        vulkan_utils::throwIfNotSuccess(vkEndCommandBuffer(mCommand),
                                        "vkEndCommandBuffer");
    }

    void kernel_invocation::submitCommand(VkQueue queue) {
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &mCommand;

        vulkan_utils::throwIfNotSuccess(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE),
                                        "vkQueueSubmit");

    }

    void kernel_invocation::run(VkQueue                     queue,
                                const kernel&               kern,
                                const WorkgroupDimensions&  num_workgroups) {
        updateDescriptorSets(kern.getLiteralSamplerDescSet(), kern.getArgumentDescSet());
        fillCommandBuffer(kern, num_workgroups);
        submitCommand(queue);

        vkQueueWaitIdle(queue);
    }

} // namespace clspv_utils

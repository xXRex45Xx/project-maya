#pragma once
// Included only by STRATA_ENABLE_HIP builds. CUDA builds use NVIDIA headers.
#ifdef STRATA_HIP_WAVE64
// On wave64 cards (gfx908, gfx906) HIP's own warp-sync builtins would run across the whole 64-lane wavefront and break
// every kernel's reduction order; disable them and let intrinsics.hpp supply the logical-32-lane-half mapping.
// Must come before hip_runtime.h. Wave32 builds leave HIP's builtins alone: they compile exactly as before.
#ifndef HIP_DISABLE_WARP_SYNC_BUILTINS
#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#endif
#endif  // STRATA_HIP_WAVE64
#include <hip/hip_runtime.h>
#include "host_lock.hpp"
#ifdef STRATA_HIP_WAVE64
#include <atomic>
#endif
// CUDA's annotation avoids per-thread copies of const kernel parameters. HIP
// passes the same const values without this CUDA-only optimization annotation.
#ifndef __grid_constant__
#define __grid_constant__
#endif
// Do not let HIP's legacy macro corrupt libstdc++ attribute names.
#ifdef __noinline__
#undef __noinline__
#endif
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrClockRate hipDeviceAttributeClockRate
#define cudaDevAttrMemoryClockRate hipDeviceAttributeMemoryClockRate
#define cudaDevAttrGlobalMemoryBusWidth hipDeviceAttributeMemoryBusWidth
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrCanUseHostPointerForRegisteredMem hipDeviceAttributeCanUseHostPointerForRegisteredMem
#define cudaDevAttrIntegrated hipDeviceAttributeIntegrated
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpyPeer hipMemcpyPeer
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorUnknown hipErrorUnknown
#define cudaErrorInvalidConfiguration hipErrorInvalidConfiguration
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorStreamCaptureUnsupported hipErrorStreamCaptureUnsupported
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaFreeHost(p) (::strata::hip_compat::host_free(p))
#ifdef STRATA_HIP_WAVE64
// Mapped host memory is how the GPU and the CPU talk inside a running graph (the doorbells, the staging buffers):
// a kernel spins on a word the CPU writes and the CPU on a word the kernel writes. On AMD that needs
// FINE-GRAINED (coherent) memory — coarse-grained host memory may sit in the GPU's L2 for the whole kernel and
// the spin never sees the other side. Wave64 builds make a mapped allocation coherent when a wave64 GPU may read it;
// wave32 GPUs keep the plain hipHostMalloc of old (the Strix Halo / RDNA and APU paths depend on that behaviour).
// A wave64 arch beside wave32 ones (gfx906;gfx1200) is one binary, so this is decided per device at run time: a
// portable allocation (any GPU may read it) is coherent when any visible GPU is wave64, any other one when the current
// GPU is.
inline bool strata_hip_device_wave64(int dev) {
    static std::atomic<int> cache[64]{};   // 0 not probed yet, 1 wave32, 2 wave64
    if (dev < 0 || dev >= 64) return true;
    int c = cache[dev].load(std::memory_order_relaxed);
    if (!c) {
        int ws = 64;
        if (hipDeviceGetAttribute(&ws, hipDeviceAttributeWarpSize, dev) != hipSuccess) ws = 64;
        c = ws == 64 ? 2 : 1;
        cache[dev].store(c, std::memory_order_relaxed);
    }
    return c == 2;
}
inline bool strata_hip_any_wave64() {
    static const bool any = [] {
        int n = 0;
        if (hipGetDeviceCount(&n) != hipSuccess) return true;
        for (int d = 0; d < n; ++d)
            if (strata_hip_device_wave64(d)) return true;
        return false;
    }();
    return any;
}
inline hipError_t cudaHostAlloc(void** p, size_t bytes, unsigned int flags) {
    if (flags & hipHostMallocMapped) {
        int dev = 0;
        const bool wave64 = (flags & hipHostMallocPortable) ? strata_hip_any_wave64()
                            : hipGetDevice(&dev) != hipSuccess || strata_hip_device_wave64(dev);
        if (wave64) flags |= hipHostMallocCoherent;
    }
    return ::strata::hip_compat::host_malloc(p, bytes, flags);
}
template <typename T>
inline hipError_t cudaHostAlloc(T** p, size_t bytes, unsigned int flags) {
    return cudaHostAlloc(reinterpret_cast<void**>(p), bytes, flags);
}
#else
inline hipError_t cudaHostAlloc(void** p, size_t bytes, unsigned int flags) {
    return ::strata::hip_compat::host_malloc(p, bytes, flags);
}
template <typename T>
inline hipError_t cudaHostAlloc(T** p, size_t bytes, unsigned int flags) {
    return cudaHostAlloc(reinterpret_cast<void**>(p), bytes, flags);
}
#endif  // STRATA_HIP_WAVE64
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload
#define cudaGraph_t hipGraph_t
#define cudaHostAllocDefault hipHostAllocDefault
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaHostAllocPortable hipHostAllocPortable
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegister(p, bytes, flags) (::strata::hip_compat::host_register((p), (bytes), (flags)))
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostUnregister(p) (::strata::hip_compat::host_unregister(p))
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaMalloc hipMalloc
#define cudaMallocHost(...) (::strata::hip_compat::malloc_host(__VA_ARGS__))
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemcpy hipMemcpy
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaSetDevice hipSetDevice
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamLegacy hipStreamLegacy
#define cudaStreamQuery hipStreamQuery
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

namespace strata::hip_compat {
template <typename T>
inline hipError_t malloc_host(T** pointer, size_t bytes) {
    return host_malloc(reinterpret_cast<void**>(pointer), bytes, hipHostMallocDefault);
}
}  // namespace strata::hip_compat

template <typename Kernel>
inline hipError_t cudaFuncSetAttribute(Kernel kernel, hipFuncAttribute attribute, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attribute, value);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph, unsigned long long flags) {
    return hipGraphInstantiateWithFlags(exec, graph, flags);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                     hipGraphNode_t* error, char* log, size_t size) {
    return hipGraphInstantiate(exec, graph, error, log, size);
}
#define __trap() __builtin_trap()   // the compiled-out sm_80 paths (never selected on AMD)
#define cudaMemcpyToSymbol(symbol, ...) hipMemcpyToSymbol(HIP_SYMBOL(symbol), __VA_ARGS__)

#include "intrinsics.hpp"

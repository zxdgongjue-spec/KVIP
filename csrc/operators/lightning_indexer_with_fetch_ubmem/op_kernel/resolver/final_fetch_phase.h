// Prefetch is already complete; all UBmem descriptors are safe in phase zero.
if(!phase) {
        if(!error && info[1]) {
            const uint32_t begin=b*kSlots,count=info[1];
            auto addresses=reinterpret_cast<__gm__ uint64_t*>(ctx->fallbackAddresses);
            auto destinations=reinterpret_cast<__gm__ int32_t*>(ctx->destinations);
            ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(addresses+begin),uint64_t(count)*8);
            ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(destinations+begin),uint64_t(count)*4);
            uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
            uint32_t stride=ctx->outputStride?uint32_t(ctx->outputStride):shape.resident;
            TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,21504);
            kvaf_device::UbmemBatchGetDeviceArgs args{addresses,destinations,
                region->peerBase,region->mappedBase,region->bytes,region->outputBase,region->outputBytes,
                begin,count,owner*stride,shape.resident,w,shape.finalWorkers,shape.tokenBytes};
            result=kvaf_device::KvafBatchGetDevice(args,scratch.Get<uint8_t>());
            pipe.Destroy();
        }
}

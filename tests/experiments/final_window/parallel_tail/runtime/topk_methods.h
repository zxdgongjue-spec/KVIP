// Included inside the private terminal QLIVector class only.
// The production hot/cold/window QLIVector is not changed.
__aicore__ inline uint32_t TerminalQueryRuntime(uint32_t query) {
    auto ctx=reinterpret_cast<__gm__ rotating::Context*>(rotationContext_);
    const uint32_t valid=ctx->queryEnds?
        reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)[query]:runtimeShape_.history;
    auto meta=IncrementalMeta(query);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);
    const bool missing=valid && !meta[0];
    if(!valid || missing) {
        // An inactive graph row must still participate in the request-local
        // publication barriers, but must not consume last replay's TopK UB.
        Duplicate(indicesOutLocal_,UINT32_MAX,2048);
        meta[3]=0;
        meta[6]=0; // No compact miss payload is valid for an inactive Q.
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(meta),128);
        PipeBarrier<PIPE_ALL>();
        return missing?1U:0U;
    }
    TerminalQuery(query);
    return 0;
}

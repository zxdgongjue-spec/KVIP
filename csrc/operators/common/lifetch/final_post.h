// Transport-independent final descriptor publication.
__aicore__ inline __gm__ FinalPost* FinalPostAt(uint32_t rt_finalPostOffset,__gm__ Context* ctx,uint32_t core) {
    return reinterpret_cast<__gm__ FinalPost*>(ctx->owners+rt_finalPostOffset)+core;
}
__aicore__ inline void PrepareFinalFetch(uint32_t rt_finalWorkers,uint32_t rt_finalPostOffset,
    __gm__ Context* ctx,uint32_t b,uint32_t w,uint32_t count,uint32_t descriptor,uint32_t error,uint32_t nonconflicting) {
    auto post=FinalPostAt(rt_finalPostOffset,ctx,b*rt_finalWorkers+w);
    post->count=count;post->pending=0;post->error=error;
    post->deferred=count && !error;post->descriptor=descriptor;post->head=0;
    post->nonconflicting=nonconflicting;
    LIFETCH_CLEAN_CACHE_LINES(reinterpret_cast<__gm__ uint8_t*>(post),sizeof(*post));
}

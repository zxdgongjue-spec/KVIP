#if defined(__CCE_AICORE__)
// SPDX-License-Identifier: Apache-2.0
// Experimental bounded final window. The original tail owns completion/Index updates.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void PrepareTerminal(__gm__ pipeline_ab::Args* a,uint32_t b) {
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);
 auto shape=rotating::LoadRuntimeShape(ctx);
 auto errors=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+b*128;
 auto row=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b;
 uint32_t total=0,reason=0;
 for(uint32_t q=0;q<4;++q){uint32_t n=asc_ldcg(errors+q*32+1);
  if(asc_ldcg(errors+q*32))reason|=1;
  if(!n || n>2049)reason|=2;
  if(n && n<=2049)total+=n-1;
  if(!threadIdx.x)asc_stcg(errors+8+q,n?n-1:UINT32_MAX);
 }
 uint32_t used=asc_ldcg(&row->slots);
 if(total>rotating::kCandidateCapacity)reason|=4;
 if(used>rotating::kLruSlots || total>rotating::kLruSlots-used)reason|=8;
 bool eligible=!reason && total>0;
 if(!threadIdx.x){
  asc_stcg(errors+2,uint32_t(eligible));asc_stcg(errors+3,total);
  asc_stcg(errors+4,0U);asc_stcg(errors+5,asc_ldcg(&row->requests));
  asc_stcg(errors+6,reason);asc_stcg(errors+7,used);
 }
 if(!eligible)return;
 auto loc=reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*8192;
 auto backup=reinterpret_cast<__gm__ uint32_t*>(ctx->sourceOffsets)+b*8192;
 for(uint32_t i=threadIdx.x;i<8192;i+=256)asc_stcg(backup+i,asc_ldcg(loc+i));
 asc_threadfence();asc_syncthreads();
 auto packed=reinterpret_cast<__gm__ uint32_t*>(ctx->sourceBlocks)+b*8192;
 auto metadata=reinterpret_cast<__gm__ uint32_t*>(rotating::TopKHeader(ctx))+b*128;
 for(uint32_t q=0;q<4;++q){uint32_t n=asc_ldcg(errors+q*32+1)-1;
  for(uint32_t i=threadIdx.x;i<n;i+=256)asc_stcg(loc+q*2048+i,asc_ldcg(packed+q*2048+i)>>11);
  if(!threadIdx.x)asc_stcg(metadata+q*32+6,n);
 }
 asc_threadfence();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void RestoreTerminal(__gm__ pipeline_ab::Args* a,uint32_t b) {
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);auto shape=rotating::LoadRuntimeShape(ctx);
 auto errors=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)+b*128;
 if(!asc_ldcg(errors+2))return;
 auto selected=reinterpret_cast<__gm__ uint32_t*>(ctx->selected)+b*8192;
 auto loc=reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*8192;
 auto backup=reinterpret_cast<__gm__ uint32_t*>(ctx->sourceOffsets)+b*8192;
 auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(b)*shape.history;
 auto lru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*2048;
 for(uint32_t i=threadIdx.x;i<8192;i+=256){uint32_t slot=asc_ldcg(backup+i),token=asc_ldcg(selected+i);
  if(slot==UINT32_MAX && token!=UINT32_MAX){
   if(token<shape.history)slot=rotating::ResolvePhysicalSlot(shape.resident,lru,asc_ldcg(map+token));
   if(slot==UINT32_MAX)asc_atomic_or(errors+(i/2048)*32,8U);
  }
  asc_stcg(loc+i,slot);
 }
 for(uint32_t q=threadIdx.x;q<4;q+=256)asc_stcg(errors+q*32+1,1U);
 if(!threadIdx.x){auto row=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b;
  asc_stcg(errors+4,asc_ldcg(&row->requests)-asc_ldcg(errors+5));}
 asc_threadfence();
}
__aicore__ inline void RunTerminalService(__gm__ pipeline_ab::Args* a,uint32_t lane) {
 auto ctx=reinterpret_cast<__gm__ rotating::Context*>(a->context);auto s=rotating::LoadRuntimeShape(ctx);
 for(uint32_t b=lane;b<s.batch;b+=s.serviceLanes){
  Simt::VF_CALL<PrepareTerminal>(Simt::Dim3{256,1,1},a,b);PipeBarrier<PIPE_ALL>();
 }
 TPipe pipe;TBuf<TPosition::VECCALC> scratch,flags,ids;
 pipe.InitBuffer(scratch,65536);pipe.InitBuffer(flags,32768);pipe.InitBuffer(ids,8192);
 rotating::FastService<false,true,true>(s.batch,s.history,s.resident,s.rounds,s.serviceLanes,s.tokenBytes,
  s.postThreads,s.postBank,s.batchedRows,s.scanTokens,reinterpret_cast<GM_ADDR>(ctx),s.rounds,lane/2,lane,
  scratch.Get<uint32_t>(),flags.Get<uint16_t>(),ids.Get<uint32_t>(),0,a->queryErrors);
 pipe.Destroy();
 for(uint32_t b=lane;b<s.batch;b+=s.serviceLanes){
  Simt::VF_CALL<RestoreTerminal>(Simt::Dim3{256,1,1},a,b);PipeBarrier<PIPE_ALL>();
 }
}

#endif

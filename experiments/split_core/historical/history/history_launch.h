#pragma once
extern "C" uint32_t aclrtlaunch_v9_li_history_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_history_topk_local(uint32_t,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,void*);
extern "C" uint32_t aclrtlaunch_split_topk_merge(uint32_t,void*,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,void*);
extern "C" uint32_t kvop_direct_update_outputs_early(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_fetch_update_kernel(uint32_t,void*,void*,void*);

inline void RunHistoryPool(Memory& m,const std::vector<uint64_t>& original,
    rotating::Context real,const std::vector<uint8_t>& query,const std::vector<uint16_t>& weight,
    const std::vector<float>& qscale,uint64_t realBinding,uint64_t update,aclrtStream stream,
    const rotating::RuntimeShape& shape,uint32_t shards,uint32_t ccap,uint32_t vcap,
    const std::string& out,const std::string& caseName) {
    uint32_t b=shape.batch,h=shape.history,q=b*4,vb=b*shards;
    uint32_t topkShards=std::getenv("KVCORE_TOPK_SHARDS")?std::stoul(std::getenv("KVCORE_TOPK_SHARDS")):shards;
    if(!topkShards)topkShards=shards;
    require(topkShards>=1 && topkShards<=32 && h/128/topkShards*128>2048,"selection range length");
    require((caseName=="standard"||caseName=="zero") && h%128==0,"history prototype case");
    require(shards>=1 && vb<=ccap && ccap<=32 && vcap>=1 && vcap<=64,"history resources");
    require(h/128/shards*128>2048,"history local length needs >2048");
    auto ends=m.get<uint32_t>(real.queryEnds,q);
    for(auto end:ends)require(end==h,"partial queryEnds require a separate clipped-local-K variant");
    uint32_t maxPages=(h/128+shards-1)/shards;
    uint32_t rowStride=(maxPages*128+255)/256*256,rowPages=rowStride/128;
    rotating::RuntimeShape vs{};vs.batch=vb;vs.history=rowStride;vs.resident=2048;
    vs.indexHeads=shape.indexHeads;vs.aicCount=ccap;vs.splitQ=0;vs.rounds=1;vs.tokenBytes=shape.tokenBytes;
    check(rotating::PrepareShape(&vs));vs.rounds=0; // Keep nonzero prepared serviceAics.
    require(2*vs.liCores<=vcap,"existing score producer still needs two AIV per active AIC");
    auto pages=m.get<int32_t>(original[7],uint64_t(b)*h/128);
    std::vector<int32_t> vpages(uint64_t(vb)*rowPages,0),qlens(vb,4);
    std::vector<uint32_t> klens(vb),vends(vb*4);
    std::vector<uint8_t> vq(uint64_t(vb)*4*shape.indexHeads*128);
    std::vector<uint16_t> vw(uint64_t(vb)*4*shape.indexHeads);
    std::vector<float> vqs(vw.size());
    for(uint32_t batch=0;batch<b;++batch)for(uint32_t s=0;s<shards;++s){
        uint32_t v=batch*shards+s,begin=h/128*s/shards,end=h/128*(s+1)/shards;
        klens[v]=(end-begin)*128;
        std::fill_n(vends.begin()+v*4,4,klens[v]);
        for(uint32_t p=0;p<end-begin;++p)vpages[uint64_t(v)*rowPages+p]=pages[uint64_t(batch)*h/128+begin+p];
        std::copy_n(query.begin()+uint64_t(batch)*4*shape.indexHeads*128,4*shape.indexHeads*128,vq.begin()+uint64_t(v)*4*shape.indexHeads*128);
        std::copy_n(weight.begin()+uint64_t(batch)*4*shape.indexHeads,4*shape.indexHeads,vw.begin()+uint64_t(v)*4*shape.indexHeads);
        std::copy_n(qscale.begin()+uint64_t(batch)*4*shape.indexHeads,4*shape.indexHeads,vqs.begin()+uint64_t(v)*4*shape.indexHeads);
    }
    auto binding=original;binding[0]=m.put(vq);binding[2]=m.put(vw);binding[3]=m.put(vqs);
    binding[5]=m.put(qlens);binding[6]=m.put(klens);binding[7]=m.put(vpages);
    binding[9]=m.put(std::vector<int32_t>(uint64_t(vb)*4*2048+128,0x5a5a5a5a));
    auto score=m.put(std::vector<uint16_t>(uint64_t(vb)*4*rowStride+128,0x5a5a));binding[12]=score;
    binding[13]=m.put(std::vector<uint8_t>(uint64_t(vb)*(4*2048*6+1024+4*rowStride*2)+128));
    auto virtualContext=real;virtualContext.shape=vs;virtualContext.lruSlots=0;virtualContext.reservedHot=0;
    virtualContext.progress=m.put(std::vector<uint32_t>(vb*4*32));virtualContext.queryEnds=m.put(vends);
    virtualContext.hits=binding[13]+uint64_t(vb)*(4*2048*6+1024);
    binding[14]=m.put(std::vector<rotating::Context>{virtualContext});auto virtualBinding=m.put(binding);
    auto ids=m.put(std::vector<uint32_t>(uint64_t(q)*topkShards*2048+128,0x5a5a5a5a));
    auto scores=m.put(std::vector<uint16_t>(uint64_t(q)*topkShards*2048+128,0x5a5a));
    auto finalScores=m.put(std::vector<uint16_t>(uint64_t(q)*2048+128,0x5a5a));
    auto localTimes=m.put(std::vector<uint64_t>(64*4)),mergeTimes=m.put(std::vector<uint64_t>(64*4));
    uint32_t lw=std::min(q*topkShards,vcap),mw=std::min(q,vcap);
    std::ofstream geometry(out+"/history-geometry.json");
    geometry<<"{\"real_batch\":"<<b<<",\"history\":"<<h<<",\"shards\":"<<shards
      <<",\"aic_budget\":"<<ccap<<",\"aiv_budget\":"<<vcap<<",\"producer_aic\":"<<vs.liCores
      <<",\"score_aiv\":"<<2*vs.liCores<<",\"local_topk_aiv\":"<<lw
      <<",\"topk_shards\":"<<topkShards<<",\"merge_aiv\":"<<(topkShards>1?mw:0)<<",\"row_stride\":"<<rowStride
      <<",\"min_chunk\":"<<h/128/shards*128<<",\"max_chunk\":"<<maxPages*128
      <<",\"query_group\":4,\"service_rounds\":0,\"source_layout\":\"b,shard,q,Hpad\"}\n";geometry.close();
    check(v9_init((void*)realBinding,stream,&shape));
    check(aclrtlaunch_v9_li_history_kernel(vs.liCores,stream,(void*)virtualBinding));
    check(aclrtlaunch_history_topk_local(lw,stream,(void*)score,(void*)(topkShards==1?original[9]:ids),
        (void*)scores,q,h,rowStride,shards,topkShards,lw,(void*)localTimes));
    if(topkShards>1)check(aclrtlaunch_split_topk_merge(mw,stream,(void*)ids,(void*)scores,
        (void*)original[9],(void*)finalScores,q,topkShards,mw,(void*)mergeTimes));
    check(v9_resolve((void*)realBinding,stream,&shape));
    check(kvop_direct_update_outputs_early(b,(void*)update,stream));
    check(aclrtlaunch_v9_fetch_update_kernel(b,stream,(void*)realBinding,(void*)update));
    check(aclrtSynchronizeStream(stream));
    auto raw=m.get<uint16_t>(score,uint64_t(vb)*4*rowStride+128);
    std::vector<uint16_t> flat(uint64_t(q)*h);
    for(uint32_t batch=0;batch<b;++batch)for(uint32_t s=0;s<shards;++s){
        uint32_t begin=h/128*s/shards*128,len=klens[batch*shards+s];
        for(uint32_t query=0;query<4;++query)
            std::copy_n(raw.begin()+(uint64_t(batch*shards+s)*4+query)*rowStride,len,
                        flat.begin()+(uint64_t(batch)*4+query)*h+begin);
        for(uint32_t query=0;query<4;++query)for(uint32_t t=len;t<rowStride;++t)
            require(raw[(uint64_t(batch*shards+s)*4+query)*rowStride+t]==0x5a5a,"history row padding guard");
    }
    save(out+"/score-keys.bin",flat);
    auto gi=m.get<uint32_t>(ids,uint64_t(q)*topkShards*2048+128);
    auto gv=m.get<uint16_t>(scores,uint64_t(q)*topkShards*2048+128);
    auto gf=m.get<uint16_t>(finalScores,uint64_t(q)*2048+128);
    for(uint32_t i=0;i<128;++i)require(raw[uint64_t(vb)*4*rowStride+i]==0x5a5a &&
        gi[uint64_t(q)*topkShards*2048+i]==0x5a5a5a5a && gv[uint64_t(q)*topkShards*2048+i]==0x5a5a &&
        gf[uint64_t(q)*2048+i]==0x5a5a,"history guard");
    save(out+"/local-ids.bin",topkShards==1?m.get<uint32_t>(original[9],uint64_t(q)*2048+128):gi);
    save(out+"/local-scores.bin",gv);
    if(caseName=="zero"){
        auto selected=m.get<uint32_t>(original[9],uint64_t(q)*2048);
        for(uint32_t query=0;query<q;++query){auto begin=selected.begin()+query*2048;std::sort(begin,begin+2048);
            for(uint32_t k=0;k<2048;++k)require(begin[k]==k,"zero history exact selection");}
    }
    std::ofstream proof(out+"/history-check.json");
    proof<<"{\"status\":\"passed\",\"score_layout_reconstructed\":true,\"allocation_guards\":true,\"partial_query_end_supported\":false}\n";
}

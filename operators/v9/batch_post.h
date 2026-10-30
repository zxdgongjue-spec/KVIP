#pragma once
namespace rotating {
// One SIMT task covers every batch assigned to this AIV. Per-batch queues
// remain independent. locations is dead until final resolve and stages IDs.
constexpr uint32_t kPostGroups=(SIM_BATCH+kServiceLanes-1)/kServiceLanes;
constexpr uint32_t kPostThreads=kPostGroups<=1?512:(kPostGroups<=2?256:(kPostGroups<=4?128:64));
constexpr uint32_t kPostBank=kPostGroups<=1?448:(kPostGroups<=2?240:(kPostGroups<=4?120:56));
// FinalBuild stores complete 64-WQE vector blocks, including tail padding.
static_assert(SIM_BATCH>16 || ((kPostGroups-1)*kPostBank+((kPostBank+63)/64)*64)*64<=32000,
              "Padded WQE banks exceed the smallest borrowed scratch");
template<bool BuildSimt>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void BatchBuild(__gm__ Context* ctx,uint32_t epoch,uint32_t lane,uint32_t begin,
    __ubuf__ uint64_t* remote,__ubuf__ uint32_t* meta,__ubuf__ uint32_t* inputs,__ubuf__ uint32_t* wqes) {
    const uint32_t g=threadIdx.x/kPostThreads,t=threadIdx.x%kPostThreads,b=lane+kServiceLanes*g;
    const bool active=b<SIM_BATCH;
    uint32_t count=0,n=0,slotBase=0,depth=0;uint64_t head=0,output=0;
    __gm__ Row* row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    __gm__ uint32_t* rowWords=reinterpret_cast<__gm__ uint32_t*>(row);
    if(active) {
        auto input=inputs+g*80;auto input64=reinterpret_cast<__ubuf__ uint64_t*>(input);
        count=input[64];
        if(count) {
            auto entity=input;
            output=input64[35];
            head=input64[25];depth=entity[19];
            uint32_t pending=input[66],cqDepth=entity[31];
            if(count>=depth || pending+1>=cqDepth) {
                if(!t){
                    asc_stcg(rowWords+11,uint32_t(kvaf_device::kInvalidArgs));
                    asc_stcg(rowWords+8,asc_ldcg(rowWords+8)-count);
                    asc_stcg(rowWords+9,asc_ldcg(rowWords+9)-count);input[64]=0;
                }count=0;
            } else {
                auto addresses=reinterpret_cast<__gm__ uint64_t*>(input64[34]);
                auto jobs=reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*8192;
                auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*SIM_HISTORY;
                slotBase=input[65];
                n=count>begin?count-begin:0;if(n>kPostBank)n=kPostBank;
                for(uint32_t i=t;i<n;i+=kPostThreads) {
                    uint32_t token=asc_ldcg(jobs+begin+i);
                    uint32_t physical=slotBase+begin+i;
                    uint64_t address;
                    if(ctx->contract) { address=contract_simt::ContractRemote(contract_simt::ContractOf(ctx),b,token); physical=contract_simt::ReserveContract(ctx,b,token,slotBase+begin+i); }
                    else address=asc_ldcg(addresses+token);
                    if constexpr(BuildSimt)remote[g*kPostBank+i]=address;
                    else {
                        auto fields=reinterpret_cast<__ubuf__ uint32_t*>(remote);
                        fields[g*kPostBank+i]=uint32_t(address);
                        fields[512+g*kPostBank+i]=uint32_t(address>>32);
                        fields[1024+g*kPostBank+i]=physical;
                    }
                    if(!ctx->contract)asc_stcg(map+token,slotBase+begin+i);
                }
                if(t<16) {
                    uint32_t value=0;
                    if(t==1)value=6U<<8;
                    if(t==2)value=(entity[20]&0xffffffU)|0x1000000U;
                    if(t==3)value=entity[6]&0xfffffU;
                    if(t>=4 && t<8)value=entity[22+t-4];
                    if(t==8)value=entity[7];
                    if(t==12)value=kTokenBytes;
                    if(t==13)value=entity[12];
                    if(t==14)value=uint32_t(output);
                    if(t==15)value=uint32_t(output>>32);
                    meta[g*32+t]=value;
                }
                if(!t) {meta[g*32+16]=uint32_t(head+begin);meta[g*32+17]=depth;}
            }
        }
    }
    asc_syncthreads();
    if(BuildSimt && active && count) {
        for(uint32_t i=t;i<n*16;i+=kPostThreads) {
            uint32_t token=i/16,field=i%16,value=meta[g*32+field];
            uint32_t physical=slotBase+begin+token;
            if(ctx->contract) { auto c=contract_simt::ContractOf(ctx); physical=contract_simt::ContractPhysical(c,b,contract_simt::ContractPtr<int32_t>(c->eligible_victims)[b*2048+physical]); }
            uint64_t address=remote[g*kPostBank+token],local=output+uint64_t(physical)*kTokenBytes;
            if(field==0)value=((begin+token+1==count)?0x30360000U:0x30150000U)|(((head+begin+token)&depth)?0U:0x80000000U);
            if(field==10)value=uint32_t(address);
            if(field==11)value=uint32_t(address>>32);
            if(field==14)value=uint32_t(local);
            if(field==15)value=uint32_t(local>>32);
            wqes[g*kPostBank*16+i]=value;
        }
    }
    asc_threadfence();asc_syncthreads();
    if(active && count && begin+n==count && n && !t) {
        asc_stcg(rowWords+211,inputs[g*80+66]+1);
    }
    asc_threadfence();
}
}

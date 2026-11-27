// Included in joint_tail after the actual per-lane/row mailbox types.
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void OneTeam(__gm__ TerminalRowSchedule* sched) {
    if(!threadIdx.x){
        asc_stcg(sched->order,tiny::first);asc_stcg(sched->order+31,tiny::second);
        asc_stcg(&sched->next,31U);asc_stcg(&sched->finishedTeams,15U);
        asc_threadfence();asc_stcg(&sched->ready,1U);asc_threadfence();
    }
}
__aicore__ inline void PrepareFixture(__gm__ terminal_runtime::Binding* r,__gm__ rotating::Context* c,
                                     Saved& s,uint32_t row,uint32_t q,bool defer){
    auto p=Bind(r,c,row,q);auto tr=reinterpret_cast<__gm__ tiny::Trace*>(p.args->indices);
    if(!Begin(s,row,q))return;
    ++tr->prepare;
    const uint32_t n=p.args->token_bytes;
    auto cuts=rotating::PlanTerminalBatches(n,row==tiny::first?302:317,0);
    const auto sq=p.Producer();tr->initialSq=sq;
    rotating::CommitTerminalPost(p.args,sq,defer?0:cuts.ends[0],p.post);
    if(!Arm(s,sq,p.Cursor(),cuts.count,cuts.ends,n,!defer,2,p.Key())){s.error=32;s.stop=true;}
    s.owner=row;s.generation=1;s.total=row==tiny::first?302:317;s.scan=131072;
    tr->refused=!Begin(s,row,q);
}
__aicore__ inline void Supply(Saved& s,Port& p){
    if(s.seen==s.issued || s.stop || s.phase!=Phase::Prepared)return;
    // Only synthesize delivery for a real published private prefix, never modify its software cursor.
    const uint32_t fault=p.args->table_capacity;
    if(s.row==tiny::first && s.query==3 && s.seen==2 && fault==2)return;
    const auto at=s.cqHead+s.seen;
    auto ptr=reinterpret_cast<__gm__ uint32_t*>(p.entity->cq_va)+(at%8)*16;
    const bool bad=s.row==tiny::first && s.query==3 && s.seen==2 && fault==1;
    *ptr=(uint32_t(1^((at/8)&1))<<2)|(bad?(1U<<16):0);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ptr),64);
}

"""Generate the experiment-only copy from the pinned production row routine."""
import argparse
from pathlib import Path

def prepare(base, out):
    text=(base / "row_kernel.h").read_text()
    def once(before, after):
        nonlocal text
        if text.count(before)!=1:
            raise ValueError(f"Baseline row hook changed: {before}")
        text=text.replace(before,after)
    once("template<bool LargeBatch>\n__aicore__ inline void RunTerminalRow(GM_ADDR binding, uint32_t b, uint32_t q){",
         '''#include "resume.h"
template<bool LargeBatch, bool Suspend=false>
__aicore__ inline void RunTerminalRow(GM_ADDR binding, uint32_t b, uint32_t q,
     joint_tail::Saved* saved=nullptr,bool defer=false){
 if constexpr(Suspend){if(!joint_tail::Begin(*saved,b,q))return;}''')
    once(" if(own){\n  if(profile)ts[13]", """ bool transportSafe=true;
 if constexpr(Suspend) {
  auto entity=reinterpret_cast<__gm__ KvafBatchGetDirectUrmaEntity*>(args->channel_entity);
  transportSafe=!queue->poisoned && (!own || (own<entity->sq_depth && entity->cq_depth>1 &&
      originalHead<=UINT64_MAX-own && queue->cq_consumer_index<=UINT64_MAX-batches.count));
 }
 if(own && transportSafe){
  if(profile)ts[13]""")
    once("rotating::CommitTerminalPost(args,originalHead,batches.ends[0],post);",
         """if constexpr(Suspend) {
  rotating::CommitTerminalPost(args,originalHead,(defer || !transportSafe)?0:batches.ends[0],post);
 } else { rotating::CommitTerminalPost(args,originalHead,batches.ends[0],post); }""")
    once(" uint64_t cqError;\n",''' if constexpr(Suspend) {
  saved->owner=owner;saved->generation=generation;saved->total=total;saved->first=first;
  saved->n=n;saved->scan=scanHistory;saved->topError=topkError;saved->packError=packError;
  saved->indexError=indexError;
  bool ok=joint_tail::Arm(*saved,originalHead,queue->cq_consumer_index,own?batches.count:0,
         batches.ends,own,own&&!defer&&transportSafe,own&&rotating::CanOverlapTerminalPrefixes(args,own)?2:1,args->queue_state);
  if(!ok || !transportSafe){saved->phase=joint_tail::Phase::Prepared;saved->error=kvaf_device::kInvalidArgs;saved->stop=true;}
  // MTE3 public outputs have already joined, all WQEs are in the private SQ and Index is done.
  pipe.Destroy();return;
 }
 uint64_t cqError;
''')
    once("if (LargeBatch && batches.count > 1 && rotating::CanOverlapTerminalPrefixes(args, own))",
         "if (false /* unmodified single-credit fallback in this experiment-only TU */)")
    (out / "row_kernel.h").write_text(text)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("base",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args();prepare(a.base,a.out)

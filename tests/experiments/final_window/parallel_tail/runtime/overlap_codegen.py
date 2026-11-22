"""Candidate parallel payload update, tested separately before integration."""
from pathlib import Path
import shutil

from private_topk import replace_once


def adapt_parallel_overlap(destination, *, reuse_scratch=False):
    shutil.copy2(Path(__file__).with_name('overlap_update.h'), destination/'overlap_update.h')
    path = destination/'kernel.cpp'
    text = replace_once(path.read_text(), '#include "helpers.h"',
                        '#include "helpers.h"\n#include "overlap_update.h"')
    start = text.index('  if(!q){\n   pipe.Reset();')
    end = text.index('  // Retirement is shared', start)
    text = text[:start] + '''  TBuf<TPosition::VECCALC> parallelScratch;
  auto parallelLocal=local;
  uint32_t setWords=128;
  while(setWords<total*2 && setWords<kParallelSet)setWords*=2;
  uint32_t neededWords=128+2*setWords+4*total+((2*total+3)/4)*kIndexRowWords;
  if(neededWords>parallelLocal.GetSize()){
   pipe.Reset();
   pipe.InitBuffer(parallelScratch,kParallelIndexWords*sizeof(uint32_t));
   parallelLocal=parallelScratch.Get<uint32_t>();
  }
  work=reinterpret_cast<__ubuf__ uint32_t*>(parallelLocal.GetPhyAddr());
  Simt::VF_CALL<ApplyParallelIndex>(Simt::Dim3{512,1,1},allSources,dst-first,total,
   s.resident,s.history,q,generation,
   reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*s.history*16,
   reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*s.history,
   reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*addressStride,
   reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*s.history,
   reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*s.resident,
   flags,ctx->reservedHot?reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+uint64_t(b)*s.resident:nullptr,work,parallelLocal.GetSize());
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  indexError=parallelLocal.GetValue(0);
''' + text[end:]
    # Counter counts only overlapping rounds, not the full graph generation.
    # Reset after every worker leaves the existing completion rendezvous.
    finish = destination/'terminal_finish.h'
    finish.write_text(replace_once(finish.read_text(),
        'if(!q && !t)asc_stcg(flags+5*32,0U);',
        'if(!q && !t){asc_stcg(flags+5*32,0U);asc_stcg(flags+7*32,0U);}'))
    text = text.replace('s.resident,s.history,q,generation,', 's.resident,s.history,q,1U,')
    if not reuse_scratch:
        # Retain the measured 40.11 us implementation. Scratch reuse was
        # slower in the same cross-worker-cycle fixture (41.38 us).
        start = text.index('  TBuf<TPosition::VECCALC> parallelScratch;')
        end = text.index('  work=reinterpret_cast<__ubuf__ uint32_t*>(parallelLocal.GetPhyAddr());', start)
        text = text[:start] + """  pipe.Reset();
  TBuf<TPosition::VECCALC> parallelScratch;
  pipe.InitBuffer(parallelScratch,kParallelIndexWords*sizeof(uint32_t));
  auto parallelLocal=parallelScratch.Get<uint32_t>();
""" + text[end:]
        text = text.replace('nullptr,work,parallelLocal.GetSize());', 'nullptr,work);')
        header = destination/'overlap_update.h'
        body = header.read_text().replace('work,uint32_t workspaceWords){', 'work){')
        body = body.replace('>workspaceWords)', '>kParallelIndexWords)')
        header.write_text(body)
    path.write_text(text)

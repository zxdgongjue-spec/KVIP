"""Extract same-run phase timings from native simulator instruction timestamps."""
import argparse
import json
from pathlib import Path
import re

def validate_audits(log):
    audits=re.findall(r'mode=(\d+) posted=(\d+) bad=(\d+)',log)
    if not audits or any(posted!='146' or bad!='0' for _,posted,bad in audits) or 'FAIL ' in log:
        raise ValueError('Full output/Index/WQE verification did not pass')
    mode=int(audits[0][0])
    expected_launches=2 if mode in (3,9,15) else 1
    if any(int(case)!=mode for case,_,_ in audits) or len(audits)!=expected_launches:
        raise ValueError(f'Expected {expected_launches} full per-launch audits; found {len(audits)}')
    if expected_launches>1 and re.findall(r'verified_launch=(\d+)',log)!=['1','2']:
        raise ValueError('Repeated-run audit lacks both pre-restore verification markers')
    return mode, expected_launches

def cq_timing_available(rows):
    """Require a real post timestamp and at least the modeled delay before exit."""
    if len(rows) != 4:
        return False
    for row in rows:
        ts = row['stamps_us']
        if not all(i in ts for i in (4, 16, 17)):
            return False
        if not ts[4] <= ts[16] <= ts[17] or ts[17] - ts[4] < 4.98:
            return False
    return True

def resolve_simt_post_clocks(events, times, stamps, query):
    """Match UB-returned clock64 values to native instructions, never estimate them."""
    values=[stamps.get(query*32+i,0) for i in (4,5)]
    if not all(values):
        return
    clocks=[e for e in events if e.get('name')=='SIMT_S2RL'
            and '__clang_cce_simt_spr.h:33' in e.get('args',{}).get('code','')]
    if len(clocks)!=2 or len(times)<2:
        raise ValueError('Ambiguous SIMT post clock evidence')
    clocks.sort(key=lambda e:e['ts'])
    low,high=min(times),max(times)
    scale=(times[high]-times[low])/(high-low)
    for value,event in zip(values,clocks):
        predicted=times[low]+(value-low)*scale
        if abs(event['ts']-predicted)>0.001:
            raise ValueError('SIMT post clock differs from native SYS_CNT calibration')
        times[value]=event['ts']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--baseline-us',type=float,default=80.0)
    a=parser.parse_args()
    log=(a.directory/'run.log').read_text()
    mode, expected_launches=validate_audits(log)
    merged_prepare='rotating::PrepareCandidateWqes(' in (a.directory/'parallel_post.h').read_text() if (a.directory/'parallel_post.h').exists() else False
    merged_prepare = merged_prepare or ('PrepareDirect(resident,ids,slots,mapping,counter[0]' in (a.directory/'parallel_post.h').read_text() if (a.directory/'parallel_post.h').exists() else False)
    source=(a.directory/'kernel.cpp').read_text() if (a.directory/'kernel.cpp').exists() else ''
    async_wqe='Emit uses separate UB storage; WQE DMA may run concurrently.' in source
    early_output='MERGED_OUTPUT_BEFORE_WQE' in source
    output_dma_overlap='Private output DMA uses disjoint storage' in source
    merged_output=' Emit(fullIds,' in (a.directory/'parallel_post.h').read_text() if (a.directory/'parallel_post.h').exists() else False
    stamps={int(k):int(v) for k,v in re.findall(r'stamp(\d+)=(\d+)',log)}
    rows=[]
    calibration=[]
    for path in sorted((a.directory/'profile').glob('**/core*.veccore*/trace.json')):
        events=json.loads(path.read_text())['traceEvents']
        times={}
        for event in events:
            detail=event.get('args',{}).get('detail','')
            if 'SPR:SYS_CNT' in detail:
                m=re.search(r'=0x([0-9a-fA-F]+)',detail)
                if m:times[int(m[1],16)]=event['ts']
        if len(times)>=2:
            lo,hi=min(times),max(times)
            if times[hi]>times[lo]:calibration.append(9000*(times[hi]-times[lo])/(hi-lo))
        core=re.search(r'core(\d+).veccore(\d+)',str(path))
        q=int(core[1])*2+int(core[2])
        if 'Simt::VF_CALL<PublishQueueAndJoin>' in source:
            resolve_simt_post_clocks(events,times,stamps,q)
        # v48's host printed only 24 fields. Recover its four omitted clock reads
        # from the instruction trace, bounded by its printed stamps 23 and 11.
        source=(a.directory/'kernel.cpp').read_text() if (a.directory/'kernel.cpp').exists() else ''
        if q*32+24 not in stamps and all(f'ts[{i}]=' in source for i in range(24,28)):
            reads=sorted(k for k in times if stamps[q*32+23]<k<stamps[q*32+11])
            if len(reads)!=4:raise ValueError('Ambiguous omitted subphase timestamps')
            for i,value in zip(range(24,28),reads):stamps[q*32+i]=value
        ts={i:times[stamps[q*32+i]] for i in list(range(18))+list(range(23,28)) if q*32+i in stamps and stamps[q*32+i] in times}
        spans={}
        for name,start,end in [('post_queue_dma_issue',3,23),('post_wait_prior_topk',23,24),('post_header_dma',24,25),('post_candidate_dma',25,26),('post_bitmap_clear',26,27),('post_dedup_compute',27,11),('setup',0,1),('topk',1,2),('topk_to_post_setup',2,3),('queue_publish',15,4),('event_release',4,5),('tail_setup',5,6),('output_adaptation_including_mapping_wait',6,7),('wait_slot_readers',7,8),('index_metadata_shared_retire_cq',8,9),('generation_finish',9,10),('dedup_and_pack',3,11),('count_handoff',11,12),('prepare_wqe_inputs',12,13),('write_wqes',13,14),('logical_output',14,15),('cq_wait_exposed',16,17)]:
            if start in ts and end in ts:
                if early_output and name=='tail_setup':continue
                if async_wqe and name in ('tail_setup','queue_publish','event_release'):continue
                if name=='logical_output' and 'Logical IDs are emitted with slots after all admissions are visible.' in source:continue
                if async_wqe and name=='write_wqes':name='build_and_issue_wqes'
                if async_wqe and name=='output_adaptation_including_mapping_wait':name='output_compute_wqe_join_doorbell_and_output_dma'
                if ts[end]<ts[start]:raise ValueError(f'Negative phase {name} on Q{q+1}; update phase definitions')
                if merged_prepare and name=='dedup_and_pack':name='dedup_pack_wqe_inputs_and_output' if merged_output else 'dedup_pack_and_wqe_inputs'
                if merged_prepare and name=='prepare_wqe_inputs':name='scalar_queue_handoff'
                if early_output and name=='output_adaptation_including_mapping_wait':name='dedup_prepare_output_compute_and_dma'
                if early_output and name=='wait_slot_readers':name='remaining_wqe_post_and_output_join'
                if early_output and name=='scalar_queue_handoff':name='output_dma_and_queue_setup'
                if output_dma_overlap and name=='dedup_prepare_output_compute_and_dma':name='dedup_output_compute_and_dma_issue'
                if output_dma_overlap and name=='write_wqes':name='write_wqes_and_complete_output_dma'
                if 'Simt::VF_CALL<PublishQueueAndJoin>' in source and name=='event_release':name='output_ready_release'
                if name=='cq_wait_exposed' and 'rotating::CompleteTerminalPost(args,post)' in source:
                    # Native completion includes queue/status publication and,
                    # on Q1, prior-prefetch completion; it is not pure CQ wait.
                    prefetch_q=1 if 'if(q==1)rotating::CompleteTerminalPrefetch(ctx,b);' in source else 0
                    name='cq_completion_and_status_with_prefetch' if q==prefetch_q else 'cq_completion_and_status'
                spans[name]=ts[end]-ts[start]
        if 'rotating::CompleteTerminalPost(args,post)' in source:
            # Exclusive tail intervals: their sum reconstructs 8 -> 10.
            for name,start,end in [('index_and_retirement',8,16),
                                   ('completion_status',16,17),
                                   ('completion_to_finish',17,9),
                                   ('request_join_and_finish',9,10)]:
                if start in ts and end in ts:
                    if ts[end]<ts[start]:raise ValueError(f'Negative native phase {name}')
                    spans[name]=ts[end]-ts[start]
            if all(i in ts for i in (8,16,17,9,10)):
                total=sum(spans[n] for n in ('index_and_retirement','completion_status',
                                            'completion_to_finish','request_join_and_finish'))
                if abs(total-(ts[10]-ts[8]))>1e-6:
                    raise ValueError('Native tail interval accounting failed')
        if 3 in ts and 4 in ts:spans['service_through_doorbell_including_output_compute' if async_wqe or early_output else 'service']=ts[4]-ts[3]
        rows.append({'q':q+1,'posted_unique':stamps.get(q*32+20),'stamps_us':ts,'durations_us':spans})
    cores=[{'core':name,'duration_us':float(d),'running_us':float(r)} for name,d,r in re.findall(r'(core\d+.veccore\d+)\s+([0-9.]+)\s+([0-9.]+)',log)]
    mode=int(re.search(r'mode=(\d+) posted=146 bad=0',log)[1])
    expected=4  # msprof exports the first matching kernel; mode 3 verifies both launches.
    if len(cores)!=expected:raise ValueError(f'Expected {expected} core intervals, got {len(cores)}')
    maximum=max(c['duration_us'] for c in cores)
    if not calibration or any(abs(value-5.0)>0.01 for value in calibration):
        raise ValueError(f'CQ model clock calibration failed: {calibration}')
    result={'case':mode,'verified_launches':expected_launches,'profiled_launches':1,'detailed_stamps':len(rows)==4 and all(all(i in q['stamps_us'] for i in (0,1,2,3,4,7,8,9,10)) for q in rows),'cq_delay_us_verified':calibration,'correctness':'PASS','time_source':'vendor simulator native microseconds','cq':'simulation: ready 9000 SYS_CNT ticks after post; verify 5 us against native timeline','fixture':'constructed scores, trace-matched 11/0/94/87 misses, union 146, prior 84','cores':cores,'queries':rows,'max_core_us':maximum,'baseline_us':a.baseline_us,'baseline_is_reference_only':True}
    # Required posting interval includes handoff and all four location outputs.
    leader=next((row for row in rows if row['q']==1),None)
    posting=None
    if leader and all(i in leader['stamps_us'] for i in (3,8)):
        posting=max(row['stamps_us'][8] for row in rows if 8 in row['stamps_us'])-min(row['stamps_us'][3] for row in rows if 3 in row['stamps_us'])
    result['posting_including_output_sync_us']=posting
    result['posting_target_us']=10.0
    result['posting_target_pass']=None if posting is None else posting<=10.0
    # The integration goal includes all tail work. The old non-CQ diagnostic
    # below must never stand in for the user's 28 us full-tail requirement.
    result['full_tail_target_us']=28.0
    result['full_tail_target_pass']=None if mode==5 else maximum<=28.0
    result['integration_performance_pass']=(None if posting is None or mode==5
                                          else posting<=10.0 and maximum<=28.0)
    result['non_cq_target_us']=35.0
    result['non_cq_target_pass']=None
    result['non_cq_note']='CQ is combined with Index in current stamps; do not subtract a fixed 5 us from overlapped execution.'
    result['cq_post_and_completion_timestamps_valid']=cq_timing_available(rows)
    if cq_timing_available(rows) and all('cq_wait_exposed' in row['durations_us'] for row in rows):
        waits={row['q']-1:row['durations_us']['cq_wait_exposed'] for row in rows}
        adjusted=[]
        for core in cores:
            match=re.search(r'core(\d+).veccore(\d+)',core['core'])
            q=int(match[1])*2+int(match[2])
            adjusted.append(core['duration_us']-(waits[q] if q==0 or 'cq_per_core=1' in log else 0.0))
        result['non_cq_us']=max(adjusted)
        result['non_cq_target_pass']=max(adjusted)<=35.0
        result['non_cq_note']='Subtract only measured exposed CQ wait per core; retain all other work and synchronization.'
    result['performance_gate_applicable']=mode!=5 and result['detailed_stamps'] and cq_timing_available(rows)
    if mode==5:
        result['posting_target_pass']=None
        result['non_cq_target_pass']=None
        result['stress_note']='Deliberately delay a producer by 36000 SYS_CNT ticks (~20 us); inspect the archived kernel for Q and placement; correctness regression only.'
    (a.directory/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()

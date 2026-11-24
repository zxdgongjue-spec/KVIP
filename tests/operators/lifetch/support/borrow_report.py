"""Compare window-borrow and four-batch candidates against their preceding measured B24 version."""
import argparse
from pathlib import Path

from last_window_borrow import CANDIDATE, COMBINED, DUAL, expected_text, summarize
from li_restore_report import saved, stats
from li_window_report import KVAF
from main_five_report import MAIN, archived_oracle, case_folder, measured_parallel, read
from peer_delta import pair_placement
from reporting import atomic_json


def report(root, group, owner, reference):
    revision = (root/'candidate.txt').read_text().strip()
    combined = revision == COMBINED
    old_revision = CANDIDATE if combined else DUAL
    reference_name = 'last_window_borrow' if combined else 'dual_cut'
    if revision not in (CANDIDATE, COMBINED):
        raise ValueError(f'Unexpected native operator in window-borrow comparison: {revision}')
    for archive, expected_rev in ((root, revision), (reference, old_revision)):
        for name, expected in (('main', MAIN), ('candidate', expected_rev), ('physical_npu', '4'), ('kvaf', KVAF)):
            if (archive/f'{name}.txt').read_text().strip() != expected:
                raise ValueError(f'Unexpected {name} in G{group} archive {archive}')
    native, receipt = archived_oracle(owner, group)
    identity = read(case_folder(owner, group, 'logical-ties').parent/'replay_input.json')
    old, _, old_place, old_proof, old_cfg = measured_parallel(
        reference, group, 'li-window', old_revision, owner, native, receipt, identity, verify_library=False)
    now, _, place, proof, cfg = measured_parallel(
        root, group, 'li-window', revision, owner, native, receipt, identity, verify_library=True)
    if Path(cfg['kvaf_root']).resolve() != Path(old_cfg['kvaf_root']).resolve() or old_place['bytes'] != place['bytes']:
        raise ValueError('Candidate and reference use different KVAF or registered DDR capacity')
    folder = case_folder(root, group, 'li-window')
    evidence = read(folder/'borrow_checks.json')
    if evidence['device_text_sha256'] != expected_text(revision):
        raise ValueError('Unverified device code in window-borrow evidence')
    for phase, key in (('measured', 'measured'), ('detail', 'terminal_only')):
        counts = summarize(dict(enumerate(evidence['calls'][phase])))
        if counts != evidence[key]:
            raise ValueError(f'{phase} mailbox counts disagree with saved per-call observations')
        print(f'G{group} {key}_row_work completed_two_queries={counts["delegated"]}/320 '
              f'owner_all_queries={counts["retained"]}/320 before_first_Q={counts["before_first_q"]} '
              f'before_second_Q={counts["before_second_q"]}')
    score = read(folder/'terminal_only/score_topk_check.json')
    if (score['calls'], score['queries']) != (20, 1280) or score['native_reference']['identity'] != receipt['native_reference']['identity']:
        raise ValueError('Incomplete terminal-only TopK proof')
    placement = pair_placement(old_place, place, local_baseline=True)
    same_runtime = old_proof['runtime'] == proof['runtime']
    if combined:
        from terminal_batch_checks import totals
        old_evidence = read(case_folder(reference, group, 'li-window')/'borrow_checks.json')
        if old_evidence['device_text_sha256'] != expected_text(old_revision):
            raise ValueError('Archived LI-only operator is missing its verified instruction hashes')
        q_cq = read(folder/'final_cq_checks.json')
        if q_cq['independent_status_allocations'] != 20:
            raise ValueError('Final CQ evidence must use 20 independently bound status allocations')
        for phase in ('measured', 'detail'):
            histogram = totals(q_cq[phase])
            print(f'G{group} {phase}_private_final_CQ batches_0/1/2/3/4='
                  + '/'.join(map(str, histogram)) + ' queries=1280/1280')
    if placement != 'matched' or not same_runtime:
        raise ValueError(f'G{group}: {reference_name} comparison unverified: placement={placement} same_runtime={int(same_runtime)}')
    print(f'G{group} {reference_name}_archive', *(f'{k}_mean/P50_us={stats(v)}' for k, v in old.items()))
    print(f'G{group} vs_{reference_name} historical_unpaired', *(f'{k}_saved_mean/P50_us={saved(old[k], now[k])}' for k in now),
          f'placement={placement} same_runtime={int(same_runtime)}')
    archive_fields = (dict(li_only_archive=str(reference.resolve()), li_only=old) if combined else
                      dict(dual_archive=str(reference.resolve()), dual=old))
    atomic_json(root/f'g{group}/borrow_comparison.json', dict(
        group=group, **archive_fields,
        candidate=now, placement=placement,
        same_runtime=same_runtime, measured_work=evidence['measured'], terminal_only_work=evidence['terminal_only']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('group', type=int, choices=(10, 4, 17, 7, 2))
    parser.add_argument('owner', type=Path)
    parser.add_argument('reference', type=Path)
    args = parser.parse_args()
    report(args.root, args.group, args.owner, args.reference)

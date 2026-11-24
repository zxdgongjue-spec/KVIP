"""Compare one normal window-borrow measurement with the verified B24 dual-cut archive."""
import argparse
from pathlib import Path

from last_window_borrow import CANDIDATE, DUAL, EXPECTED_TEXT, summarize
from li_restore_report import saved, stats
from li_window_report import KVAF
from main_five_report import MAIN, archived_oracle, case_folder, measured_parallel, read
from peer_delta import pair_placement
from reporting import atomic_json


def report(root, group, owner, dual):
    for archive, revision in ((root, CANDIDATE), (dual, DUAL)):
        for name, expected in (('main', MAIN), ('candidate', revision), ('physical_npu', '4'), ('kvaf', KVAF)):
            if (archive/f'{name}.txt').read_text().strip() != expected:
                raise ValueError(f'Unexpected {name} in G{group} archive {archive}')
    native, receipt = archived_oracle(owner, group)
    identity = read(case_folder(owner, group, 'logical-ties').parent/'replay_input.json')
    old, _, old_place, old_proof, old_cfg = measured_parallel(
        dual, group, 'li-window', DUAL, owner, native, receipt, identity, verify_library=False)
    now, _, place, proof, cfg = measured_parallel(
        root, group, 'li-window', CANDIDATE, owner, native, receipt, identity, verify_library=True)
    if Path(cfg['kvaf_root']).resolve() != Path(old_cfg['kvaf_root']).resolve() or old_place['bytes'] != place['bytes']:
        raise ValueError('Candidate and archived dual-cut use different KVAF or registered DDR capacity')
    folder = case_folder(root, group, 'li-window')
    evidence = read(folder/'borrow_checks.json')
    if evidence['device_text_sha256'] != EXPECTED_TEXT:
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
    if placement != 'matched' or not same_runtime:
        raise ValueError(f'G{group}: dual-cut comparison unverified: placement={placement} same_runtime={int(same_runtime)}')
    print(f'G{group} dual_cut_archive', *(f'{k}_mean/P50_us={stats(v)}' for k, v in old.items()))
    print(f'G{group} vs_dual_cut historical_unpaired', *(f'{k}_saved_mean/P50_us={saved(old[k], now[k])}' for k in now),
          f'placement={placement} same_runtime={int(same_runtime)}')
    atomic_json(root/f'g{group}/borrow_comparison.json', dict(
        group=group, dual_archive=str(dual.resolve()), dual=old, candidate=now, placement=placement,
        same_runtime=same_runtime, measured_work=evidence['measured'], terminal_only_work=evidence['terminal_only']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('group', type=int, choices=(10, 4, 17, 7, 2))
    parser.add_argument('owner', type=Path)
    parser.add_argument('dual', type=Path)
    args = parser.parse_args()
    report(args.root, args.group, args.owner, args.dual)

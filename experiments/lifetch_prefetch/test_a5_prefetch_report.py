# SPDX-License-Identifier: Apache-2.0
"""Exercise the A5 runner report gate with synthetic host-side evidence."""

from contextlib import redirect_stdout
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from run_a5_prefetch import (BATCHES, DEFAULT_BATCH, KVAF, diagnostic_sample, diagnostics,
                             expected_geometry, main, save, summarize_batch, verified_route)


def sample(batch=DEFAULT_BATCH):
    windows = expected_geometry(batch)['windows']

    def window_rows(posted):
        return [dict(window=w, posted=posted if w == 1 else 0) for w in range(1, windows + 1)]

    audit = dict(early_duplicates=0, final_duplicates=0, early_final_overlap=0)
    requests = [dict(prefetch=0, final=0, windows=window_rows(0), prefetch_transfers=[], posted_identity_audit=audit.copy())
                for _ in range(batch)]
    requests[0] = dict(prefetch=2, final=1, windows=window_rows(2), prefetch_transfers=[
        dict(token=11, position=11), dict(token=12, position=12)], posted_identity_audit=audit.copy())
    cores = [
        dict(block=block, stages=[
            dict(name='initialize', start_cycles=200),
            dict(name='final_CQ_join', duration_us=1.5 if block == 0 else 100),
            dict(name='prefetch_CQ_join', duration_us=0.5 if block == 1 else 100),
        ]) for block in range(batch * 4)
    ]
    head = dict(
        prefetch_bandwidth=dict(entries=2, first_cycles=100, span_us=15,
                                note='Prefetch first doorbell to CQ join return'),
        final_get_bandwidth=dict(entries=1, span_us=4, groups=[dict(block=0, slot_base=2, entries=1)]),
        prefetch_services=[dict(lane=0, epoch=0, begin_cycles=80, end_cycles=180,
                                posted_count=2, ub_candidates=False, doorbells=1,
                                first_doorbell_cycles=100, last_doorbell_cycles=110)],
        terminal_timeline=dict(cores=cores, clock_mhz=10),
    )
    return dict(round=0, source_round=0, algorithm=dict(requests=requests),
                streaming_cycles=[head] + [{}] * (batch - 1))


def change_post_count(value, count):
    request = value['algorithm']['requests'][0]
    request.update(prefetch=count, prefetch_transfers=[dict(token=i, position=i) for i in range(count)])
    request['windows'][0]['posted'] = count
    head = value['streaming_cycles'][0]
    head['prefetch_bandwidth']['entries'] = count
    head['final_get_bandwidth']['groups'][0]['slot_base'] = count
    head['prefetch_services'][0]['posted_count'] = count
    return head


class ProtocolReportTest(unittest.TestCase):
    def test_counts_and_independent_cq_waits(self):
        result = diagnostic_sample(sample())
        self.assertEqual(result['early_entries'], 2)
        self.assertEqual(result['final_entries'], 1)
        self.assertEqual(result['duplicate_early_posts'], 0)
        self.assertEqual(result['service_to_first_doorbell_us'], 2)
        self.assertEqual(result['early_doorbell_to_terminal_us'], 10)
        self.assertEqual(result['early_cq_join_max_us'], 0.5)
        self.assertEqual(result['final_cq_join_max_us'], 1.5)
        self.assertEqual(result['early_doorbells'], 1)
        self.assertEqual(result['possible_chunked_doorbells'], 1)

    def test_duplicate_uses_request_local_logical_identity_from_actual_journal(self):
        value = sample()
        value['algorithm']['requests'][0]['prefetch_transfers'][1]['token'] = 11
        value['algorithm']['requests'][0]['posted_identity_audit']['early_duplicates'] = 1
        self.assertEqual(diagnostic_sample(value)['duplicate_early_posts'], 1)
        value['algorithm']['requests'][0]['prefetch_transfers'][1]['position'] = 11
        with self.assertRaisesRegex(ValueError, 'Admission journal'):
            diagnostic_sample(value)

    def test_final_intervals_and_cross_phase_duplicate_report(self):
        value = sample()
        value['algorithm']['requests'][0]['posted_identity_audit']['early_final_overlap'] = 1
        self.assertEqual(diagnostic_sample(value)['early_final_overlap'], 1)
        value['streaming_cycles'][0]['final_get_bandwidth']['groups'][0]['slot_base'] = 3
        with self.assertRaisesRegex(ValueError, 'not contiguous'):
            diagnostic_sample(value)

    def test_reject_service_entry_fallback(self):
        value = sample()
        value['streaming_cycles'][0]['prefetch_bandwidth']['note'] = 'Prefetch service entry to CQ join return'
        with self.assertRaisesRegex(ValueError, 'captured first doorbell'):
            diagnostic_sample(value)

    def test_split_doorbell_must_match_actual_request_window(self):
        value = sample()
        head = change_post_count(value, 65)
        head['prefetch_services'][0].update(doorbells=2, last_doorbell_cycles=140)
        self.assertEqual(diagnostic_sample(value, post_chunk=64)['split_early_windows'], 1)
        with self.assertRaisesRegex(ValueError, 'Doorbell count'):
            diagnostic_sample(value)
        head['prefetch_services'][0]['doorbells'] = 1
        result = diagnostic_sample(value, post_chunk=64)
        self.assertEqual(result['split_early_windows'], 0)
        self.assertEqual(result['grouped_fallback_windows'], 1)

    def test_bs1_native_window_count_and_grouped_doorbells(self):
        value = sample(1)
        head = change_post_count(value, 128)
        result = diagnostic_sample(value, batch=1, post_chunk=64)
        self.assertEqual(result['early_doorbells'], 1)
        self.assertEqual(result['split_early_windows'], 0)
        self.assertEqual(result['grouped_fallback_windows'], 0)
        head['prefetch_services'][0]['doorbells'] = 2
        with self.assertRaisesRegex(ValueError, 'Doorbell count'):
            diagnostic_sample(value, batch=1, post_chunk=64)
        change_post_count(value, 129)
        with self.assertRaisesRegex(ValueError, 'prefetch counters'):
            diagnostic_sample(value, batch=1)

    def test_service_lane_and_terminal_coverage_across_batch_sizes(self):
        for batch in BATCHES:
            with self.subTest(batch=batch):
                value = sample(batch)
                requests = value['algorithm']['requests']
                requests[0], requests[-1] = requests[-1], requests[0]
                head = value['streaming_cycles'][0]
                head['final_get_bandwidth']['groups'][0]['block'] = (batch - 1) * 4
                head['prefetch_services'][0]['lane'] = batch - 1
                self.assertEqual(diagnostic_sample(value, batch=batch)['early_entries'], 2)
                head['prefetch_services'][0]['lane'] = expected_geometry(batch)['service_lanes']
                with self.assertRaisesRegex(ValueError, 'Service lane/window'):
                    diagnostic_sample(value, batch=batch)

    def test_reject_replayed_sample_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raw = folder / 'diagnose/round-0'
            raw.mkdir(parents=True)
            save(raw / 'detail_samples.json', [sample(), sample()])
            with self.assertRaisesRegex(ValueError, 'independent checked diagnostic samples'):
                diagnostics(folder, 2)
            save(raw / 'detail_samples.json', [sample()])
            result = diagnostics(folder, 1)
            self.assertEqual(result['diagnostic_early_entries_mean'], 2)
            self.assertEqual(result['diagnostic_early_doorbells_mean'], 1)
            self.assertTrue((folder / 'prefetch_protocol.json').is_file())

    def test_verify_compiled_variant_options(self):
        row = dict(effective_aic=32, soc='Ascend950PR_9599', windows=2, service_lanes=2, local_counts=False,
                   service_ub_candidates=False, service_post_chunk=64)
        variant = dict(service_ub_candidates=False, service_post_chunk=64)
        self.assertTrue(verified_route(row, variant, 1))
        self.assertFalse(verified_route(row, variant, 2))
        self.assertFalse(verified_route(row, dict(variant, service_post_chunk=0), 1))

    def test_dry_run_default_and_repeat(self):
        for options, batches in (([], [16]), (['--batch', '1', '--batch', '8'], [1, 8])):
            with self.subTest(options=options), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['--dry-run', *options]), 0)
                plan = json.loads(output.getvalue())
                self.assertEqual(plan['batches'], batches)
                for batch in batches:
                    case = plan['cases'][str(batch)]
                    self.assertEqual(case['geometry']['windows'], 2 if batch == 1 else 4)
                    self.assertEqual(case['geometry']['service_lanes'], max(2, batch))
                    self.assertEqual(case['geometry']['window_cap'], 128)
                    self.assertEqual(case['geometry']['early_cap'], 256 if batch == 1 else 512)
                    cli = case['arguments']
                    self.assertEqual(cli[cli.index('--batches') + 1], str(batch))
                    self.assertEqual(cli[cli.index('--intermediate-fetches') + 1], '4')

    def test_case_worker_isolation_and_build_reuse(self):
        launches = []

        def launch(config, folder):
            launches.append((config, folder))
            folder.mkdir(parents=True)
            (folder / 'worker-result.json').write_text('{}')
            return int(config.mr25_handoff == 'ub')

        fake_modules = dict(execution=SimpleNamespace(launch=launch),
                            workloads=SimpleNamespace(configuration=lambda args: SimpleNamespace(args=args)))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, fake_modules), \
                patch.dict(os.environ, KVIP_AB_OUTPUT=directory), patch('run_a5_prefetch.git', return_value=KVAF), \
                patch('run_a5_prefetch.summarize_batch', return_value=[]), \
                patch('run_a5_prefetch.summarize', return_value=False), redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--batch', '1', '--batch', '8']), 1)
            self.assertEqual(len(launches), 6)
            self.assertEqual([folder.parent.name for _, folder in launches], ['bs1'] * 3 + ['bs8'] * 3)
            self.assertEqual([cfg.handoff_build_dir for cfg, _ in launches[:3]],
                             [cfg.handoff_build_dir for cfg, _ in launches[3:]])

    def test_matrix_stops_after_an_unchecked_worker(self):
        launches = []

        def launch(config, folder):
            launches.append((config, folder))
            return 1

        fake_modules = dict(execution=SimpleNamespace(launch=launch),
                            workloads=SimpleNamespace(configuration=lambda args: SimpleNamespace(args=args)))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, fake_modules), \
                patch.dict(os.environ, KVIP_AB_OUTPUT=directory), patch('run_a5_prefetch.git', return_value=KVAF), \
                patch('run_a5_prefetch.summarize_batch', return_value=[]), \
                patch('run_a5_prefetch.summarize', return_value=False), redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--batch', '1', '--batch', '8']), 1)
            self.assertEqual(len(launches), 1)
            manifest = json.loads(next(Path(directory).glob('*/manifest.json')).read_text())
            self.assertEqual(manifest['stopped_after']['batch'], 1)
            self.assertEqual(manifest['stopped_after']['variant'], 'baseline')

    def test_bs1_grouped_and_ub_fallback_remain_checked_comparisons(self):
        variants = dict(baseline=dict(service_ub_candidates=False, service_post_chunk=0),
                        ub=dict(service_ub_candidates=True, service_post_chunk=0),
                        batch=dict(service_ub_candidates=False, service_post_chunk=64))
        records = {mode: dict(variant=mode, accepted=True, reason='', effective_aic=32,
                              soc='Ascend950PR_9599', windows=2, service_lanes=2, local_counts=False,
                              fixture_sha256='matched', kv_ready_us=2, library_sha256='built',
                              prefetch_observed=True, final_get_requests_per_call=1, **options)
                   for mode, options in variants.items()}
        observed = dict(diagnostic_ub_early_post_fraction_mean=0, diagnostic_split_early_windows_mean=0,
                        diagnostic_duplicate_early_posts_mean=0, diagnostic_duplicate_final_posts_mean=0,
                        diagnostic_early_final_overlap_mean=0)
        completed = dict.fromkeys(records, 0)
        with patch('run_a5_prefetch.collect', side_effect=lambda _, mode, batch: records[mode].copy()), \
                patch('run_a5_prefetch.diagnostics', return_value=observed), \
                patch('run_a5_prefetch.completed_reads', return_value={(0, 0, 0): ([128, 0], [1, 0, 0, 0])}), \
                redirect_stdout(io.StringIO()):
            baseline, ub, batched = summarize_batch(Path('/unused'), variants, completed, 1, 1)
            self.assertTrue(all(row['protocol_verified'] for row in (baseline, ub, batched)))
            self.assertEqual(ub['matched_total_kv_ready_ratio'], 1)
            self.assertFalse(ub['optimization_exercised'])
            self.assertFalse(batched['optimization_exercised'])
            self.assertIn('Effective W2', batched['effect_note'])
            records['batch']['fixture_sha256'] = 'different'
            self.assertIsNone(summarize_batch(Path('/unused'), variants, completed, 1, 1)[2]
                              ['observed_kv_ready_ratio'])
            observed['diagnostic_early_final_overlap_mean'] = 1
            rejected = summarize_batch(Path('/unused'), variants, completed, 1, 1)[1]
            self.assertFalse(rejected['protocol_verified'])
            self.assertIsNone(rejected['observed_kv_ready_ratio'])


if __name__ == '__main__':
    unittest.main()

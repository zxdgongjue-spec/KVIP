# SPDX-License-Identifier: Apache-2.0
"""Redraw the latest checked MR28 capture from the shared A5 results."""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import mkdtemp

from lifetch_msprof_timeline import ROUTE_PHASES, device_business_view, events, verified_vendor_timeline

SUPPORT = Path(__file__).resolve().parents[2] / 'tests/operators/lifetch/support'
sys.path.insert(0, str(SUPPORT))
from baseline_stage_trace import decode, render, verify_route  # noqa: E402
from reporting import atomic_json  # noqa: E402

LEGACY_ROUTES = {'main_p2_parallel_terminal': 'p2', 'dedicated_service_parallel_terminal': 'dedicated'}


def read_json(path):
    return json.loads(path.read_text())


def capture_arm(directory, manifest):
    if (manifest.get('status') != 'PASS' or manifest.get('worker_exit_code') != 0
            or manifest.get('business_status', 'PASS') != 'PASS' or not manifest.get('msprof_timeline')):
        raise ValueError(f'Incomplete marked capture in {directory.name}')
    reports = list(directory.glob('worker/miss*/case-b*/baseline_stage_report.json'))
    if len(reports) != 1:
        raise ValueError(f'Expected one checked stage report in {directory.name}')
    report_path = reports[0]
    report = read_json(report_path)
    if report.get('correctness') != 'PASS' or report.get('execution') != 'COMPLETED':
        raise ValueError(f'Unverified source replay in {directory.name}')
    route = report.get('cold_route') or LEGACY_ROUTES.get(manifest.get('route'), manifest.get('route'))
    if route not in ROUTE_PHASES:
        raise ValueError(f'Unknown saved route {route}')
    raw = report_path.with_name('baseline_stage_trace.bin')
    native = Path(manifest['msprof_timeline'])
    if not native.is_file() or not native.resolve().is_relative_to(directory):
        native = directory / 'lifetch_msprof_trace.json'
    if not raw.is_file() or not native.is_file():
        raise ValueError(f'Source binary or native timeline missing in {directory.name}')
    return dict(route=route, batch=int(report_path.parent.name.removeprefix('case-b')),
                counter_mhz=float(report['counter_mhz']), score_parts=report.get('hot_parts', 2),
                raw=raw, native=native)


def render_arm(arm):
    route = arm['route']
    decoded = decode(arm['raw'].read_bytes())
    verify_route(decoded, arm['batch'], cold_route=route, score_parts=arm['score_parts'])
    source, _ = render(decoded, arm['counter_mhz'], cold_route=route)
    _, vendor, intervals = verified_vendor_timeline([arm['native']], cold_route=route)
    device = device_business_view(vendor, intervals)
    # Keep only the derived tracks in the new device file. The capture still holds the original events.
    device_tracks = dict(traceEvents=device['traceEvents'][len(events(vendor)):],
                         kvip_inter_kernel_handoffs=device['kvip_inter_kernel_handoffs'])
    return source, device_tracks


def latest_complete(root, route_filter):
    manifests = []
    for path in root.glob('mr28.*/manifest.json'):
        try:
            manifests.append((path.stat().st_mtime_ns, path))
        except FileNotFoundError:
            continue
    for _, path in sorted(manifests, reverse=True):
        directory = path.parent.resolve()
        try:
            manifest = read_json(path)
            if manifest.get('status') != 'PASS':
                continue
            routes = manifest.get('routes')
            if routes:
                planned = manifest.get('planned_routes') or tuple(routes)
                if set(planned) != set(routes) or not set(planned) <= ROUTE_PHASES.keys():
                    raise ValueError('Comparison lacks its planned routes')
                arms = [capture_arm(directory / route, routes[route]) for route in planned]
                if [arm['route'] for arm in arms] != list(planned):
                    raise ValueError('Saved comparison route differs from its stage report')
            else:
                arms = [capture_arm(directory, manifest)]
            selected = [arm for arm in arms if route_filter is None or arm['route'] == route_filter]
            if selected:
                return directory, [(arm, *render_arm(arm)) for arm in selected]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            print(f'Skipping {directory.name}: {error}', file=sys.stderr)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=ROUTE_PHASES, help='only redraw the latest complete run for this route')
    args = parser.parse_args(argv)
    a5_root = Path(os.environ.get('KVIP_A5_ROOT') or '/home/w00568585/li-aligned-a5')
    root = Path(os.environ.get('KVIP_AB_OUTPUT') or a5_root / 'results').resolve()
    result = latest_complete(root, args.route)
    if result is None:
        parser.exit(1, f'No complete MR28 source and device capture under {root}\n')
    capture, arms = result
    output = Path(mkdtemp(prefix='handoff-redraw.', dir=capture))
    print(f'Capture: {capture}')
    print(f'Redraw output: {output}')
    for arm, source, device in arms:
        destination = output if len(arms) == 1 else output / arm['route']
        source_handoffs = source['metadata']['source_handoffs']
        device_handoffs = device['kvip_inter_kernel_handoffs']
        atomic_json(destination / 'lifetch_business_activity.json', source)
        atomic_json(destination / 'lifetch_msprof_activity.json', device)
        atomic_json(destination / 'handoff_source.json', dict(
            cold_route=arm['route'], counter_mhz=arm['counter_mhz'], clock=source['metadata']['clock'],
            source_business_handoffs=source_handoffs))
        atomic_json(destination / 'handoff_device.json', dict(
            cold_route=arm['route'], clock='CANN device timestamps in microseconds',
            native_device_handoffs=device_handoffs))
        print(f"{arm['route']}: {destination / 'lifetch_business_activity.json'}")
        print(f"{arm['route']} device tasks: {destination / 'lifetch_msprof_activity.json'}")
        print(f"{arm['route']} reports: {destination / 'handoff_source.json'}, {destination / 'handoff_device.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

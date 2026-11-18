"""Report measured kernel intervals, independently of the headline timing."""
import json
import sys
from pathlib import Path


def main(path):
    data = json.loads(path.read_text())
    if data['status'] != 'pass':
        raise ValueError('Failed replay')
    metrics = {}
    for name in ('reverse', 'reverse-inc-register', 'reverse-inc-light-ties', 'reverse-inc-segments'):
        entry, = [e for e in data['cases'][name]['system_profiles']['passes'] if e['metric'] == 'none']
        t = entry.get('timeline') or {}
        if not all(t.get(k) for k in ('device', 'li', 'streaming')):
            raise ValueError(f"Missing kernel intervals for {name} in {entry['directory']}: {t.get('reason')}")
        metrics[name] = {k: t[k]['span_us'] for k in ('device', 'li', 'streaming')}
        print('KERNEL ' + name + ' ' + ' '.join(f'{k}_us={v:.3f}' for k,v in metrics[name].items()))
    baseline, old, light, new = metrics.values()
    print(f"LI_DELTA old_extra_us={old['li']-baseline['li']:+.3f} "
          f"new_extra_us={new['li']-baseline['li']:+.3f} "
          f"saved_vs_old_us={old['li']-new['li']:+.3f} "
          f"saved_vs_light_us={light['li']-new['li']:+.3f} "
          f"streaming_saved_vs_reverse_us={baseline['streaming']-new['streaming']:+.3f}")


if __name__ == '__main__':
    main(Path(sys.argv[1]))

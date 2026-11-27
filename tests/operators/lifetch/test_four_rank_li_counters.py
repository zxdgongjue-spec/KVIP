# SPDX-License-Identifier: Apache-2.0
import ctypes as ct
import csv
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import four_rank_li_counters as pmu


@pytest.mark.parametrize("fails", (False, True))
def test_profiler_lifecycle_calls_only_once(tmp_path, monkeypatch, fails):
    calls = []
    def stub(name, rc=0):
        def op(*args):
            calls.append(name)
            if name == "device":
                ct.cast(args[0], ct.POINTER(ct.c_int))[0] = 0
            if name == "create":
                assert (args[0]._obj.value, *args[1:]) == (0, 1, 1, None, 6)
            return rc
        return op
    api = SimpleNamespace(aclrtGetDevice=stub("device"), aclprofInit=stub("init"),
                          aclprofCreateConfig=stub("create", 123), aclprofStart=stub("start"),
                          aclprofStop=stub("stop"), aclprofFinalize=stub("finish"),
                          aclprofDestroyConfig=stub("destroy"))
    monkeypatch.setattr(pmu.ct, "CDLL", lambda _: api)
    def invoke():
        calls.append("forward")
        if fails:
            raise RuntimeError("synchronize")
    if fails:
        with pytest.raises(RuntimeError, match="synchronize"):
            pmu.capture(tmp_path / "pmu", invoke)
    else:
        pmu.capture(tmp_path / "pmu", invoke)
    assert calls == ["device", "init", "create", "start", "forward", "stop", "destroy", "finish"]


def test_counter_table_preserves_units_and_reports_missing_data(tmp_path):
    path = tmp_path / "op_summary_0.csv"
    def write(vector):
        with path.open("w") as out:
            rows = csv.writer(out)
            rows.writerow(["Op Name", "Task ID", "aic_mte2_time(us)", "aic_mac_ratio", "aiv_vec_ratio"])
            for s in pmu.SYMBOLS.values():
                rows.writerow([s, "123", "4.5", "0.4", vector])
            rows.writerow(["v9_li_large_four_ranks_trace", "124", "1", "2", "3"])
    write("15%")
    complete = pmu.read_pipe_rows(tmp_path)
    assert complete["status"] == "COMPLETE" and len(complete["stages"]["LI23"]) == 1
    assert complete["stages"]["LI1"][0]["counters"]["aiv_vec_ratio"] == "15%"
    write("N/A")
    assert pmu.read_pipe_rows(tmp_path)["status"] == "PARTIAL"

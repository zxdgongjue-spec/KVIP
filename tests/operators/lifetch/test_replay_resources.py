# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from execution import create_kvaf_runtime


@pytest.mark.parametrize(
    "transport,workers,get_jettys,expected_put",
    [("urma", 2, 2, 2), ("urma", 40, 2, 2), ("urma", 32, 4, 2),
     ("urma", 32, 5, 2), ("ubmem", 2, None, None)],
)
def test_replay_put_and_get_have_separate_jetty_budgets(transport, workers, get_jettys, expected_put):
    sdk = SimpleNamespace(
        PutExecutionConfig=lambda *args: args, GetExecutionConfig=lambda *args: args,
        ExecutionResources=dict, NpuRuntime=dict,
    )
    extra = {"get_jettys": get_jettys} if get_jettys in (4, 5) else {}
    runtime = create_kvaf_runtime(sdk, "get.o", "put.o", 656, 128, 0, workers, transport, **extra)
    for path, budget in (("put", expected_put), ("get", get_jettys)):
        (entry,) = runtime[path + "_resources"]
        config, resources = entry
        assert config == runtime["default_" + path]
        expected = {"cores": workers, "kernel_path": path + ".o"}
        if budget is not None:
            expected["jettys"] = budget
        assert resources == expected

from device_trace import account_native


def events():
    return [dict(name='replay', ph='X', ts=100, dur=100),
            dict(name='QuantLightningIndexer_1', ph='X', cat='kernel', ts=110, dur=20),
            dict(name='a5_lidu_fused_batch_get_prepare_mtp_2', ph='X', cat='kernel', ts=135, dur=10),
            dict(name='KvafExecuteBatchGetAivKernelV2_3', ph='X', cat='kernel', ts=150, dur=30)]


def test_native_names_and_gaps_are_accounted_without_ordinal_guessing():
    result = account_native(events(), 'replay', 80)
    assert 'error' not in result
    assert result['stages_us'] == dict(li=20, lidu_prepare=10, fetch=30)
    assert result['kernel_span_us'] == 70
    assert result['inter_kernel_gap_us'] == 10
    assert result['boundary_residual_us'] == 10


def test_unknown_native_kernel_remains_visible_and_incomplete():
    trace = events()
    trace[-1]['name'] = 'DifferentVendorGet'
    result = account_native(trace, 'replay', 80)
    assert 'error' in result
    assert result['missing_stages'] == ['fetch']
    assert result['unclassified'][0]['name'] == 'DifferentVendorGet'


def test_cpu_launch_names_are_not_device_timings():
    trace = events()
    trace[-1]['cat'] = 'cpu_op'
    assert 'error' in account_native(trace, 'replay', 80)

"""Adapt only the generated terminal copy of QLIVector for the live caller."""
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Private TopK source site changed: {old[:100]}')
    return text.replace(old, new)


def adapt(directory: Path):
    methods = directory / 'indexer/incremental_methods.h'
    text = methods.read_text()
    # The public Context remains unchanged: intermediate windows still consume
    # its configured threshold and budget. Only the private final invocation
    # requests all final misses, including a single Q with > windowLimit/4.
    old = ('uint32_t count=kvip_prefetch_rank::SelectBitmap(indicesOutLocal_,scoreOutLocal_,compact,\n'
           '        topkSharedTmpLocal_[128],covered,runtimeShape_.history,valid,rank,ctx->windowLimit/4,rankTrace_?rankClocks:nullptr,(ctx->windowLimit/4)/2,windowCoveredEmpty_);')
    new = ('const bool terminal=rankedEpoch_==runtimeShape_.rounds;\n'
           '    const uint32_t limit=terminal?2048U:ctx->windowLimit/4;\n'
           '    uint32_t count=kvip_prefetch_rank::SelectBitmap(indicesOutLocal_,scoreOutLocal_,compact,\n'
           '        topkSharedTmpLocal_[128],covered,runtimeShape_.history,valid,terminal?2048U:rank,limit,rankTrace_?rankClocks:nullptr,terminal?0U:limit/2,windowCoveredEmpty_);')
    methods.write_text(replace_once(text, old, new))
    vector = directory / 'indexer/indexer_vector.h'
    text = vector.read_text()
    marker = '    __aicore__ inline void TerminalLogical('
    if text.count(marker) != 1:
        raise ValueError('Expected generated terminal methods')
    helper = Path(__file__).with_name('topk_methods.h').read_text()
    vector.write_text(text.replace(marker, helper + '\n' + marker))

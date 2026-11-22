"""Copy and print failed state only, outside every measured device interval."""
import json

FIELDS = ('combined', 'topk', 'pack', 'index', 'cq', 'prefetch', 'total_slots',
          'index_begin', 'index_count', 'posted', 'pending', 'generation',
          'bad_ordinal', 'index_reason_bits', 'source', 'destination', 'previous_ordinal')


def decode(errors, status):
    return [dict(request=b, status=value,
                 queries=[dict(q=q+1, **dict(zip(FIELDS, record)))
                          for q, record in enumerate(errors[b])])
            for b, value in enumerate(status) if value]


def save_failure(output, forward, rotating, status):
    errors = forward.query_errors.cpu()
    flags = forward.binding.flags.cpu()
    posts = rotating.final_posts.cpu()
    report = decode(errors.tolist(), status)
    for row in report:
        for query in row['queries']:
            bits = query['index_reason_bits']
            query['index_reasons'] = ([name for bit, name in (
                (1, 'source_below_resident'), (2, 'source_outside_history'),
                (4, 'destination_outside_resident'), (8, 'duplicate_source'),
                (16, 'duplicate_destination')) if bits & bit]
                if bits != 0xffffffff else [])
    (output / 'failure_details.json').write_text(json.dumps(report, indent=2)+'\n')
    forward.torch.save(dict(query_errors=errors, flags=flags, final_posts=posts,
                           row_status=rotating.row_status.cpu()), output / 'terminal_failure.pt')
    print('TERMINAL FAILURE: '+json.dumps(report), flush=True)

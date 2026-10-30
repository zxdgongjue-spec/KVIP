"""Host admission before ownership changes or queue submissions.

TODO: uniform Q2/MTP1 and Q3/MTP2; unequal candidate lengths with uniform Q.
No mixed-query-count batches are planned. Device ABI arrays remain per request.
"""


def validate_request(*, queries_per_request, candidate_lengths, resident_budgets):
    if queries_per_request != 4:
        raise ValueError("V9 currently requires uniform MTP3/Q4; MTP1/2 are TODO")
    lengths, budgets = tuple(candidate_lengths), tuple(resident_budgets)
    if not lengths or len(lengths) != len(budgets):
        raise ValueError("candidate lengths and resident budgets need one entry per request")
    if len(set(lengths)) != 1 or len(set(budgets)) != 1:
        raise ValueError("V9 currently requires equal candidate lengths and resident budgets")
    if any(v < 0 for v in lengths) or any(v < 2048 or v > 16384 for v in budgets):
        raise ValueError("invalid candidate length or resident budget")

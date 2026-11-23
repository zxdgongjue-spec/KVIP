"""Validate a candidate against the archived B24 strong-serial-equivalent score multisets."""

from hashlib import sha256

import numpy as np

from correctness import topk_fingerprint
from native_li_reference import native_ends


# main-five-tail.tgz contains the five strong serial score-equivalence receipts
# for main 609fa26f's NPU4 replay. main-tail-pages.tgz supplies the corresponding
# 20 raw measured logical-ID tensors, captured layouts and causal bounds.
# Map the eight-window audit's original physical uint16 score tensors through
# those layouts, require all twenty selected score vectors per Q to equal the
# independent highest 2048 valid scores, then hash one per-query sorted copy.
# main-five-tail.tgz SHA-256: e096137f4396e7bd5e307aca67afedee69b25468eabc8fb4b606e511d872c198
# main-tail-pages.tgz SHA-256: 4d31bda95fe39bc4b4be8877258075124607ae4cd74fe8fc141150d782cfcd08
# Columns: logical score domain digest, winner score digest.
ARCHIVE = {
    10: (
        "f105e6052bc78d2b442faf8e9e3db310430e8857d14382af62634ad64a024fc7",
        "44dcca9ad536842df5197217c1457eadb2e5b6bfe34f9133c8a4bf4da84342fd",
    ),
    4: (
        "439425aa23ac695b7a66307f48b3b471383dc8e4f576ca40f5706810f6574c6d",
        "9be606488c0ad57526e6ac575ea3b206c5b2ec309fe1d177787baa715a6707f2",
    ),
    17: (
        "0b33cc178900c99c52465b4912436f7b8ac85b1ba35186c403cfbcb61c540f37",
        "fa8440f6f837cbffe6cfeaa6f0d9a856333bffd4e6cf39108135c93e969dcd26",
    ),
    7: (
        "9163349e5a1fd7cecb0f82bc947ab10f8182ee7d92bfae3675abbdb7ef543614",
        "96a13adba0830389e02af4e00b4ae7d1067064cddd0f6b1850c4c3662c1586d0",
    ),
    2: (
        "ae1405546c1db2082060fa57994132f6a5a9ecacfe724cf7e3cc809a7c7b9fbc",
        "021162d4eed9bede0335fde9c68a0e3b0ea443c8a5720a4cdd6f3be002ad51ed",
    ),
}


# Hot validity, LRU slot ordering and resident tail reservation from the same
# B24 request snapshot. Physical transport page addresses may change by host.
REQUEST_STATE = {
    10: "7b5132a0cba556ecfcb025c3aa47e7ec114368e4fefaf470ad441e902d01a42e",
    4: "41b9607113d7b7b15d76d4da32b41d56f45211f3fc4dd6fe3ae0388b4b3c2271",
    17: "ec4806d5ba7641204d1053871cb425425f3fcb1d926012511c7de1a61b2e3b46",
    7: "154355d8bf881b87ffd938349950fd6d6abb414d841119c203cc971ff127226a",
    2: "29a69b19bc9ec75d47c1499e1f6c10cd62226ed491456d5a705c420792f9ee81",
}


def request_state_digest(inputs):
    batch, resident = inputs.geometry["batch"], inputs.geometry["resident"]
    hot = np.asarray(inputs.captured_hot_valid)
    lru = np.asarray(inputs.captured_lru)
    tail = np.asarray(inputs.tail_bounds)
    if (
        hot.shape != (batch, resident)
        or hot.dtype.kind not in "biu"
        or np.any((hot != 0) & (hot != 1))
        or lru.shape != (batch, 2048)
        or lru.dtype.kind not in "iu"
        or np.any(lru < 0)
        or np.any(lru >= resident)
        or tail.shape != (batch, 2)
        or tail.dtype.kind not in "iu"
        or np.any(tail < 0)
        or np.any(tail > inputs.geometry["history"])
        or np.any(tail[:, 1] < tail[:, 0])
    ):
        raise ValueError("Invalid captured request state")
    digest = sha256()
    for values, dtype in ((hot, "u1"), (lru, "<i4"), (tail, "<i4")):
        digest.update(np.ascontiguousarray(values, dtype=dtype).tobytes())
    return digest.hexdigest()


def domain_digest(inputs):
    layout = np.asarray(inputs.captured_layout)
    ends = np.asarray(inputs.query_ends)
    scores = np.asarray(inputs.logical_score_keys)
    valid = np.asarray(inputs.logical_score_valid)
    history = inputs.geometry["history"]
    batch = inputs.geometry["batch"]
    if (
        layout.shape != (batch, history)
        or layout.dtype.kind not in "iu"
        or np.any(layout < 0)
        or np.any(layout >= history)
        or ends.shape != (batch, 4)
        or ends.dtype.kind not in "iu"
        or np.any(ends < 2048)
        or np.any(ends > history)
        or scores.shape != (batch, 4, history)
        or scores.dtype.kind != "u"
        or scores.dtype.itemsize != 2
        or valid.shape != scores.shape
        or valid.dtype != np.bool_
    ):
        raise ValueError("Invalid capture layout, causal bounds, uint16 scores or validity mask")
    digest = sha256()
    digest.update(np.ascontiguousarray(ends, dtype="<i4").tobytes())
    digest.update(np.ascontiguousarray(layout, dtype="<i4").tobytes())
    digest.update(np.packbits(valid, axis=-1, bitorder="little").tobytes(order="C"))
    # Invalid score words are outside the population and may contain uninitialized data.
    digest.update(np.where(valid, scores, 0).astype("<u2", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def preflight(group, inputs):
    if group not in ARCHIVE or inputs.geometry["batch"] != 16:
        raise ValueError(f"Unsupported portable score oracle input for G{group}")
    actual = domain_digest(inputs)
    if actual != ARCHIVE[group][0]:
        raise ValueError(f"G{group}: capture score domain differs from the B24 archive ({actual})")
    if request_state_digest(inputs) != REQUEST_STATE[group]:
        raise ValueError(f"G{group}: captured request state differs from the B24 archive")
    return actual


def score_digest(scores, ids):
    chosen = np.take_along_axis(scores, ids, axis=-1)
    return sha256(np.ascontiguousarray(np.sort(chosen, axis=-1), dtype="<u2").tobytes()).hexdigest()


def verify(group, inputs, outputs):
    domain = preflight(group, inputs)
    outputs = np.asarray(outputs)
    if outputs.shape != (20, 16, 4, 2048) or outputs.dtype.kind not in "iu":
        raise ValueError("Expected 20 measured 16-row Q4 logical integer TopK outputs")
    ends = native_ends(inputs)[..., None]
    reference = ARCHIVE[group][1]
    ids_digests = []
    for call, ids in enumerate(outputs, 1):
        # FusedReplayState has already decoded any tagged physical IDs to public
        # logical IDs. Never strip flags from a malformed public ID.
        if np.any(ids < 0) or np.any(ids >= ends):
            raise ValueError(f"G{group} call {call}: token exceeds the native logical query bounds")
        inputs.validate_topk(ids)
        if score_digest(inputs.logical_score_keys, ids) != reference:
            raise ValueError(
                f"G{group} call {call}: winner score multisets differ from the strong serial archive"
            )
        ids_digests.append(topk_fingerprint(ids))
    return dict(
        group=group,
        calls=20,
        queries=1280,
        score_multiset_sha256=reference,
        score_domain_sha256=domain,
        request_state_sha256=REQUEST_STATE[group],
        topk_set_digests=ids_digests,
    )

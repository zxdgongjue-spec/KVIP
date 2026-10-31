"""Deterministic packed-C8 payload; also usable by the CPU-only lifecycle model."""

import numpy as np
from kvip.v9.contract import TOKEN_BYTES


def payload(pool_layer, tokens):
    tokens = np.asarray(tokens, dtype=np.int64).reshape(-1, 1)
    columns = np.arange(512, dtype=np.int64)[None, :]
    code = np.asarray([0xC4, 0xC0, 0xB8, 0, 0x38, 0x40, 0x44], np.uint8)
    nope = code[((tokens >> ((columns // 8) % 18)) + columns + pool_layer * 3) % 7]
    rope_columns = np.arange(64, dtype=np.int64)[None, :]
    rope = 0x3E80 | (((tokens >> ((rope_columns // 4) % 18)) + rope_columns + pool_layer * 11) & 127)
    rope = rope.astype("<u2").view(np.uint8)
    scales = np.broadcast_to(np.asarray([0.03125, 0.0625, 0.125, 0.25], "<f4"), (len(tokens), 4)).copy()
    return np.concatenate((nope, rope, scales.view(np.uint8)), axis=1).reshape(-1, TOKEN_BYTES)


def unpack(packed):
    packed = np.asarray(packed, np.uint8).reshape(-1, TOKEN_BYTES)
    raw = packed[:, :512]
    sign = np.where(raw & 128, -1.0, 1.0)
    exponent = ((raw >> 3) & 15).astype(np.int32)
    mantissa = raw & 7
    values = sign * np.where(exponent == 0, mantissa / 8.0 * 2.0**-6, (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    scales = packed[:, 640:].copy().view("<f4")
    nope = (values * np.repeat(scales, 128, axis=1)).astype(np.float32)
    rope = (packed[:, 512:640].copy().view("<u2").astype(np.uint32) << 16).view(np.float32)
    return np.concatenate((nope, rope), axis=1), nope

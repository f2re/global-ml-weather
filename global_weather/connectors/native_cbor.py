"""Bounded reader for definite-length CBOR emitted by nlohmann::json.

Metadata only, no object construction, tagged objects, network or code execution.
Unsupported encodings fail closed; this is not a general-purpose CBOR library.
"""
import math
import struct


def loads(data: bytes, *, max_bytes=8*1024*1024, max_items=200_000, max_depth=32):
    if not isinstance(data, bytes) or len(data) > max_bytes:
        raise ValueError('CBOR size/type limit.')
    pos, count = 0, 0

    def take(n):
        nonlocal pos
        if n < 0 or n > len(data)-pos:
            raise ValueError('Truncated CBOR.')
        value = data[pos:pos+n]; pos += n
        return value

    def read(depth):
        nonlocal count
        count += 1
        if depth > max_depth or count > max_items:
            raise ValueError('CBOR nesting/item limit.')
        first = take(1)[0]; major, arg = first >> 5, first & 31
        if major == 7:
            if arg in (20, 21, 22): return {20: False, 21: True, 22: None}[arg]
            if arg in (25, 26, 27):
                value = struct.unpack({25: '>e', 26: '>f', 27: '>d'}[arg], take({25: 2, 26: 4, 27: 8}[arg]))[0]
                if not math.isfinite(value): raise ValueError('Nonfinite CBOR number.')
                return value
            raise ValueError('Unsupported CBOR simple value.')
        if major == 6 or arg >= 28:
            raise ValueError('CBOR tags/indefinite lengths are not admitted.')
        n = arg if arg < 24 else int.from_bytes(take(1 << (arg-24)), 'big')
        if major == 0: return n
        if major == 1: return -1-n
        if major == 2: raise ValueError('Binary CBOR values are not admitted as product metadata.')
        if major == 3: return take(n).decode('utf-8')
        if n > max_items-count: raise ValueError('CBOR item limit.')
        if major == 4: return [read(depth+1) for _ in range(n)]
        if major == 5:
            out = {}
            for _ in range(n):
                key = read(depth+1)
                if not isinstance(key, str) or key in out:
                    raise ValueError('CBOR map needs unique string keys.')
                out[key] = read(depth+1)
            return out
        raise ValueError('Unsupported CBOR type.')

    value = read(0)
    if pos != len(data) or not isinstance(value, dict):
        raise ValueError('Expected exactly one CBOR metadata map.')
    return value

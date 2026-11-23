#!/usr/bin/env python3
"""Check executable sections of the embedded Ascend device images in three linked bridges."""
import argparse
import hashlib
from pathlib import Path
import struct
import sys


HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
SECTION = struct.Struct("<IIQQQQIIQQ")
KERNEL_TAGS = {"parallel_terminal_kernel": {1}, "v9_kernel": {0, 1}}


def block(data, offset, size):
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError("ELF or image field extends beyond its enclosing file")
    return data[offset:offset + size]


def sections(data, machine, file_type):
    h = HEADER.unpack(block(data, 0, HEADER.size))
    ident = h[0]
    if ident[:7] != b"\x7fELF\x02\x01\x01" or h[1] != file_type or h[3] != 1 or (machine is not None and h[2] != machine):
        raise ValueError(f"expected ELF64 little-endian type={file_type} machine={machine}")
    offset, flags, entry_size, count, names_idx = h[6], h[7], h[11], h[12], h[13]
    if entry_size != SECTION.size or not 0 < names_idx < count:
        raise ValueError("unsupported ELF section table")
    headers = list(SECTION.iter_unpack(block(data, offset, count * entry_size)))
    name_header = headers[names_idx]
    if name_header[1] != 3:
        raise ValueError("invalid section name string table")
    names = block(data, name_header[4], name_header[5])
    result = []
    for header in headers:
        name_offset = header[0]
        end = names.find(b"\0", name_offset)
        if end < 0:
            raise ValueError("invalid section name offset")
        result.append((names[name_offset:end].decode("ascii"), header))
    return flags, h[4], result


def extract(path):
    data = path.read_bytes()
    # The host architecture does not affect how ELF wraps the device images.
    _, _, host = sections(data, machine=None, file_type=3)
    found = {}
    for name, header in host:
        parts = name.split(".")
        if len(parts) != 5 or parts[:3] != ["", "ascend", "kernel"] or parts[4] not in KERNEL_TAGS:
            continue
        arch, kernel = parts[3:]
        if kernel in found or header[1] != 1:
            raise ValueError(f"{path}: duplicate or invalid {name}")
        payload = block(data, header[4], header[5])
        version, count = struct.unpack("<II", block(payload, 0, 8))
        if version != 1 or count != len(KERNEL_TAGS[kernel]):
            raise ValueError(f"{path}: unexpected image table for {name}")
        offset = 8
        tags = set()
        instructions = {}
        for _ in range(count):
            tag, disk_size, raw_size = struct.unpack("<III", block(payload, offset, 12))
            offset += 12
            if tag not in KERNEL_TAGS[kernel] or tag in tags or disk_size != raw_size:
                raise ValueError(f"{path}: unexpected core tag or image encoding in {name}")
            tags.add(tag)
            image = block(payload, offset, disk_size)
            offset += disk_size
            flags, entry, image_sections = sections(image, machine=0x1029, file_type=2)
            exec_count = 0
            for section_name, sh in image_sections:
                if not (sh[2] & 4) or not sh[5]:
                    continue
                if sh[1] != 1:
                    raise ValueError(f"{path}: unexpected executable section type in {name}")
                key = (arch, tag, flags, entry, section_name, sh[3])
                if key in instructions:
                    raise ValueError(f"{path}: duplicate executable section {key}")
                instructions[key] = block(image, sh[4], sh[5])
                exec_count += 1
            if not exec_count:
                raise ValueError(f"{path}: missing executable code in {name} tag={tag}")
        if offset != len(payload):
            raise ValueError(f"{path}: unparsed trailing data in {name}")
        found[kernel] = instructions
    if found.keys() != KERNEL_TAGS.keys():
        raise ValueError(f"{path}: missing device kernel(s) {sorted(KERNEL_TAGS.keys() - found.keys())}")
    return found


def compare(label, wanted, actual):
    if wanted.keys() != actual.keys():
        raise ValueError(f"{label}_CODE_MISMATCH executable section identities differ")
    for key in sorted(wanted):
        if wanted[key] != actual[key]:
            raise ValueError(f"{label}_CODE_MISMATCH {key}: reference={len(wanted[key])} bytes fixed={len(actual[key])} bytes")
    digests = [f"tag{key[1]}/{key[4]}={len(code)}B,{hashlib.sha256(code).hexdigest()}"
               for key, code in sorted(wanted.items())]
    return f"{label}_CODE_MATCH " + " ".join(digests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("main", "prior", "fixed"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    main_lib, prior_lib, fixed_lib = (extract(path) for path in (args.main, args.prior, args.fixed))
    tail = "parallel_terminal_kernel"
    li = "v9_kernel"
    print_lines = [compare("TAIL", main_lib[tail], fixed_lib[tail]),
                   compare("LI", prior_lib[li], fixed_lib[li])]
    print(*print_lines, sep="\n")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, struct.error) as error:
        print(f"device check failed: {error}", file=sys.stderr)
        sys.exit(1)

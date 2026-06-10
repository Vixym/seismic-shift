#!/usr/bin/env python3
"""Truncate a Seismic documents.bin to the first N vectors.

Binary format (little-endian):
  uint32 n_vecs
  repeated n_vecs times:
    uint32 nnz
    nnz * uint32  components
    nnz * float32 values
"""
import sys
import struct

def main():
    if len(sys.argv) != 4:
        print("usage: make_subset.py <in.bin> <out.bin> <n_docs>")
        sys.exit(1)
    src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        total = struct.unpack("<I", fin.read(4))[0]
        keep = min(n, total)
        print(f"source has {total} vectors; writing {keep}")
        fout.write(struct.pack("<I", keep))
        for i in range(keep):
            hdr = fin.read(4)
            nnz = struct.unpack("<I", hdr)[0]
            body = fin.read(nnz * 8)  # nnz uint32 + nnz float32
            fout.write(hdr)
            fout.write(body)
            if (i + 1) % 200000 == 0:
                print(f"  {i+1}/{keep}")
    print("done")

if __name__ == "__main__":
    main()

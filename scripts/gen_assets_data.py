#!/usr/bin/env python3
"""
Generate a single C header that embeds binary assets into a ".assets"
linker section as named symbols (header-only output).

Behavior:
 - For .obj files: convert to compact binary blob
 - For other files: not accepted currently.
 - Output: header with numbers of static const int16_t <`.obj`_filename>[], 
           consisting the b3d_mesh_t structure data according to the C 
           source code `b3d-obj.h`, placed into section ".assets" and 
           16-byte aligned.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Iterable, List, Tuple

DEFAULT_BINARY_OUTPUT = "assets/globe-countries.bin"
DEFAULT_HEADER_OUTPUT = "examples/globe-data.h"

# Scale factor for Q15.16 fixed point
Q16 = 1 << 16

def float_to_q15_16(x: float) -> int:
    """Convert float to Q15.16 fixed point signed int32.

    Since MyCPU lacks floating point support, we convert .obj files
    to a compact binary format with Q15.16 fixed point coordinates.
    """
    q = int(round(x * Q16))
    if q > 0x7FFFFFFF:
        return 0x7FFFFFFF
    if q < -0x80000000:
        return -0x80000000
    return q

def to_hex32(x: int) -> str:
    """Output the singned integer with hex formatting."""
    return f"0x{(x & 0xFFFFFFFF):08X}"

def resolve_index(obj_index: int, vertex_count: int) -> int:
    """Convert OBJ index to 0-based Python index."""
    if obj_index > 0:
        return obj_index - 1
    else:
        return vertex_count + obj_index

def parse_obj(path: str) -> List[float]:
    """Parse OBJ and return triangle soup as flat float list:
      [ax, ay, az, bx, by, bz, cx, cy, cz, ...]
    """
    vertices: List[Tuple[float, float, float]] = []
    triangles: List[float] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            # Vertex: v x y z
            if parts[0] == "v":
                vertices.append(
                    (float(parts[1]), float(parts[2]), float(parts[3]))
                )
            # Face: f v1 v2 v3 ... (supports n-gons with fan triangulation)
            elif parts[0] == "f":
                idx = []
                for tok in parts[1:]:
                    v = int(tok.split("/")[0])
                    vi = resolve_index(v, len(vertices))
                    idx.append(vi)

                # fan triangulation
                for i in range(1, len(idx) - 1):
                    for vi in (idx[0], idx[i], idx[i + 1]):
                        x, y, z = vertices[vi]
                        triangles.extend((x, y, z))

    return triangles

def collect_obj_files(inputs: Iterable[str]) -> List[str]:
    """Load the Waveform .obj file."""
    obj_files = set()

    for inp in inputs:
        if any(c in inp for c in "*?[]"):
            for p in glob.glob(inp):
                if p.lower().endswith(".obj"):
                    obj_files.add(p)

        elif os.path.isdir(inp):
            for p in glob.glob(os.path.join(inp, "*.obj")):
                obj_files.add(p)

        elif inp.lower().endswith(".obj") and os.path.isfile(inp):
            obj_files.add(inp)

    return sorted(obj_files)

def sanitize(path: str) -> str:
    """Sanitize the symbol name from the file path
    """
    name = os.path.splitext(os.path.basename(path))[0]
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    return f"{name}"

def write_c_array(triangles: List[float], symbol: str, out) -> int:
    """Write the C array for the given triangle list

    Typically, the byte array would be struct as follows:
        typedef struct {
            int triangle_count; /* Number of triangles */
            int vertex_count;   /* Total vertex components (triangle_count * 9) */
            float *triangles;   /* Triangle vertices: 9 floats per tri (ax,ay,az,...) */
        } b3d_mesh_t;
    with triangles being in Q15.16 fixed point format.

    Therefore, the C array format is:
        int32_t triangle_count, vertex_count
        int32_t v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, 
                v2.y, v2.z, ...
    """
    tri_count = len(triangles) // 9
    vtx_components = len(triangles)

    # C array definition with alignment and linker section
    out.write(
        f"static const int32_t {symbol}[] "
        f"__attribute__((section(\".assets\"), aligned(16))) = {{\n"
    )

    # metadata (triangle_count & vertex_count)
    out.write(f"    {to_hex32(tri_count)}, {to_hex32(vtx_components)},\n")

    line = "    "
    for f in triangles:
        q = float_to_q15_16(f)
        line += f"{to_hex32(q)}, " # fan triangulated face value
        if len(line) > 78:
            out.write(line + "\n")
            line = "    "

    if line.strip():
        out.write(line + "\n")

    out.write("};\n\n")

    # Calculate the size of bytes
    byte_size = (2 + vtx_components) * 4
    return byte_size

def write_header(obj_files: List[str], output: str) -> int:
    """Write the C header file of the object assets
    """
    total_byte_size = 0
    with open(output, "w", encoding="utf-8") as out:
        # Header information and includes
        out.write("/* Auto-generated mesh assets (Q15.16) */\n")
        out.write("#pragma once\n")
        out.write("#include <stdint.h>\n\n")

        for obj in obj_files:
            print(f"[OBJ] {obj}")
            triangles = parse_obj(obj)
            symbol = sanitize(obj)
            byte_size = write_c_array(triangles, symbol, out)
            # Update total byte size
            total_byte_size += byte_size
    return total_byte_size

def main():
    p = argparse.ArgumentParser(
        description="Generate C header embedding asset object files into .assets section",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert object file(s) (.obj) to C header
  %(prog)s header -i assets/*.obj -o examples/globe-data.h
""",    
    )

    subparsers = p.add_subparsers(dest="command", help="Commands")

    # Header subcommand
    hdr_parser = subparsers.add_parser(
        "header", help="Convert Wavefront object asset to C header"
    )
    hdr_parser.add_argument(
        "-i",
        "--input", 
        nargs="+",
        help="OBJ files, directories (e.g. assets/*.obj)",
    )
    hdr_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output C header file",
    )

    args = p.parse_args()

    if args.command == "header":
        obj_files = collect_obj_files(args.input)
        if not obj_files:
            print("Error: no OBJ files found")
            sys.exit(1)

        print(f"Found {len(obj_files)} OBJ file(s)")
        total_byte_size = write_header(obj_files, args.output)
        print(f"Written: {args.output}")
        print(f"Total asset size: {total_byte_size} bytes")
    else:
        p.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
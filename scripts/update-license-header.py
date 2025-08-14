#!/usr/bin/env python3
# BEGIN LICENSE
#   SupaSim, a GPGPU and simulation toolkit.
#   Copyright (C) 2025 SupaMaggie70 (Magnus Larsson)
#
#
#   SupaSim is free software; you can redistribute it and/or
#   modify it under the terms of the GNU General Public License
#   as published by the Free Software Foundation; either version 3
#   of the License, or (at your option) any later version.
#
#   SupaSim is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
# END LICENSE

import sys
import os.path

if __name__ == "__main__":
    header_file = ""
    files_to_update: list[str] = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in ["--header", "-h"]:
            header_file = sys.argv[i + 1]
            i += 1
        else:
            files_to_update.append(sys.argv[i])
        i += 1
    header_lines = open(header_file, "r").readlines()
    header_rust = ["/* BEGIN LICENSE\n"]
    for line in header_lines:
        if len(line.strip()) != 0:
            header_rust.append("  " + line)
        else:
            header_rust.append("\n")
    header_rust.append("END LICENSE */\n")
    header_toml = ["# BEGIN LICENSE\n"]
    for line in header_lines:
        if len(line.strip()) != 0:
            header_toml.append("#   " + line)
        else:
            header_toml.append("#\n")
    header_toml.append("# END LICENSE\n")

    for fname in files_to_update:
        print(f"Editing {fname}")
        extension = os.path.splitext(fname)[1]
        if extension in [".rs", ".slang", ".wgsl"]:
            header = header_rust
        elif extension in [".toml", ".py", ".yml"]:
            header = header_toml
        else:
            continue

        with open(fname, "r") as file:
            lines: list[str] = file.readlines()

        out_parts = []
        shebang = None
        if lines and lines[0].startswith("#!"):
            shebang = lines[0]
            lines = lines[1:]

        # Remove existing license header if present
        if lines and lines[0] == header[0]:
            header_ended = False
            for line in lines[1:]:
                if not header_ended:
                    if line == header[-1]:
                        header_ended = True
                    continue
                else:
                    out_parts.append(line)
            # Remove blank lines immediately after old header
            while out_parts and out_parts[0].strip() == "":
                out_parts.pop(0)
        else:
            out_parts = lines[:]

        # Prepend shebang if it existed
        if shebang:
            final_parts = [shebang]
        else:
            final_parts = []

        # Insert new header with exactly one blank line after
        final_parts.extend(header)
        if len(out_parts) != 0 and out_parts[0] != "":
            final_parts.append("\n")
        final_parts.extend(out_parts)

        with open(fname, "w") as file:
            file.writelines(final_parts)

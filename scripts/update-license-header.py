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
        header: list[str]
        if extension == ".rs":
            header = header_rust
        elif extension in [".toml", ".py", ".yml"]:
            header = header_toml
        file = open(fname, "r")
        lines: list[str] = file.readlines()
        out_parts = []
        header_started = False
        if len(lines) > 0 and lines[0].startswith("#!"):
            out_parts.append(lines[0])
            lines = lines[1:]
        if len(lines) > 0 and lines[0] == header[0]:
            out_parts.extend(header)
            header_ended = False
            for line in lines[1:]:
                if header_ended:
                    out_parts.append(line)
                if line == header[-1]:
                    header_ended = True
        else:
            out_parts.extend(header)
            out_parts.extend(lines)
        file.close()
        file = open(fname, "w")
        file.writelines(out_parts)
        file.truncate()

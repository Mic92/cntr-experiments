#!/usr/bin/env python3

import pandas as pd
import re
import sys
import numpy as np
from pathlib import Path
from typing import Any
from natsort import natsort_keygen
import warnings

from plot import (
    apply_aliases,
    catplot,
    column_alias,
    explode,
    sns,
    PAPER_MODE,
    plt,
    format,
    magnitude_formatter,
)
from plot import ROW_ALIASES, COLUMN_ALIASES, FORMATTER

if PAPER_MODE:
    out_format = ".pdf"
else:
    out_format = ".png"

ROW_ALIASES.update(
    {
        "direction": dict(read_mean="read", write_mean="write"),
        "system": dict(
            direct_host1="native",
            direct_host2="native #2",
            direct_detached_qemublk=r"$\ast \dag$ qemu-blk",
            direct_ws_qemublk=r"$\dag$ wrap_syscall qemu-blk",
            direct_ws_javdev=r"$\ast$ wrap_syscall vmsh-blk",
            direct_iorefd_qemublk=r"$\dag$ ioregionfd qemu-blk",
            direct_iorefd_javdev=r"$\ast$ ioregionfd vmsh-blk",
            detached_qemublk=r"$\ddag$ qemu-blk",
            detached_qemu9p=r"$\ddag$ qemu-9p",
            attached_ws_javdev="wrap_syscall vmsh-blk",
            attached_iorefd_javdev="ioregionfd vmsh-blk",
        ),
        "iotype": dict(
            direct="Direct/Block IO",
            file="File IO",
        ),
        "benchmark_id": {
            "Compile Bench: Test: Compile [MB/s]": "Compile Bench: Compile",
            "Compile Bench: Test: Initial Create [MB/s]": "Compile Bench: Create",
            "Compile Bench: Test: Read Compiled Tree [MB/s]": "Compile Bench: Read tree",
            "Dbench: 1 Clients [MB/s]": "Dbench: 1 Client",
            "Dbench: 12 Clients [MB/s]": "Dbench: 12 Clients",
            "FS-Mark: Test: 1000 Files, 1MB Size [Files/s]": "FS-Mark: 1000 Files, 1MB",
            "FS-Mark: Test: 1000 Files, 1MB Size, No Sync/FSync [Files/s]": "FS-Mark: 1k Files, No Sync",
            "FS-Mark: Test: 4000 Files, 32 Sub Dirs, 1MB Size [Files/s]": "FS-Mark: 4k Files, 32 Dirs",
            "FS-Mark: Test: 5000 Files, 1MB Size, 4 Threads [Files/s]": "FS-Mark: 5k Files, 1MB, 4 Threads",
            "Flexible IO Tester: Type: Random Read - IO Engine: Linux AIO - Buffered: No - Direct: Yes - Block Size: 4KB - Disk Target: Default Test Directory [IOPS]": "Fio: Rand read, 4KB",
            "Flexible IO Tester: Type: Random Read - IO Engine: Linux AIO - Buffered: No - Direct: Yes - Block Size: 2MB - Disk Target: Default Test Directory [IOPS]": "Fio: Rand read, 2MB",
            "Flexible IO Tester: Type: Random Write - IO Engine: Linux AIO - Buffered: No - Direct: Yes - Block Size: 4KB - Disk Target: Default Test Directory [IOPS]": "Fio: Rand write, 4KB",
            "Flexible IO Tester: Type: Random Write - IO Engine: Linux AIO - Buffered: No - Direct: Yes - Block Size: 2MB - Disk Target: Default Test Directory [IOPS]": "Fio: Rand write, 2MB",
            "Flexible IO Tester: Type: Sequential Read - IO Engine: Linux AIO - Buffered: No - Direct: Yes - Block Size: 4KB - Disk Target: Default Test Directory [IOPS]": "Fio: Sequential read, 4KB",
            "Flexible IO Tester: Type: Sequential Read - IO Engine: Linux AIO - Buffered: No - Direct: Yes - Block Size: 2MB - Disk Target: Default Test Directory [IOPS]": "Fio: Sequential read, 2MB",
            "Flexible IO Tester: Type: Sequential Write - IO Engine: Linux AIO - Buffered: No - Direct: Yes - Block Size: 4KB - Disk Target: Default Test Directory [IOPS]": "Fio: Sequential write, 2KB",
            "Flexible IO Tester: Type: Sequential Write - IO Engine: Linux AIO - Buffered: No - Direct: Yes - Block Size: 2MB - Disk Target: Default Test Directory [IOPS]": "Fio: Sequential write, 2MB",
            "IOR: Block Size: 2MB - Disk Target: Default Test Directory [MB/s]": "IOR: 2MB",
            "IOR: Block Size: 4MB - Disk Target: Default Test Directory [MB/s]": "IOR: 4MB",
            "IOR: Block Size: 8MB - Disk Target: Default Test Directory [MB/s]": "IOR: 8MB",
            "IOR: Block Size: 16MB - Disk Target: Default Test Directory [MB/s]": "IOR: 16MB",
            "IOR: Block Size: 32MB - Disk Target: Default Test Directory [MB/s]": "IOR: 32MB",
            "IOR: Block Size: 64MB - Disk Target: Default Test Directory [MB/s]": "IOR: 64MB",
            "IOR: Block Size: 256MB - Disk Target: Default Test Directory [MB/s]": "IOR: 256MB",
            "IOR: Block Size: 512MB - Disk Target: Default Test Directory [MB/s]": "IOR: 512MB",
            "IOR: Block Size: 1024MB - Disk Target: Default Test Directory [MB/s]": "IOR: 1025MB",
            "PostMark: Disk Transaction Performance [TPS]": "PostMark: Disk transactions",
            "SQLite: Threads / Copies: 1 [Seconds]": "Sqlite: 1 Threads",
            "SQLite: Threads / Copies: 8 [Seconds]": "Sqlite: 8 Threads",
            "SQLite: Threads / Copies: 32 [Seconds]": "Sqlite: 32 Threads",
            "SQLite: Threads / Copies: 64 [Seconds]": "Sqlite: 64 Threads",
            "SQLite: Threads / Copies: 128 [Seconds]": "Sqlite: 128 Threads",
            "AIO-Stress: Random Write": "AIO-Stress: Random Write",
            "SQLite: Timed SQLite Insertions": "SQlite",
            "FS-Mark: 1000 Files, 1MB Size": "FS-Mark",
        },
    }
)

ROW_ALIASES["system"]["vmsh-console"] = "vmsh-console"

COLUMN_ALIASES.update(
    {
        "container_size": "image size [MB]",
        "iops": "IOPS [k]",
        "io_throughput": "Throughput [GB/s]",
        "direction": "Direction",
        "seconds": "latency [ms]",
    }
)
FORMATTER.update(
    {
        "iops": magnitude_formatter(3),
        "io_throughput": magnitude_formatter(6),
        "seconds": magnitude_formatter(-3),
    }
)


def compute_ratio(x: pd.DataFrame) -> pd.Series:
    title = x.benchmark_title.iloc[0]
    scale = x.scale.iloc[0]
    x.median = x.raw_string.map(lambda v: np.median(list(map(float, v.split(":")))))

    cntr_idx = -1
    for i, name in enumerate(x.identifier):
        if name == "cntr":
            cntr_idx = i
            break
    if cntr_idx == -1:
        raise Exception(f"no cntr for benchmark {title}")
    native = x.median.iloc[cntr_idx]

    if x.proportion.iloc[0] == "LIB":
        diff = x.median / native
        proportion = "lower is better"
    else:
        diff = native / x.median
        proportion = "higher is better"

    diff *= 1.1

    llen = len(x.median)
    result = dict(
        identifier=list(x.identifier),
        title=[x.title.iloc[0]] * llen,
        benchmark_title=[title] * llen,
        benchmark_group=[x.benchmark_name] * llen,
        diff=list(diff),
        median=list(x.median),
        scale=x.scale,
        proportion=[proportion] * llen,
    )
    return pd.Series(result, name="metrics")


CONVERSION_MAPPING = {
    "MB": 10e6,
    "KB": 10e3,
}

ALL_UNITS = "|".join(CONVERSION_MAPPING.keys())
UNIT_FINDER = re.compile(r"(\d+)\s*({})".format(ALL_UNITS), re.IGNORECASE)


def unit_replacer(matchobj: re.Match) -> str:
    """Given a regex match object, return a replacement string where units are modified"""
    number = matchobj.group(1)
    unit = matchobj.group(2)
    new_number = int(number) * CONVERSION_MAPPING[unit]
    return f"{new_number} B"


def sort_row(val: pd.Series) -> Any:
    return natsort_keygen()(val.apply(lambda v: UNIT_FINDER.sub(unit_replacer, v)))


def bar_colors(graph: Any, df: pd.Series, num_colors: int) -> None:
    colors = sns.color_palette(n_colors=num_colors)
    groups = 0
    last_group = df[0].iloc[0]
    for i, patch in enumerate(graph.axes[0][0].patches):
        if last_group != df[i].iloc[0]:
            last_group = df[i].iloc[0]
            groups += 1
        patch.set_facecolor(colors[groups])


def phoronix(df: pd.DataFrame) -> Any:
    #df = df[df["identifier"].isin(["vmsh-blk", "qemu-blk"])]
    groups = len(df.benchmark_name.unique())
    # same benchmark with different units
    df = df[~((df.benchmark_name.str.startswith("pts/fio")) & (df.scale == "MB/s"))]
    df = df.sort_values(by=["benchmark_id", "identifier"], key=sort_row)
    df = df.groupby("benchmark_id").apply(compute_ratio).reset_index()
    columns = [
        "identifier",
        "title",
        "benchmark_title",
        "benchmark_group",
        "diff",
        "median",
        "scale",
        "proportion"
    ]
    df = df.explode(columns)
    df = df[df.identifier != "cntr"]
    df = df.sort_values(by=["benchmark_id"], key=sort_row)
    g = catplot(
        data=apply_aliases(df),
        y=column_alias("benchmark_id"),
        x=column_alias("diff"),
        hue=column_alias("identifier"),
        kind="bar",
        palette=None,
        aspect=0.7,
        height=12,
    )
    #bar_colors(g, df.benchmark_group, groups)
    g.ax.set_xlabel("")
    g.ax.set_ylabel("")
    FONT_SIZE = 9
    g.ax.annotate(
        "Lower is better",
        xycoords="axes fraction",
        xy=(0, 0),
        xytext=(0.1, -0.08),
        fontsize=FONT_SIZE,
        color="navy",
        weight="bold",
    )
    g.ax.annotate(
        "",
        xycoords="axes fraction",
        xy=(0.0, -0.07),
        xytext=(0.1, -0.07),
        fontsize=FONT_SIZE,
        arrowprops=dict(arrowstyle="-|>", color="navy"),
    )
    g.ax.axvline(x=1, color="gray", linestyle=":")
    g.ax.annotate(
        "baseline",
        xy=(1.1, -0.2),
        fontsize=FONT_SIZE,
    )
    return g


def main() -> None:
    if len(sys.argv) < 2:
        print(f"USAGE: {sys.argv[0]} graph.tsv...")
    graphs = []
    for arg in sys.argv[1:]:
        tsv_path = Path(arg)
        df = pd.read_csv(tsv_path, sep="\t")
        assert isinstance(df, pd.DataFrame)
        name = tsv_path.stem

        if name.startswith("phoronix"):
            graphs.append(("phoronix", phoronix(df)))
        else:
            print(f"unhandled graph name: {tsv_path}", file=sys.stderr)
            sys.exit(1)

    for prefix, graph in graphs:
        fname = f"{prefix}{out_format}"
        print(f"write {fname}")
        graph.savefig(fname)


if __name__ == "__main__":
    main()

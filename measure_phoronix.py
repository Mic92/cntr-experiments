from contextlib import contextmanager
import measure_helpers as util
from root import TEST_ROOT
from typing import Iterator, List, Dict, IO, Union
from pathlib import Path
from dataclasses import dataclass
import subprocess
import pandas as pd
import phoronix
import os
import time
from functools import partial
import shutil

from nix import nix_build

STATS_PATH = util.MEASURE_RESULTS.joinpath("phoronix-stats.tsv")


@contextmanager
def fresh_fs_ssd() -> Iterator[None]:
    with util.fresh_fs_ssd(nix_build(".#phoronix-image")[0]["outputs"]["out"]):
        yield


def phoronix_test_suite() -> Path:
    p = Path(nix_build(".#phoronix-test-suite")[0]["outputs"]["out"])
    return p.joinpath("bin/phoronix-test-suite")


@dataclass
class Command:
    args: List[str]
    env: Dict[str, str]

    @property
    def env_vars(self) -> List[str]:
        env_vars = []
        for k, v in self.env.items():
            env_vars.append(f"{k}={v}")
        return env_vars


def phoronix_command(phoronix_path: Path, skip_tests: List[str]) -> Command:
    exe = phoronix_test_suite()
    env = dict(
        SKIP_TESTS="iozone",
        PTS_USER_PATH_OVERRIDE=f"{phoronix_path}/",
        PTS_DOWNLOAD_CACHE=f"{phoronix_path.joinpath('download-cache')}/",
        TEST_RESULTS_NAME="vmsh",
        TEST_RESULTS_IDENTIFIER="vmsh",
        PTS_NPROC="4",
        NUMBER_OF_PROCESSORS="4",
        PTS_PHYSICAL_CORES="4",
        # no goddamn auto-updates
        http_proxy="127.0.1.2:28201",
    )
    # IOzone is marked as deprecated & unmaintained and does not work
    env["SKIP_TESTS"] = ",".join(skip_tests + ["iozone"])

    return Command([str(exe), "run", "pts/disk"], env)


REPORT_PATH = Path("test-results/vmsh/composite.xml")


def parse_result(report: Union[Path, str], identifier: str) -> pd.DataFrame:
    df = phoronix.parse_xml(report)
    df["identifier"] = identifier
    return df


def yes_please() -> IO[bytes]:
    yes_please = subprocess.Popen(["yes"], stdout=subprocess.PIPE)
    assert yes_please.stdout is not None
    return yes_please.stdout


def link_phoronix_test_suite(test_suite: Path) -> List[str]:
    return [
        "ln",
        "-sfn",
        str(test_suite),
        "/root/.phoronix-test-suite",
    ]


def systemd_run(
    cpus: int = 4, memory_gigabytes: int = 16, env: Dict[str, str] = {}
) -> List[str]:
    """
    Since we limit memory inside our VM we also limit the number of CPUs for the benchmark
    """
    assert memory_gigabytes >= 1
    # if 0 this is an empty string, which means no restrictions
    mask = ",".join(map(str, range(cpus)))
    high_mem = (memory_gigabytes - 0.5) * 1000
    cmd = [
        "systemd-run",
        "--pty",
        "--wait",
        "--collect",
        "-p",
        f"MemoryHigh={high_mem}M",
        "-p",
        f"MemoryMax={memory_gigabytes}G",
        "-p",
        f"AllowedCPUs={mask}",
    ]
    for k, v in env.items():
        cmd.append(f"--setenv={k}={v}")
    cmd.append("--")
    return cmd


def native(skip_tests: List[str]) -> pd.DataFrame:
    with fresh_fs_ssd():
        test_suite = util.HOST_DIR_PATH.joinpath("phoronix-test-suite")
        report_path = test_suite.joinpath(REPORT_PATH)
        cmd = phoronix_command(test_suite, skip_tests)
        # this is gross but so is phoronix test suite
        util.run(["sudo"] + link_phoronix_test_suite(test_suite))
        prefix = systemd_run(env=cmd.env)

        util.run(
            ["sudo"] + prefix + cmd.args,
            extra_env=cmd.env,
            stdout=None,
            stderr=None,
            stdin=yes_please(),
            # we check at the end if phoronix passed tests when we parse results
            check=False,
        )
        if not report_path.exists():
            raise OSError(f"phoronix did not create a report at {report_path}")
        return parse_result(report_path, "native")


def cntr(optimizations: List[str], name: str, skip_tests: List[str]) -> pd.DataFrame:
    with fresh_fs_ssd():
        test_suite = util.HOST_DIR_PATH.joinpath("phoronix-test-suite")
        fuse_mnt = util.HOST_DIR_PATH.joinpath("fuse")
        fuse_mnt.mkdir()
        report_path = test_suite.joinpath(REPORT_PATH)
        cmd = phoronix_command(fuse_mnt, skip_tests)
        # this is gross but so is phoronix test suite
        util.run(["sudo"] + link_phoronix_test_suite(fuse_mnt))

        util.run(["sudo", "umount", "-l", str(test_suite)], check=False)
        cntr = [
            "cargo",
            "build",
            "--release",
        ]
        cntr_dir = TEST_ROOT.joinpath("cntr")
        cntr_bin = cntr_dir.joinpath("target", "release", "cntrfs-test")
        util.run(cntr, cwd=str(cntr_dir))
        cntr_cmd = [
            str(cntr_bin),
            str(test_suite),
            str(fuse_mnt),
        ]
        env = cmd.env.copy()
        for opt in optimizations:
            env[opt] = "1"
        paths = map(os.path.dirname, [
            shutil.which("sleep"),
            shutil.which("kill"),
            shutil.which("mountpoint"),
        ])
        env["PATH"] = ':'.join(paths)
        try:
            prefix = systemd_run(env=env)
            # Super hässlich!
            script = f"""
{' '.join(cntr_cmd)}
while ! mountpoint -q {fuse_mnt}; do
  sleep 1;
done
{' '.join(cmd.args)}
"""
            util.run(
                ["sudo"] + prefix + ["sh", "-x", "-c", script],
                stdout=None,
                stderr=None,
                stdin=yes_please(),
                # we check at the end if phoronix passed tests when we parse results
                check=False,
            )
            if not report_path.exists():
                raise OSError(f"phoronix did not create a report at {report_path}")
            return parse_result(report_path, name)
        finally:
            #time.sleep(1)
            #print("terminate")
            #p.terminate()
            #p.wait(timeout=1)
            #time.sleep(1)
            #p.kill()
            #p.wait(timeout=1)
            print("umount")
            util.run(["sudo", "umount", "-l", str(fuse_mnt)], check=False)


def main() -> None:
    util.check_ssd()
    util.check_intel_turbo()
    all = set(
        [
            "SPLICE_READ",
            "SPLICE_WRITE",
            "FOPEN_KEEPCACHE",
            "WRITEBACK_CACHE",
            "PARALLEL_DIROPS",
        ]
    )

    benchmarks = [
        ("cntr-nosplice", partial(cntr, all - set(["SPLICE_READ", "SPLICE_WRITE"]))),
        ("cntr-nofopen-keepcache", partial(cntr, all - set(["FOPEN_KEEPCACHE"]))),
        ("cntr-nowriteback-cache", partial(cntr, all - set(["WRITEBACK_CACHE"]))),
        ("cntr-noparallel-dirops", partial(cntr, all - set(["PARALLEL_DIROPS"]))),
        ("cntr", partial(cntr, all)),
    ]
    df = None
    if STATS_PATH.exists():
        df = pd.read_csv(STATS_PATH, sep="\t")

    for name, benchmark in benchmarks:
        # Useful for testing
        #skip_tests = "fio,sqlite,dbench,ior,compilebench,postmark".split(",")
        skip_tests = "sqlite,dbench,ior,compilebench,postmark,fio".split(",")
        #skip_tests = []
        #if df is not None:
        #    skip_tests = list(df[df.identifier == name].benchmark_name.unique())
        #    if len(skip_tests) == 7:
        #        # result of len(df.benchmark_name.unique())
        #        continue
        new_df = benchmark(name, skip_tests)
        if df is not None:
            df = pd.concat([df, new_df])
        else:
            df = new_df

        STATS_PATH.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(STATS_PATH, index=True, sep="\t")


if __name__ == "__main__":
    main()

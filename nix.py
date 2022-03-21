#!/usr/bin/env python3

import functools
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator

from root import PROJECT_ROOT


@functools.lru_cache(maxsize=None)
def nix_build(what: str) -> Any:
    path = PROJECT_ROOT.joinpath(".git/nix-results")
    path.mkdir(parents=True, exist_ok=True)
    # gc root to improve caching
    link_name = path.joinpath(what.lstrip(".#"))
    result = subprocess.run(
        ["nix", "build", "--out-link", str(link_name), "--json", what],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
        cwd=PROJECT_ROOT,
    )
    return json.loads(result.stdout)

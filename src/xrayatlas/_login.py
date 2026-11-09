"""Login helper that delegates to the ``xatlas`` CLI.

The client does not implement a native login flow. `login` runs ``xatlas login`` when the CLI is
installed and then reads the credential the CLI saved.
"""

from __future__ import annotations

import shutil
import subprocess

from xrayatlas._credentials import Credential, load_credential
from xrayatlas.errors import AuthError


def login(host: str | None = None) -> Credential:
    """Sign in through the ``xatlas`` CLI and return the saved credential.

    Args:
        host: Atlas origin to sign in to, or ``None`` for the CLI default.

    Returns:
        The credential the CLI saved.

    Raises:
        AuthError: The CLI is not installed, exited with an error, or saved no credential.
    """
    executable = shutil.which("xatlas")
    if executable is None:
        raise AuthError(
            "The xatlas CLI is not installed. Install it from "
            "https://github.com/WSU-Carbon-Lab/atlas-cli, run `xatlas login`, then retry."
        )
    args = [executable, "login"]
    if host:
        args.extend(["--host", host])
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise AuthError("`xatlas login` did not complete.")
    credential = load_credential()
    if credential is None:
        raise AuthError("`xatlas login` finished but saved no credential.")
    return credential

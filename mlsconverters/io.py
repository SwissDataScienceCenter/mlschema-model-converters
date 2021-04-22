import os
import json
from pathlib import Path

MLS_DIR = "ml"
ENV_RENKU_HOME = "RENKU_HOME"
COMMON_DIR = "latest"


def log_renku_mls(mls, hash, force=False):
    if ENV_RENKU_HOME in os.environ:
        renku_project_root = os.environ[ENV_RENKU_HOME]
    elif force:
        # hope for the best...
        renku_project_root = ".renku"
    else:
        # we are not running as part of renku run
        # hence NOP
        return

    path = Path(os.path.join(renku_project_root, MLS_DIR, COMMON_DIR))
    if not path.exists():
        path.mkdir(parents=True)

    path = path / (hash + ".jsonld")
    with path.open(mode="w") as f:
        f.write(mls)

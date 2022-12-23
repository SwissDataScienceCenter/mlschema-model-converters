import os
from pathlib import Path

import psutil
from renku.core.util.contexts import renku_project_context
from renku.domain_model.project_context import project_context

MLS_DIR = "ml"
ENV_RENKU_HOME = "RENKU_HOME"
COMMON_DIR = "latest"


def log_renku_mls(mls, hash, force=False):
    inside_renku = False

    parent = psutil.Process().parent()

    while parent is not None:
        if parent.name() == "renku" or "renku.ui.cli" in parent.cmdline():
            inside_renku = True
            break
        parent = parent.parent()

    if force or inside_renku:
        with renku_project_context("."):
            renku_project_root = project_context.metadata_path
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

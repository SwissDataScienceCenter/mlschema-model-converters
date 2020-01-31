from __future__ import absolute_import, print_function
import os
import json
from pathlib import Path
from mls.config import MLS_DIR, MLS_METADATA_FILE
from . import sklearn


def export(model, force=False):
    if 'RENKU_HOME' in os.environ:
        renku_project_root = os.environ['RENKU_HOME']
    elif force:
        # hope for the best...
        renku_project_root = ".renku"
    else:
        # we are not running as part of renku run
        # hence NOP
        return

    if model.__module__.startswith("sklearn"):
        mls = sklearn.to_mls(model)
    else:
        raise ValueError("Unsupported library")

    path = Path(os.path.join(renku_project_root, MLS_DIR, str(model.__hash__())))
    if not path.exists():
        path.mkdir(parents=True)

    path = path / MLS_METADATA_FILE
    with path.open(mode='w') as f:
        json.dump(mls, f)

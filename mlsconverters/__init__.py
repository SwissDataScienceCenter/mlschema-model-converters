from __future__ import absolute_import, print_function
import os
from pathlib import Path


def _extract_mls(model, **kwargs):
    if model.__module__.startswith("sklearn"):
        from . import sklearn
        return sklearn.to_mls(model, **kwargs)
    else:
        raise ValueError("Unsupported library")


def export_to_file(model, filename, **kwargs):
    mls = _extract_mls(model, **kwargs)
    with open(filename, 'w') as f:
        f.write(mls)


def export(model, force=False, **kwargs):
    if 'RENKU_HOME' in os.environ:
        renku_project_root = os.environ['RENKU_HOME']
    elif force:
        # hope for the best...
        renku_project_root = ".renku"
    else:
        # we are not running as part of renku run
        # hence NOP
        return

    mls = _extract_mls(model, **kwargs)

    path = Path(os.path.join(renku_project_root, MLS_DIR, str(model.__hash__())))
    if not path.exists():
        path.mkdir(parents=True)

    path = path / MLS_METADATA_FILE
    with path.open(mode='w') as f:
        f.write(mls)

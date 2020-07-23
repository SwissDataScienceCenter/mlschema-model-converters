import os
import json
from pathlib import Path

MLS_DIR = 'ml'
MLS_METADATA_FILE = 'metadata.jsonld'
ENV_RENKU_HOME = 'RENKU_HOME'

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

    path = Path(os.path.join(renku_project_root, MLS_DIR, hash))
    if not path.exists():
        path.mkdir(parents=True)

    path = path / MLS_METADATA_FILE
    with path.open(mode='w') as f:
        json.dump(mls.asjsonld(), f)

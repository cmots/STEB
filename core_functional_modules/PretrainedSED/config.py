"""Configuration for the BEATs sound-event detector used by STEB eval."""

import os


RESOURCES_FOLDER = os.environ.get(
    "PRETRAINED_SED_RESOURCES",
    os.path.join(os.path.expanduser("~"), ".cache", "steb", "pretrained_sed"),
)
GITHUB_RELEASE_URL = "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/"

CHECKPOINT_URLS = {
    "BEATs_strong_1": GITHUB_RELEASE_URL + "BEATs_strong_1.pt",
}

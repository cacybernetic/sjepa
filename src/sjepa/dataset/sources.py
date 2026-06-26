"""Find audio files inside a folder, a zip, or a tar archive.

The training data can come in three shapes:

  * a plain folder with audio files (maybe in sub-folders);
  * a single ".zip" archive that holds the audio files;
  * a single ".tar" (or ".tar.gz") archive that holds the audio files.

This module walks any of these shapes and returns a flat list of `AudioRef`
objects. Each ref says where one audio file lives, so we can read it later
without unpacking the whole archive.

The discovery is recursive: files in the root and in any sub-folder are found.
"""

from __future__ import annotations

import os
import tarfile
import zipfile
from dataclasses import dataclass

# Audio file endings we accept. The check is case-insensitive.
AUDIO_EXTENSIONS = (
    ".wav", ".mp3", ".mp2", ".flac", ".ogg", ".oga",
    ".opus", ".m4a", ".aac", ".aif", ".aiff", ".au", ".w64",
)


def is_audio_name(name):
    """Return True when a file name ends with a known audio extension."""
    lowered = name.lower()
    return lowered.endswith(AUDIO_EXTENSIONS)


@dataclass(frozen=True)
class AudioRef:
    """A pointer to one audio file.

    Fields:
        archive: path to the zip or tar archive, or None for a loose file.
        member: the inner archive path, or the file path for a loose file.
    """

    archive: object
    member: str

    @property
    def container(self):
        """Return the kind of container: 'file', 'zip', or 'tar'."""
        if self.archive is None:
            return "file"
        if self.archive.lower().endswith(".zip"):
            return "zip"
        return "tar"

    def to_dict(self):
        """Return a small dict so the ref can be saved in a JSON cache."""
        return {"archive": self.archive, "member": self.member}

    @classmethod
    def from_dict(cls, data):
        """Rebuild a ref from a dict made by `to_dict`."""
        return cls(archive=data["archive"], member=data["member"])


class _FolderSource:
    """List audio files inside a plain folder, walking sub-folders too."""

    def __init__(self, root):
        self.root = root

    def list_refs(self):
        """Return one AudioRef per audio file found under the folder."""
        refs = []
        for current, _dirs, files in os.walk(self.root):
            for name in files:
                if is_audio_name(name):
                    refs.append(AudioRef(None, os.path.join(current, name)))
        return refs


class _ZipSource:
    """List audio files inside a zip archive without unpacking it."""

    def __init__(self, path):
        self.path = path

    def list_refs(self):
        """Return one AudioRef per audio entry inside the zip."""
        refs = []
        with zipfile.ZipFile(self.path) as archive:
            for info in archive.infolist():
                if not info.is_dir() and is_audio_name(info.filename):
                    refs.append(AudioRef(self.path, info.filename))
        return refs


class _TarSource:
    """List audio files inside a tar archive without unpacking it."""

    def __init__(self, path):
        self.path = path

    def list_refs(self):
        """Return one AudioRef per audio entry inside the tar."""
        refs = []
        with tarfile.open(self.path) as archive:
            for member in archive.getmembers():
                if member.isfile() and is_audio_name(member.name):
                    refs.append(AudioRef(self.path, member.name))
        return refs


def _pick_source(path):
    """Choose the right source class for a folder or an archive path."""
    if os.path.isdir(path):
        return _FolderSource(path)
    lowered = path.lower()
    if lowered.endswith(".zip"):
        return _ZipSource(path)
    if lowered.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
        return _TarSource(path)
    raise ValueError(f"path is not a folder, zip, or tar: {path}")


def discover_audio(path):
    """Find every audio file under a folder or archive.

    Args:
        path: a folder, a ".zip", or a ".tar" path.

    Returns:
        A sorted list of `AudioRef`. The order is stable across runs.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset path does not exist: {path}")
    source = _pick_source(path)
    refs = source.list_refs()
    refs.sort(key=lambda ref: (ref.archive or "", ref.member))
    return refs

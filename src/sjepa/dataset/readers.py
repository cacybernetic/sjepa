"""Read the raw bytes of one audio file from a folder, zip, or tar.

A reader turns an `AudioRef` into bytes. We keep one open archive handle per
process so we do not reopen the zip or tar for every single read. This class
has one job: give the bytes of a referenced audio file.

The handles are not shared between processes. Each DataLoader worker builds its
own `ArchiveReader`, so there is no cross-process file sharing problem.
"""

from __future__ import annotations

import io
import tarfile
import zipfile


class ArchiveReader:
    """Read audio bytes from loose files or from cached archive handles."""

    def __init__(self):
        self._zip_handles = {}
        self._tar_handles = {}

    def _zip(self, path):
        """Return a cached open zip handle for a path."""
        handle = self._zip_handles.get(path)
        if handle is None:
            handle = zipfile.ZipFile(path)
            self._zip_handles[path] = handle
        return handle

    def _tar(self, path):
        """Return a cached open tar handle for a path."""
        handle = self._tar_handles.get(path)
        if handle is None:
            handle = tarfile.open(path)
            self._tar_handles[path] = handle
        return handle

    def read_bytes(self, ref):
        """Return the raw file bytes for one `AudioRef`.

        Args:
            ref: an `AudioRef` pointing at a loose file or an archive member.

        Returns:
            A `bytes` object with the file content.
        """
        kind = ref.container
        if kind == "file":
            with open(ref.member, "rb") as handle:
                return handle.read()
        if kind == "zip":
            return self._zip(ref.archive).read(ref.member)
        member = self._tar(ref.archive).getmember(ref.member)
        extracted = self._tar(ref.archive).extractfile(member)
        if extracted is None:
            raise IOError(f"cannot read tar member: {ref.member}")
        return extracted.read()

    def read_stream(self, ref):
        """Return a seekable in-memory stream for one `AudioRef`."""
        return io.BytesIO(self.read_bytes(ref))

    def close(self):
        """Close every cached archive handle."""
        for handle in self._zip_handles.values():
            handle.close()
        for handle in self._tar_handles.values():
            handle.close()
        self._zip_handles.clear()
        self._tar_handles.clear()

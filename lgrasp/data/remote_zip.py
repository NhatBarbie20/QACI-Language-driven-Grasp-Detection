"""Read selected members of a remote ZIP archive with HTTP range requests.

Grasp-Anything ships its images as a single 65 GB ZIP that is split into two
files on the Hugging Face Hub. Downloading it whole is not an option on a laptop
or on Colab, so this module:

1. locates and stream-parses the ZIP central directory (the table of contents),
2. keeps only the members the caller asks for,
3. fetches each member with one ranged GET and checks its CRC-32.

Only the ZIP features the dataset uses are supported: stored / deflated members,
ZIP64, and archives split into parts with the Unix ``split`` tool.
"""

from __future__ import annotations

import struct
import threading
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Callable, Iterator, Sequence
from urllib.parse import urljoin, urlparse

import requests

HF_RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"

_EOCD_SIG = b"PK\x05\x06"
_EOCD64_LOCATOR_SIG = b"PK\x06\x07"
_EOCD64_SIG = b"PK\x06\x06"
_CENTRAL_SIG = b"PK\x01\x02"
_LOCAL_SIG = b"PK\x03\x04"

_CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")  # 46 bytes
_LOCAL_HEADER_SIZE = 30
_LOCAL_EXTRA_SLACK = 128  # local extra fields are usually shorter than this
_MAX_RETRIES = 6
_TIMEOUT = (15, 120)  # connect, read (seconds)


class RemoteZipError(RuntimeError):
    """The archive is malformed or a member failed its integrity check."""


@dataclass(frozen=True)
class ZipEntry:
    """One archive member, as described by the central directory."""

    name: str
    offset: int  # position of the local file header in the (concatenated) archive
    compressed_size: int
    size: int
    crc32: int
    method: int  # 0 = stored, 8 = deflate
    name_length: int  # length of the encoded file name, in bytes

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ZipEntry":
        return cls(**data)


class RemoteBlob:
    """A single file on the Hugging Face Hub, readable by byte range.

    The Hub answers ``resolve`` URLs with a redirect to a signed CDN URL. That URL
    is resolved once and reused, so thousands of small reads go straight to the
    CDN instead of counting against the Hub's request limits. It is refreshed
    automatically when it expires.
    """

    def __init__(self, repo: str, filename: str, revision: str = "main", token: str | None = None):
        self.url = HF_RESOLVE_URL.format(repo=repo, revision=revision, filename=filename)
        self.name = f"{repo}/{filename}"
        self._token = token
        self._signed_url: str | None = None
        self._size: int | None = None
        self._lock = threading.Lock()
        self._local = threading.local()

    # -- connection handling -------------------------------------------------

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def _resolve(self, stale: str | None = None) -> str:
        """Return the signed URL, resolving it again if ``stale`` is the current one."""
        with self._lock:
            if self._signed_url is not None and self._signed_url != stale:
                return self._signed_url
            url = self.url
            for _ in range(5):
                on_hub = urlparse(url).netloc.endswith("huggingface.co")
                headers = {"Range": "bytes=0-0"}
                if on_hub and self._token:
                    headers["Authorization"] = f"Bearer {self._token}"
                response = self._session().get(url, headers=headers, allow_redirects=False, timeout=_TIMEOUT)
                if response.status_code in (301, 302, 303, 307, 308):
                    url = urljoin(url, response.headers["Location"])
                    if not urlparse(url).netloc.endswith("huggingface.co"):
                        break
                    continue
                response.raise_for_status()
                break
            self._signed_url = url
            return url

    def _get(self, start: int, end: int, stream: bool = False) -> requests.Response:
        """GET bytes ``[start, end)`` with retries. Raises after ``_MAX_RETRIES`` failures."""
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            url = self._resolve()
            try:
                response = self._session().get(
                    url, headers={"Range": f"bytes={start}-{end - 1}"}, stream=stream, timeout=_TIMEOUT
                )
                if response.status_code in (401, 403):  # signed URL expired
                    response.close()
                    self._resolve(stale=url)
                    continue
                if response.status_code == 206:
                    return response
                last_error = RemoteZipError(f"{self.name}: HTTP {response.status_code} for a range request")
                response.close()
            except requests.RequestException as error:
                last_error = error
            time.sleep(min(2**attempt, 30))
        raise RemoteZipError(f"{self.name}: giving up on bytes {start}-{end}: {last_error}")

    # -- public API ----------------------------------------------------------

    @property
    def size(self) -> int:
        if self._size is None:
            response = self._get(0, 1)
            self._size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
        return self._size

    def read(self, start: int, end: int) -> bytes:
        """Return bytes ``[start, end)``."""
        for attempt in range(_MAX_RETRIES):
            try:
                data = self._get(start, end).content
            except requests.RequestException:
                data = b""
            if len(data) == end - start:
                return data
            time.sleep(min(2**attempt, 30))
        raise RemoteZipError(f"{self.name}: short read for bytes {start}-{end}")

    def stream(self, start: int, end: int, chunk_size: int = 1 << 20) -> Iterator[bytes]:
        """Yield bytes ``[start, end)`` in chunks, resuming after dropped connections."""
        position = start
        failures = 0
        while position < end:
            try:
                with self._get(position, end, stream=True) as response:
                    for chunk in response.iter_content(chunk_size):
                        position += len(chunk)
                        failures = 0
                        yield chunk
            except requests.RequestException as error:
                failures += 1
                if failures >= _MAX_RETRIES:
                    raise RemoteZipError(f"{self.name}: stream failed at byte {position}: {error}") from error
                time.sleep(min(2**failures, 30))


class RemoteZip:
    """A ZIP archive stored as one or more consecutive remote parts."""

    def __init__(self, parts: Sequence[RemoteBlob]):
        if not parts:
            raise ValueError("RemoteZip needs at least one part")
        self.parts = list(parts)
        self._starts: list[int] = []
        total = 0
        for part in self.parts:
            self._starts.append(total)
            total += part.size
        self.size = total

    # -- byte access across parts -------------------------------------------

    def _segments(self, start: int, end: int) -> Iterator[tuple[RemoteBlob, int, int]]:
        """Map the global range ``[start, end)`` onto per-part ranges."""
        for part, part_start in zip(self.parts, self._starts):
            lo = max(start, part_start)
            hi = min(end, part_start + part.size)
            if lo < hi:
                yield part, lo - part_start, hi - part_start

    def read(self, start: int, end: int) -> bytes:
        return b"".join(part.read(lo, hi) for part, lo, hi in self._segments(start, end))

    def stream(self, start: int, end: int) -> Iterator[bytes]:
        for part, lo, hi in self._segments(start, end):
            yield from part.stream(lo, hi)

    # -- central directory ---------------------------------------------------

    def _locate_central_directory(self) -> tuple[int, int, int]:
        """Return ``(start, size, offset_bias)`` of the central directory.

        ``offset_bias`` corrects member offsets when extra bytes precede the
        archive; it is 0 for a well-formed file.
        """
        tail_start = max(0, self.size - (1 << 16) - 128)
        tail = self.read(tail_start, self.size)
        eocd_at = tail.rfind(_EOCD_SIG)
        if eocd_at < 0:
            raise RemoteZipError("End-of-central-directory record not found: not a ZIP archive")
        _, _, _, _, _, cd_size, cd_offset, _ = struct.unpack_from("<4s4H2LH", tail, eocd_at)
        record_position = tail_start + eocd_at

        locator_at = eocd_at - 20
        if locator_at >= 0 and tail[locator_at : locator_at + 4] == _EOCD64_LOCATOR_SIG:
            _, _, eocd64_position, _ = struct.unpack_from("<4sLQL", tail, locator_at)
            raw = self.read(eocd64_position, eocd64_position + 56)
            if raw[:4] != _EOCD64_SIG:
                # The locator stores an offset without the bias; search the tail instead.
                eocd64_at = tail.rfind(_EOCD64_SIG, 0, locator_at)
                if eocd64_at < 0:
                    raise RemoteZipError("ZIP64 end-of-central-directory record not found")
                raw = tail[eocd64_at : eocd64_at + 56]
                eocd64_position = tail_start + eocd64_at
            _, _, _, _, _, _, _, _, cd_size, cd_offset = struct.unpack("<4sQ2H2L4Q", raw)
            record_position = eocd64_position

        cd_start = record_position - cd_size
        return cd_start, cd_size, cd_start - cd_offset

    def scan(
        self,
        keep: Callable[[str], bool],
        on_progress: Callable[[int], None] | None = None,
    ) -> tuple[list[ZipEntry], int]:
        """Stream the central directory and return the members accepted by ``keep``.

        Returns ``(entries, total_members)``. ``on_progress`` receives the number of
        directory bytes consumed since the previous call.
        """
        cd_start, cd_size, bias = self._locate_central_directory()
        entries: list[ZipEntry] = []
        total = 0
        buffer = bytearray()
        for chunk in self.stream(cd_start, cd_start + cd_size):
            buffer += chunk
            position = 0
            while len(buffer) - position >= _CENTRAL_HEADER.size:
                (signature, _, _, flags, method, _, _, crc, csize, usize,
                 name_len, extra_len, comment_len, _, _, _, offset) = _CENTRAL_HEADER.unpack_from(buffer, position)
                if signature != _CENTRAL_SIG:
                    raise RemoteZipError("Corrupt central directory (bad member signature)")
                record_size = _CENTRAL_HEADER.size + name_len + extra_len + comment_len
                if len(buffer) - position < record_size:
                    break
                name_start = position + _CENTRAL_HEADER.size
                name = bytes(buffer[name_start : name_start + name_len]).decode(
                    "utf-8" if flags & 0x800 else "cp437"
                )
                total += 1
                if keep(name):
                    if flags & 0x1:
                        raise RemoteZipError(f"{name}: encrypted members are not supported")
                    extra = bytes(buffer[name_start + name_len : name_start + name_len + extra_len])
                    usize, csize, offset = _apply_zip64_extra(extra, usize, csize, offset)
                    entries.append(ZipEntry(name, offset + bias, csize, usize, crc, method, name_len))
                position += record_size
            del buffer[:position]
            if on_progress is not None:
                on_progress(len(chunk))
        if buffer:
            raise RemoteZipError("Central directory ended in the middle of a record")
        return entries, total

    # -- members -------------------------------------------------------------

    def fetch(self, entry: ZipEntry) -> bytes:
        """Download one member, decompress it and verify its size and CRC-32."""
        guess = _LOCAL_HEADER_SIZE + entry.name_length + _LOCAL_EXTRA_SLACK + entry.compressed_size
        raw = self.read(entry.offset, min(entry.offset + guess, self.size))
        if raw[:4] != _LOCAL_SIG:
            raise RemoteZipError(f"{entry.name}: local file header not found at offset {entry.offset}")
        name_len, extra_len = struct.unpack_from("<2H", raw, 26)
        data_start = _LOCAL_HEADER_SIZE + name_len + extra_len
        data_end = data_start + entry.compressed_size
        if data_end > len(raw):  # unusually long local extra field
            raw += self.read(entry.offset + len(raw), entry.offset + data_end)
        payload = raw[data_start:data_end]

        if entry.method == 0:
            data = payload
        elif entry.method == 8:
            data = zlib.decompress(payload, wbits=-15)
        else:
            raise RemoteZipError(f"{entry.name}: unsupported compression method {entry.method}")

        if len(data) != entry.size or (zlib.crc32(data) & 0xFFFFFFFF) != entry.crc32:
            raise RemoteZipError(f"{entry.name}: size or CRC-32 mismatch after download")
        return data


def _apply_zip64_extra(extra: bytes, usize: int, csize: int, offset: int) -> tuple[int, int, int]:
    """Replace 32-bit placeholder values with the 64-bit ones from the ZIP64 extra field."""
    position = 0
    while position + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<2H", extra, position)
        position += 4
        if field_id == 0x0001:
            cursor = position
            if usize == 0xFFFFFFFF:
                (usize,) = struct.unpack_from("<Q", extra, cursor)
                cursor += 8
            if csize == 0xFFFFFFFF:
                (csize,) = struct.unpack_from("<Q", extra, cursor)
                cursor += 8
            if offset == 0xFFFFFFFF:
                (offset,) = struct.unpack_from("<Q", extra, cursor)
            break
        position += field_size
    return usize, csize, offset

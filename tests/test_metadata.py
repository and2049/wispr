from pathlib import Path
from types import SimpleNamespace

from wispr.metadata import MutagenMetadataReader


def test_mutagen_metadata_reader_maps_common_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"mock")

    def fake_file(path: Path, easy: bool) -> SimpleNamespace:
        assert path == audio
        assert easy is True
        return SimpleNamespace(tags={"title": ["Title"], "artist": ["Artist"], "album": ["Album"]})

    monkeypatch.setattr("wispr.metadata.File", fake_file)

    metadata = MutagenMetadataReader().read(audio)

    assert metadata.title == "Title"
    assert metadata.artist == "Artist"
    assert metadata.album == "Album"
    assert metadata.source_path == audio


def test_mutagen_metadata_reader_skips_unreadable_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"mock")

    def fake_file(path: Path, easy: bool) -> None:
        raise RuntimeError("bad file")

    monkeypatch.setattr("wispr.metadata.File", fake_file)

    metadata = MutagenMetadataReader().read(audio)

    assert metadata.title is None
    assert metadata.artist is None
    assert metadata.album is None
    assert metadata.source_path == audio

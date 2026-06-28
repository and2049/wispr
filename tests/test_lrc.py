from wispr.lrc import format_timestamp, serialize_lrc
from wispr.models import LrcDocument, LyricLine, TrackMetadata


def test_format_timestamp_uses_centiseconds() -> None:
    assert format_timestamp(65.432) == "01:05.43"


def test_serialize_lrc_orders_metadata_and_lines() -> None:
    document = LrcDocument(
        metadata=TrackMetadata(title="Song", artist="Artist", album="Album"),
        lines=(LyricLine(line_number=1, text="hello", start=1.2),),
    )

    assert serialize_lrc(document) == "[ar:Artist]\n[al:Album]\n[ti:Song]\n[00:01.20]hello\n"


def test_serialize_lrc_omits_empty_metadata() -> None:
    document = LrcDocument(
        metadata=TrackMetadata(title="Song"),
        lines=(LyricLine(line_number=1, text="hello", start=0),),
    )

    assert serialize_lrc(document) == "[ti:Song]\n[00:00.00]hello\n"

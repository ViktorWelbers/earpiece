import time

from earpiece.audio.playback import PlaybackQueue


def test_suppress_capture_covers_playing_and_tail():
    q = PlaybackQueue(device=None)
    assert q.suppress_capture is False  # idle, nothing played recently

    q.playing = True
    assert q.suppress_capture is True  # actively playing

    q.playing = False
    q._last_audio = time.monotonic()
    assert q.suppress_capture is True  # tail window after the last block

    q._last_audio = time.monotonic() - 5.0
    assert q.suppress_capture is False  # tail expired

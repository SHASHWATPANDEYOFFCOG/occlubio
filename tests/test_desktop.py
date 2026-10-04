import pytest

from occlubio.desktop import DEFAULT_WINDOW, MIN_WINDOW, window_geometry


@pytest.mark.parametrize("screen", [(1024, 768), (1280, 800), (1366, 768), (1440, 900), (800, 600)])
def test_window_fits_on_screen(screen):
    w, h, (min_w, min_h) = window_geometry(*screen)
    assert w <= screen[0] and h <= screen[1]
    assert min_w <= w and min_h <= h


def test_large_screen_keeps_default_size():
    assert window_geometry(2560, 1440)[:2] == DEFAULT_WINDOW
    assert window_geometry(2560, 1440)[2] == MIN_WINDOW


def test_unknown_screen_uses_defaults():
    assert window_geometry(0, 0) == (*DEFAULT_WINDOW, MIN_WINDOW)

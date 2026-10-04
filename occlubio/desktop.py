from __future__ import annotations

import argparse
import html
import inspect
import json
import os
import socket
import sys
import threading
import time

from occlubio import __version__
from occlubio.platform_support import (is_frozen, open_url, os_name, user_data_dir,
                                       user_log_dir)

APP_TITLE = "occlubio"
DEFAULT_PORT = 8001
DEFAULT_WINDOW = (1280, 840)
MIN_WINDOW = (900, 600)
APP_MENU = "__app__"

_PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;height:100vh;display:grid;place-items:center;background:#080b14;color:#cbd5f5;
font:15px -apple-system,system-ui,"Segoe UI",sans-serif}}
main{{max-width:520px;padding:24px;text-align:center}} h1{{font-size:20px;color:#fff}}
p{{line-height:1.5}} code{{color:#93c5fd}}</style></head><body><main>{body}</main></body></html>"""

LOADING_HTML = _PAGE.format(body="<h1>Starting occlubio…</h1><p>Loading face models. The first launch "
                                 "downloads about 300&nbsp;MB, so it can take a few minutes.</p>")


def _error_html(message: str, log_path: str) -> str:
    return _PAGE.format(body=f"<h1>occlubio could not start</h1><p>{html.escape(message)}</p>"
                             f"<p>Details are in <code>{html.escape(log_path)}</code>.</p>")


def _port_free(host: str, port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex((host, port)) != 0


def _pick_port(host: str, wanted: int) -> int:
    if wanted and _port_free(host, wanted):
        return wanted
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _redirect_output_to_log() -> str:
    log_dir = user_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "occlubio.log"
    stream = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = stream
    return str(path)


class ServerThread:
    def __init__(self, host: str, port: int, certfile: str | None, keyfile: str | None):
        import uvicorn

        from occlubio.api.app import app

        self.config = uvicorn.Config(app, host=host, port=port, log_level="info",
                                     ssl_certfile=certfile, ssl_keyfile=keyfile)
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self._run, name="occlubio-server", daemon=True)
        self.error: BaseException | None = None

    def _run(self) -> None:
        try:
            self.server.run()
        except BaseException as e:
            self.error = e

    def start(self, timeout: float = 900.0) -> bool:
        self.thread.start()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.server.started:
                return True
            if not self.thread.is_alive():
                return False
            time.sleep(0.2)
        return False

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def window_geometry(screen_w: int, screen_h: int) -> tuple[int, int, tuple[int, int]]:
    width, height = DEFAULT_WINDOW
    min_w, min_h = MIN_WINDOW
    if screen_w > 0 and screen_h > 0:
        width = min(width, int(screen_w * 0.9))
        height = min(height, int(screen_h * 0.9))
        min_w, min_h = min(min_w, width), min(min_h, height)
    return width, height, (min_w, min_h)


def _run_window(server: ServerThread, url: str, log_path: str) -> None:
    import webview
    from webview.menu import Menu, MenuAction

    def show_authority_code():
        from occlubio.api.app import AUTHORITY_CODE

        window.evaluate_js("alert(%s)" % json.dumps(
            "Authority sign-up code:\n\n" + AUTHORITY_CODE
            + "\n\nShare it only with operators who should be able to create authority accounts."))

    menu = [Menu(APP_MENU, [MenuAction("Open in Browser", lambda: open_url(url)),
                            MenuAction("Show Authority Sign-up Code", show_authority_code)])]
    screens = webview.screens
    width, height, min_size = window_geometry(*((screens[0].width, screens[0].height) if screens else (0, 0)))
    kwargs = dict(width=width, height=height, min_size=min_size, html=LOADING_HTML)
    start_kwargs = dict(private_mode=False, storage_path=str(user_data_dir() / "webview"))
    if "menu" in inspect.signature(webview.create_window).parameters:
        kwargs["menu"] = menu
    else:
        start_kwargs["menu"] = menu
    window = webview.create_window(APP_TITLE, **kwargs)

    def boot():
        if server.start():
            window.load_url(url)
        else:
            reason = str(server.error) if server.error else "the server did not come up in time"
            window.load_html(_error_html(reason, log_path))

    webview.start(boot, **start_kwargs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="occlubio", description="occlubio desktop launcher")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("OCCLUBIO_PORT", DEFAULT_PORT)))
    ap.add_argument("--headless", action="store_true", help="serve only; no window (CI / servers)")
    ap.add_argument("--browser", action="store_true", help="open the default browser instead of a window")
    ap.add_argument("--version", action="version", version=f"occlubio {__version__}")
    args, _ = ap.parse_known_args(argv)

    log_path = "stderr"
    if is_frozen() and not args.headless:
        log_path = _redirect_output_to_log()

    certfile = os.environ.get("OCCLUBIO_SSL_CERT") or None
    keyfile = os.environ.get("OCCLUBIO_SSL_KEY") or None
    scheme = "https" if certfile and keyfile else "http"
    port = _pick_port(args.host, args.port)
    url = f"{scheme}://{args.host}:{port}/"
    server = ServerThread(args.host, port, certfile, keyfile)
    print(f"occlubio {__version__} on {os_name()} -> {url}", flush=True)

    if args.headless or args.browser:
        if not server.start():
            print(f"server failed to start: {server.error}", flush=True)
            return 1
        if args.browser:
            open_url(url)
        try:
            while server.thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        server.stop()
        return 0

    try:
        _run_window(server, url, log_path)
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

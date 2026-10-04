"""Browser end-to-end tests for the web console.

Opt-in (needs Playwright, the buffalo_l face models and a few minutes):

    pip install playwright && python -m playwright install webkit
    OCCLUBIO_E2E=1 OCCLUBIO_E2E_ENGINES=msedge,webkit pytest tests/e2e -v

Engines: ``msedge`` / ``chrome`` (installed browser via Playwright channel), ``chromium``,
``webkit`` (Safari's engine; on Windows/Linux this is Playwright's WebKit build, not Safari).
Screenshots land in ``OCCLUBIO_E2E_ARTIFACTS`` (default: a temp dir, printed at the end).
Tests marked xfail(strict=True) document known open issues from TEST_REPORT.md.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

if os.environ.get("OCCLUBIO_E2E") != "1":
    pytest.skip("browser e2e tests are opt-in: set OCCLUBIO_E2E=1", allow_module_level=True)

playwright_sync = pytest.importorskip("playwright.sync_api")
httpx = pytest.importorskip("httpx")
cv2 = pytest.importorskip("cv2")
insightface_data = pytest.importorskip("insightface.data")

from occlubio.platform_support import open_video_writer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CODE = "e2e-code"
PW = "e2e-password-1"
ENGINES = [e.strip() for e in os.environ.get("OCCLUBIO_E2E_ENGINES", "msedge,webkit").split(",") if e.strip()]
ARTIFACTS = Path(os.environ.get("OCCLUBIO_E2E_ARTIFACTS") or tempfile.mkdtemp(prefix="occlubio-e2e-shots-"))
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, home: Path):
        self.home = home
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc = None
        self.startup_s = None
        self.log = home / "server.log"

    def start(self) -> None:
        env = dict(os.environ, OCCLUBIO_HOME=str(self.home), OCCLUBIO_AUTHORITY_CODE=CODE)
        env.pop("OCCLUBIO_DB", None)
        t0 = time.time()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "occlubio.api.app:app", "--port", str(self.port)],
            cwd=ROOT, env=env, stdout=open(self.log, "ab"), stderr=subprocess.STDOUT)
        deadline = t0 + 900
        while time.time() < deadline:
            try:
                if httpx.get(self.base + "/login", timeout=2).status_code == 200:
                    self.startup_s = time.time() - t0
                    return
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(self.log.read_text(errors="ignore")[-3000:])
            time.sleep(0.5)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    d = tmp_path_factory.mktemp("assets")
    from occlubio.pipeline.face_analyzer import FaceAnalyzer

    img = insightface_data.get_image("t1")
    faces = FaceAnalyzer(model_name="buffalo_l", ctx_id=-1, detection_only=True).analyze(img)
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = f.bbox
    m = 0.4 * max(x2 - x1, y2 - y1)
    h, w = img.shape[:2]
    crop = img[max(0, int(y1 - m)):min(h, int(y2 + m)), max(0, int(x1 - m)):min(w, int(x2 + m))]
    face = d / "face photo é.jpg"
    cv2.imencode(".jpg", crop)[1].tofile(str(face))
    clip = d / "gate clip.mp4"
    writer, codec = open_video_writer(clip, 10.0, (w, h))
    for _ in range(20):
        writer.write(img)
    writer.release()
    return {"face": face, "clip": clip, "codec": codec, "n_faces": len(faces)}


@pytest.fixture(scope="module")
def pw():
    with playwright_sync.sync_playwright() as p:
        yield p


def _launch(pw, engine):
    if engine in ("msedge", "chrome"):
        return pw.chromium.launch(channel=engine)
    return getattr(pw, engine).launch()


@pytest.fixture(scope="module", params=ENGINES)
def env(request, pw, tmp_path_factory, assets):
    engine = request.param
    home = tmp_path_factory.mktemp(f"home-{engine}")
    server = Server(home)
    server.start()
    browser = _launch(pw, engine)
    state = {"engine": engine, "server": server, "browser": browser, "assets": assets}
    yield state
    browser.close()
    server.stop()
    print(f"\n[{engine}] server startup {server.startup_s:.1f}s, screenshots in {ARTIFACTS}")


def _page(env, **ctx):
    context = env["browser"].new_context(**ctx)
    page = context.new_page()
    page.set_default_timeout(30000)
    page.on("dialog", lambda d: d.accept())
    return page


def _shot(page, env, name):
    page.screenshot(path=str(ARTIFACTS / f"{env['engine']}-{name}.png"), full_page=True)


def _login(page, env, user, pw=PW):
    base = env["server"].base
    page.goto(base + "/login")
    page.fill("#l_user", user)
    page.fill("#l_pass", pw)
    page.click("#form_login button[type=submit]")
    page.wait_for_url(base + "/")
    page.wait_for_selector("#nav a")


def _nav(page, view):
    page.click(f'.nav a[data-view="{view}"]')
    page.wait_for_selector(f'section[data-view="{view}"]:not(.hidden)')


def _api(env):
    return httpx.Client(base_url=env["server"].base, timeout=60)


def _register_api(env, **body):
    with _api(env) as c:
        r = c.post("/api/register", json=body)
        assert r.status_code == 200, r.text
        return r.json()


def test_register_validation_messages(env):
    page = _page(env)
    page.goto(env["server"].base + "/login")
    page.click("#tab_reg")
    page.click("#form_reg button[type=submit]")
    assert "fix the highlighted fields" in page.inner_text("#r_msg")
    page.fill("#r_email", "not-an-email")
    page.fill("#r_pass", "short")
    page.click("#form_reg button[type=submit]")
    assert "fix the highlighted fields" in page.inner_text("#r_msg")
    _shot(page, env, "register-validation")
    page.context.close()


def test_student_registers_via_ui_and_sees_profile(env):
    page = _page(env)
    page.goto(env["server"].base + "/login")
    page.click("#tab_reg")
    page.fill("#r_roll", "STU-001")
    page.fill("#r_name", "Aarav Sharma")
    page.fill("#r_email", "aarav@example.com")
    page.fill("#r_pass", PW)
    page.click("#form_reg button[type=submit]")
    page.wait_for_url(env["server"].base + "/")
    page.wait_for_selector('section[data-view="profile"]:not(.hidden)')
    page.wait_for_function("document.getElementById('p_roll').textContent.includes('STU-001')")
    assert "Aarav Sharma" in page.inner_text("#p_name")
    _shot(page, env, "student-profile")
    page.context.close()


def test_duplicate_roll_number_feedback(env):
    page = _page(env)
    page.goto(env["server"].base + "/login")
    page.click("#tab_reg")
    page.fill("#r_roll", "STU-001")
    page.locator("#r_roll").blur()
    page.wait_for_function("document.getElementById('h_roll').textContent.includes('already taken')")
    page.context.close()


def test_authority_registers_and_sees_overview(env):
    page = _page(env)
    page.goto(env["server"].base + "/login")
    page.click("#tab_reg")
    page.click("#role_auth")
    page.fill("#r_user", "operator.one")
    page.fill("#r_name", "Operator One")
    page.fill("#r_email", "op1@example.com")
    page.fill("#r_pass", PW)
    page.fill("#r_code", CODE)
    page.click("#form_reg button[type=submit]")
    page.wait_for_url(env["server"].base + "/")
    page.wait_for_selector('section[data-view="overview"]:not(.hidden)')
    page.wait_for_function("document.getElementById('s_users').textContent.trim() === '2'")
    _shot(page, env, "authority-overview")
    page.context.close()


def test_wrong_authority_code_is_rejected(env):
    page = _page(env)
    page.goto(env["server"].base + "/login")
    page.click("#tab_reg")
    page.click("#role_auth")
    page.fill("#r_user", "intruder")
    page.fill("#r_name", "Intruder")
    page.fill("#r_email", "intruder@example.com")
    page.fill("#r_pass", PW)
    page.fill("#r_code", "wrong-code")
    page.click("#form_reg button[type=submit]")
    page.wait_for_function("document.getElementById('r_msg').textContent.length > 0 && "
                           "!document.getElementById('r_msg').textContent.includes('Creating')")
    assert page.url.endswith("/login")
    page.context.close()


def test_authority_enrolls_student_from_photo(env):
    page = _page(env)
    _login(page, env, "operator.one")
    _nav(page, "enroll")
    page.wait_for_function("document.querySelectorAll('#e_subject option').length > 1")
    page.select_option("#e_subject", label="STU-001 — Aarav Sharma")
    page.set_input_files("#e_files", str(env["assets"]["face"]))
    page.click('section[data-view="enroll"] button:has-text("Enroll")')
    page.wait_for_function("document.getElementById('e_out').textContent.includes('Enrolled')")
    _shot(page, env, "enroll-done")
    page.context.close()


def test_identify_clip_finds_enrolled_subject(env):
    page = _page(env)
    _login(page, env, "operator.one")
    _nav(page, "identify")
    page.wait_for_function("document.querySelectorAll('#t_subject option').length > 1")
    page.select_option("#t_subject", index=1)
    page.set_input_files("#v_file", str(env["assets"]["clip"]))
    page.fill("#v_start", "2026-10-01T09:00")
    page.fill("#v_end", "2026-10-01T09:00:02")
    page.click('section[data-view="identify"] button:has-text("Search clip")')
    page.wait_for_function("document.getElementById('v_result').textContent.length > 0", timeout=300000)
    text = page.inner_text("#v_result")
    assert "not found" not in text.lower(), text
    has_player = page.locator("#v_result video").count() == 1
    assert has_player == (env["assets"]["codec"] == "avc1")
    _shot(page, env, "identify-result")
    page.context.close()


def test_alerts_listed_and_clip_modal_opens_and_closes_with_escape(env):
    page = _page(env)
    _login(page, env, "operator.one")
    _nav(page, "alerts")
    page.wait_for_selector(".alertcard")
    assert page.locator(".alertcard").count() >= 1
    page.locator(".alertcard button").first.click()
    page.wait_for_selector("#clip_modal.open")
    _shot(page, env, "alert-clip-modal")
    page.keyboard.press("Escape")
    page.wait_for_selector("#clip_modal:not(.open)")
    page.context.close()


def test_broadcast_notice_reaches_student(env):
    page = _page(env)
    _login(page, env, "operator.one")
    _nav(page, "messages")
    page.wait_for_selector("#m_all")
    page.check("#m_all")
    page.fill("#m_body", "Gate 2 closed — use Gate 1 (गेट १) today.")
    page.click('section[data-view="messages"] button:has-text("Send notice")')
    page.wait_for_function("document.getElementById('sent_body').textContent.includes('Gate 2 closed')")
    page.context.close()

    student = _page(env)
    _login(student, env, "STU-001")
    _nav(student, "notices")
    student.wait_for_function("document.getElementById('notices_list').textContent.includes('गेट १')")
    _shot(student, env, "student-notices")
    student.context.close()


def test_subject_activity_page(env):
    page = _page(env)
    _login(page, env, "operator.one")
    _nav(page, "users")
    page.wait_for_selector("#users_body tr td")
    page.locator('#users_body button:has-text("Activity")').first.click()
    page.wait_for_selector('section[data-view="activity"]:not(.hidden)')
    page.wait_for_function("!document.getElementById('act_body').textContent.includes('Loading')")
    _shot(page, env, "activity")
    page.context.close()


def test_remove_user_with_confirmation(env):
    _register_api(env, email="temp@example.com", password=PW, full_name="Temp User",
                  role="user", roll_number="TMP-001")
    page = _page(env)
    _login(page, env, "operator.one")
    _nav(page, "users")
    row = page.locator("#users_body tr", has_text="TMP-001")
    row.locator("button.danger").click()
    page.wait_for_function("!document.getElementById('users_body').textContent.includes('TMP-001')")
    page.context.close()


def test_data_persists_across_server_restart(env):
    server = env["server"]
    server.stop()
    server.start()
    page = _page(env)
    _login(page, env, "STU-001")
    page.wait_for_function("document.getElementById('p_enrolled').textContent.trim().length > 0")
    assert "yes" in page.inner_text("#p_enrolled").lower()
    assert (server.home / "occlubio.db").exists()
    assert any((server.home / "data" / "uploads").iterdir())
    page.context.close()


def test_logout_clears_session(env):
    page = _page(env)
    _login(page, env, "STU-001")
    page.click("button.logout")
    page.wait_for_url(env["server"].base + "/login")
    assert page.evaluate("localStorage.getItem('occlubio_session')") is None
    page.goto(env["server"].base + "/")
    page.wait_for_url(env["server"].base + "/login")
    page.context.close()


def test_camera_denied_shows_message(env):
    page = _page(env)
    _login(page, env, "STU-001")
    _nav(page, "enroll")
    page.click('button:has-text("Start camera")')
    page.wait_for_function("[...document.querySelectorAll('div')].some(d => /Camera|camera/.test(d.textContent) "
                           "&& getComputedStyle(d).position === 'fixed')")
    page.context.close()


def test_insecure_context_camera_message(env):
    page = _page(env)
    page.add_init_script("Object.defineProperty(navigator, 'mediaDevices', {value: undefined});"
                         "Object.defineProperty(window, 'isSecureContext', {value: false});")
    _login(page, env, "STU-001")
    _nav(page, "enroll")
    page.click('button:has-text("Start camera")')
    page.wait_for_function("document.body.textContent.includes('Live camera needs HTTPS')")
    page.context.close()


@pytest.mark.xfail(strict=True, reason="ISSUE-001: stored XSS via full_name in Subjects/recipients lists")
def test_full_name_is_rendered_as_text_not_html(env):
    _register_api(env, email="xss@example.com", password=PW, role="user", roll_number="XSS-001",
                  full_name='<img src=x onerror="window.__xss=1">')
    page = _page(env)
    _login(page, env, "operator.one")
    _nav(page, "users")
    page.wait_for_selector("#users_body tr td")
    page.wait_for_timeout(500)
    assert page.evaluate("window.__xss") is None
    page.context.close()


@pytest.mark.xfail(strict=True, reason="ISSUE-006: sidebar links, login tabs and role cards are not keyboard-focusable")
def test_navigation_is_keyboard_reachable(env):
    def tab_walk(page, n=60):
        seen = []
        for _ in range(n):
            page.keyboard.press("Tab")
            seen.append(page.evaluate("""() => { const e = document.activeElement;
                return e ? (e.id || e.dataset.view || e.className || e.tagName) : null; }"""))
        return seen

    fresh = _page(env)
    fresh.goto(env["server"].base + "/login")
    login_focus = tab_walk(fresh, 25)
    fresh.context.close()

    page = _page(env)
    _login(page, env, "operator.one")
    dash_focus = tab_walk(page)
    print("login tab order:", login_focus)
    print("dashboard tab order:", dash_focus)
    page.context.close()
    assert "tab_reg" in login_focus, "the 'Create account' tab cannot be reached with Tab"
    assert any(v in dash_focus for v in ("users", "alerts", "identify")), "sidebar views cannot be reached with Tab"


PHONE_DEVICES = ["iPhone SE", "iPhone 15 Pro Max", "iPad Pro 11 landscape", "iPad (gen 7)"]


@pytest.mark.parametrize("device", PHONE_DEVICES)
def test_mobile_layout_has_no_horizontal_overflow(env, pw, device):
    if env["engine"] != "webkit":
        pytest.skip("device emulation checks run on the WebKit engine")
    page = _page(env, **pw.devices[device])
    slug = device.replace(" ", "_").replace("(", "").replace(")", "")
    page.goto(env["server"].base + "/login")
    page.wait_for_timeout(400)
    over = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    _shot(page, env, f"login-{slug}")
    assert over <= 1, f"login overflows by {over}px on {device}"
    _login(page, env, "operator.one")
    small = page.evaluate("window.innerWidth <= 780")
    for view in ["overview", "identify", "users", "alerts", "messages", "enroll"]:
        if small:
            page.click(".menu-btn")
            page.wait_for_selector("body.nav-open")
        page.click(f'.nav a[data-view="{view}"]')
        page.wait_for_selector(f'section[data-view="{view}"]:not(.hidden)')
        page.wait_for_timeout(300)
        over = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        _shot(page, env, f"{view}-{slug}")
        assert over <= 1, f"view '{view}' overflows by {over}px on {device}"
    page.context.close()


@pytest.mark.xfail(strict=True, reason="ISSUE-008: notice textarea keeps an inline 0.92rem font size, so iOS zooms on focus")
@pytest.mark.parametrize("device", ["iPhone SE", "iPhone 15 Pro Max"])
def test_phone_inputs_do_not_trigger_zoom(env, pw, device):
    if env["engine"] != "webkit":
        pytest.skip("device emulation checks run on the WebKit engine")
    page = _page(env, **pw.devices[device])
    page.goto(env["server"].base + "/login")
    sizes = page.evaluate("[...document.querySelectorAll('input')].map(i => parseFloat(getComputedStyle(i).fontSize))")
    assert min(sizes) >= 16, sizes
    _login(page, env, "operator.one")
    sizes = page.evaluate("[...document.querySelectorAll('main input:not([type=file]):not([type=radio]):not([type=checkbox]), main select, main textarea')]"
                          ".map(i => parseFloat(getComputedStyle(i).fontSize))")
    assert min(sizes) >= 16, sizes
    page.context.close()


@pytest.mark.xfail(strict=True, reason="ISSUE-007: several touch targets are smaller than 44x44 pt on phones")
def test_phone_touch_targets_are_at_least_44pt(env, pw):
    if env["engine"] != "webkit":
        pytest.skip("device emulation checks run on the WebKit engine")
    page = _page(env, **pw.devices["iPhone SE"])
    _login(page, env, "operator.one")
    page.click(".menu-btn")
    _nav(page, "users")
    page.wait_for_selector("#users_body tr td")
    small = page.evaluate("""() => [...document.querySelectorAll('button, a, input[type=radio], input[type=checkbox], select')]
        .filter(e => e.offsetParent !== null)
        .map(e => { const r = e.getBoundingClientRect(); return [e.textContent.trim().slice(0, 24) || e.id || e.className, Math.round(r.width), Math.round(r.height)]; })
        .filter(([, w, h]) => w < 44 || h < 44)""")
    print("small touch targets:", small)
    assert not small
    page.context.close()


def test_users_page_with_2000_subjects_renders_quickly(env):
    from sqlalchemy import create_engine, text

    db = env["server"].home / "occlubio.db"
    eng = create_engine(f"sqlite:///{db.as_posix()}")
    with eng.begin() as c:
        cols = [r[1] for r in c.execute(text("PRAGMA table_info(users)"))]
        have = c.execute(text("SELECT COUNT(*) FROM users")).scalar()
        rows = []
        for i in range(2000):
            row = {"username": f"BULK-{i:05d}", "email": f"bulk{i}@example.com", "full_name": f"Bulk User {i}",
                   "role": "user", "password_hash": "x:y", "roll_number": f"BULK-{i:05d}"}
            if "created_at" in cols:
                row["created_at"] = "2026-10-01 09:00:00"
            rows.append({k: v for k, v in row.items() if k in cols})
        keys = list(rows[0])
        c.execute(text(f"INSERT INTO users ({', '.join(keys)}) VALUES ({', '.join(':' + k for k in keys)})"), rows)
    page = _page(env)
    _login(page, env, "operator.one")
    t0 = time.time()
    _nav(page, "users")
    page.wait_for_function(f"document.querySelectorAll('#users_body tr').length >= {have + 2000}", timeout=60000)
    elapsed = time.time() - t0
    print(f"[{env['engine']}] users page with {have + 2000} rows rendered in {elapsed:.2f}s")
    assert elapsed < 5.0
    page.context.close()

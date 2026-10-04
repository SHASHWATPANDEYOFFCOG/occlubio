# Fix report: cross-platform issues

Branch `fix/cross-platform-issues` (from `feature/cross-platform` at `5f80c5e`). 15 of the 17 issues
raised by the cross-platform QA pass are closed: 14 fixed, 1 cannot be reproduced. Two issues
(ISSUE-003, ISSUE-016) have no record left and need the owner (see the end of this report).

> `TEST_REPORT.md`, the QA report these issue IDs come from, was never committed and is no longer
> on disk. Titles and root causes below are taken from the fix commits, which are the surviving
> record. Severities were not carried into the commits, so they are not repeated here. The titles of
> ISSUE-015 and ISSUE-017 are inferred from what their commits change.

## Summary

| Issue | Title | Status | Commit |
|---|---|---|---|
| ISSUE-001 | Stored XSS: user data inserted into console HTML unescaped | Fixed | `218eb4f` |
| ISSUE-002 | Anyone could register an authority with the public default code | Fixed | `3a86c5e` |
| ISSUE-003 | (no record; see below) | Needs owner | none |
| ISSUE-004 | FAISS gallery save/load fails under non-ASCII paths on Windows | Fixed | `42b41cc` |
| ISSUE-005 | Server timestamps in UTC, shown and compared as local time | Fixed | `5cf9c91` |
| ISSUE-006 | Login tabs, role cards and sidebar not keyboard-operable | Fixed | `98ac8e7` |
| ISSUE-007 | Touch targets smaller than 44 pt on phones | Fixed | `ce551f2` |
| ISSUE-008 | iOS Safari zooms when the notice box is focused | Fixed | `ce551f2` |
| ISSUE-009 | Training extra breaks on Intel Macs (torch 2.2.2 vs NumPy 2) | Fixed | `e72bbdd` |
| ISSUE-010 | Identify upload read whole clip into RAM | Fixed | `c4f76b9` |
| ISSUE-011 | Idle pages burn CPU on decorative animation | Fixed | `b7477fe` |
| ISSUE-012 | macOS window taller than small screens | Fixed | `003bbcf` |
| ISSUE-013 | macOS menu bar order non-standard (File after View) | Fixed | `003bbcf` |
| ISSUE-014 | Packaged app can't find insightface `objects/meanshape_68.pkl` | Fixed | `7718efb` |
| ISSUE-015 | iOS app slow to load the console via `localhost` | Cannot reproduce | `6573e78` (measurement only) |
| ISSUE-016 | (no record; see below) | Needs owner | none |
| ISSUE-017 | Screen-reader landmarks/labels missing on login and console | Fixed | `98ac8e7` |

## Root causes and fixes

- **Unescaped HTML (001).** Values went into `innerHTML` raw. One `esc()` helper now wraps every
  interpolated value in `web/index.html` and `web/login.html`. Inline handlers pass only numeric IDs.
- **Public authority code (002).** Without `OCCLUBIO_AUTHORITY_CODE` the documented default worked
  for anyone. A random per-install code is now generated into `authority_code.txt` in the data
  directory (mode 0600 where supported) and compared with `secrets.compare_digest`. The macOS app
  shows it under occlubio → Show Authority Sign-up Code.
- **FAISS narrow-char `fopen` (004).** `faiss.write_index`/`read_index` can't open non-ASCII paths on
  Windows. The index is now serialized in memory and written by Python. The bytes are identical, so
  existing galleries load unchanged.
- **Mixed clocks (005).** Defaults used `datetime.utcnow()` while user input was local. All
  server-generated stamps now use local time. Explicit offsets in API input are converted to local.
- **Non-semantic controls (006, 017).** `<div onclick>` and `<a>` without `href` are now focusable
  with ARIA roles and states, plus one Enter/Space handler and a focus ring. The login card is in
  `<main>`, and the actions column header is labelled.
- **Touch sizing and zoom (007, 008).** Under `@media (pointer:coarse)` the controls get 44 pt
  minimums, with no change for mouse users. An inline `font-size` on the notice textarea overrode the
  16 px phone rule; it was removed.
- **Intel-Mac wheels (009).** Torch's last Intel-Mac build (2.2.2) needs NumPy 1.x. An
  environment-marked `numpy<2` applies to darwin/x86_64 only.
- **Upload buffering (010).** `await file.read()` is replaced by 1 MB chunked copies. Server RSS
  during a 200 MB upload went from +199 MB to under 50 MB.
- **Endless animation (011).** A shared idle controller pauses CSS animations and the login canvas
  after 20 s idle, on blur or when hidden. Idle CPU on Windows/Edge dropped from 96 % to 2.2 % of a core.
- **macOS window and menus (012, 013).** The window is capped at 90 % of the main screen. Custom items
  moved into the application menu through pywebview's `__app__` menu (requires pywebview ≥ 6.1).
- **Frozen insightface lookup (014).** When frozen, insightface looks in `sys._MEIPASS/objects`. The
  PyInstaller spec now also places `objects/` at the bundle root.
- **ISSUE-015, cannot reproduce.** CI job `ios-load-latency` (run `37177436455`) built the iOS app
  once for each URL and launched each build 3 times on the Simulator:

  | Server URL | Launch to painted | Resume to painted | Timeouts |
  |---|---|---|---|
  | `http://localhost:8001` | 2, 2, 3 s | 1, 1, 1 s | 0 |
  | `http://127.0.0.1:8001` | 2, 1, 1 s | 1, 1, 2 s | 0 |

  IPv6 `::1` is refused immediately (`curl -6` exits 7 in 0.2 ms), so WebKit falls back to IPv4 with
  no measurable delay. The default stays as it is. The job is still in `qa.yml` if you want to
  re-measure on other hardware.

## Tests added or strengthened

| File | Covers |
|---|---|
| `tests/test_authority_code.py` | Env override, generated code persisted and reused, 0600 mode, constant-time compare (002) |
| `tests/test_timestamps.py` | Default clip start and `created_at` on local clock; offset conversion (005), run with `TZ=Asia/Kolkata` on POSIX CI |
| `tests/test_upload_memory.py` | Server RSS stays under +50 MB during a 200 MB identify upload (010) |
| `tests/test_desktop.py` | `window_geometry()` on small and large screens (012) |
| `tests/test_training_env.py` | torch ↔ NumPy interop (009); run in the Intel QA job |
| `tests/test_crossplatform_risks.py` | Non-ASCII FAISS round-trip, no longer xfail on Windows (004); timestamp parsing (005) |
| `tests/e2e/test_console_ui.py` | XSS regression (001); Tab + Enter/Space walk and axe-core audit (006, 017); 44 pt targets and no input zoom on iPhone SE / Pro Max (007, 008); animations stop when idle (011). Former strict xfails are now regular tests. |

## Final build and test results

Windows (local, Windows 11, Python 3.12.6, HEAD `b7477fe`):

- `pytest -q`: 67 passed, 3 skipped. Skips: browser e2e is opt-in; 2 tests need torch, which the
  inference install doesn't include.
- `OCCLUBIO_E2E=1 pytest -q tests/e2e`: 47 passed, 9 skipped (iPhone/iPad emulation runs on WebKit only,
  which macOS CI covers; the axe-core audit needs `OCCLUBIO_AXE`).
- `python scripts/smoke_api.py`: passed (register → enroll → identify, sanitised upload, video served).

GitHub Actions on `b7477fe`: all 12 jobs green.

| Workflow | Job | Result |
|---|---|---|
| CI `37179854255` | test: windows-latest, macos-latest (arm64), macos-15-intel | Pass (1m52s, 1m41s, 3m15s) |
| | macOS app build + smoke: arm64, x86_64 | Pass (3m38s, 7m41s) |
| | iOS Simulator build + launch against live server | Pass (7m28s) |
| QA `37179854251` | macOS app QA (installed `.dmg`, menus, Cmd+Q, relaunch): arm64, x86_64 | Pass (5m08s, 10m30s) |
| | x86_64 app under Rosetta on Apple Silicon | Pass (2m55s) |
| | Browser e2e on macOS (WebKit + Chromium) | Pass (4m28s) |
| | iOS Simulator QA device matrix | Pass (22m38s) |
| | iOS load latency (ISSUE-015 measurement) | Pass (16m46s) |

## Remaining open or blocked issues

- **ISSUE-003 and ISSUE-016: needs owner.** No commit, test or note on any branch mentions them, and
  the QA report that defined them is gone. Send me their titles, or the text from wherever you keep
  the QA output, and I'll fix them or close them the same way.
- **Re-create `TEST_REPORT.md` (optional).** If you have the original, commit it and I'll add each
  issue's final status, commit and verification from the table above.
- **Apple Developer items** are unchanged from [PLATFORM_NOTES.md](PLATFORM_NOTES.md#needs-your-action):
  Developer ID signing and notarization secrets, iOS signing team, and a pass on a physical iPhone/iPad
  (camera, HTTPS trust of the mkcert CA, local-network prompt).

## Release readiness

| Platform | Verdict | Basis |
|---|---|---|
| Windows | Ready | Full unit, e2e and smoke suites pass locally; CI test job |
| macOS (arm64 + x86_64) | Ready for ad-hoc / internal use | `.dmg` builds, installs and passes GUI QA in CI on both arches and under Rosetta. Public distribution needs Developer ID signing + notarization. |
| iOS / iPadOS | Ready for Simulator / internal testing | Builds and loads the console across the Simulator device matrix. Device builds and TestFlight need an Apple Developer team. |

Readiness holds unless ISSUE-003 or ISSUE-016 turns out to be Critical or High.

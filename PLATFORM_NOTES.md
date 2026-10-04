# Platform notes

What differs between Windows, macOS and iOS, what is disabled where, and what needs your action.
All OS-specific code lives in [`occlubio/platform_support.py`](occlubio/platform_support.py).
Launchers and packaging are under [`run_server.*`](run_server.sh), [`packaging/macos/`](packaging/macos/)
and [`ios-client/`](ios-client/).

## How each platform runs occlubio

| | Windows | macOS (Apple Silicon + Intel) | iOS / iPadOS |
|---|---|---|---|
| What runs locally | Full system: server + browser console | Full system: server + browser console, or the packaged `occlubio.app` | **Client only**: the console, loaded from an occlubio server |
| Launcher | `run_server.ps1` / `run_server.bat` | `./run_server.sh` or `occlubio.app` | `occlubio` app (Capacitor) or Safari / Home Screen web app |
| Face detection + recognition | On this machine (CPU, or CUDA if present) | On this machine (CPU; CoreML optional, see below) | On the server it connects to |
| Data location (source checkout) | `occlubio.db`, `data/` in the folder you start from | same | n/a |
| Data location (packaged app or `OCCLUBIO_HOME`) | `%LOCALAPPDATA%\occlubio` | `~/Library/Application Support/occlubio` | n/a |
| Logs (packaged app) | `%LOCALAPPDATA%\occlubio\logs` | `~/Library/Logs/occlubio/occlubio.log` | n/a |
| Face model cache | `%USERPROFILE%\.insightface` | `~/.insightface` | n/a |

## Features that behave differently

- **Annotated output video.** On macOS, OpenCV writes H.264 (`avc1`), so the console plays the result
  inline in Safari, Chrome and on iOS. On Windows the output stays `mp4v`, which browsers can't play;
  the console shows a download link and suggests VLC, exactly as before. If a Mac's OpenCV build can't
  write H.264, it falls back to `mp4v` automatically.
- **iPhone clips (HEVC/H.265 `.mov`).** OpenCV on macOS decodes them. On Windows decoding depends on the
  OpenCV build. If decoding fails, the job now reports "re-export as H.264 / Most Compatible" instead of
  an empty result. The iOS photo picker usually hands Safari an H.264 copy anyway.
- **Webcam enrollment.**
  - Desktop browsers on the same machine (`localhost`): works.
  - iPhone/iPad or any other device over the LAN: the live camera **requires HTTPS**. Over plain HTTP
    the console says so and points to the Photos button, which on iOS can take a picture directly.
  - Inside `occlubio.app` (macOS): the window uses WebKit. If the camera doesn't start there, use
    **occlubio → Open in Browser**.
- **macOS app window.** `occlubio.app` shows the console in a native window. It has the standard app
  menu (About, Open in Browser, Show Authority Sign-up Code, Hide, **Quit with Cmd+Q**), then the
  Edit (Cmd+C/V/X/A) and View menus. The window opens at up to 1280×840, shrunk to fit smaller
  screens. Closing the window quits the
  app and stops the server, as with other single-window utility apps. The Dock icon behaves normally.
  Keyboard shortcuts in the console itself are limited to `Esc` (close the clip player), which is the
  same on all platforms.
- **Downloads in the iOS app.** Tapping "Download" opens the video in the in-app viewer instead of
  saving it. To save a file to Files, open the console in Safari.
- **Timestamps.** All times are the server's local wall-clock time, the same clock as the
  "clip starts/ends" fields in the console. Explicit offsets sent to the API (`…Z`, `+05:30`) are
  converted to server-local time. Run the server in the same time zone as its operators. Rows
  written before this convention (by versions that used UTC) keep their old values.
- **Training (`occlubio.training`).** Uses CUDA if available, then Apple-Silicon `mps`, then CPU.
  It is not included in the packaged macOS app.
- **CoreML acceleration on macOS (opt-in).** onnxruntime's `CoreMLExecutionProvider` is not enabled by
  default because its accuracy hasn't been verified against the CPU path. To try it, put
  `CoreMLExecutionProvider` first under `device.providers` in `configs/default.yaml`. Unavailable
  providers are dropped automatically on every OS.
- **Scalability report PDF (`scalebench`).** Uses headless Edge or Chrome on Windows, Chrome/Edge/Chromium
  from `/Applications` on macOS, and `chromium` on Linux. With none installed it skips the PDF; the HTML
  report is still complete.

## Disabled or not available

| Feature | Where | Why |
|---|---|---|
| Running detection/recognition on the device | iOS / iPadOS | insightface, onnxruntime-python, OpenCV-python and FAISS have no iOS builds, and an iOS app can't host a long-running background server. Doing it on-device would mean a separate Swift + Core ML rewrite (not done; see PORTING_PLAN.md §3) |
| Using the app without a server | iOS / iPadOS | Same reason. With no server it shows a "Can't reach the occlubio server" screen with a retry button |
| Live camera over plain HTTP from another device | iOS (and all browsers) | Browsers only allow the camera in a secure context. Use HTTPS (below) or photo upload |
| Saving downloads to Files from inside the iOS app | iOS app | The Capacitor web view doesn't handle file downloads. Use Safari |
| Universal2 (single fat) macOS binary | macOS | onnxruntime, faiss-cpu and opencv ship single-arch wheels. Build once per arch (CI does both) |
| DeepStream / TensorRT edge deployment | Windows, macOS, iOS | NVIDIA Jetson / Linux + CUDA only (unchanged) |
| GPU training | macOS | CUDA only. Apple-Silicon `mps` works but is slower; no CUDA on Macs |

## Serving the console to phones and tablets

1. Make a certificate the devices trust. The easiest way is [mkcert](https://github.com/FiloSottile/mkcert):
   `mkcert -install` and `mkcert 192.168.1.20 my-laptop.local` (use your machine's LAN IP/hostname).
   To make the iPhone/iPad trust it: AirDrop or email it the mkcert root CA (`mkcert -CAROOT`), install
   the profile, then enable it under *Settings → General → About → Certificate Trust Settings*.
2. Start the server on the LAN with HTTPS:
   - Windows: `.\run_server.ps1 -Bind 0.0.0.0 -CertFile 192.168.1.20+1.pem -KeyFile 192.168.1.20+1-key.pem`
   - macOS: `./run_server.sh --bind 0.0.0.0 --cert 192.168.1.20+1.pem --key 192.168.1.20+1-key.pem`
3. Open `https://192.168.1.20:8001` in Safari and use *Share → Add to Home Screen*, or build the iOS
   app pointing at that URL (README → iOS).

Binding beyond `127.0.0.1` exposes a biometric system to the network. Read §0 (responsible use) of
`OCCLUSION_ROBUST_FR_ARCHITECTURE.md` first. The default stays localhost-only.

## Needs your action

- **Apple Developer Program membership** (paid, $99/year). You need it to sign with a Developer ID,
  notarize the `.dmg`, and run the iOS app on a physical device or distribute it via TestFlight / App
  Store. Without it:
  - the `.dmg` is ad-hoc signed and Gatekeeper warns (right-click → Open the first time);
  - the iOS app runs only in the Simulator, or on your own device with a free personal team (7-day
    provisioning).
- **macOS signing + notarization secrets.** Set these repository secrets for CI, or export them locally
  before running `packaging/macos/build_macos.sh`:
  - `APPLE_CERT_P12` (base64 of the Developer ID Application `.p12`) and `APPLE_CERT_PASSWORD`;
  - `APPLE_SIGNING_IDENTITY`, e.g. `Developer ID Application: Your Name (TEAMID)`;
  - `APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_APP_PASSWORD` (an app-specific password), or locally
    `APPLE_NOTARY_PROFILE` from `xcrun notarytool store-credentials`.

  Nothing is hardcoded. With no secrets set, CI still builds and smoke-tests an ad-hoc-signed `.dmg`.
- **iOS signing.** Open `ios-client/ios/App/App.xcodeproj` in Xcode, pick your team under *Signing &
  Capabilities*, and change the bundle ID if `io.github.shashwatpandeyoffcog.occlubio` doesn't suit you.
  The bundle ID can also be set with `OCCLUBIO_BUNDLE_ID` at sync time.
- **Bundle identifier.** The macOS and iOS apps default to `io.github.shashwatpandeyoffcog.occlubio`.
  Override it with `OCCLUBIO_BUNDLE_ID`.
- **Physical-device testing.** CI covers the iOS Simulator only. Camera capture, HTTPS trust of your
  mkcert CA, and local-network permission prompts need a pass on a real iPhone and iPad.
- **First macOS run of the camera inside `occlubio.app`.** Grant camera access when macOS asks. Verify
  on a real Mac, because CI runners have no camera.
- **App Store review** (only if you publish). Apple's guideline 4.2 rejects apps that are "just a
  website". This client fits enterprise / TestFlight / ad-hoc distribution better than a public App Store
  listing. Facial-recognition apps also face extra privacy review.

import type { CapacitorConfig } from '@capacitor/cli';

// The app is a thin shell around the occlubio web console: face detection,
// recognition and storage run on the server (Windows / macOS / Linux / DGX).
// Point it at your server when syncing, e.g.
//   OCCLUBIO_SERVER_URL=https://192.168.1.20:8001 npm run sync
// The default suits the iOS Simulator talking to a server on the same Mac.
const serverUrl = process.env.OCCLUBIO_SERVER_URL ?? 'http://localhost:8001';
const host = new URL(serverUrl).hostname;

const config: CapacitorConfig = {
  appId: process.env.OCCLUBIO_BUNDLE_ID ?? 'io.github.shashwatpandeyoffcog.occlubio',
  appName: 'occlubio',
  webDir: 'www',
  server: {
    url: serverUrl,
    cleartext: serverUrl.startsWith('http:'),
    allowNavigation: [host],
    errorPath: 'error.html',
  },
  ios: {
    contentInset: 'never',
    backgroundColor: '#080b14',
  },
};

export default config;

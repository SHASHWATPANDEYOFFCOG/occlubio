import { writeFileSync } from 'node:fs';

const url = process.env.OCCLUBIO_SERVER_URL ?? 'http://localhost:8001';
new URL(url);
writeFileSync(new URL('../www/server-config.js', import.meta.url),
  `window.OCCLUBIO_SERVER_URL = ${JSON.stringify(url)};\n`);
console.log(`occlubio server URL: ${url}`);

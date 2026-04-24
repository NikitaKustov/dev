/**
 * Скачивает JS/CSS зависимости в ./vendor для офлайн-запуска.
 * Запуск: node scripts/download-vendor.mjs
 * Требуется Node.js 18+ (fetch).
 */
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, '..');
const vendorDir = path.join(root, 'vendor');

const TF = '4.22.0';
const POSE = '2.1.3';
const BOOTSTRAP = '4.6.2';

const files = [
  [
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@${TF}/dist/tf-core.min.js`,
    'tf-core.min.js',
  ],
  [
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter@${TF}/dist/tf-converter.min.js`,
    'tf-converter.min.js',
  ],
  [
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@${TF}/dist/tf-backend-webgl.min.js`,
    'tf-backend-webgl.min.js',
  ],
  [
    `https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@${POSE}/dist/pose-detection.min.js`,
    'pose-detection.min.js',
  ],
  [
    `https://cdn.jsdelivr.net/npm/bootstrap@${BOOTSTRAP}/dist/css/bootstrap.min.css`,
    'bootstrap.min.css',
  ],
];

async function download(url, dest) {
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${url}`);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  await fs.writeFile(dest, buf);
}

async function main() {
  await fs.mkdir(vendorDir, { recursive: true });
  for (const [url, name] of files) {
    const dest = path.join(vendorDir, name);
    process.stdout.write(`Downloading ${name}... `);
    await download(url, dest);
    console.log('ok');
  }
  console.log('Done. Vendor files are in', vendorDir);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

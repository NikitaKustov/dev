const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const readline = require('readline');
const express = require('express');
const cors = require('cors');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const multer = require('multer');
const db = require('./db');

let ffmpegPath;
try {
    ffmpegPath = require('ffmpeg-static');
} catch {
    ffmpegPath = null;
}

const JWT_SECRET = process.env.JWT_SECRET || 'dev-body-tracker-secret-change-me';
const PORT = Number(process.env.PORT) || 3000;
const ROOT = path.join(__dirname, '..');

const uploadsRoot = path.join(__dirname, 'uploads');
const userVideosDir = path.join(uploadsRoot, 'user_videos');
const eventsDir = path.join(uploadsRoot, 'events');
const testDir = path.join(uploadsRoot, 'test');
const avatarsDir = path.join(uploadsRoot, 'avatars');
for (const d of [uploadsRoot, userVideosDir, eventsDir, testDir, avatarsDir]) {
    fs.mkdirSync(d, { recursive: true });
}

function diskStorageFor(destDir) {
    return multer.diskStorage({
        destination: (_req, _file, cb) => cb(null, destDir),
        filename: (_req, file, cb) => {
            const ext = path.extname(file.originalname || '').toLowerCase() || '.webm';
            cb(null, `${Date.now()}-${Math.random().toString(36).slice(2, 10)}${ext}`);
        },
    });
}

const videoFilter = (_req, file, cb) => {
    const ok =
        /^video\//.test(file.mimetype) ||
        /\.(mp4|webm|mkv|avi|mov)$/i.test(file.originalname);
    if (ok) cb(null, true);
    else cb(new Error('Only video files are allowed'));
};

const uploadResult = multer({
    storage: diskStorageFor(userVideosDir),
    limits: { fileSize: 250 * 1024 * 1024 },
    fileFilter: videoFilter,
});
const uploadEventVideo = multer({
    storage: diskStorageFor(eventsDir),
    limits: { fileSize: 250 * 1024 * 1024 },
    fileFilter: videoFilter,
});
const uploadTestVideo = multer({
    storage: diskStorageFor(testDir),
    limits: { fileSize: 250 * 1024 * 1024 },
    fileFilter: videoFilter,
});
const uploadAvatar = multer({
    storage: diskStorageFor(avatarsDir),
    limits: { fileSize: 10 * 1024 * 1024 },
    fileFilter: (_req, file, cb) => {
        const ok =
            /^image\//.test(file.mimetype) ||
            /\.(png|jpg|jpeg|webp)$/i.test(file.originalname);
        if (ok) cb(null, true);
        else cb(new Error('Only image files are allowed'));
    },
});
const uploadImage = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 20 * 1024 * 1024 },
});

const app = express();
app.use(cors({ origin: true, credentials: true }));
app.use(express.json());

function authMiddleware(req, res, next) {
    const h = req.headers.authorization || '';
    const m = /^Bearer\s+(.+)$/i.exec(h);
    if (!m) return res.status(401).json({ error: 'Missing token' });
    try {
        const payload = jwt.verify(m[1], JWT_SECRET);
        req.userId = payload.id_users;
        next();
    } catch {
        res.status(401).json({ error: 'Invalid token' });
    }
}

function authAdmin(req, res, next) {
    const u = db.get('SELECT id_users, name FROM users WHERE id_users = ?', [req.userId]);
    if (!u || String(u.name).toLowerCase() !== 'admin') {
        return res.status(403).json({ error: 'Admin only' });
    }
    req.user = u;
    next();
}

function makeUserRow(row) {
    return {
        id_users: row.id_users,
        name: row.name,
        lastname: row.lastname,
        firstname: row.firstname,
        email: row.email,
        avatar: row.avatar || '',
    };
}

app.get('/api/health', (_req, res) => res.json({ ok: true }));

app.get('/api/inference/health', (_req, res) => {
    const modelPath =
        process.env.POSE_MODEL_PATH ||
        path.join(ROOT, 'ml-pipeline', 'exports', 'best.onnx');
    res.json({
        ok: true,
        modelPath,
        modelExists: fs.existsSync(modelPath),
        pythonCommand: process.env.PYTHON_BIN || 'python',
    });
});

/** True if YOLO returned at least one detection with at least one keypoint. */
function hasValidPoseResult(result) {
    if (!result || !Array.isArray(result.detections)) return false;
    return result.detections.some(
        (d) => d && Array.isArray(d.keypoints) && d.keypoints.length > 0
    );
}

/** One-shot spawn (no daemon). Optional overrides: { conf, imgsz } */
function runOneShotPoseInference(imagePath, overrides = {}) {
    return new Promise((resolveOne, rejectOne) => {
        const pythonBin =
            process.env.PYTHON_BIN || (process.platform === 'win32' ? 'py' : 'python');
        const modelPath =
            process.env.POSE_MODEL_PATH ||
            path.join(ROOT, 'ml-pipeline', 'exports', 'best.onnx');
        const scriptPath = path.join(ROOT, 'ml-pipeline', 'inference', 'pose_infer.py');
        const timeoutMs = Number(process.env.POSE_TIMEOUT_MS || 60000);
        const confDefault = process.env.POSE_CONF || '0.1';
        const imgszDefault = process.env.POSE_IMGSZ || '960';
        if (!fs.existsSync(scriptPath)) {
            rejectOne(new Error(`Inference script not found: ${scriptPath}`));
            return;
        }
        if (!fs.existsSync(modelPath)) {
            rejectOne(new Error(`Pose model not found: ${modelPath}`));
            return;
        }
        const conf =
            overrides.conf !== undefined && overrides.conf !== null
                ? String(overrides.conf)
                : confDefault;
        const imgsz =
            overrides.imgsz !== undefined && overrides.imgsz !== null
                ? String(overrides.imgsz)
                : imgszDefault;
        const args = [
            ...(pythonBin === 'py' ? ['-3', '-u'] : ['-u']),
            scriptPath,
            '--model',
            modelPath,
            '--image',
            imagePath,
            '--imgsz',
            imgsz,
            '--conf',
            conf,
            '--device',
            process.env.POSE_DEVICE || 'cpu',
        ];
        const proc = spawn(pythonBin, args, { stdio: ['ignore', 'pipe', 'pipe'] });
        let out = '';
        let err = '';
        let isTimedOut = false;
        const timer = setTimeout(() => {
            isTimedOut = true;
            try {
                proc.kill('SIGKILL');
            } catch (_) {}
        }, timeoutMs);
        proc.stdout.on('data', (d) => (out += d.toString()));
        proc.stderr.on('data', (d) => (err += d.toString()));
        proc.on('error', rejectOne);
        proc.on('close', (code) => {
            clearTimeout(timer);
            if (isTimedOut) {
                rejectOne(new Error(`Inference timeout after ${timeoutMs}ms`));
                return;
            }
            if (code !== 0) {
                rejectOne(new Error(`Inference failed with code ${code}: ${err || out}`));
                return;
            }
            try {
                const stdoutText = (out || '').trim();
                if (!stdoutText) {
                    resolveOne({});
                    return;
                }
                const lines = stdoutText
                    .split(/\r?\n/)
                    .map((l) => l.trim())
                    .filter(Boolean);
                let jsonLine = lines[lines.length - 1] || '{}';
                for (let i = lines.length - 1; i >= 0; i--) {
                    if (lines[i].startsWith('{') || lines[i].startsWith('[')) {
                        jsonLine = lines[i];
                        break;
                    }
                }
                resolveOne(JSON.parse(jsonLine));
            } catch (e) {
                rejectOne(new Error(`Invalid inference JSON: ${e.message}`));
            }
        });
    });
}

/**
 * Daemon is fast but may return empty detections on failure; always fall back to one-shot.
 * poseDaemon is defined below (used at call time).
 */
function runPoseInference(imagePath) {
    return (async () => {
        let result = null;
        if (poseDaemon && poseDaemon.isReady()) {
            try {
                result = await poseDaemon.infer({ imagePath });
            } catch {
                result = null;
            }
        }
        if (hasValidPoseResult(result)) return result;

        try {
            result = await runOneShotPoseInference(imagePath, {});
        } catch {
            result = {};
        }
        if (hasValidPoseResult(result)) return result;

        try {
            result = await runOneShotPoseInference(imagePath, { conf: 0.05 });
        } catch {
            result = {};
        }
        if (hasValidPoseResult(result)) return result;

        try {
            result = await runOneShotPoseInference(imagePath, { conf: 0.05, imgsz: 960 });
        } catch {
            result = { detections: [] };
        }
        return result;
    })();
}

// ---------------- Pose daemon (single process) ----------------
function createPoseDaemon() {
    const pythonBin =
        process.env.PYTHON_BIN || (process.platform === 'win32' ? 'py' : 'python');
    const modelPath =
        process.env.POSE_MODEL_PATH || path.join(ROOT, 'ml-pipeline', 'exports', 'best.onnx');
    const scriptPath = path.join(ROOT, 'ml-pipeline', 'inference', 'pose_daemon.py');
    const imgsz = Number(process.env.POSE_IMGSZ || 960);
    const conf = Number(process.env.POSE_CONF || 0.1);
    const device = String(process.env.POSE_DEVICE || 'cpu');
    const timeoutMs = Number(process.env.POSE_TIMEOUT_MS || 60000);
    const maxQueue = Number(process.env.POSE_QUEUE_MAX || 2);

    let proc = null;
    let rl = null;
    let ready = false;
    const queue = [];
    const pending = [];

    function start() {
        if (!fs.existsSync(scriptPath) || !fs.existsSync(modelPath)) {
            return;
        }
        const args = [
            ...(pythonBin === 'py' ? ['-3', '-u'] : ['-u']),
            scriptPath,
        ];
        proc = spawn(pythonBin, args, { stdio: ['pipe', 'pipe', 'pipe'] });
        rl = readline.createInterface({ input: proc.stdout });
        ready = true;
        proc.stderr.on('data', (d) => {
            const t = String(d || '').trim();
            if (t) console.error('[pose-daemon]', t);
        });

        rl.on('line', (line) => {
            const job = pending.shift();
            if (!job) return;
            clearTimeout(job.timer);
            try {
                const txt = String(line || '').trim();
                job.resolve(txt ? JSON.parse(txt) : { detections: [] });
            } catch (e) {
                job.reject(new Error('Invalid daemon JSON: ' + e.message));
            } finally {
                pump();
            }
        });

        proc.on('exit', () => {
            ready = false;
            try {
                rl && rl.close();
            } catch (_) {}
            rl = null;
            proc = null;
            // reject everything
            while (pending.length) {
                const j = pending.shift();
                clearTimeout(j.timer);
                j.reject(new Error('Pose daemon exited'));
            }
            while (queue.length) {
                const j = queue.shift();
                j.reject(new Error('Pose daemon not available'));
            }
        });
    }

    function isReady() {
        return !!(ready && proc && proc.stdin && !proc.killed);
    }

    function pump() {
        if (!isReady()) return;
        if (pending.length > 0) return; // only 1 in-flight (keeps ordering simple)
        const job = queue.shift();
        if (!job) return;
        const payload = {
            model: modelPath,
            image: job.imagePath,
            imgsz,
            conf,
            device,
        };
        const timer = setTimeout(() => {
            // kill daemon on hard hang; it will auto-restart on next request
            try {
                proc.kill('SIGKILL');
            } catch (_) {}
            job.reject(new Error(`Inference timeout after ${timeoutMs}ms`));
        }, timeoutMs);
        pending.push({ ...job, timer });
        try {
            proc.stdin.write(JSON.stringify(payload) + '\n');
        } catch (e) {
            clearTimeout(timer);
            pending.pop();
            job.reject(e);
        }
    }

    async function infer({ imagePath }) {
        if (!isReady()) {
            start();
        }
        if (!isReady()) {
            throw new Error('Pose daemon not available');
        }
        if (queue.length >= maxQueue) {
            throw new Error('Pose daemon busy');
        }
        return await new Promise((resolve, reject) => {
            queue.push({ imagePath, resolve, reject });
            pump();
        });
    }

    start();
    return { infer, isReady };
}

const poseDaemon = createPoseDaemon();

app.post('/api/inference/pose', authMiddleware, (req, res) => {
    uploadImage.single('image')(req, res, async (err) => {
        if (err) return res.status(400).json({ error: err.message || 'Image upload failed' });
        if (!req.file) return res.status(400).json({ error: 'image file is required' });
        const ext = path.extname(req.file.originalname || '').toLowerCase() || '.jpg';
        const safeExt = ['.jpg', '.jpeg', '.png', '.webp', '.bmp'].includes(ext)
            ? ext
            : '.jpg';
        const tmpPath = path.join(
            uploadsRoot,
            `pose-${Date.now()}-${Math.random().toString(36).slice(2, 10)}${safeExt}`
        );
        try {
            fs.writeFileSync(tmpPath, req.file.buffer);
            const result = await runPoseInference(tmpPath);
            res.json({ ok: true, result });
        } catch (e) {
            console.error(e);
            res.status(500).json({ error: e.message || 'Pose inference error' });
        } finally {
            try {
                fs.unlinkSync(tmpPath);
            } catch (_) {}
        }
    });
});

app.post('/api/register', (req, res) => {
    const { name, lastname, firstname, email, password } = req.body || {};
    if (!name || !lastname || !firstname || !email || !password) {
        return res.status(400).json({ error: 'All fields are required' });
    }
    if (String(password).length < 6) {
        return res.status(400).json({ error: 'Password must be at least 6 characters' });
    }
    const hash = bcrypt.hashSync(String(password), 10);
    const em = String(email).toLowerCase();
    const login = String(name).trim();
    if (!login) {
        return res.status(400).json({ error: 'All fields are required' });
    }
    try {
        const existingEmail = db.get('SELECT id_users FROM users WHERE email = ?', [em]);
        if (existingEmail) {
            return res.status(409).json({ error: 'Email already registered' });
        }
        const existingLogin = db.get('SELECT id_users FROM users WHERE lower(name) = lower(?)', [login]);
        if (existingLogin) {
            return res.status(409).json({ error: 'Login already registered' });
        }
        db.run(
            'INSERT INTO users (name, lastname, firstname, email, password) VALUES (?, ?, ?, ?, ?)',
            [login, lastname, firstname, em, hash]
        );
        const row = db.get(
            'SELECT id_users, name, lastname, firstname, email, avatar FROM users WHERE lower(email) = ?',
            [em]
        );
        if (!row) {
            console.error('Register: insert succeeded but row not found for email', em);
            return res.status(500).json({ error: 'Registration failed' });
        }
        const token = jwt.sign({ id_users: row.id_users }, JWT_SECRET, { expiresIn: '7d' });
        res.status(201).json({ token, user: makeUserRow(row) });
    } catch (e) {
        const msg = String(e.message || e);
        if (msg.includes('UNIQUE') || msg.includes('constraint')) {
            return res.status(409).json({ error: 'Email already registered' });
        }
        console.error(e);
        res.status(500).json({ error: 'Registration failed' });
    }
});

app.post('/api/login', (req, res) => {
    const { identity, password } = req.body || {};
    if (!identity || !password) {
        return res.status(400).json({ error: 'Email/login and password required' });
    }
    const raw = String(identity).trim();
    const em = raw.toLowerCase();
    const row = db.get(
        'SELECT id_users, name, lastname, firstname, email, avatar, password FROM users WHERE lower(email) = ? OR lower(name) = ?',
        [em, em]
    );
    if (!row || !bcrypt.compareSync(String(password), row.password)) {
        return res.status(401).json({ error: 'Invalid credentials' });
    }
    const token = jwt.sign({ id_users: row.id_users }, JWT_SECRET, { expiresIn: '7d' });
    res.json({ token, user: makeUserRow(row) });
});

app.get('/api/me', authMiddleware, (req, res) => {
    const row = db.get(
        'SELECT id_users, name, lastname, firstname, email, avatar FROM users WHERE id_users = ?',
        [req.userId]
    );
    if (!row) return res.status(404).json({ error: 'User not found' });
    res.json(makeUserRow(row));
});

app.put('/api/me', authMiddleware, (req, res) => {
    const { name, lastname, firstname, email } = req.body || {};
    if (!name || !lastname || !firstname || !email) {
        return res.status(400).json({ error: 'All fields are required' });
    }
    const em = String(email).toLowerCase();
    const login = String(name).trim();
    try {
        const sameEmail = db.get(
            'SELECT id_users FROM users WHERE lower(email) = ? AND id_users <> ?',
            [em, req.userId]
        );
        if (sameEmail) return res.status(409).json({ error: 'Email already registered' });
        const sameLogin = db.get(
            'SELECT id_users FROM users WHERE lower(name) = lower(?) AND id_users <> ?',
            [login, req.userId]
        );
        if (sameLogin) return res.status(409).json({ error: 'Login already registered' });
        db.run(
            'UPDATE users SET name=?, lastname=?, firstname=?, email=? WHERE id_users=?',
            [login, lastname, firstname, em, req.userId]
        );
        const row = db.get(
            'SELECT id_users, name, lastname, firstname, email, avatar FROM users WHERE id_users = ?',
            [req.userId]
        );
        res.json({ ok: true, user: makeUserRow(row) });
    } catch (e) {
        const msg = String(e.message || e);
        if (msg.includes('UNIQUE') || msg.includes('constraint')) {
            return res.status(409).json({ error: 'Email already registered' });
        }
        res.status(500).json({ error: 'Could not update profile' });
    }
});

app.post('/api/me/avatar', authMiddleware, (req, res) => {
    uploadAvatar.single('avatar')(req, res, (err) => {
        if (err) return res.status(400).json({ error: err.message || 'Avatar upload failed' });
        if (!req.file) return res.status(400).json({ error: 'avatar file is required' });
        const avatarPath = `/uploads/avatars/${req.file.filename}`;
        const oldRow = db.get('SELECT avatar FROM users WHERE id_users = ?', [req.userId]);
        db.run('UPDATE users SET avatar = ? WHERE id_users = ?', [avatarPath, req.userId]);
        if (oldRow && oldRow.avatar && /^\/uploads\/avatars\//.test(oldRow.avatar)) {
            fs.unlink(path.join(__dirname, oldRow.avatar.replace(/^\/+/, '')), () => {});
        }
        const row = db.get(
            'SELECT id_users, name, lastname, firstname, email, avatar FROM users WHERE id_users = ?',
            [req.userId]
        );
        res.json({ ok: true, user: makeUserRow(row) });
    });
});

app.delete('/api/me', authMiddleware, (req, res) => {
    try {
        const row = db.get(
            'SELECT id_users, avatar FROM users WHERE id_users = ?',
            [req.userId]
        );
        if (!row) {
            return res.status(404).json({ error: 'User not found' });
        }
        const videos = db.all('SELECT user_video FROM results WHERE id_users = ?', [req.userId]);
        for (const v of videos || []) {
            if (v && v.user_video) {
                const rel = String(v.user_video).replace(/^\/+/, '');
                fs.unlink(path.join(__dirname, rel), () => {});
            }
        }
        db.run('DELETE FROM results WHERE id_users = ?', [req.userId]);
        if (row.avatar && /^\/uploads\/avatars\//.test(row.avatar)) {
            fs.unlink(path.join(__dirname, row.avatar.replace(/^\/+/, '')), () => {});
        }
        db.run('DELETE FROM users WHERE id_users = ?', [req.userId]);
        res.json({ ok: true });
    } catch (e) {
        console.error(e);
        res.status(500).json({ error: 'Could not delete account' });
    }
});

app.get('/api/events', (_req, res) => {
    const rows = db.all(
        "SELECT id_trainies, name, description, video FROM events WHERE name IN ('Отжимания','Приседания','Пресс') ORDER BY id_trainies ASC",
        []
    );
    res.json(rows);
});

app.post('/api/admin/events/:name/video', authMiddleware, authAdmin, (req, res) => {
    uploadEventVideo.single('video')(req, res, (err) => {
        if (err) return res.status(400).json({ error: err.message || 'Video upload failed' });
        if (!req.file) return res.status(400).json({ error: 'video file is required' });
        const name = String(req.params.name || '');
        if (!['Отжимания', 'Приседания', 'Пресс'].includes(name)) {
            return res.status(400).json({ error: 'Unsupported training name' });
        }
        const video = `/uploads/events/${req.file.filename}`;
        db.run('UPDATE events SET video = ? WHERE name = ?', [video, name]);
        res.json({ ok: true, name, video });
    });
});

app.post('/api/admin/dev/save', authMiddleware, authAdmin, (req, res) => {
    uploadTestVideo.single('video')(req, res, (err) => {
        if (err) return res.status(400).json({ error: err.message || 'Video upload failed' });
        if (!req.file) return res.status(400).json({ error: 'video file is required' });
        const annotations = req.body.annotations || '{}';
        const frameIndex = Number(req.body.frameIndex || 0);
        const metaName = `${path.parse(req.file.filename).name}.json`;
        const metaPath = path.join(testDir, metaName);
        fs.writeFileSync(
            metaPath,
            JSON.stringify(
                {
                    sourceVideo: '',
                    frameIndex,
                    linkedEvent: null,
                    annotations: (() => {
                        try {
                            return JSON.parse(annotations);
                        } catch {
                            return annotations;
                        }
                    })(),
                    savedAt: new Date().toISOString(),
                    byUserId: req.userId,
                },
                null,
                2
            )
        );
        fs.unlink(req.file.path, () => {});
        res.json({
            ok: true,
            meta: `/uploads/test/${metaName}`,
            linkedEvent: null,
        });
    });
});

app.get('/api/results/mine', authMiddleware, (req, res) => {
    const rows = db.all(
        `SELECT r.id, r.user_result_no, r.id_trainies, r.id_users, r.user_video, r.sets_done,
              r.analysis_json, r.trimmed_from_sec, r.trimmed_to_sec,
              e.name AS event_name, e.description AS event_description, e.video AS event_video
       FROM results r
       JOIN events e ON e.id_trainies = r.id_trainies
       WHERE r.id_users = ?
       ORDER BY r.user_result_no DESC, r.id DESC`,
        [req.userId]
    );
    res.json(rows);
});

function convertToMp4(inputPath) {
    return new Promise((resolve, reject) => {
        if (!ffmpegPath) {
            reject(new Error('ffmpeg-static not available'));
            return;
        }
        const outPath = inputPath.replace(/\.[^.]+$/, '') + '.mp4';
        const args = [
            '-y',
            '-i',
            inputPath,
            '-c:v',
            'libx264',
            '-preset',
            'medium',
            '-crf',
            '16',
            '-pix_fmt',
            'yuv420p',
            '-movflags',
            '+faststart',
            '-an',
            outPath,
        ];
        const proc = spawn(ffmpegPath, args, { stdio: 'ignore' });
        proc.on('error', reject);
        proc.on('close', (code) => {
            if (code === 0) {
                fs.unlink(inputPath, () => {});
                resolve(path.basename(outPath));
            } else reject(new Error(`ffmpeg exited ${code}`));
        });
    });
}

app.post('/api/video/convert-mp4', authMiddleware, (req, res) => {
    uploadTestVideo.single('video')(req, res, async (err) => {
        if (err) return res.status(400).json({ error: err.message || 'Upload failed' });
        if (!req.file) return res.status(400).json({ error: 'video file is required' });
        try {
            const ext = path.extname(req.file.filename).toLowerCase();
            let filename = req.file.filename;
            if (ext !== '.mp4') {
                filename = await convertToMp4(req.file.path);
            }
            res.json({ ok: true, url: `/uploads/test/${filename}` });
        } catch (e) {
            console.error(e);
            res.status(500).json({ error: e.message || 'Conversion failed' });
        }
    });
});

app.post('/api/results', authMiddleware, (req, res) => {
    uploadResult.single('user_video')(req, res, (err) => {
        if (err) return res.status(400).json({ error: err.message || 'Upload failed' });
        const id_trainies = Number(req.body.id_trainies);
        const setsDone = Number(req.body.sets_done || 0);
        const trimmedFromSec = Number(req.body.trimmed_from_sec || 0);
        const trimmedToSec = Number(req.body.trimmed_to_sec || 0);
        let analysisJson = null;
        if (req.body.analysis_json) {
            try {
                analysisJson = JSON.stringify(JSON.parse(String(req.body.analysis_json)));
            } catch {
                analysisJson = null;
            }
        }
        if (!id_trainies || !req.file) {
            return res
                .status(400)
                .json({ error: 'id_trainies and user_video file are required' });
        }
        const ev = db.get('SELECT id_trainies FROM events WHERE id_trainies = ?', [id_trainies]);
        if (!ev) {
            fs.unlink(req.file.path, () => {});
            return res.status(400).json({ error: 'Unknown training event' });
        }
        (async () => {
            let filename = req.file.filename;
            const ext = path.extname(filename).toLowerCase();
            if (ext !== '.mp4' && ffmpegPath) {
                try {
                    filename = await convertToMp4(req.file.path);
                } catch (e) {
                    console.warn('MP4 conversion failed, keeping source file:', e.message);
                }
            }
            const publicPath = `/uploads/user_videos/${filename}`;
            const nextNoRow = db.get(
                'SELECT COALESCE(MAX(user_result_no), 0) + 1 AS next_no FROM results WHERE id_users = ?',
                [req.userId]
            );
            const userResultNo = Number(nextNoRow && nextNoRow.next_no ? nextNoRow.next_no : 1);
            db.run(
                'INSERT INTO results (id_trainies, id_users, user_result_no, user_video, sets_done, analysis_json, trimmed_from_sec, trimmed_to_sec) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                [
                    id_trainies,
                    req.userId,
                    userResultNo,
                    publicPath,
                    setsDone > 0 ? setsDone : null,
                    analysisJson,
                    Number.isFinite(trimmedFromSec) ? trimmedFromSec : null,
                    Number.isFinite(trimmedToSec) ? trimmedToSec : null,
                ]
            );
            const id = db.lastInsertRowid();
            res.status(201).json({
                id,
                user_result_no: userResultNo,
                id_trainies,
                id_users: req.userId,
                user_video: publicPath,
                sets_done: setsDone > 0 ? setsDone : null,
                analysis_json: analysisJson,
                trimmed_from_sec: Number.isFinite(trimmedFromSec) ? trimmedFromSec : null,
                trimmed_to_sec: Number.isFinite(trimmedToSec) ? trimmedToSec : null,
            });
        })().catch((e) => {
            console.error(e);
            res.status(500).json({ error: 'Could not save result' });
        });
    });
});

app.delete('/api/results/:id', authMiddleware, (req, res) => {
    const id = Number(req.params.id);
    if (!id) return res.status(400).json({ error: 'Invalid result id' });
    const row = db.get('SELECT id, id_users, user_video FROM results WHERE id = ?', [id]);
    if (!row) return res.status(404).json({ error: 'Result not found' });
    if (Number(row.id_users) !== Number(req.userId)) {
        return res.status(403).json({ error: 'Forbidden' });
    }
    db.run('DELETE FROM results WHERE id = ?', [id]);
    if (row.user_video) {
        fs.unlink(path.join(__dirname, String(row.user_video).replace(/^\/+/, '')), () => {});
    }
    res.json({ ok: true, id });
});

app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use(express.static(ROOT));

async function main() {
    await db.initDatabase();
    app.listen(PORT, () => {
        console.log(`Server http://localhost:${PORT}`);
        console.log(`SQLite file: ${path.join(__dirname, 'data', 'app.db')}`);
    });
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});

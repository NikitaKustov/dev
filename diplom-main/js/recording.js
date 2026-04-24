/**
 * Запись canvas (видео + скелет) и сохранение на сервер в user_videos как MP4 (конвертация на сервере).
 */
const recording = (function () {
    let mediaRecorder = null;
    let chunks = [];
    let stoppedBlob = null;
    let mimeType = '';
    let mirrorCanvas = null;
    let mirrorCtx = null;
    let mirrorReq = null;
    const RECORD_WIDTH = 1702;
    const RECORD_HEIGHT = 834;
    let livePreviewStream = null;
    let waitPreviewTimer = null;
    let processedBlob = null;
    let analysisMeta = null;
    let recordingActive = false;
    const referencePoseCache = new Map();
    const YOLO_SKELETON = [
        ['left_eye', 'right_eye'],
        ['left_eye', 'nose'],
        ['right_eye', 'nose'],
        ['left_ear', 'left_eye'],
        ['right_ear', 'right_eye'],
        ['left_shoulder', 'right_shoulder'],
        ['left_shoulder', 'left_elbow'],
        ['left_elbow', 'left_wrist'],
        ['right_shoulder', 'right_elbow'],
        ['right_elbow', 'right_wrist'],
        ['left_shoulder', 'left_hip'],
        ['right_shoulder', 'right_hip'],
        ['left_hip', 'right_hip'],
        ['left_hip', 'left_knee'],
        ['left_knee', 'left_ankle'],
        ['right_hip', 'right_knee'],
        ['right_knee', 'right_ankle'],
    ];
    /** Согласовано с mirrorCanvas.captureStream(30) — поля по кадрам в processRecordedVideo. */
    const RECORD_NOMINAL_FPS = 30;
    const DRAW_POST_ANALYSIS_POSE = true;

    function pickMime() {
        const opts = [
            'video/mp4',
            'video/webm;codecs=vp9,opus',
            'video/webm;codecs=vp8,opus',
            'video/webm',
        ];
        for (let i = 0; i < opts.length; i++) {
            if (MediaRecorder.isTypeSupported(opts[i])) return opts[i];
        }
        return '';
    }

    function getCanvas() {
        return document.querySelector('#canvas');
    }

    function getRawVideo() {
        return document.querySelector('#video');
    }

    function getRecordingSource() {
        const raw = getRawVideo();
        if (raw && raw.videoWidth && raw.videoHeight && raw.readyState >= 2) {
            return raw;
        }
        return getCanvas();
    }

    function stopMirrorLoop() {
        if (mirrorReq != null) {
            cancelAnimationFrame(mirrorReq);
            mirrorReq = null;
        }
    }

    function drawCover(srcNode) {
        const sw = srcNode && srcNode.videoWidth ? srcNode.videoWidth : srcNode.width;
        const sh = srcNode && srcNode.videoHeight ? srcNode.videoHeight : srcNode.height;
        const dw = RECORD_WIDTH;
        const dh = RECORD_HEIGHT;
        if (!sw || !sh || !mirrorCtx) return;
        mirrorCtx.fillStyle = '#000000';
        mirrorCtx.fillRect(0, 0, dw, dh);
        const srcRatio = sw / sh;
        const dstRatio = dw / dh;
        let sx = 0;
        let sy = 0;
        let sWidth = sw;
        let sHeight = sh;
        if (srcRatio > dstRatio) {
            sWidth = sh * dstRatio;
            sx = (sw - sWidth) / 2;
        } else {
            sHeight = sw / dstRatio;
            sy = (sh - sHeight) / 2;
        }
        mirrorCtx.drawImage(srcNode, sx, sy, sWidth, sHeight, 0, 0, dw, dh);
    }

    function attachLivePreview() {
        const preview = document.getElementById('record_preview');
        if (!preview || !mirrorCanvas) return;
        if (livePreviewStream) {
            livePreviewStream.getTracks().forEach((t) => t.stop());
            livePreviewStream = null;
        }
        livePreviewStream = mirrorCanvas.captureStream(30);
        preview.srcObject = livePreviewStream;
        preview.muted = true;
        preview.play().catch(() => {});
    }

    function detachLivePreview() {
        const preview = document.getElementById('record_preview');
        if (!preview) return;
        preview.pause();
        preview.srcObject = null;
        if (livePreviewStream) {
            livePreviewStream.getTracks().forEach((t) => t.stop());
            livePreviewStream = null;
        }
    }

    function startMirrorLoop(srcCanvas) {
        if (!srcCanvas) return;
        if (mirrorCanvas && mirrorCtx && mirrorReq != null) return;
        mirrorCanvas = document.createElement('canvas');
        mirrorCanvas.width = RECORD_WIDTH;
        mirrorCanvas.height = RECORD_HEIGHT;
        mirrorCtx = mirrorCanvas.getContext('2d');
        const tick = () => {
            if (!mirrorCtx || !srcCanvas) return;
            drawCover(srcCanvas);
            mirrorReq = requestAnimationFrame(tick);
        };
        mirrorReq = requestAnimationFrame(tick);
    }

    function clearWaitPreviewTimer() {
        if (waitPreviewTimer != null) {
            clearInterval(waitPreviewTimer);
            waitPreviewTimer = null;
        }
    }

    function startLivePreviewFromCanvas() {
        const source = getRecordingSource();
        if (!source) return false;
        const sw = source.videoWidth || source.width || 0;
        const sh = source.videoHeight || source.height || 0;
        if (!sw || !sh) return false;
        startMirrorLoop(source);
        attachLivePreview();
        const preview = document.getElementById('record_preview');
        if (preview) {
            preview.controls = false;
        }
        return true;
    }

    function setButtons(state) {
        const start = document.getElementById('btn_rec_start');
        const stop = document.getElementById('btn_rec_stop');
        const save = document.getElementById('btn_rec_save');
        if (!start || !stop || !save) return;
        if (state === 'idle') {
            start.hidden = false;
            stop.hidden = true;
            save.hidden = true;
            save.disabled = true;
        } else if (state === 'recording') {
            start.hidden = true;
            stop.hidden = false;
            save.hidden = true;
        } else if (state === 'stopped') {
            start.hidden = false;
            stop.hidden = true;
            save.hidden = false;
            save.disabled = !stoppedBlob;
        }
    }

    function updatePreview() {
        const preview = document.getElementById('record_preview');
        if (!preview) return;
        const viewBlob = processedBlob || stoppedBlob;
        if (!viewBlob) {
            preview.removeAttribute('src');
            preview.srcObject = null;
            preview.load();
            return;
        }
        detachLivePreview();
        const url = URL.createObjectURL(viewBlob);
        preview.srcObject = null;
        preview.src = url;
        preview.onloadeddata = () => {
            URL.revokeObjectURL(url);
        };
    }

    function angleDeg(a, b, c) {
        if (!a || !b || !c) return null;
        const abx = a.x - b.x;
        const aby = a.y - b.y;
        const cbx = c.x - b.x;
        const cby = c.y - b.y;
        const denom = Math.hypot(abx, aby) * Math.hypot(cbx, cby);
        if (!denom) return null;
        const cos = Math.max(-1, Math.min(1, (abx * cbx + aby * cby) / denom));
        return (Math.acos(cos) * 180) / Math.PI;
    }

    function getKp(frame, name) {
        const map = frame && frame.map;
        if (!map) return null;
        return map[name] || null;
    }

    function estimateActiveRange(frames) {
        if (!frames.length) return { start: 0, end: 0 };
        const energies = frames.map((f, i) => {
            if (i === 0 || !f.pose || !frames[i - 1].pose) return 0;
            const names = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'];
            let s = 0;
            let n = 0;
            names.forEach((name) => {
                const p = f.map[name];
                const q = frames[i - 1].map[name];
                if (p && q) {
                    s += Math.hypot(p.x - q.x, p.y - q.y);
                    n += 1;
                }
            });
            return n ? s / n : 0;
        });
        const maxE = Math.max(0.001, ...energies);
        const thr = Math.max(2, maxE * 0.16);
        let start = 0;
        let end = frames.length - 1;
        while (start < end && energies[start] < thr) start += 1;
        while (end > start && energies[end] < thr) end -= 1;
        return { start, end };
    }

    function parseExerciseName() {
        const events = training && training._eventByName ? training._eventByName : {};
        const selectedId = Number(training.getSelectedTrainingId ? training.getSelectedTrainingId() : 0);
        const found = Object.values(events).find((e) => Number(e.id_trainies) === selectedId);
        return found && found.name ? found.name : 'Отжимания';
    }

    function getSelectedEventMeta() {
        const events = training && training._eventByName ? training._eventByName : {};
        const selectedId = Number(training.getSelectedTrainingId ? training.getSelectedTrainingId() : 0);
        const found = Object.values(events).find((e) => Number(e.id_trainies) === selectedId);
        return found || null;
    }

    function repSignal(frame, exercise) {
        if (!frame || !frame.pose) return null;
        if (exercise === 'Приседания') {
            const lk = angleDeg(getKp(frame, 'left_hip'), getKp(frame, 'left_knee'), getKp(frame, 'left_ankle'));
            const rk = angleDeg(getKp(frame, 'right_hip'), getKp(frame, 'right_knee'), getKp(frame, 'right_ankle'));
            if (lk == null && rk == null) return null;
            return ((lk || 0) + (rk || 0)) / (lk && rk ? 2 : 1);
        }
        if (exercise === 'Пресс') {
            const lh = angleDeg(getKp(frame, 'left_shoulder'), getKp(frame, 'left_hip'), getKp(frame, 'left_knee'));
            const rh = angleDeg(getKp(frame, 'right_shoulder'), getKp(frame, 'right_hip'), getKp(frame, 'right_knee'));
            if (lh == null && rh == null) return null;
            return ((lh || 0) + (rh || 0)) / (lh && rh ? 2 : 1);
        }
        const le = angleDeg(getKp(frame, 'left_shoulder'), getKp(frame, 'left_elbow'), getKp(frame, 'left_wrist'));
        const re = angleDeg(getKp(frame, 'right_shoulder'), getKp(frame, 'right_elbow'), getKp(frame, 'right_wrist'));
        if (le == null && re == null) return null;
        return ((le || 0) + (re || 0)) / (le && re ? 2 : 1);
    }

    function analyzeTechnique(frames, exercise, setsTarget) {
        const errors = [];
        const values = [];
        frames.forEach((f) => {
            const s = repSignal(f, exercise);
            if (s != null) values.push(s);
        });
        if (!values.length) {
            return {
                summary: 'Недостаточно данных для анализа техники',
                reps: 0,
                setsTarget,
                errors: ['Камера не видит тело стабильно'],
            };
        }
        const minV = Math.min(...values);
        const maxV = Math.max(...values);
        const span = maxV - minV;
        let reps = 0;
        let prev = values[0];
        const lo = minV + span * 0.35;
        const hi = minV + span * 0.65;
        let state = prev > hi ? 'up' : 'down';
        for (let i = 1; i < values.length; i++) {
            const v = values[i];
            if (state === 'up' && v < lo) {
                state = 'down';
            } else if (state === 'down' && v > hi) {
                state = 'up';
                reps += 1;
            }
            prev = v;
        }
        if (span < 20) {
            errors.push('Слишком малая амплитуда движения');
        }
        if (exercise === 'Отжимания' && minV > 85) {
            errors.push('Недостаточная глубина отжиманий');
        }
        if (exercise === 'Приседания' && minV > 95) {
            errors.push('Присед слишком неглубокий');
        }
        if (exercise === 'Пресс' && minV > 105) {
            errors.push('Недостаточный подъем корпуса');
        }
        if (setsTarget > 0 && reps < setsTarget) {
            errors.push('Количество повторов меньше указанного числа подходов');
        }
        const summary = errors.length
            ? `Найдены ошибки техники (${errors.length})`
            : 'Критичных ошибок не обнаружено';
        return { summary, reps, setsTarget, errors };
    }

    function signalFromFrames(frames, exercise) {
        const values = [];
        for (let i = 0; i < frames.length; i++) {
            const s = repSignal(frames[i], exercise);
            if (s != null && Number.isFinite(s)) values.push(s);
        }
        return values;
    }

    function resampleSeries(values, targetLen) {
        if (!Array.isArray(values) || !values.length || targetLen < 2) return [];
        if (values.length === 1) return Array(targetLen).fill(values[0]);
        const out = [];
        const last = values.length - 1;
        for (let i = 0; i < targetLen; i++) {
            const p = (i / (targetLen - 1)) * last;
            const l = Math.floor(p);
            const r = Math.min(last, Math.ceil(p));
            const t = p - l;
            out.push(values[l] * (1 - t) + values[r] * t);
        }
        return out;
    }

    function normalizeSeries(values) {
        if (!Array.isArray(values) || !values.length) return [];
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + (b - mean) * (b - mean), 0) / values.length;
        const std = Math.sqrt(Math.max(variance, 1e-6));
        return values.map((v) => (v - mean) / std);
    }

    function meanAbsDiff(a, b) {
        if (!a.length || !b.length || a.length !== b.length) return null;
        let s = 0;
        for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s / a.length;
    }

    function compareWithReference(userFrames, refFrames, exercise) {
        if (!userFrames.length || !refFrames.length) {
            return {
                score: null,
                notes: ['Эталон недоступен: сравнение выполнено только по вашим ключевым точкам'],
            };
        }
        const userSig = signalFromFrames(userFrames, exercise);
        const refSig = signalFromFrames(refFrames, exercise);
        if (userSig.length < 6 || refSig.length < 6) {
            return {
                score: null,
                notes: ['Недостаточно стабильных ключевых точек для точного сравнения с эталоном'],
            };
        }
        const targetLen = 72;
        const userN = normalizeSeries(resampleSeries(userSig, targetLen));
        const refN = normalizeSeries(resampleSeries(refSig, targetLen));
        const shapeDiff = meanAbsDiff(userN, refN);
        const uMin = Math.min(...userSig);
        const uMax = Math.max(...userSig);
        const rMin = Math.min(...refSig);
        const rMax = Math.max(...refSig);
        const ampUser = Math.max(0.001, uMax - uMin);
        const ampRef = Math.max(0.001, rMax - rMin);
        const ampRatio = ampUser / ampRef;
        const minDelta = uMin - rMin;
        const notes = [];
        if (shapeDiff != null && shapeDiff > 0.95) {
            notes.push('Траектория движения заметно отличается от эталонной');
        }
        if (ampRatio < 0.72) {
            notes.push('Амплитуда движения ниже эталона');
        } else if (ampRatio > 1.4) {
            notes.push('Амплитуда движения выше эталона (возможен переразгиб)');
        }
        if (exercise === 'Отжимания' && minDelta > 10) {
            notes.push('В нижней точке опускаетесь выше, чем на эталоне');
        }
        if (exercise === 'Приседания' && minDelta > 12) {
            notes.push('Глубина приседа ниже эталона');
        }
        if (exercise === 'Пресс' && minDelta > 12) {
            notes.push('Подъем корпуса ниже эталонного');
        }
        const quality = shapeDiff == null ? 0 : Math.max(0, 100 - shapeDiff * 35 - Math.abs(1 - ampRatio) * 24);
        return {
            score: Math.round(quality),
            notes,
            metrics: {
                shapeDiff: Number(shapeDiff != null ? shapeDiff.toFixed(3) : 0),
                ampRatio: Number(ampRatio.toFixed(3)),
                userSignalRange: [Number(uMin.toFixed(2)), Number(uMax.toFixed(2))],
                refSignalRange: [Number(rMin.toFixed(2)), Number(rMax.toFixed(2))],
            },
        };
    }

    async function buildReferenceFrames(eventMeta, estimatePoses, sampleFps) {
        if (!eventMeta || !eventMeta.video) return [];
        const cacheKey = `${eventMeta.id_trainies || eventMeta.name || 'x'}|${eventMeta.video}|${sampleFps}`;
        const cached = referencePoseCache.get(cacheKey);
        if (cached) return cached;
        const loader = (async () => {
            const v = document.createElement('video');
            v.crossOrigin = 'anonymous';
            v.preload = 'auto';
            v.src = eventMeta.video;
            v.muted = true;
            v.playsInline = true;
            await new Promise((resolve, reject) => {
                v.onloadedmetadata = () => resolve();
                v.onerror = () => reject(new Error('Не удалось загрузить эталонное видео'));
            });
            const frames = await buildFrameSeries(v, estimatePoses, sampleFps);
            const active = estimateActiveRange(frames);
            return frames.slice(active.start, active.end + 1);
        })().catch(() => []);
        referencePoseCache.set(cacheKey, loader);
        return loader;
    }

    async function estimatePosesViaYoloApi(canvas, timeoutMs) {
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.72));
        if (!blob || !window.auth || typeof auth.apiFetch !== 'function') {
            return null;
        }
        const fd = new FormData();
        fd.append('image', blob, 'frame.jpg');
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs || 3000);
        try {
            const r = await auth.apiFetch('/api/inference/pose', {
                method: 'POST',
                body: fd,
                signal: controller.signal,
            });
            const dets = r && r.result && Array.isArray(r.result.detections) ? r.result.detections : [];
            return dets.map((d) => ({ keypoints: Array.isArray(d.keypoints) ? d.keypoints : [] }));
        } finally {
            clearTimeout(timer);
        }
    }

    async function buildFrameSeries(video, estimatePoses, fps) {
        const c = document.createElement('canvas');
        c.width = video.videoWidth;
        c.height = video.videoHeight;
        const ctx = c.getContext('2d');
        const frames = [];
        const duration = video.duration || 0;
        if (!duration || !fps) return frames;
        const step = 1 / fps;
        for (let t = 0; t < duration; t += step) {
            await new Promise((resolve) => {
                const onSeeked = () => resolve();
                video.addEventListener('seeked', onSeeked, { once: true });
                video.currentTime = Math.min(duration - 0.001, t);
            });
            ctx.drawImage(video, 0, 0, c.width, c.height);
            const poses = await estimatePoses(c);
            const pose = poses && poses.length ? poses[0] : null;
            const map = {};
            if (pose && Array.isArray(pose.keypoints)) {
                pose.keypoints.forEach((k) => {
                    map[k.name] = k;
                });
            }
            frames.push({ t, pose, map });
        }
        return frames;
    }

    function drawOverlay(ctx, pose, analysis, t, duration) {
        ctx.fillStyle = 'rgba(18,24,30,0.48)';
        ctx.fillRect(14, 14, 620, 102);
        ctx.fillStyle = '#e6edf3';
        ctx.font = '24px Segoe UI';
        ctx.fillText(`Анализ: ${analysis.summary}`, 26, 48);
        ctx.font = '20px Segoe UI';
        ctx.fillText(`Повторы: ${analysis.reps}`, 26, 76);
        ctx.fillText(`Время: ${t.toFixed(1)} / ${duration.toFixed(1)} c`, 230, 76);
        if (analysis.errors && analysis.errors.length) {
            ctx.fillStyle = '#ff9b6a';
            ctx.fillText(`Ошибки: ${analysis.errors.slice(0, 2).join(', ')}`, 26, 102);
        }
        if (!DRAW_POST_ANALYSIS_POSE || !pose || !pose.keypoints) return;
        const map = {};
        pose.keypoints.forEach((k) => {
            if (k && k.name) map[k.name] = k;
        });
        for (let i = 0; i < YOLO_SKELETON.length; i++) {
            const a = map[YOLO_SKELETON[i][0]];
            const b = map[YOLO_SKELETON[i][1]];
            if (!a || !b) continue;
            const s = ((a.score || 0) + (b.score || 0)) / 2;
            if (s < 0.32) continue;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = 'rgba(90,220,255,0.82)';
            ctx.lineWidth = 2.4;
            ctx.stroke();
        }
        pose.keypoints.forEach((kp) => {
            if (!kp || (kp.score || 0) < 0.35) return;
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#8cff8c';
            ctx.fill();
        });
    }

    async function processRecordedVideo(inputBlob) {
        if (!window.poseDetection || !window.tf) {
            return { blob: inputBlob, analysis: null, trim: null };
        }
        await tf.ready();
        const exercise = parseExerciseName();
        const setsTarget = training.getSetsCount ? Number(training.getSetsCount()) : 0;
        const videoUrl = URL.createObjectURL(inputBlob);
        const srcVideo = document.createElement('video');
        srcVideo.src = videoUrl;
        srcVideo.muted = true;
        srcVideo.playsInline = true;
        await new Promise((resolve, reject) => {
            srcVideo.onloadedmetadata = () => resolve();
            srcVideo.onerror = () => reject(new Error('Не удалось прочитать запись для анализа'));
        });
        let detector = null;
        let yoloFails = 0;
        const estimatePoses = async (canvas) => {
            try {
                if (yoloFails < 3) {
                    const yoloPoses = await estimatePosesViaYoloApi(canvas, 2800);
                    if (yoloPoses) {
                        return yoloPoses;
                    }
                }
            } catch (_) {
                yoloFails += 1;
            }
            if (!detector) {
                detector = await poseDetection.createDetector(
                    poseDetection.SupportedModels.BlazePose,
                    { runtime: 'tfjs', enableSmoothing: true, modelType: 'heavy' }
                );
            }
            return detector.estimatePoses(canvas, { flipHorizontal: false }, performance.now());
        };
        const durationSec = srcVideo.duration || 0;
        const sampleFps = 8;
        const sampledFrames = await buildFrameSeries(srcVideo, estimatePoses, sampleFps);
        const active = estimateActiveRange(sampledFrames);
        const activeFrames = sampledFrames.slice(active.start, active.end + 1);
        const analysisFrames = activeFrames;
        const analysis = analyzeTechnique(analysisFrames, exercise, setsTarget);
        analysis.analysisWindow = {
            sampleFps,
            sampleFramesTotal: sampledFrames.length,
            activeStartSample: active.start,
            activeEndSample: active.end,
            activeFramesUsed: analysisFrames.length,
            activeStartSec: Number((active.start / sampleFps).toFixed(2)),
            activeEndSec: Number((((active.end + 1) / sampleFps)).toFixed(2)),
        };
        const eventMeta = getSelectedEventMeta();
        const refFrames = await buildReferenceFrames(eventMeta, estimatePoses, sampleFps);
        const refCmp = compareWithReference(analysisFrames, refFrames, exercise);
        if (refCmp && refCmp.notes && refCmp.notes.length) {
            analysis.errors = [...analysis.errors, ...refCmp.notes];
        }
        if (refCmp && typeof refCmp.score === 'number') {
            analysis.referenceScore = refCmp.score;
        }
        if (refCmp && refCmp.metrics) {
            analysis.referenceMetrics = refCmp.metrics;
        }
        if (refCmp && refCmp.notes && refCmp.notes.length) {
            analysis.summary = `Найдены отклонения от эталона (${refCmp.notes.length})`;
        }
        const dur = srcVideo.duration || 0;
        // Keep full recording to preserve all sets and avoid frozen lead-in
        // artifacts caused by aggressive motion-based trimming.
        const trimStart = 0;
        const trimEnd = Math.max(0, dur - 0.01);

        const outCanvas = document.createElement('canvas');
        outCanvas.width = srcVideo.videoWidth;
        outCanvas.height = srcVideo.videoHeight;
        const outCtx = outCanvas.getContext('2d');
        const stream = outCanvas.captureStream(30);
        const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')
            ? 'video/webm;codecs=vp9,opus'
            : 'video/webm';
        const recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 9000000 });
        const chunks = [];
        recorder.ondataavailable = (e) => {
            if (e.data && e.data.size) chunks.push(e.data);
        };
        srcVideo.currentTime = 0;
        await new Promise((resolve) => {
            srcVideo.addEventListener('seeked', resolve, { once: true });
        });
        await srcVideo.play();
        recorder.start(250);
        await new Promise((resolve) => {
            let lastPose = null;
            const tick = () => {
                if (srcVideo.ended || srcVideo.currentTime >= trimEnd) {
                    try {
                        srcVideo.pause();
                    } catch (_) {}
                    recorder.stop();
                    resolve();
                    return;
                }
                outCtx.drawImage(srcVideo, 0, 0, outCanvas.width, outCanvas.height);
                let pose = null;
                if (sampledFrames.length) {
                    const sampleIndex = Math.max(
                        0,
                        Math.min(sampledFrames.length - 1, Math.round(srcVideo.currentTime * sampleFps))
                    );
                    pose = sampledFrames[sampleIndex] && sampledFrames[sampleIndex].pose
                        ? sampledFrames[sampleIndex].pose
                        : null;
                }
                if (pose && pose.keypoints && pose.keypoints.length) {
                    lastPose = pose;
                } else if (lastPose) {
                    pose = lastPose;
                }
                const segLen = Math.max(0.01, trimEnd - trimStart);
                drawOverlay(outCtx, pose, analysis, srcVideo.currentTime - trimStart, segLen);
                requestAnimationFrame(tick);
            };
            tick();
        });
        const outputBlob = await new Promise((resolve) => {
            recorder.onstop = () => resolve(new Blob(chunks, { type: 'video/webm' }));
        });
        if (detector) detector.dispose();
        URL.revokeObjectURL(videoUrl);
        return {
            blob: outputBlob,
            analysis,
            trim: { from: trimStart, to: trimEnd },
        };
    }

    async function saveToServer() {
        const save = document.getElementById('btn_rec_save');
        const errEl = document.getElementById('recording_err');
        if (errEl) {
            errEl.textContent = '';
            errEl.hidden = true;
        }
        if (!stoppedBlob) return;
        const id_trainies =
            typeof training.getSelectedTrainingId === 'function'
                ? Number(training.getSelectedTrainingId())
                : 0;
        if (!id_trainies) {
            if (errEl) {
                errEl.textContent = 'Сначала выберите упражнение.';
                errEl.hidden = false;
            }
            return;
        }
        const sourceBlob = processedBlob || stoppedBlob;
        const ext = mimeType.indexOf('mp4') !== -1 ? 'mp4' : 'webm';
        const file = new File([sourceBlob], `capture.${ext}`, { type: sourceBlob.type || mimeType || 'video/webm' });
        const fd = new FormData();
        fd.append('id_trainies', String(id_trainies));
        fd.append('sets_done', String(training.getSetsCount ? training.getSetsCount() : 0));
        if (analysisMeta && analysisMeta.analysis) {
            fd.append('analysis_json', JSON.stringify(analysisMeta.analysis));
        }
        if (analysisMeta && analysisMeta.trim) {
            fd.append('trimmed_from_sec', String(analysisMeta.trim.from || 0));
            fd.append('trimmed_to_sec', String(analysisMeta.trim.to || 0));
        }
        fd.append('user_video', file);
        if (save) save.disabled = true;
        try {
            await auth.apiFetch('/api/results', { method: 'POST', body: fd });
            stoppedBlob = null;
            processedBlob = null;
            analysisMeta = null;
            setButtons('idle');
            updatePreview();
            if (typeof training.refreshResults === 'function') await training.refreshResults();
        } catch (e) {
            if (errEl) {
                errEl.textContent = e.message || 'Ошибка сохранения';
                errEl.hidden = false;
            }
        } finally {
            if (save) save.disabled = false;
        }
    }

    function init() {
        const start = document.getElementById('btn_rec_start');
        const stop = document.getElementById('btn_rec_stop');
        const save = document.getElementById('btn_rec_save');
        if (!start || !stop || !save) return;

        start.addEventListener('click', () => {
            const source = getRecordingSource();
            const sw = source && (source.videoWidth || source.width || 0);
            if (!source || !sw) {
                const errEl = document.getElementById('recording_err');
                if (errEl) {
                    errEl.textContent = 'Дождитесь запуска видео.';
                    errEl.hidden = false;
                }
                return;
            }
            mimeType = pickMime();
            if (!mimeType || !window.MediaRecorder) {
                alert('Запись не поддерживается в этом браузере.');
                return;
            }
            stopMirrorLoop();
            startMirrorLoop(source);
            attachLivePreview();
            const preview = document.getElementById('record_preview');
            if (preview) {
                preview.controls = false;
            }
            const stream = mirrorCanvas.captureStream(30);
            try {
                mediaRecorder = mimeType
                    ? new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 3500000 })
                    : new MediaRecorder(stream, { videoBitsPerSecond: 3500000 });
            } catch {
                mediaRecorder = new MediaRecorder(stream);
            }
            const rec = mediaRecorder;
            chunks = [];
            stoppedBlob = null;
            rec.ondataavailable = (e) => {
                if (e.data && e.data.size) chunks.push(e.data);
            };
            rec.onstop = () => {
                recordingActive = false;
                stoppedBlob = new Blob(chunks, {
                    type: rec.mimeType || mimeType || 'video/webm',
                });
                processedBlob = null;
                analysisMeta = null;
                mediaRecorder = null;
                stopMirrorLoop();
                setButtons('stopped');
                const saveBtn = document.getElementById('btn_rec_save');
                if (saveBtn) saveBtn.disabled = true;
                updatePreview();
                const preview = document.getElementById('record_preview');
                if (preview) {
                    preview.controls = true;
                }
                const errEl = document.getElementById('recording_err');
                if (errEl) {
                    errEl.textContent = 'Анализ техники и обработка видео...';
                    errEl.hidden = false;
                }
                processRecordedVideo(stoppedBlob)
                    .then((res) => {
                        if (res && res.blob) {
                            processedBlob = res.blob;
                        }
                        analysisMeta = res;
                        updatePreview();
                        if (errEl) {
                            errEl.hidden = true;
                            errEl.textContent = '';
                        }
                        setButtons('stopped');
                    })
                    .catch((e) => {
                        console.warn(e);
                        if (errEl) {
                            errEl.textContent = 'Не удалось выполнить полный анализ, сохранится исходная запись.';
                            errEl.hidden = false;
                        }
                        if (saveBtn) saveBtn.disabled = false;
                    });
            };
            rec.start(250);
            recordingActive = true;
            setButtons('recording');
            const errEl = document.getElementById('recording_err');
            if (errEl) {
                errEl.textContent = '';
                errEl.hidden = true;
            }
        });

        stop.addEventListener('click', () => {
            if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
            try {
                if (mediaRecorder.state === 'recording') {
                    mediaRecorder.requestData();
                }
            } catch (_) {}
            mediaRecorder.stop();
        });

        save.addEventListener('click', () => saveToServer());

        const startCamera = document.getElementById('btn_start_camera');
        if (startCamera) {
            startCamera.addEventListener('click', () => {
                clearWaitPreviewTimer();
                if (startLivePreviewFromCanvas()) return;
                let tries = 0;
                waitPreviewTimer = setInterval(() => {
                    tries += 1;
                    if (startLivePreviewFromCanvas() || tries > 100) {
                        clearWaitPreviewTimer();
                    }
                }, 100);
            });
        }

        setButtons('idle');
    }

    function isRecording() {
        return recordingActive;
    }

    return { init, isRecording };
})();

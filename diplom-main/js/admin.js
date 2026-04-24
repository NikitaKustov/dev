const admin = (function () {
    let isAdmin = false;
    let currentPose = null;
    let dragKey = null;
    let customPointSeq = 0;
    let linkMode = false;
    let pendingLinkStart = null;
    let actionHistory = [];
    let frameAnnotations = {};
    let autoDetectOnFrameChange = true;
    let detectRunToken = 0;
    let lastAutoPose = null;

    const skeleton = [
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

    function byName(list) {
        const m = {};
        for (const p of list || []) m[p.name] = p;
        return m;
    }

    function meanScore(pose) {
        if (!pose || !pose.keypoints || !pose.keypoints.length) return 0;
        let s = 0;
        for (const kp of pose.keypoints) s += kp && typeof kp.score === 'number' ? kp.score : 0;
        return s / pose.keypoints.length;
    }

    function pickPrimaryPose(poses) {
        if (!poses || poses.length <= 1) return poses;
        let bestI = 0;
        let best = -1;
        for (let i = 0; i < poses.length; i++) {
            const m = meanScore(poses[i]);
            if (m > best) {
                best = m;
                bestI = i;
            }
        }
        return [poses[bestI]];
    }

    function mapKeypointsFromVideoToCanvas(keypoints, v, c) {
        const vw = v.videoWidth || 0;
        const vh = v.videoHeight || 0;
        if (!vw || !vh || !c) return keypoints;
        return keypoints.map((k) => {
            let x =
                typeof k.x === 'number' && k.x >= 0 && k.x <= 1
                    ? k.x * c.width
                    : (Number(k.x) / vw) * c.width;
            let y =
                typeof k.y === 'number' && k.y >= 0 && k.y <= 1
                    ? k.y * c.height
                    : (Number(k.y) / vh) * c.height;
            if (!Number.isFinite(x) || !Number.isFinite(y)) {
                x = 0;
                y = 0;
            }
            return {
                name: k.name,
                x: Math.max(0, Math.min(c.width, x)),
                y: Math.max(0, Math.min(c.height, y)),
                score: k.score,
            };
        });
    }

    function getCurrentFrameIndex() {
        const slider = document.getElementById('dev_frame_slider');
        return Number(slider && slider.value ? slider.value : 0);
    }

    function clonePose(pose) {
        return pose ? JSON.parse(JSON.stringify(pose)) : null;
    }

    function getNearestAnnotatedPose(frameIndex) {
        const keys = Object.keys(frameAnnotations)
            .map((k) => Number(k))
            .filter((n) => Number.isFinite(n));
        if (!keys.length) return null;
        let bestKey = keys[0];
        let bestDist = Math.abs(keys[0] - frameIndex);
        for (let i = 1; i < keys.length; i++) {
            const d = Math.abs(keys[i] - frameIndex);
            if (d < bestDist) {
                bestDist = d;
                bestKey = keys[i];
            }
        }
        return clonePose(frameAnnotations[bestKey]) || null;
    }

    function loadFramePose(frameIndex) {
        currentPose = clonePose(frameAnnotations[frameIndex]) || getNearestAnnotatedPose(frameIndex) || null;
        actionHistory = [];
        pendingLinkStart = null;
        draw();
    }

    function updateFrameSavedStateLabel(frameIndex) {
        const msg = document.getElementById('admin_msg');
        if (!msg) return;
        if (frameAnnotations[frameIndex]) {
            msg.textContent = `Кадр ${frameIndex}: разметка сохранена локально.`;
            return;
        }
        msg.textContent = `Кадр ${frameIndex}: разметка не сохранена.`;
    }

    function saveFrameAnnotations() {
        const frameIndex = getCurrentFrameIndex();
        if (!currentPose || !currentPose.keypoints || !currentPose.keypoints.length) {
            updateFrameSavedStateLabel(frameIndex);
            return;
        }
        frameAnnotations[frameIndex] = clonePose(currentPose);
        updateFrameSavedStateLabel(frameIndex);
    }

    function draw() {
        const c = document.getElementById('dev_canvas');
        const v = document.getElementById('dev_video');
        if (!c || !v) return;
        const ctx = c.getContext('2d');
        ctx.clearRect(0, 0, c.width, c.height);
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, c.width, c.height);
        if (v.readyState >= 2 && v.videoWidth && v.videoHeight) {
            ctx.drawImage(v, 0, 0, c.width, c.height);
        }
        if (!currentPose || !currentPose.keypoints) return;
        const map = byName(currentPose.keypoints);
        for (const [a, b] of skeleton) {
            if (!map[a] || !map[b]) continue;
            ctx.beginPath();
            ctx.moveTo(map[a].x, map[a].y);
            ctx.lineTo(map[b].x, map[b].y);
            ctx.strokeStyle = 'rgba(80,210,255,0.9)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        for (const [a, b] of currentPose.links || []) {
            if (!map[a] || !map[b]) continue;
            ctx.beginPath();
            ctx.moveTo(map[a].x, map[a].y);
            ctx.lineTo(map[b].x, map[b].y);
            ctx.strokeStyle = 'rgba(255,125,79,0.95)';
            ctx.lineWidth = 3;
            ctx.stroke();
        }
        for (const kp of currentPose.keypoints) {
            const isCustom = String(kp.name || '').startsWith('custom_');
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, isCustom ? 7 : 5, 0, Math.PI * 2);
            const isPending = pendingLinkStart && pendingLinkStart.name === kp.name;
            ctx.fillStyle = isPending ? '#31d2f2' : isCustom ? '#e056fd' : '#ffde59';
            ctx.fill();
        }
    }

    function nearestKeypoint(x, y) {
        if (!currentPose || !currentPose.keypoints) return null;
        let best = null;
        let dBest = 24;
        for (const kp of currentPose.keypoints) {
            const d = Math.hypot(kp.x - x, kp.y - y);
            if (d < dBest) {
                dBest = d;
                best = kp;
            }
        }
        return best;
    }

    async function detectPosesViaYolo(frameCanvas) {
        if (!frameCanvas) return [];
        const blob = await new Promise((resolve) => frameCanvas.toBlob(resolve, 'image/jpeg', 0.95));
        if (!blob) {
            throw new Error('Не удалось получить кадр для YOLO.');
        }
        const fd = new FormData();
        fd.append('image', blob, 'frame.jpg');
        const serverRes = await auth.apiFetch('/api/inference/pose', { method: 'POST', body: fd });
        const detections = (serverRes && serverRes.result && serverRes.result.detections) || [];
        if (!detections.length) return [];
        const candidate = detections
            .filter((d) => Array.isArray(d.keypoints) && d.keypoints.length > 0)
            .sort((a, b) => Number(b.score || 0) - Number(a.score || 0))[0];
        if (!candidate) return [];
        return [{ keypoints: candidate.keypoints || [] }];
    }

    function scheduleAutoDetect() {
        if (!autoDetectOnFrameChange) return;
        const token = ++detectRunToken;
        setTimeout(() => {
            if (token !== detectRunToken) return;
            detectOnFrame({ silent: true }).catch(() => {});
        }, 70);
    }

    async function detectOnFrame(opts) {
        const options = opts || {};
        const msg = document.getElementById('admin_msg');
        const v = document.getElementById('dev_video');
        const c = document.getElementById('dev_canvas');
        if (!v || !c) return;
        if (v.seeking) {
            if (msg) msg.textContent = 'Ожидание загрузки выбранного кадра...';
            await new Promise((resolve) => {
                const done = () => {
                    v.removeEventListener('seeked', done);
                    resolve();
                };
                v.addEventListener('seeked', done, { once: true });
            });
        }
        if (v.readyState < 2) {
            if (msg) msg.textContent = 'Дождитесь загрузки кадра.';
            return;
        }
        if (!v.videoWidth || !v.videoHeight) {
            if (msg) msg.textContent = 'Нет размеров видео — выберите другой файл или кадр.';
            return;
        }
        try {
            let poses = [];
            const frameCanvas = document.createElement('canvas');
            frameCanvas.width = v.videoWidth;
            frameCanvas.height = v.videoHeight;
            frameCanvas.getContext('2d').drawImage(v, 0, 0, frameCanvas.width, frameCanvas.height);
            try {
                poses = await detectPosesViaYolo(frameCanvas);
            } catch (e) {
                if (msg && !options.silent) {
                    msg.textContent = 'YOLO недоступен: ' + (e && e.message ? e.message : 'ошибка детекции');
                }
                poses = [];
            }
            if (!poses || !poses.length) {
                // Keep last successful pose to avoid empty frames in dev panel.
                if (lastAutoPose && lastAutoPose.keypoints && lastAutoPose.keypoints.length) {
                    currentPose = clonePose(lastAutoPose);
                    if (msg && !options.silent) {
                        msg.textContent = 'YOLO не нашёл позу на этом кадре — показана последняя найденная поза.';
                    }
                    draw();
                    return;
                }
                if (msg) msg.textContent = 'Поза не найдена YOLO-моделью — проверьте кадр/освещение.';
                return;
            }
            // If YOLO returns no detections for a specific frame, keep last pose to avoid "empty frames" in UI.
            const frameIndex = getCurrentFrameIndex();
            const framePose = frameAnnotations[frameIndex] || currentPose;
            const prevCustom = (framePose && framePose.keypoints
                ? framePose.keypoints.filter((k) => String(k.name || '').startsWith('custom_'))
                : []) || [];
            const prevLinks = (framePose && framePose.links ? framePose.links.slice() : []) || [];
            const mapped = mapKeypointsFromVideoToCanvas(poses[0].keypoints, v, c);
            currentPose = { keypoints: mapped.concat(prevCustom), links: prevLinks };
            lastAutoPose = clonePose(currentPose);
            if (msg && !options.silent) {
                msg.textContent = 'Точки расставлены. Можно править и сохранять.';
            }
            draw();
        } catch (e) {
            console.error(e);
            if (msg && !options.silent) msg.textContent = e.message || 'Ошибка детекции';
        }
    }

    function addCustomPoint() {
        const c = document.getElementById('dev_canvas');
        if (!c) return;
        pushHistory();
        if (!currentPose) currentPose = { keypoints: [] };
        if (!currentPose.keypoints) currentPose.keypoints = [];
        if (!currentPose.links) currentPose.links = [];
        customPointSeq += 1;
        currentPose.keypoints.push({
            name: 'custom_' + customPointSeq,
            x: c.width * 0.5,
            y: c.height * 0.5,
            score: 1,
        });
        draw();
    }

    function removeLastCustomPoint() {
        if (!currentPose || !currentPose.keypoints) return;
        for (let i = currentPose.keypoints.length - 1; i >= 0; i--) {
            const kp = currentPose.keypoints[i];
            if (!String(kp.name || '').startsWith('custom_')) continue;
            pushHistory();
            currentPose.keypoints.splice(i, 1);
            currentPose.links = (currentPose.links || []).filter(
                ([a, b]) => a !== kp.name && b !== kp.name
            );
            pendingLinkStart = null;
            draw();
            return;
        }
    }

    function snapshotState() {
        if (!currentPose) return null;
        return JSON.parse(JSON.stringify(currentPose));
    }

    function pushHistory() {
        const snap = snapshotState();
        if (!snap) return;
        actionHistory.push(snap);
        if (actionHistory.length > 100) actionHistory.shift();
    }

    function undoLastAction() {
        if (!actionHistory.length) return;
        currentPose = actionHistory.pop();
        pendingLinkStart = null;
        draw();
    }

    function toggleLinkMode() {
        linkMode = !linkMode;
        pendingLinkStart = null;
        const btn = document.getElementById('dev_link_points_btn');
        if (btn) {
            btn.classList.toggle('active', linkMode);
            btn.textContent = linkMode ? 'Режим связывания: ВКЛ' : 'Связать точки линией';
        }
        draw();
    }

    function hasCustomLink(a, b) {
        const links = (currentPose && currentPose.links) || [];
        return links.some(([x, y]) => (x === a && y === b) || (x === b && y === a));
    }

    function addCustomLink(a, b) {
        if (!currentPose || !currentPose.keypoints || a === b) return;
        if (!currentPose.links) currentPose.links = [];
        if (hasCustomLink(a, b)) return;
        pushHistory();
        currentPose.links.push([a, b]);
        draw();
    }

    function clickCanvasPoint(x, y) {
        const kp = nearestKeypoint(x, y);
        if (!kp || !linkMode) return;
        if (!pendingLinkStart) {
            pendingLinkStart = kp;
            draw();
            return;
        }
        addCustomLink(pendingLinkStart.name, kp.name);
        pendingLinkStart = null;
    }

    async function saveToTest() {
        const fileInput = document.getElementById('dev_video_file');
        if (!fileInput || !fileInput.files || !fileInput.files[0]) return;
        const frameIndex = getCurrentFrameIndex();
        if (!frameAnnotations[frameIndex] && currentPose && currentPose.keypoints && currentPose.keypoints.length) {
            frameAnnotations[frameIndex] = clonePose(currentPose);
        }
        const fd = new FormData();
        fd.append('video', fileInput.files[0]);
        fd.append('frameIndex', String(frameIndex));
        fd.append(
            'annotations',
            JSON.stringify({
                currentFrame: frameIndex,
                frames: frameAnnotations,
            })
        );
        const res = await auth.apiFetch('/api/admin/dev/save', { method: 'POST', body: fd });
        const el = document.getElementById('admin_msg');
        if (el) {
            el.textContent = 'Разметка сохранена: ' + (res.meta || '');
        }
    }

    function bindVideoFile() {
        const f = document.getElementById('dev_video_file');
        const v = document.getElementById('dev_video');
        const s = document.getElementById('dev_frame_slider');
        if (!f || !v || !s) return;
        f.addEventListener('change', () => {
            const file = f.files && f.files[0];
            if (!file) return;
            currentPose = null;
            lastAutoPose = null;
            frameAnnotations = {};
            actionHistory = [];
            pendingLinkStart = null;
            v.src = URL.createObjectURL(file);
            v.onloadedmetadata = () => {
                const frames = Math.max(0, Math.floor((v.duration || 0) * 25));
                s.max = String(Math.max(1, frames));
                loadFramePose(0);
                scheduleAutoDetect();
            };
        });
        s.addEventListener('input', () => {
            if (!v.duration) return;
            const fIndex = Number(s.value);
            currentPose = null;
            pendingLinkStart = null;
            draw();
            v.currentTime = Math.min(v.duration - 1e-4, fIndex / 25);
        });
        v.addEventListener('seeked', () => {
            loadFramePose(getCurrentFrameIndex());
            updateFrameSavedStateLabel(getCurrentFrameIndex());
            if (!frameAnnotations[getCurrentFrameIndex()]) {
                scheduleAutoDetect();
            }
        });
    }

    function bindDrag() {
        const c = document.getElementById('dev_canvas');
        if (!c) return;
        c.addEventListener('mousedown', (e) => {
            const r = c.getBoundingClientRect();
            const sx = c.width / r.width;
            const sy = c.height / r.height;
            const x = (e.clientX - r.left) * sx;
            const y = (e.clientY - r.top) * sy;
            dragKey = nearestKeypoint(x, y);
            if (dragKey) {
                pushHistory();
            }
        });
        c.addEventListener('mousemove', (e) => {
            if (!dragKey) return;
            const r = c.getBoundingClientRect();
            const sx = c.width / r.width;
            const sy = c.height / r.height;
            dragKey.x = (e.clientX - r.left) * sx;
            dragKey.y = (e.clientY - r.top) * sy;
            draw();
        });
        c.addEventListener('click', (e) => {
            const r = c.getBoundingClientRect();
            const sx = c.width / r.width;
            const sy = c.height / r.height;
            clickCanvasPoint((e.clientX - r.left) * sx, (e.clientY - r.top) * sy);
        });
        window.addEventListener('mouseup', () => {
            dragKey = null;
        });
    }

    function setAdminRowPreview(row, videoUrl) {
        const vid = row.querySelector('.admin-ref-preview');
        const empty = row.querySelector('.admin-ref-empty');
        if (!vid) return;
        if (videoUrl) {
            if (vid.dataset.loadedUrl !== videoUrl) {
                vid.src = videoUrl;
                vid.dataset.loadedUrl = videoUrl;
            }
            vid.hidden = false;
            if (empty) empty.hidden = true;
        } else {
            vid.removeAttribute('src');
            delete vid.dataset.loadedUrl;
            vid.load();
            vid.hidden = true;
            if (empty) empty.hidden = false;
        }
    }

    async function syncAdminReferencePreviews() {
        const root = document.getElementById('admin_training_ref_rows');
        if (!root) return;
        try {
            const events = await auth.apiFetch('/api/events');
            const byName = {};
            for (const e of events) byName[e.name] = e;
            root.querySelectorAll('.admin-training-ref-row').forEach((row) => {
                const name = row.dataset.exName;
                const ev = byName[name];
                const url = ev && ev.video && String(ev.video).trim();
                setAdminRowPreview(row, url || '');
            });
        } catch (_) {}
    }

    function bindTrainingEditor() {
        const root = document.getElementById('admin_training_ref_rows');
        if (!root) return;
        root.querySelectorAll('.admin-training-ref-row').forEach((row) => {
            const btn = row.querySelector('.admin-event-upload');
            const fileInput = row.querySelector('.admin-event-file');
            if (!btn || !fileInput) return;
            btn.addEventListener('click', async () => {
                if (!fileInput.files || !fileInput.files[0]) return;
                const name = row.dataset.exName;
                const msg = document.getElementById('admin_msg');
                try {
                    const fd = new FormData();
                    fd.append('video', fileInput.files[0]);
                    const enc = encodeURIComponent(name);
                    const r = await auth.apiFetch('/api/admin/events/' + enc + '/video', {
                        method: 'POST',
                        body: fd,
                    });
                    if (msg) msg.textContent = 'Эталон обновлён: ' + r.name;
                    setAdminRowPreview(row, r.video);
                    fileInput.value = '';
                    if (typeof training !== 'undefined' && training.refreshEventCacheAndRef) {
                        await training.refreshEventCacheAndRef();
                    }
                } catch (e) {
                    if (msg) msg.textContent = e.message || 'Ошибка загрузки эталона';
                }
            });
        });
    }

    function drawPoseOnContext(ctx, pose, sx, sy) {
        if (!pose || !pose.keypoints) return;
        const map = byName(pose.keypoints);
        for (const [a, b] of skeleton) {
            if (!map[a] || !map[b]) continue;
            ctx.beginPath();
            ctx.moveTo(map[a].x * sx, map[a].y * sy);
            ctx.lineTo(map[b].x * sx, map[b].y * sy);
            ctx.strokeStyle = 'rgba(80,210,255,0.9)';
            ctx.lineWidth = 3;
            ctx.stroke();
        }
        for (const [a, b] of pose.links || []) {
            if (!map[a] || !map[b]) continue;
            ctx.beginPath();
            ctx.moveTo(map[a].x * sx, map[a].y * sy);
            ctx.lineTo(map[b].x * sx, map[b].y * sy);
            ctx.strokeStyle = 'rgba(255,125,79,0.95)';
            ctx.lineWidth = 4;
            ctx.stroke();
        }
        for (const kp of pose.keypoints) {
            const isCustom = String(kp.name || '').startsWith('custom_');
            ctx.beginPath();
            ctx.arc(kp.x * sx, kp.y * sy, isCustom ? 8 : 6, 0, Math.PI * 2);
            ctx.fillStyle = isCustom ? '#e056fd' : '#ffde59';
            ctx.fill();
        }
    }

    function mapKeypointsBetweenFrames(keypoints, srcW, srcH, dstW, dstH) {
        if (!Array.isArray(keypoints)) return [];
        return keypoints.map((k) => {
            const rawX = typeof k.x === 'number' ? k.x : 0;
            const rawY = typeof k.y === 'number' ? k.y : 0;
            const x = rawX >= 0 && rawX <= 1 ? rawX * dstW : (rawX / Math.max(1, srcW)) * dstW;
            const y = rawY >= 0 && rawY <= 1 ? rawY * dstH : (rawY / Math.max(1, srcH)) * dstH;
            return { name: k.name, x, y, score: k.score };
        });
    }

    async function renderAnnotatedVideoMp4() {
        const msg = document.getElementById('admin_msg');
        const fileInput = document.getElementById('dev_video_file');
        if (!fileInput || !fileInput.files || !fileInput.files[0]) {
            if (msg) msg.textContent = 'Сначала выберите видео.';
            return;
        }
        if (!Object.keys(frameAnnotations).length) {
            if (msg) msg.textContent = 'Нет сохраненных точек по кадрам.';
            return;
        }
        if (msg) msg.textContent = 'Обработка видео...';
        const srcVideo = document.createElement('video');
        srcVideo.src = URL.createObjectURL(fileInput.files[0]);
        srcVideo.muted = true;
        srcVideo.playsInline = true;

        await new Promise((resolve, reject) => {
            srcVideo.onloadedmetadata = () => resolve();
            srcVideo.onerror = () => reject(new Error('Не удалось загрузить видео для рендера'));
        });
        const fps = 25;
        const w = srcVideo.videoWidth;
        const h = srcVideo.videoHeight;
        const off = document.createElement('canvas');
        off.width = w;
        off.height = h;
        const ctx = off.getContext('2d');
        const stream = off.captureStream(fps);
        const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')
            ? 'video/webm;codecs=vp9,opus'
            : 'video/webm';
        const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 9000000 });
        const chunks = [];
        rec.ondataavailable = (e) => {
            if (e.data && e.data.size) chunks.push(e.data);
        };

        await new Promise(async (resolve, reject) => {
            let lastPose = null;
            rec.onstop = resolve;
            rec.onerror = reject;
            rec.start(250);
            srcVideo.currentTime = 0;
            await srcVideo.play();
            const drawTick = async () => {
                if (srcVideo.paused || srcVideo.ended) {
                    rec.stop();
                    return;
                }
                ctx.drawImage(srcVideo, 0, 0, w, h);
                const frameI = Math.round(srcVideo.currentTime * fps);
                let framePose = frameAnnotations[frameI];
                if (!framePose) {
                    try {
                        const frameCanvas = document.createElement('canvas');
                        frameCanvas.width = w;
                        frameCanvas.height = h;
                        const frameCtx = frameCanvas.getContext('2d');
                        frameCtx.drawImage(srcVideo, 0, 0, w, h);
                        const poses = await detectPosesViaYolo(frameCanvas);
                        if (poses && poses.length && poses[0].keypoints) {
                            framePose = {
                                keypoints: mapKeypointsBetweenFrames(poses[0].keypoints, w, h, w, h),
                                links: [],
                            };
                        }
                    } catch (_) {
                        framePose = null;
                    }
                } else {
                    framePose = {
                        keypoints: mapKeypointsBetweenFrames(framePose.keypoints, 960, 540, w, h),
                        links: framePose.links || [],
                    };
                }
                if (framePose && framePose.keypoints && framePose.keypoints.length) {
                    lastPose = framePose;
                } else if (lastPose) {
                    framePose = lastPose;
                }
                if (framePose) {
                    drawPoseOnContext(ctx, framePose, 1, 1);
                }
                requestAnimationFrame(() => {
                    drawTick().catch(() => {
                        rec.stop();
                    });
                });
            };
            drawTick().catch(reject);
            srcVideo.onended = () => rec.stop();
        });

        const webmBlob = new Blob(chunks, { type: 'video/webm' });
        const fd = new FormData();
        fd.append('video', new File([webmBlob], 'annotated.webm', { type: 'video/webm' }));
        const converted = await auth.apiFetch('/api/video/convert-mp4', { method: 'POST', body: fd });
        if (!converted || !converted.url) {
            throw new Error('Сервер не вернул MP4');
        }
        if (msg) msg.textContent = 'Готово. MP4 сформирован.';
        const a = document.createElement('a');
        a.href = converted.url;
        a.download = 'annotated_video.mp4';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(srcVideo.src);
    }

    function init() {
        const user = auth.getUser ? auth.getUser() : null;
        isAdmin = !!user && String(user.name).toLowerCase() === 'admin';
        const btn = document.getElementById('btn_admin');
        if (btn) btn.hidden = !isAdmin;
        const page = document.getElementById('page_admin');
        if (page) page.hidden = !isAdmin;
        document.getElementById('main_app')?.classList.toggle('user-is-admin', isAdmin);
        if (!isAdmin) return;
        bindVideoFile();
        bindDrag();
        bindTrainingEditor();
        syncAdminReferencePreviews();
        document.getElementById('dev_detect_btn')?.addEventListener('click', () => detectOnFrame());
        document.getElementById('dev_save_frame_btn')?.addEventListener('click', () => saveFrameAnnotations());
        document.getElementById('dev_save_btn')?.addEventListener('click', () => {
            saveToTest().catch((e) => {
                const m = document.getElementById('admin_msg');
                if (m) m.textContent = e.message || 'Ошибка';
            });
        });
        document.getElementById('dev_add_point_btn')?.addEventListener('click', () => addCustomPoint());
        document.getElementById('dev_remove_last_point_btn')?.addEventListener('click', () =>
            removeLastCustomPoint()
        );
        document.getElementById('dev_link_points_btn')?.addEventListener('click', () => toggleLinkMode());
        document.getElementById('dev_undo_btn')?.addEventListener('click', () => undoLastAction());
        document.getElementById('dev_render_mp4_btn')?.addEventListener('click', () => {
            renderAnnotatedVideoMp4().catch((e) => {
                const m = document.getElementById('admin_msg');
                if (m) m.textContent = e.message || 'Ошибка обработки видео';
            });
        });
    }

    return { init };
})();

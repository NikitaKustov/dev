// Copyright Marcin "szczyglis" Szczyglinski, 2022. All Rights Reserved.
// Email: szczyglis@protonmail.com
// WWW: https://github.com/szczyglis-dev/js-ai-body-tracker
// Library: tracker.js
// Version: 1.0.0
// This file is licensed under the MIT License.
// License text available at https://opensource.org/licenses/MIT

const tracker = {
    // config options (переопределяется setModel; по умолчанию BlazePose lite)
    detectorModel: poseDetection.SupportedModels.BlazePose,
    detectorConfig: {
        runtime: 'tfjs',
        enableSmoothing: false,
        modelType: 'lite'
    },
    autofit: false, // bool, enable autofit on canvas scaling
    enableAI: true, // bool, enable or disable tracking
    enableVideo: true, // bool, enable or disable display original video on canvas on canvas
    pointWidth: 3, // width of line between points
    pointRadius: 5, // point circle radius
    minScore: 0.5, // minimum threshold for estimated point
    useCustomBlazePoseLayout: true, // draw custom body-only layout for BlazePose
    enableTemporalFilter: true, // reduce jitter and outliers frame-to-frame
    smoothingMinAlpha: 0.28, // lower alpha = smoother motion
    smoothingMaxAlpha: 0.9, // higher alpha = faster reaction
    maxJumpPxPerFrame: 32, // clamp sudden jumps for unstable keypoints
    filterStateTtlMs: 1500, // stale keypoint state lifetime
    /** Extra Y offset (px) added when drawing keypoints; negative shifts skeleton up (e.g. toward eye level). */
    poseVerticalNudgePx: -18,
    /** Fixed internal canvas size (px). Inference uses video native size; drawing is scaled here. */
    outputWidth: 1702,
    outputHeight: 834,
    useFixedOutputSize: true,
    /** При падении средней уверенности (ладонь закрыла камеру) — показывать последнюю стабильную позу. */
    enableOcclusionHold: true,
    occlusionMinMeanScore: 0.38,
    occlusionScoreDrop: 0.2,
    occlusionHoldMaxFrames: 8,
    personMinScore: 0.45,
    /** Доля продления луча запястье→палец за пределы ключевой точки (визуально «кончик»). */
    fingertipExtrapolate: 0.42,
    // Performance tuning: estimate poses at capped frequency, draw every frame.
    // Network inference is expensive; keep UI smooth by sampling.
    inferenceIntervalMs: 260,
    /** Во время записи чаще обновляем позу; кадр для YOLO меньше — меньше задержка сети без смены разрешения записи. */
    recordingInferenceIntervalMs: 200,
    inferenceProviderPreferred: 'yolo_api', // yolo_api | blazepose
    inferenceProviderActive: 'blazepose',
    yoloHealthPath: '/api/inference/health',
    yoloPosePath: '/api/inference/pose',
    // Live optimization: keep inference cheap; recording quality is unaffected (we record the canvas).
    yoloFrameMaxWidth: 320,
    /** Узкий кадр для API во время записи — быстрее ответ, запись с canvas по-прежнему в полном разрешении. */
    yoloFrameMaxWidthRecording: 256,
    yoloJpegQuality: 0.62,
    yoloJpegQualityRecording: 0.56,
    yoloRequestTimeoutMs: 12000,
    yoloMaxErrorsBeforeFallback: 3,
    yoloRecoveryCheckMs: 4500,
    renderCache: null,
    _lastInferenceTs: 0,
    _lastModeRecording: false,
    _yoloCanvas: null,
    _yoloCtx: null,
    _yoloErrorStreak: 0,
    _lastYoloHealthCheckTs: 0,
    _lastYoloHealthOk: false,
    _yoloInFlight: false,
    _inferencePromise: null,
    _yoloLastGoodPose: null,
    _yoloHoldFramesLeft: 0,
    // Keep last pose for a bit to avoid "empty" overlays on short misses.
    yoloHoldMaxFrames: 20,
    log: true, // bool, enable logging to console
    hooks: { // user defined hooks/events
        'beforeupdate': [], // before poses update
        'afterupdate': [], // after poses update
        'statuschange': [], // when status change
        'detectorerror': [], // if detector error 
        'videoerror': [] // if video error
    },

    // HTML elements
    elCanvas: '#canvas', // HTML element for canvas
    elVideo: '#video', // HTML element for video
    
    // internals
    detector: null, // tensor flow detector instance
    reqID: null, // requested frame ID
    isPlaying: false, // bool, current playback state
    isWaiting: false, // bool, waiting for video state
    poses: null, // estimated poses
    video: null, // DOMElement with vidoe
    canvas: null, // DOMElement with canvas 
    ctx: null, // canvas context instance
    container: null, // container for video
    status: '', // current status message
    poseFilterState: {}, // temporal keypoint smoothing state
    _lastStablePose: null,
    _occHoldTicks: 0,
    _prevMeanScore: null,
    paths: {
        'blaze_pose': {
            // left hip > left knee
            'l_hip_l_knee': {
                'from_x': ['left_hip'],
                'from_y': ['left_hip'],
                'to_x': ['left_knee'],
                'to_y': ['left_knee'],
                'scores': ['left_knee'],
                'rgb': [42, 163, 69]
            },
            // right hip > right knee
            'r_hip_r_knee': {
                'from_x': ['right_hip'],
                'from_y': ['right_hip'],
                'to_x': ['right_knee'],
                'to_y': ['right_knee'],
                'scores': ['right_knee'],
                'rgb': [42, 163, 69]
            },
            // hips (mid-point)
            'hip_l_m': { // left
                'from_x': ['left_hip'],
                'from_y': ['left_hip'],
                'to_x': ['left_hip', 'right_hip'],
                'to_y': ['left_hip', 'right_hip'],
                'scores': ['left_hip', 'right_hip'],
                'rgb': [140, 232, 90]
            },
            'hip_r_m': { // right
                'from_x': ['right_hip'],
                'from_y': ['right_hip'],
                'to_x': ['left_hip', 'right_hip'],
                'to_y': ['left_hip', 'right_hip'],
                'scores': ['left_hip', 'right_hip'],
                'rgb': [140, 232, 90]
            },
            // hip to shoulders
            'hip_l_shoulder_l': { // left
                'from_x': ['left_hip'],
                'from_y': ['left_hip'],
                'to_x': ['left_shoulder'],
                'to_y': ['left_shoulder'],
                'scores': ['left_hip', 'left_shoulder'],
                'rgb': [242, 85, 240]
            },
            'hip_r_shoulder_r': { // right
                'from_x': ['right_hip'],
                'from_y': ['right_hip'],
                'to_x': ['right_shoulder'],
                'to_y': ['right_shoulder'],
                'scores': ['right_hip', 'right_shoulder'],
                'rgb': [242, 85, 240]
            },
            // left knee > left ankle
            'l_knee_l_ankle': {
                'from_x': ['left_knee'],
                'from_y': ['left_knee'],
                'to_x': ['left_ankle'],
                'to_y': ['left_ankle'],
                'scores': ['left_ankle'],
                'rgb': [140, 232, 90]
            },
            // right knee > right ankle
            'r_knee_r_ankle': {
                'from_x': ['right_knee'],
                'from_y': ['right_knee'],
                'to_x': ['right_ankle'],
                'to_y': ['right_ankle'],
                'scores': ['right_ankle'],
                'rgb': [140, 232, 90]
            },
            // left ankle > left heel
            'l_ankle_l_heel': {
                'from_x': ['left_ankle'],
                'from_y': ['left_ankle'],
                'to_x': ['left_heel'],
                'to_y': ['left_heel'],
                'scores': ['left_ankle', 'left_heel'],
                'rgb': [42, 163, 69]
            },
            // left heel > left foot_index
            'l_heel_l_foot_index': {
                'from_x': ['left_heel'],
                'from_y': ['left_heel'],
                'to_x': ['left_foot_index'],
                'to_y': ['left_foot_index'],
                'scores': ['left_heel', 'left_foot_index'],
                'rgb': [42, 163, 69]
            },
            // left foot_index > left ankle
            'l_foot_index_l_ankle': {
                'from_x': ['left_foot_index'],
                'from_y': ['left_foot_index'],
                'to_x': ['left_ankle'],
                'to_y': ['left_ankle'],
                'scores': ['left_foot_index', 'left_ankle'],
                'rgb': [42, 163, 69]
            },
            // right ankle > right heel
            'r_ankle_r_heel': {
                'from_x': ['right_ankle'],
                'from_y': ['right_ankle'],
                'to_x': ['right_heel'],
                'to_y': ['right_heel'],
                'scores': ['right_ankle', 'right_heel'],
                'rgb': [42, 163, 69]
            },
            // right heel > right foot_index
            'r_heel_r_foot_index': {
                'from_x': ['right_heel'],
                'from_y': ['right_heel'],
                'to_x': ['right_foot_index'],
                'to_y': ['right_foot_index'],
                'scores': ['right_heel', 'right_foot_index'],
                'rgb': [42, 163, 69]
            },
            // right foot_index > right ankle
            'r_foot_index_r_ankle': {
                'from_x': ['right_foot_index'],
                'from_y': ['right_foot_index'],
                'to_x': ['right_ankle'],
                'to_y': ['right_ankle'],
                'scores': ['right_foot_index', 'right_ankle'],
                'rgb': [42, 163, 69]
            },
            // hips > shoulders
            'hips_shoulders_m': {
                'from_x': ['left_hip', 'right_hip'],
                'from_y': ['left_hip', 'right_hip'],
                'to_x': ['left_shoulder', 'right_shoulder'],
                'to_y': ['left_shoulder', 'right_shoulder'],
                'scores': ['left_hip', 'right_hip'],
                'rgb': [242, 85, 240]
            },
            // shoulders (mid-point)
            'shoulder_l_m': { // left
                'from_x': ['left_shoulder'],
                'from_y': ['left_shoulder'],
                'to_x': ['left_shoulder', 'right_shoulder'],
                'to_y': ['left_shoulder', 'right_shoulder'],
                'scores': ['left_shoulder', 'right_shoulder'],
                'rgb': [92, 70, 235]
            },
            'shoulder_r_m': { // right
                'from_x': ['right_shoulder'],
                'from_y': ['right_shoulder'],
                'to_x': ['left_shoulder', 'right_shoulder'],
                'to_y': ['left_shoulder', 'right_shoulder'],
                'scores': ['left_shoulder', 'right_shoulder'],
                'rgb': [92, 70, 235]
            },
            // shoulders (mid-point) > nose (neck)
            'neck': {
                'from_x': ['left_shoulder', 'right_shoulder'],
                'from_y': ['left_shoulder', 'right_shoulder'],
                'to_x': ['left_ear', 'right_ear'],
                'to_y': ['left_ear', 'right_ear'],
                'scores': ['left_shoulder', 'right_shoulder'],
                'rgb': [92, 108, 145]
            },
            // left shoulder > left elbow
            'l_shoulder_l_elbow': {
                'from_x': ['left_shoulder'],
                'from_y': ['left_shoulder'],
                'to_x': ['left_elbow'],
                'to_y': ['left_elbow'],
                'scores': ['left_elbow'],
                'rgb': [245, 129, 66]
            },
            // right shoulder > right elbow
            'r_shoulder_r_elbow': {
                'from_x': ['right_shoulder'],
                'from_y': ['right_shoulder'],
                'to_x': ['right_elbow'],
                'to_y': ['right_elbow'],
                'scores': ['right_elbow'],
                'rgb': [245, 129, 66]
            },
            // left elbow > left wrist
            'l_elbow_l_wrist': {
                'from_x': ['left_elbow'],
                'from_y': ['left_elbow'],
                'to_x': ['left_wrist'],
                'to_y': ['left_wrist'],
                'scores': ['left_wrist'],
                'rgb': [227, 156, 118]
            },
            // right elbow > right wrist
            'r_elbow_r_wrist': {
                'from_x': ['right_elbow'],
                'from_y': ['right_elbow'],
                'to_x': ['right_wrist'],
                'to_y': ['right_wrist'],
                'scores': ['right_wrist'],
                'rgb': [227, 156, 118]
            },

            // left wrist > left_thumb
            'l_wrist_l_thumb': {
                'from_x': ['left_wrist'],
                'from_y': ['left_wrist'],
                'to_x': ['left_thumb'],
                'to_y': ['left_thumb'],
                'scores': ['left_wrist', 'left_thumb'],
                'rgb': [245, 129, 66]
            },
            // left wrist > left_pinky
            'l_wrist_l_pinky': {
                'from_x': ['left_wrist'],
                'from_y': ['left_wrist'],
                'to_x': ['left_pinky'],
                'to_y': ['left_pinky'],
                'scores': ['left_wrist', 'left_pinky'],
                'rgb': [245, 129, 66]
            },
            // left pinky > left index
            'l_pinky_l_index': {
                'from_x': ['left_pinky'],
                'from_y': ['left_pinky'],
                'to_x': ['left_index'],
                'to_y': ['left_index'],
                'scores': ['left_pinky', 'left_index'],
                'rgb': [245, 129, 66]
            },
            // left index > left wrist
            'l_index_l_wrist': {
                'from_x': ['left_index'],
                'from_y': ['left_index'],
                'to_x': ['left_wrist'],
                'to_y': ['left_wrist'],
                'scores': ['left_index', 'left_wrist'],
                'rgb': [245, 129, 66]
            },
            // right wrist > right_thumb
            'r_wrist_r_thumb': {
                'from_x': ['right_wrist'],
                'from_y': ['right_wrist'],
                'to_x': ['right_thumb'],
                'to_y': ['right_thumb'],
                'scores': ['right_wrist', 'right_thumb'],
                'rgb': [245, 129, 66]
            },
            // right wrist > right_pinky
            'r_wrist_r_pinky': {
                'from_x': ['right_wrist'],
                'from_y': ['right_wrist'],
                'to_x': ['right_pinky'],
                'to_y': ['right_pinky'],
                'scores': ['right_wrist', 'right_pinky'],
                'rgb': [245, 129, 66]
            },
            // right pinky > right index
            'r_pinky_r_index': {
                'from_x': ['right_pinky'],
                'from_y': ['right_pinky'],
                'to_x': ['right_index'],
                'to_y': ['right_index'],
                'scores': ['right_pinky', 'right_index'],
                'rgb': [245, 129, 66]
            },
            // right index > right wrist
            'r_index_r_wrist': {
                'from_x': ['right_index'],
                'from_y': ['right_index'],
                'to_x': ['right_wrist'],
                'to_y': ['right_wrist'],
                'scores': ['right_index', 'right_wrist'],
                'rgb': [245, 129, 66]
            },
            'bridge_eyes': {
                'from_x': ['left_eye_inner'],
                'from_y': ['left_eye_inner'],
                'to_x': ['right_eye_inner'],
                'to_y': ['right_eye_inner'],
                'scores': ['left_eye_inner', 'right_eye_inner'],
                'rgb': [255, 100, 140]
            },
            // mouth_left > mouth_right
            'l_mouth_r_mouth': {
                'from_x': ['mouth_left'],
                'from_y': ['mouth_left'],
                'to_x': ['mouth_right'],
                'to_y': ['mouth_right'],
                'scores': ['mouth_left', 'mouth_right'],
                'rgb': [150, 0, 0]
            },
            // mouth_right > mouth_left
            'r_mouth_l_mouth': {
                'from_x': ['mouth_right'],
                'from_y': ['mouth_right'],
                'to_x': ['mouth_left'],
                'to_y': ['mouth_left'],
                'scores': ['mouth_right', 'mouth_left'],
                'rgb': [150, 0, 0]
            },

            // left eye > left eye_outer
            'l_eye_l_eye_outer': {
                'from_x': ['left_eye'],
                'from_y': ['left_eye'],
                'to_x': ['left_eye_outer'],
                'to_y': ['left_eye_outer'],
                'scores': ['left_eye_outer'],
                'rgb': [197, 117, 15]
            },
            // left eye_outer > left ear
            'l_eye_outer_l_ear': {
                'from_x': ['left_eye_outer'],
                'from_y': ['left_eye_outer'],
                'to_x': ['left_ear'],
                'to_y': ['left_ear'],
                'scores': ['left_ear'],
                'rgb': [197, 117, 15]
            },
            // left eye_inner > left eye
            'l_eye_inner_l_eye': {
                'from_x': ['left_eye_inner'],
                'from_y': ['left_eye_inner'],
                'to_x': ['left_eye'],
                'to_y': ['left_eye'],
                'scores': ['left_eye'],
                'rgb': [197, 217, 15]
            },
            // right eye > right eye_outer
            'r_eye_r_eye_outer': {
                'from_x': ['right_eye'],
                'from_y': ['right_eye'],
                'to_x': ['right_eye_outer'],
                'to_y': ['right_eye_outer'],
                'scores': ['right_eye_outer'],
                'rgb': [197, 117, 15]
            },
            // right eye_outer > right ear
            'r_eye_outer_r_ear': {
                'from_x': ['right_eye_outer'],
                'from_y': ['right_eye_outer'],
                'to_x': ['right_ear'],
                'to_y': ['right_ear'],
                'scores': ['right_ear'],
                'rgb': [197, 117, 15]
            },
            // right eye_inner > right eye
            'r_eye_inner_r_eye': {
                'from_x': ['right_eye_inner'],
                'from_y': ['right_eye_inner'],
                'to_x': ['right_eye'],
                'to_y': ['right_eye'],
                'scores': ['right_eye'],
                'rgb': [197, 217, 15]
            },
        }
    },
    yoloSkeleton: [
        // Face
        ['left_eye', 'right_eye'],
        ['left_eye', 'nose'],
        ['right_eye', 'nose'],
        ['left_ear', 'left_eye'],
        ['right_ear', 'right_eye'],
        // Torso
        ['left_shoulder', 'right_shoulder'],
        ['left_shoulder', 'left_hip'],
        ['right_shoulder', 'right_hip'],
        ['left_hip', 'right_hip'],
        // Arms
        ['left_shoulder', 'left_elbow'],
        ['left_elbow', 'left_wrist'],
        ['right_shoulder', 'right_elbow'],
        ['right_elbow', 'right_wrist'],
        // Legs
        ['left_hip', 'left_knee'],
        ['left_knee', 'left_ankle'],
        ['right_hip', 'right_knee'],
        ['right_knee', 'right_ankle'],
    ],
    // Rendering: analysis uses keypoints, skeleton links are visual-only.
    // Keep enabled to provide stable visual feedback during exercise.
    yoloDrawSkeleton: true,

    /*
        Run predictions
     */
    run: function(source) {
        switch (source) {
            case 'video':
                tracker.initVideo();
                break;
            case 'camera':
                tracker.initCamera();
                break;
        }
    },

    /** Остановить камеру и цикл отрисовки (освободить getUserMedia). */
    stopCamera: function() {
        if (tracker.reqID != null) {
            window.cancelAnimationFrame(tracker.reqID);
            tracker.reqID = null;
        }
        tracker.isPlaying = false;
        tracker._inferencePromise = null;
        tracker.poses = null;
        if (tracker.video && tracker.video.srcObject) {
            const stream = tracker.video.srcObject;
            try {
                stream.getTracks().forEach((t) => t.stop());
            } catch (_) {}
            tracker.video.srcObject = null;
        }
        try {
            tracker.video.pause();
        } catch (_) {}
        tracker.clearCanvas();
        tracker.setStatus('');
    },

    /*
        Initialize core elements
     */
    init: function() {
        tracker.log('Initializing...');

        tracker.video = document.querySelector(tracker.elVideo);
        tracker.canvas = document.querySelector(tracker.elCanvas);
        tracker.ctx = tracker.canvas.getContext('2d');
    },

    applyOutputCanvasSize: function() {
        if (
            tracker.useFixedOutputSize &&
            tracker.outputWidth > 0 &&
            tracker.outputHeight > 0
        ) {
            tracker.canvas.width = tracker.outputWidth;
            tracker.canvas.height = tracker.outputHeight;
        } else if (tracker.container && tracker.container.video) {
            tracker.canvas.width = tracker.container.video.videoWidth;
            tracker.canvas.height = tracker.container.video.videoHeight;
        }
    },

    /*
        Initialize video
     */
    initVideo: async function() {
        // initialize
        tracker.init();

        // setup video
        tracker.video.autoPlay = true;
        tracker.video.loop = true;
        tracker.container = {
            video: tracker.video,
            ready: true,
        };

        // setup video events
        tracker.video.addEventListener('loadedmetadata', function() {
            tracker.log('Event: loadedmetadata');
            tracker.container.ready = true;
            tracker.showPlaybackControls();
            //this.currentTime = 210; // optional - set video start time
        }, false);

        tracker.video.addEventListener('playing', function() {
            tracker.log('Event: playing');
            tracker.isWaiting = false;
            if (!tracker.isPlaying) {
                tracker.onVideoReady();
                tracker.isPlaying = true;
            }
        }, false);

        tracker.video.addEventListener('play', function() {
            tracker.log('Event: play');
        }, false);

        tracker.video.addEventListener('error', function(e) {
            console.error(e);
            tracker.dispatch('videoerror', e);
            tracker.setStatus('Error');
        }, true);

        // setup play/pause click event
        tracker.canvas.addEventListener("click", function() {
            tracker.playPauseClick();
        });
    },

    /*
        Launch video when ready
     */
    onVideoReady: async function(e) {
        tracker.log('On Video ready');

        // cancel current frame update if present
        if (tracker.reqID != null) {
            window.cancelAnimationFrame(tracker.reqID);
        }

        await tracker.ensureInferenceReady();

        tracker.video.width = tracker.container.video.videoWidth;
        tracker.video.height = tracker.container.video.videoHeight;
        tracker.applyOutputCanvasSize();
        tracker.container.ready = true;

        // init frame update
        tracker.reqID = window.requestAnimationFrame(tracker.videoFrame);
    },

    /*
        Load video from source address
     */
    loadVideo: function(src) {
        tracker.log('Loading source: ' + src);
        tracker.setStatus('Please wait...loading...');

        tracker.isPlaying = false; // allow new initialization

        // cancel current frame update if present
        if (tracker.reqID != null) {
            window.cancelAnimationFrame(tracker.reqID);
        }

        // dispose current detector
        if (tracker.detector != null) {
            tracker.detector.dispose();
        }
        tracker.detector = null;

        // pause, change source and play new
        tracker.video.pause();
        tracker.video.src = src;
        tracker.container = {
            video: tracker.video,
            ready: true,
        };
        tracker.video.play();
    },

    /*
        Render video frame
     */
    videoFrame: async function() {

        tracker.setStatus('');

        // check if video is ready
        if (tracker.container !== undefined && tracker.container.ready) {
            if (tracker.enableAI && tracker.container.video != null) {
                // try to detect poses
                try {
                    const now = performance.now();
                    const recordingMode = !!(window.recording && window.recording.isRecording && window.recording.isRecording());
                    const minInterval = recordingMode
                        ? tracker.recordingInferenceIntervalMs
                        : tracker.inferenceIntervalMs;
                    if (now - tracker._lastInferenceTs >= minInterval || tracker._lastModeRecording !== recordingMode) {
                        // Do not block the render loop while waiting for network inference.
                        if (!tracker._inferencePromise) {
                            tracker._inferencePromise = tracker
                                .getPosesForFrame(tracker.container.video, now)
                                .then((poses) => {
                                    tracker.poses = tracker.applyOcclusionHold(
                                        tracker.filterPoses(tracker.pickPrimaryPose(poses))
                                    );
                                })
                                .catch(() => {})
                                .finally(() => {
                                    tracker._inferencePromise = null;
                                });
                        }
                        tracker._lastInferenceTs = now;
                        tracker._lastModeRecording = recordingMode;
                    }
                } catch (err) {
                    tracker.dispatch('detectorerror', err);
                    console.error(err);
                }
            }

            // clear canvas
            tracker.clearCanvas();

            // draw video frame on canvas
            if (tracker.enableVideo) {
                if (
                    tracker.useFixedOutputSize &&
                    tracker.outputWidth > 0 &&
                    tracker.outputHeight > 0
                ) {
                    const v = tracker.container.video;
                    const videoSize = { width: v.videoWidth, height: v.videoHeight };
                    const canvasSize = {
                        width: tracker.canvas.width,
                        height: tracker.canvas.height,
                    };
                    const renderSize = tracker.calculateSize(videoSize, canvasSize);
                    let xOffset = (canvasSize.width - renderSize.width) / 2;
                    if (!tracker.autofit) {
                        xOffset = 0;
                    }
                    tracker.ctx.drawImage(
                        v,
                        xOffset,
                        0,
                        renderSize.width,
                        renderSize.height
                    );
                } else {
                    tracker.ctx.drawImage(
                        tracker.container.video,
                        0,
                        0,
                        tracker.container.video.videoWidth,
                        tracker.container.video.videoHeight
                    );
                }
            }

            // handle detected poses
            if (tracker.enableAI) {
                tracker.handlePoses();
            }

            // if video is paused then show controls
            if (tracker.container.video.paused) {
                tracker.showPlaybackControls();
            }
        }

        // next frame
        tracker.reqID = window.requestAnimationFrame(tracker.videoFrame);
    },

    /*
        Initialize camera
     */
    initCamera: async function() {
        tracker.init();

        // init camera first to reduce time-to-first-frame
        try {
            tracker.video = await tracker.setupCamera();
            tracker.video.play();
            await tracker.ensureInferenceReady();
            tracker.cameraFrame();
        } catch (e) {
            tracker.dispatch('videoerror', e);
            console.error(e);
        }
    },

    /*
        Set-up camera
     */
    setupCamera: async function() {
        tracker.setStatus('Please wait...initializing camera...');
        // init device
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error(
                "Browser API navigator.mediaDevices.getUserMedia not available"
            );
        }
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                width: { ideal: tracker.outputWidth || 1280 },
                height: { ideal: tracker.outputHeight || 720 },
                frameRate: { ideal: 30, max: 30 },
            },
        });
        tracker.video.srcObject = stream; // attach camera stream to video

        // get width and height of the camera video stream
        let stream_settings = stream.getVideoTracks()[0].getSettings();
        let stream_width = stream_settings.width;
        let stream_height = stream_settings.height;

        // re-init width and height with info from stream
        tracker.video.width = stream_width;
        tracker.video.height = stream_height;

        return new Promise((resolve) => {
            tracker.video.onloadedmetadata = () => resolve(tracker.video);
        });
    },

    /*
        Render camera frame
     */
    cameraFrame: async function() {
        tracker.setStatus('');

        // predict poses
        try {
            const now = performance.now();
            const recordingMode = !!(window.recording && window.recording.isRecording && window.recording.isRecording());
            const minInterval = recordingMode
                ? tracker.recordingInferenceIntervalMs
                : tracker.inferenceIntervalMs;
            if (now - tracker._lastInferenceTs >= minInterval || tracker._lastModeRecording !== recordingMode) {
                if (!tracker._inferencePromise) {
                    tracker._inferencePromise = tracker
                        .getPosesForFrame(tracker.video, now)
                        .then((poses) => {
                            tracker.poses = tracker.applyOcclusionHold(
                                tracker.filterPoses(tracker.pickPrimaryPose(poses))
                            );
                        })
                        .catch(() => {})
                        .finally(() => {
                            tracker._inferencePromise = null;
                        });
                }
                tracker._lastInferenceTs = now;
                tracker._lastModeRecording = recordingMode;
            }
        } catch (err) {
            tracker.dispatch('detectorerror', err);
            console.error(err);
        }

        tracker.applyOutputCanvasSize();
        if (tracker.video.readyState === tracker.video.HAVE_ENOUGH_DATA) {
            let xOffset = 0;
            const videoSize = {
                width: tracker.video.videoWidth,
                height: tracker.video.videoHeight
            };
            const canvasSize = {
                width: tracker.canvas.width,
                height: tracker.canvas.height
            };
            const renderSize = tracker.calculateSize(videoSize, canvasSize);
            xOffset = (canvasSize.width - renderSize.width) / 2;

            // clear canvas
            tracker.clearCanvas();

            // draw video frame from camera on canvas
            if (tracker.enableVideo) {
                tracker.ctx.drawImage(tracker.video, xOffset, 0, renderSize.width, renderSize.height);
            }
        }

        // handle poses
        if (tracker.enableAI) {
            tracker.handlePoses();
        }

        // next frame
        tracker.reqID = window.requestAnimationFrame(tracker.cameraFrame);
    },

    /*
        Find and return pose keypoints by keypoint's name
     */
    findKeypoint: function(name, pose) {
        for (const keypoint of pose.keypoints) {
            if (keypoint.name == name) {
                return keypoint;
            }
        }
    },

    /*
        Clamp number to [min, max]
     */
    clamp: function(v, min, max) {
        return Math.max(min, Math.min(max, v));
    },

    /*
        Remove stale temporal filter state
     */
    cleanupFilterState: function(now) {
        for (const k in tracker.poseFilterState) {
            if (tracker.poseFilterState.hasOwnProperty(k)) {
                if (now - tracker.poseFilterState[k].t > tracker.filterStateTtlMs) {
                    delete tracker.poseFilterState[k];
                }
            }
        }
    },

    /*
        Temporal filtering to reduce jitter and mis-detections
     */
    filterPoses: function(poses) {
        if (!tracker.enableTemporalFilter || !poses || poses.length === 0) {
            return poses;
        }
        const now = performance.now();
        tracker.cleanupFilterState(now);

        for (let p = 0; p < poses.length; p++) {
            const pose = poses[p];
            if (!pose || !pose.keypoints) {
                continue;
            }
            for (let i = 0; i < pose.keypoints.length; i++) {
                const kp = pose.keypoints[i];
                if (!kp || typeof kp.x !== 'number' || typeof kp.y !== 'number') {
                    continue;
                }
                const stateKey = p + ':' + kp.name;
                const prev = tracker.poseFilterState[stateKey];
                const confidence = tracker.clamp(kp.score || 0, 0, 1);

                if (!prev) {
                    tracker.poseFilterState[stateKey] = {
                        x: kp.x,
                        y: kp.y,
                        score: confidence,
                        t: now,
                    };
                    continue;
                }

                const dt = Math.max(1, now - prev.t);
                const maxStep = tracker.maxJumpPxPerFrame * (dt / 16.67);
                let rawX = kp.x;
                let rawY = kp.y;

                // Clamp abrupt jumps when confidence is not strong enough.
                const dxRaw = rawX - prev.x;
                const dyRaw = rawY - prev.y;
                const dist = Math.sqrt(dxRaw * dxRaw + dyRaw * dyRaw);
                if (dist > maxStep && confidence < 0.88) {
                    const ratio = maxStep / dist;
                    rawX = prev.x + dxRaw * ratio;
                    rawY = prev.y + dyRaw * ratio;
                }

                // Higher confidence -> faster adaptation.
                const alpha =
                    tracker.smoothingMinAlpha +
                    confidence * (tracker.smoothingMaxAlpha - tracker.smoothingMinAlpha);

                const fx = prev.x + alpha * (rawX - prev.x);
                const fy = prev.y + alpha * (rawY - prev.y);

                kp.x = fx;
                kp.y = fy;

                tracker.poseFilterState[stateKey] = {
                    x: fx,
                    y: fy,
                    score: confidence,
                    t: now,
                };
            }
        }
        return poses;
    },

    cloneKeypoints: function(keypoints) {
        return keypoints.map((k) =>
            k
                ? {
                      name: k.name,
                      x: k.x,
                      y: k.y,
                      z: k.z,
                      score: k.score,
                  }
                : k
        );
    },

    clonePoseSnapshot: function(pose) {
        if (!pose || !pose.keypoints) {
            return null;
        }
        return {
            keypoints: tracker.cloneKeypoints(pose.keypoints),
        };
    },

    meanKeypointScore: function(pose) {
        if (!pose || !pose.keypoints || !pose.keypoints.length) {
            return 0;
        }
        let s = 0;
        for (const kp of pose.keypoints) {
            s += kp && typeof kp.score === 'number' ? kp.score : 0;
        }
        return s / pose.keypoints.length;
    },

    /**
        Body-centric score: robust when face leaves the frame.
     */
    meanBodyCoreScore: function(pose) {
        if (!pose || !pose.keypoints || !pose.keypoints.length) {
            return 0;
        }
        // For COCO-17 (YOLO pose) many joints may be low-confidence; keep core minimal.
        const isCoco17 =
            pose.keypoints.length <= 17 &&
            pose.keypoints.some((k) => k && (k.name === 'nose' || k.name === 'left_hip'));
        const core = isCoco17
            ? ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
            : [
                  'left_shoulder', 'right_shoulder',
                  'left_hip', 'right_hip',
                  'left_elbow', 'right_elbow',
                  'left_wrist', 'right_wrist',
                  'left_knee', 'right_knee',
              ];
        let s = 0;
        let n = 0;
        for (const name of core) {
            const kp = pose.keypoints.find((k) => k && k.name === name);
            if (kp && typeof kp.score === 'number') {
                s += kp.score;
                n += 1;
            }
        }
        return n ? s / n : 0;
    },

    /** Оставляем одну позу с максимальной средней уверенностью (убираем ложные фигуры по краям кадра). */
    pickPrimaryPose: function(poses) {
        if (!poses || poses.length <= 1) {
            return poses;
        }
        let bestI = 0;
        let bestS = -1;
        for (let i = 0; i < poses.length; i++) {
            const m = Math.max(
                tracker.meanKeypointScore(poses[i]),
                tracker.meanBodyCoreScore(poses[i])
            );
            if (m > bestS) {
                bestS = m;
                bestI = i;
            }
        }
        const best = poses[bestI];
        const isCoco17 =
            best &&
            best.keypoints &&
            best.keypoints.length <= 17 &&
            best.keypoints.some((k) => k && (k.name === 'nose' || k.name === 'left_hip'));
        const personScore = Math.max(
            typeof best.score === 'number' ? best.score : 0,
            tracker.meanKeypointScore(best),
            tracker.meanBodyCoreScore(best)
        );
        const minPerson = isCoco17 ? 0.18 : tracker.personMinScore;
        if (personScore < minPerson) {
            return [];
        }
        return [best];
    },

    isKeypointRenderable: function(point) {
        if (!point || typeof point.x !== 'number' || typeof point.y !== 'number') {
            return false;
        }
        const dynamicMinScore =
            tracker.inferenceProviderActive === 'yolo_api'
                ? Math.min(tracker.minScore, 0.18)
                : tracker.minScore;
        if ((point.score || 0) < dynamicMinScore) {
            return false;
        }
        const v = tracker.video;
        if (!v || !v.videoWidth || !v.videoHeight) {
            return true;
        }
        const margin = 4;
        return (
            point.x >= margin &&
            point.y >= margin &&
            point.x <= v.videoWidth - margin &&
            point.y <= v.videoHeight - margin
        );
    },

    applyOcclusionHold: function(poses) {
        if (!tracker.enableOcclusionHold) {
            return poses;
        }
        const maxF = tracker.occlusionHoldMaxFrames;
        const minM = tracker.occlusionMinMeanScore;
        const drop = tracker.occlusionScoreDrop;

        if (!poses || poses.length === 0) {
            tracker._occHoldTicks = (tracker._occHoldTicks || 0) + 1;
            if (tracker._lastStablePose && tracker._occHoldTicks <= maxF) {
                const snap = tracker.clonePoseSnapshot(tracker._lastStablePose);
                return snap ? [snap] : poses;
            }
            return poses;
        }

        const pose = poses[0];
        if (!pose.keypoints || !pose.keypoints.length) {
            return poses;
        }

        const mean = tracker.meanKeypointScore(pose);
        const ema = tracker._prevMeanScore;
        const suddenDrop =
            ema != null && mean < ema - drop && mean < minM + 0.1;
        const low = mean < minM;

        if ((low || suddenDrop) && tracker._lastStablePose) {
            tracker._occHoldTicks = (tracker._occHoldTicks || 0) + 1;
            if (tracker._occHoldTicks <= maxF) {
                const snap = tracker.clonePoseSnapshot(tracker._lastStablePose);
                const merged = { ...pose, keypoints: snap.keypoints };
                return [merged];
            }
        }

        tracker._occHoldTicks = 0;
        tracker._prevMeanScore =
            ema == null ? mean : ema * 0.82 + mean * 0.18;
        if (mean >= minM) {
            tracker._lastStablePose = tracker.clonePoseSnapshot(pose);
        }
        return poses;
    },

    extrapolateFingertip: function(wrist, joint) {
        if (!wrist || !joint) {
            return joint;
        }
        const t = tracker.fingertipExtrapolate;
        const dx = joint.x - wrist.x;
        const dy = joint.y - wrist.y;
        return {
            x: joint.x + dx * t,
            y: joint.y + dy * t,
            score: joint.score,
        };
    },

    /*
        Draw segment using two keypoint-like objects {x,y,score}
     */
    drawSegmentPoints: function(fromPoint, toPoint, rgb) {
        if (!fromPoint || !toPoint) {
            return;
        }
        if (!tracker.isKeypointRenderable(fromPoint) || !tracker.isKeypointRenderable(toPoint)) {
            return;
        }
        const score = ((fromPoint.score || 0) + (toPoint.score || 0)) / 2;
        if (score < tracker.minScore) {
            return;
        }
        tracker.drawPath(
            fromPoint.x,
            fromPoint.y,
            toPoint.x,
            toPoint.y,
            rgb[0],
            rgb[1],
            rgb[2],
            score
        );
    },

    /*
        Midpoint helper for keypoints
     */
    midpoint: function(a, b) {
        if (!a || !b) {
            return null;
        }
        return {
            x: (a.x + b.x) / 2,
            y: (a.y + b.y) / 2,
            score: ((a.score || 0) + (b.score || 0)) / 2,
        };
    },

    /*
        Custom BlazePose: лицо без носа, квадрат головы, пальцы к кончикам.
     */
    drawBlazeFaceKeypointDots: function(kp) {
        const faceNames = [
            'left_eye_inner',
            'left_eye',
            'left_eye_outer',
            'right_eye_inner',
            'right_eye',
            'right_eye_outer',
            'left_ear',
            'right_ear',
            'mouth_left',
            'mouth_right',
        ];
        const r = Math.max(3, Math.round(tracker.pointRadius * 0.55));
        const faceLiftPx = -4;
        for (const name of faceNames) {
            const p = kp[name];
            if (!p || (p.score || 0) < tracker.minScore) {
                continue;
            }
            const a = Math.min(1, (p.score || 0) - 0.1);
            tracker.ctx.beginPath();
            tracker.ctx.arc(
                tracker.scaleX(p.x),
                tracker.scaleY(p.y) + faceLiftPx,
                r,
                0,
                2 * Math.PI
            );
            tracker.ctx.fillStyle = 'rgba(120,200,255,' + Math.max(0, a) + ')';
            tracker.ctx.fill();
            tracker.ctx.closePath();
        }
    },

    drawHandFingertipDots: function(kp) {
        const f = tracker.extrapolateFingertip;
        const pairs = [
            [kp.left_wrist, kp.left_thumb],
            [kp.left_wrist, kp.left_index],
            [kp.left_wrist, kp.left_pinky],
            [kp.right_wrist, kp.right_thumb],
            [kp.right_wrist, kp.right_index],
            [kp.right_wrist, kp.right_pinky],
        ];
        const r = Math.max(5, Math.round(tracker.pointRadius * 1.05));
        const handDropPx = 2;
        for (const [w, j] of pairs) {
            const tip = f(w, j);
            if (!tip || (tip.score || 0) < tracker.minScore) {
                continue;
            }
            const a = Math.min(1, (tip.score || 0) - 0.08);
            tracker.ctx.beginPath();
            tracker.ctx.arc(
                tracker.scaleX(tip.x),
                tracker.scaleY(tip.y) + handDropPx,
                r,
                0,
                2 * Math.PI
            );
            tracker.ctx.fillStyle = 'rgba(255,230,120,' + Math.max(0, a) + ')';
            tracker.ctx.strokeStyle = 'rgba(255,255,255,' + Math.max(0, a * 0.9) + ')';
            tracker.ctx.lineWidth = 2;
            tracker.ctx.fill();
            tracker.ctx.stroke();
            tracker.ctx.closePath();
        }
    },

    drawCustomBlazePose: function(pose) {
        const kp = {};
        for (const k of pose.keypoints) {
            kp[k.name] = k;
        }

        const neck = tracker.midpoint(kp.left_shoulder, kp.right_shoulder);
        const hips = tracker.midpoint(kp.left_hip, kp.right_hip);

        const seg = (a, b, rgb) => tracker.drawSegmentPoints(a, b, rgb);
        const f = tracker.extrapolateFingertip;

        // Лицо без носа
        seg(kp.left_eye_inner, kp.right_eye_inner, [120, 200, 255]);
        seg(kp.left_eye_inner, kp.left_eye, [120, 200, 255]);
        seg(kp.left_eye, kp.left_eye_outer, [120, 200, 255]);
        seg(kp.right_eye_inner, kp.right_eye, [120, 200, 255]);
        seg(kp.right_eye, kp.right_eye_outer, [120, 200, 255]);
        seg(kp.left_eye_outer, kp.left_ear, [160, 210, 255]);
        seg(kp.right_eye_outer, kp.right_ear, [160, 210, 255]);
        seg(kp.mouth_left, kp.mouth_right, [200, 120, 255]);
        tracker.drawBlazeFaceKeypointDots(kp);

        // Torso
        seg(kp.left_shoulder, kp.right_shoulder, [92, 70, 235]);
        seg(kp.left_hip, kp.right_hip, [140, 232, 90]);
        seg(kp.left_shoulder, kp.left_hip, [242, 85, 240]);
        seg(kp.right_shoulder, kp.right_hip, [242, 85, 240]);
        seg(kp.left_shoulder, kp.right_hip, [190, 120, 240]); // abdomen diagonal
        seg(kp.right_shoulder, kp.left_hip, [190, 120, 240]); // abdomen diagonal
        seg(neck, hips, [255, 214, 10]); // spine/back center line

        // Arms
        seg(kp.left_shoulder, kp.left_elbow, [245, 129, 66]);
        seg(kp.left_elbow, kp.left_wrist, [227, 156, 118]);
        seg(kp.right_shoulder, kp.right_elbow, [245, 129, 66]);
        seg(kp.right_elbow, kp.right_wrist, [227, 156, 118]);

        // Кисти: лучи к визуальным кончикам (экстраполяция за ключ BlazePose)
        const lT = f(kp.left_wrist, kp.left_thumb);
        const lI = f(kp.left_wrist, kp.left_index);
        const lP = f(kp.left_wrist, kp.left_pinky);
        const rT = f(kp.right_wrist, kp.right_thumb);
        const rI = f(kp.right_wrist, kp.right_index);
        const rP = f(kp.right_wrist, kp.right_pinky);
        seg(kp.left_wrist, lT, [255, 220, 80]);
        seg(kp.left_wrist, lI, [255, 220, 80]);
        seg(kp.left_wrist, lP, [255, 220, 80]);
        seg(kp.right_wrist, rT, [255, 220, 80]);
        seg(kp.right_wrist, rI, [255, 220, 80]);
        seg(kp.right_wrist, rP, [255, 220, 80]);
        tracker.drawHandFingertipDots(kp);

        // Legs
        seg(kp.left_hip, kp.left_knee, [42, 163, 69]);
        seg(kp.left_knee, kp.left_ankle, [140, 232, 90]);
        seg(kp.right_hip, kp.right_knee, [42, 163, 69]);
        seg(kp.right_knee, kp.right_ankle, [140, 232, 90]);
    },

    drawYoloPose: function(pose) {
        if (!pose || !pose.keypoints || !pose.keypoints.length) return;
        const kp = {};
        for (const k of pose.keypoints) kp[k.name] = k;
        if (tracker.yoloDrawSkeleton) {
            const segments = tracker.yoloSkeleton || [];
            for (const [a, b] of segments) {
                tracker.drawSegmentPoints(kp[a], kp[b], [120, 220, 255]);
            }
        }
        for (const k of pose.keypoints) {
            if (!tracker.isKeypointRenderable(k)) continue;
            const score = typeof k.score === 'number' ? k.score : 0;
            const alpha = Math.max(0.2, Math.min(1, score));
            tracker.ctx.beginPath();
            tracker.ctx.arc(tracker.scaleX(k.x), tracker.scaleY(k.y), tracker.pointRadius, 0, 2 * Math.PI);
            tracker.ctx.fillStyle = 'rgba(140,255,140,' + alpha + ')';
            tracker.ctx.fill();
            tracker.ctx.closePath();
        }
    },

    /*
        Find and return pose keypoint coordinate (X or Y) by keypoint's name
     */
    findPosePoint: function(axis, name, pose) {
        const kp = tracker.findKeypoint(name, pose);
        return kp[axis];
    },

    /*
        Return coordinate (X or Y) for points in path
     */
    getCoord: function(axis, points, pose) {
        // if only one point then return coordinate for this one
        if (points.length == 1) {
            return tracker.findPosePoint(axis, points[0], pose);
        } else {
            // if multiple points then calculate coordinate between them
            let sum = 0.0;
            for (const el of points) {
                sum += tracker.findPosePoint(axis, el, pose);
            }
            return sum / points.length;
        }
    },

    /*
        Return coordinates for path
     */
    getCoords: function(path, pose) {
        return {
            'from_x': tracker.getCoord('x', path.from_x, pose),
            'from_y': tracker.getCoord('y', path.from_y, pose),
            'to_x': tracker.getCoord('x', path.to_x, pose),
            'to_y': tracker.getCoord('y', path.to_y, pose),
        };
    },

    /*
        Get score for path
     */
    getScore: function(path, pose) {
        // if only one point then check score for this one
        if (path.scores.length == 1) {
            return tracker.findKeypoint(path.scores[0], pose).score;
        } else {
            // if multiple points then check score for all
            let sum = 0.0;
            for (const el of path.scores) {
                sum += tracker.findKeypoint(el, pose).score;
            }
            return sum / path.scores.length;
        }
    },

    /*
        Checks if path has required minimum score do draw it on canvas
     */
    hasScore: function(path, pose) {
        let res = true;
        // if only one point then check score for this one
        if (path.scores.length == 1) {
            if (tracker.findKeypoint(path.scores[0], pose).score < tracker.minScore) {
                res = false;
            }
        } else {
            // if multiple points then check score for all
            for (const el of path.scores) {
                if (tracker.findKeypoint(el, pose).score < tracker.minScore) {
                    res = false;
                    break;
                }
            }
        }
        return res;
    },

    /*
        Re-calculate size between source and destination area
     */
    calculateSize: function(srcSize, dstSize) {
        const srcRatio = srcSize.width / srcSize.height;
        const dstRatio = dstSize.width / dstSize.height;
        if (dstRatio > srcRatio) {
            return {
                width: dstSize.height * srcRatio,
                height: dstSize.height
            };
        } else {
            return {
                width: dstSize.width,
                height: dstSize.width / srcRatio
            };
        }
    },

    /*
        Re-calculate/scale X position of point
     */
    scaleX: function(x) {
        const c = tracker.renderCache;
        if (!c) return x;
        return Math.ceil(x * c.factorX) + c.xOffset;
    },

    /*
        Re-calculate/scale Y position of point
     */
    scaleY: function(y) {
        const c = tracker.renderCache;
        if (!c) return y;
        const nudge = typeof tracker.poseVerticalNudgePx === 'number' ? tracker.poseVerticalNudgePx : 0;
        return Math.ceil(y * c.factorY) + c.yOffset + nudge;
    },

    updateRenderCache: function(videoW, videoH, canvasW, canvasH) {
        if (!videoW || !videoH || !canvasW || !canvasH) {
            tracker.renderCache = null;
            return;
        }
        const renderSize = tracker.calculateSize(
            { width: videoW, height: videoH },
            { width: canvasW, height: canvasH }
        );
        let xOffset = (canvasW - renderSize.width) / 2;
        if (!tracker.autofit) {
            xOffset = 0;
        }
        let yOffset = (canvasH - renderSize.height) / 2;
        if (window.innerHeight > window.innerWidth || !tracker.autofit) {
            yOffset = 0;
        }
        tracker.renderCache = {
            factorX: renderSize.width / videoW,
            factorY: renderSize.height / videoH,
            xOffset: xOffset,
            yOffset: yOffset,
        };
    },

    /*
        Handle poses and draw them on canvas
     */
    handlePoses: function() {
        const v = tracker.video;
        tracker.updateRenderCache(
            v && v.videoWidth ? v.videoWidth : 0,
            v && v.videoHeight ? v.videoHeight : 0,
            tracker.canvas.width,
            tracker.canvas.height
        );
        // run user defined hooks
        tracker.dispatch('beforeupdate', tracker.poses);

        if (tracker.poses && tracker.poses.length > 0) {
            const pathlist = tracker.paths['blaze_pose'];

            let point, score;

            // loop on all finded poses
            for (let pose of tracker.poses) {
                const hasYoloShape =
                    pose &&
                    pose.keypoints &&
                    pose.keypoints.some((k) => k && (k.name === 'nose' || k.name === 'left_hip'));
                if (tracker.inferenceProviderActive === 'yolo_api' || hasYoloShape) {
                    tracker.drawYoloPose(pose);
                    continue;
                }
                if (tracker.detectorModel == poseDetection.SupportedModels.BlazePose && tracker.useCustomBlazePoseLayout) {
                    tracker.drawCustomBlazePose(pose);
                    continue;
                }

                // loop on pathslist
                for (let k in pathlist) {
                    
                    if (pathlist.hasOwnProperty(k)) {
                        
                        // if there is no required threeshold (score) then next
                        if (!tracker.hasScore(pathlist[k], pose)) {
                            continue;
                        }
                        point = tracker.getCoords(pathlist[k], pose); // get X,Y coords of path
                        score = tracker.getScore(pathlist[k], pose); // calculate score for path

                        // draw path on canvas
                        tracker.drawPath(point.from_x,
                            point.from_y,
                            point.to_x,
                            point.to_y,
                            pathlist[k].rgb[0],
                            pathlist[k].rgb[1],
                            pathlist[k].rgb[2],
                            score);
                    }
                }
            }
        }

        // run user defined hooks
        tracker.dispatch('afterupdate', tracker.poses);
    },

    /*
        Draw point and bone on canvas
     */
    drawPath: function(fromX, fromY, toX, toY, r, g, b, score) {
        // use score to calculate alpha
        let a = score - 0.15;
        if (a < 0) {
            a = 0.0;
        }
        // draw connection
        tracker.drawLine(tracker.scaleX(fromX), tracker.scaleY(fromY), 
            tracker.scaleX(toX), tracker.scaleY(toY), 
            r, g, b, a);

        // draw joint
        tracker.drawCircle(tracker.scaleX(fromX), tracker.scaleY(fromY), 
            r, g, b, a);
    },

    /*
        Draw connection between points on canvas
     */
    drawLine: function(fromX, fromY, toX, toY, r, g, b, a) {
        tracker.ctx.beginPath();
        tracker.ctx.lineWidth = tracker.pointWidth;
        tracker.ctx.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
        tracker.ctx.moveTo(fromX, fromY);
        tracker.ctx.lineTo(toX, toY);
        tracker.ctx.stroke();
        tracker.ctx.closePath();
    },

    /*
        Draw point on canvas
     */
    drawCircle: function(fromX, fromY, r, g, b, a) {
        tracker.ctx.beginPath();
        tracker.ctx.arc(fromX, fromY, tracker.pointRadius, 0, 2 * Math.PI);
        tracker.ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
        tracker.ctx.fill();
        tracker.ctx.closePath();
    },

    /*
        Clear canvas area
     */
    clearCanvas: function() {
        tracker.ctx.save();
        tracker.ctx.setTransform(1, 0, 0, 1, 0, 0);
        tracker.ctx.clearRect(0, 0, tracker.canvas.width, tracker.canvas.height);
        tracker.ctx.fillStyle = '#000000';
        tracker.ctx.fillRect(0, 0, tracker.canvas.width, tracker.canvas.height);
        tracker.ctx.restore();
    },

    /*
        Display play/pause icon
     */
    showPlaybackControls: function() {
        let size = (tracker.canvas.height / 2) * 0.5;

        tracker.ctx.fillStyle = "black";
        tracker.ctx.globalAlpha = 0.5;
        tracker.ctx.fillRect(0, 0, tracker.canvas.width, tracker.canvas.height);
        tracker.ctx.fillStyle = "#DDD";
        tracker.ctx.globalAlpha = 0.75;
        tracker.ctx.beginPath();
        tracker.ctx.moveTo(tracker.canvas.width / 2 + size / 2, tracker.canvas.height / 2);
        tracker.ctx.lineTo(tracker.canvas.width / 2 - size / 2, tracker.canvas.height / 2 + size);
        tracker.ctx.lineTo(tracker.canvas.width / 2 - size / 2, tracker.canvas.height / 2 - size);
        tracker.ctx.closePath();
        tracker.ctx.fill();
        tracker.ctx.globalAlpha = 1;
    },

    /*
        Handle play/pause click on video
     */
    playPauseClick: function() {
        if (tracker.container !== undefined && tracker.container.ready) {
            if (tracker.container.video.paused) {
                tracker.log('click: Play');
                tracker.play();
                tracker.isWaiting = true;
                tracker.setStatus('Please wait...');
            } else {
                // abort if waiting for playing
                if (!tracker.isWaiting) {
                    tracker.log('click: Pause');
                    tracker.pause();
                    tracker.setStatus('Paused.');
                }
            }
        }
    },

    /*
        Play video
     */
    play: function() {
        tracker.container.video.play();
    },

    /*
        Pause video
     */
    pause: function() {
        tracker.container.video.pause();
    },

    /*
        Log message
     */
    log: function(...args) {
        if (tracker.log) {
            console.log(...args);
        }
    },

    /*
        Set status message
     */
    setStatus: function(msg) {
        tracker.status = msg;
        tracker.dispatch('statuschange', tracker.status);
    },

    /*
        Append external hook/event
     */
    on: function(name, hook) {
        if (typeof tracker.hooks[name] === 'undefined') {
            return;
        }
        tracker.hooks[name].push(hook);
    },

    /*
        Dispatch hook/event
     */
    dispatch: function(name, event) {
        if (typeof tracker.hooks[name] === 'undefined') {
            return;
        }
        for (const hook of tracker.hooks[name]) {
            hook(event);
        }
    },

    /*
        Pre-initialize model by name
     */
    setModel: function(model) {
        switch (model) {
            case 'BlazePoseLite':
                tracker.detectorModel = poseDetection.SupportedModels.BlazePose;
                tracker.detectorConfig = {
                    runtime: 'tfjs',
                    enableSmoothing: false,
                    modelType: 'lite'
                };
                tracker.minScore = 0.35;
                break;

            case 'BlazePoseHeavy':
                tracker.detectorModel = poseDetection.SupportedModels.BlazePose;
                tracker.detectorConfig = {
                    runtime: 'tfjs',
                    enableSmoothing: true,
                    modelType: 'heavy'
                };
                tracker.minScore = 0.65;
                break;

            case 'BlazePoseFull':
                tracker.detectorModel = poseDetection.SupportedModels.BlazePose;
                tracker.detectorConfig = {
                    runtime: 'tfjs',
                    enableSmoothing: true,
                    modelType: 'full'
                };
                tracker.minScore = 0.65;
                break;
        }
    },

    ensureInferenceReady: async function() {
        if (tracker.inferenceProviderPreferred === 'yolo_api') {
            const ok = await tracker.checkYoloHealth(true);
            if (ok) {
                tracker.inferenceProviderActive = 'yolo_api';
                return;
            }
        }
        tracker.inferenceProviderActive = 'blazepose';
        await tracker.ensureBlazeDetector();
    },

    ensureBlazeDetector: async function() {
        if (tracker.detector) return;
        tracker.detector = await poseDetection.createDetector(
            tracker.detectorModel,
            tracker.detectorConfig
        );
    },

    checkYoloHealth: async function(force) {
        const now = performance.now();
        if (!force && now - tracker._lastYoloHealthCheckTs < tracker.yoloRecoveryCheckMs) {
            return tracker._lastYoloHealthOk;
        }
        tracker._lastYoloHealthCheckTs = now;
        try {
            const r = await auth.apiFetch(tracker.yoloHealthPath, { method: 'GET' });
            tracker._lastYoloHealthOk = !!(r && r.ok && r.modelExists);
        } catch (_) {
            tracker._lastYoloHealthOk = false;
        }
        return tracker._lastYoloHealthOk;
    },

    getYoloCanvas: function() {
        if (!tracker._yoloCanvas) {
            tracker._yoloCanvas = document.createElement('canvas');
            tracker._yoloCtx = tracker._yoloCanvas.getContext('2d', { alpha: false });
        }
        return tracker._yoloCanvas;
    },

    yoloToPoses: function(apiResult, scaleX, scaleY) {
        const dets = apiResult && apiResult.detections ? apiResult.detections : [];
        return dets.map((d) => ({
            score: d.score || 0,
            keypoints: (d.keypoints || []).map((k) => {
                const name = k && k.name ? k.name : 'unknown';
                const x = typeof k.x === 'number' ? k.x * scaleX : 0;
                const y = typeof k.y === 'number' ? k.y * scaleY : 0;
                const score = typeof k.score === 'number' ? k.score : 0;
                return { name, x, y, score };
            }),
        }));
    },

    applyYoloHold: function(poses) {
        if (tracker.inferenceProviderActive !== 'yolo_api') return poses;
        const maxF = Math.max(0, tracker.yoloHoldMaxFrames || 0);
        if (poses && poses.length && poses[0] && poses[0].keypoints && poses[0].keypoints.length) {
            tracker._yoloLastGoodPose = tracker.clonePoseSnapshot(poses[0]);
            tracker._yoloHoldFramesLeft = maxF;
            return poses;
        }
        if (tracker._yoloLastGoodPose && tracker._yoloHoldFramesLeft > 0) {
            tracker._yoloHoldFramesLeft -= 1;
            const snap = tracker.clonePoseSnapshot(tracker._yoloLastGoodPose);
            return snap ? [snap] : poses;
        }
        return poses;
    },

    estimatePosesYoloApi: async function(source) {
        if (tracker._yoloInFlight) return tracker.poses || [];
        const sw = source && source.videoWidth ? source.videoWidth : 0;
        const sh = source && source.videoHeight ? source.videoHeight : 0;
        if (!sw || !sh) return [];

        const c = tracker.getYoloCanvas();
        const recordingMode =
            typeof window !== 'undefined' &&
            window.recording &&
            typeof window.recording.isRecording === 'function' &&
            window.recording.isRecording();
        const baseCap = recordingMode
            ? tracker.yoloFrameMaxWidthRecording ?? 256
            : tracker.yoloFrameMaxWidth ?? 320;
        const maxW = Math.min(sw, Math.max(160, baseCap));
        const targetW = maxW;
        const targetH = Math.max(1, Math.round((targetW / sw) * sh));
        c.width = targetW;
        c.height = targetH;
        tracker._yoloCtx.drawImage(source, 0, 0, targetW, targetH);
        const jpegQ = recordingMode
            ? tracker.yoloJpegQualityRecording ?? tracker.yoloJpegQuality
            : tracker.yoloJpegQuality;
        const blob = await new Promise((resolve) =>
            c.toBlob(resolve, 'image/jpeg', jpegQ)
        );
        if (!blob) return [];

        const fd = new FormData();
        fd.append('image', blob, 'frame.jpg');
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), tracker.yoloRequestTimeoutMs);
        tracker._yoloInFlight = true;
        try {
            const r = await auth.apiFetch(tracker.yoloPosePath, {
                method: 'POST',
                body: fd,
                signal: controller.signal,
            });
            const poses = tracker.yoloToPoses(r && r.result ? r.result : null, sw / targetW, sh / targetH);
            tracker._yoloErrorStreak = 0;
            return poses;
        } finally {
            clearTimeout(timer);
            tracker._yoloInFlight = false;
        }
    },

    maybeRecoverYolo: async function() {
        if (tracker.inferenceProviderPreferred !== 'yolo_api') return;
        const ok = await tracker.checkYoloHealth(false);
        if (!ok) return;
        tracker.inferenceProviderActive = 'yolo_api';
        tracker._yoloErrorStreak = 0;
    },

    getPosesForFrame: async function(source, now) {
        if (tracker.inferenceProviderActive === 'yolo_api') {
            try {
                const poses = await tracker.estimatePosesYoloApi(source);
                return tracker.applyYoloHold(poses);
            } catch (e) {
                tracker._yoloErrorStreak += 1;
                if (tracker._yoloErrorStreak >= tracker.yoloMaxErrorsBeforeFallback) {
                    tracker.inferenceProviderActive = 'blazepose';
                    await tracker.ensureBlazeDetector();
                }
                if (tracker.detector) {
                    return await tracker.detector.estimatePoses(
                        source,
                        { flipHorizontal: false },
                        now
                    );
                }
                return [];
            }
        }
        await tracker.maybeRecoverYolo();
        if (tracker.inferenceProviderActive === 'yolo_api') {
            const poses = await tracker.estimatePosesYoloApi(source);
            return tracker.applyYoloHold(poses);
        }
        await tracker.ensureBlazeDetector();
        return await tracker.detector.estimatePoses(
            source,
            { flipHorizontal: false },
            now
        );
    },
}

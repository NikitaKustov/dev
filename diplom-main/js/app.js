const app = {
    debug: false,
    cameraStarted: false,
    themeStorageKey: 'body_tracker_theme',

    init() {
        this.initTheme();
        this.bindStaticUI();
        this.renderProfile();
        this.showPage('home');
    },

    async bootstrapTrainingUi() {
        await training.init();
    },

    bindStaticUI() {
        document.getElementById('btn_home')?.addEventListener('click', () => this.showPage('home'));
        document.getElementById('btn_go_training')?.addEventListener('click', () => this.showPage('training'));
        document.getElementById('btn_about')?.addEventListener('click', () => this.showPage('about'));
        document.getElementById('btn_profile')?.addEventListener('click', () => {
            this.renderProfile();
            this.showPage('profile');
        });
        document.getElementById('btn_admin')?.addEventListener('click', () => this.showPage('admin'));
        document.getElementById('btn_about_back')?.addEventListener('click', () => {
            this.showPage(this.cameraStarted ? 'training' : 'home');
        });
        document.getElementById('btn_profile_back')?.addEventListener('click', () => {
            this.showPage(this.cameraStarted ? 'training' : 'home');
        });
        document.getElementById('btn_theme_toggle')?.addEventListener('click', () => {
            this.toggleTheme();
        });
        const btnStartCam = document.getElementById('btn_start_camera');
        const btnStopCam = document.getElementById('btn_stop_camera');
        btnStartCam?.addEventListener('click', function () {
            if (app.cameraStarted) return;
            app.cameraStarted = true;
            this.disabled = true;
            this.textContent = 'Запуск...';
            if (btnStopCam) btnStopCam.disabled = true;
            tracker.setStatus('Запуск камеры...');
            setTimeout(() => {
                tracker.run('camera');
                this.textContent = 'Камера включена';
                if (btnStopCam) btnStopCam.disabled = false;
            }, 0);
        });
        btnStopCam?.addEventListener('click', () => {
            if (!app.cameraStarted) return;
            tracker.stopCamera();
            app.cameraStarted = false;
            if (btnStartCam) {
                btnStartCam.disabled = false;
                btnStartCam.textContent = 'Включить камеру';
            }
            btnStopCam.disabled = true;
        });

        document.getElementById('profile_form')?.addEventListener('submit', async (e) => {
            e.preventDefault();
            const body = {
                name: document.getElementById('profile_name').value.trim(),
                firstname: document.getElementById('profile_firstname').value.trim(),
                lastname: document.getElementById('profile_lastname').value.trim(),
                email: document.getElementById('profile_email').value.trim(),
            };
            const msg = document.getElementById('profile_msg');
            try {
                const r = await auth.apiFetch('/api/me', {
                    method: 'PUT',
                    body: JSON.stringify(body),
                });
                auth.updateUser(r.user);
                msg.textContent = 'Профиль сохранен';
            } catch (err) {
                msg.textContent = err.message || 'Ошибка сохранения';
            }
        });

        document.getElementById('profile_avatar_file')?.addEventListener('change', async (e) => {
            const f = e.target.files && e.target.files[0];
            if (!f) return;
            const fd = new FormData();
            fd.append('avatar', f);
            const msg = document.getElementById('profile_msg');
            try {
                const r = await auth.apiFetch('/api/me/avatar', { method: 'POST', body: fd });
                auth.updateUser(r.user);
                app.renderProfile();
                msg.textContent = 'Фото обновлено';
            } catch (err) {
                msg.textContent = err.message || 'Ошибка загрузки фото';
            }
        });

        document.getElementById('btn_delete_account')?.addEventListener('click', async () => {
            const msg = document.getElementById('profile_msg');
            if (
                !confirm(
                    'Удалить аккаунт без возможности восстановления? Все сохранённые видео тренировок будут удалены.'
                )
            ) {
                return;
            }
            if (!confirm('Подтвердите удаление аккаунта.')) {
                return;
            }
            try {
                await auth.apiFetch('/api/me', { method: 'DELETE' });
                if (msg) msg.textContent = 'Аккаунт удалён.';
                auth.logout();
            } catch (err) {
                const m = err && err.message ? String(err.message) : '';
                if (msg) {
                    msg.textContent =
                        m === 'Could not delete account' ? 'Не удалось удалить аккаунт' : m || 'Не удалось удалить аккаунт';
                }
            }
        });
    },

    initTheme() {
        let savedTheme = 'dark';
        try {
            savedTheme = localStorage.getItem(this.themeStorageKey) || 'dark';
        } catch (_) {}
        this.applyTheme(savedTheme === 'light' ? 'light' : 'dark');
    },

    applyTheme(theme) {
        const isLight = theme === 'light';
        document.body.classList.toggle('light-theme', isLight);
        const toggleBtn = document.getElementById('btn_theme_toggle');
        const cornerBtn = document.getElementById('btn_theme_corner');
        [toggleBtn, cornerBtn].forEach((btn) => {
            if (!btn) return;
            btn.textContent = isLight ? 'Темная тема' : 'Светлая тема';
            btn.classList.toggle('btn-outline-dark', isLight);
            btn.classList.toggle('btn-outline-light', !isLight);
        });
        try {
            localStorage.setItem(this.themeStorageKey, isLight ? 'light' : 'dark');
        } catch (_) {}
    },

    toggleTheme() {
        const next = document.body.classList.contains('light-theme') ? 'dark' : 'light';
        this.applyTheme(next);
    },

    showPage(page) {
        if (page === 'admin') {
            const u = auth.getUser ? auth.getUser() : null;
            if (!u || String(u.name || '').toLowerCase() !== 'admin') {
                return;
            }
        }
        document.getElementById('page_home')?.classList.toggle('active', page === 'home');
        document.getElementById('page_training')?.classList.toggle('active', page === 'training');
        document.getElementById('page_about')?.classList.toggle('active', page === 'about');
        document.getElementById('page_profile')?.classList.toggle('active', page === 'profile');
        document.getElementById('page_admin')?.classList.toggle('active', page === 'admin');
    },

    async renderProfile() {
        const user = auth.getUser ? auth.getUser() : null;
        if (!user) return;
        document.getElementById('profile_name').value = user.name || '';
        document.getElementById('profile_firstname').value = user.firstname || '';
        document.getElementById('profile_lastname').value = user.lastname || '';
        document.getElementById('profile_email').value = user.email || '';
        const img = document.getElementById('profile_avatar_img');
        if (img) {
            img.src = user.avatar || '';
            img.style.display = user.avatar ? 'block' : 'none';
        }
        const btnAdmin = document.getElementById('btn_admin');
        const isAd = String(user.name || '').toLowerCase() === 'admin';
        if (btnAdmin) btnAdmin.hidden = !isAd;
        document.getElementById('main_app')?.classList.toggle('user-is-admin', isAd);
        const pageAd = document.getElementById('page_admin');
        if (pageAd) pageAd.hidden = !isAd;
        await this.renderProfileSavedTrainings();
    },

    async renderProfileSavedTrainings() {
        const root = document.getElementById('profile_saved_trainings');
        if (!root) return;
        root.innerHTML = '<p class="text-muted small mb-0">Загрузка...</p>';
        try {
            const rows = await auth.apiFetch('/api/results/mine');
            if (!rows.length) {
                root.innerHTML = '<p class="text-muted small mb-0">Пока нет сохраненных тренировок.</p>';
                return;
            }
            const byExercise = {};
            for (const row of rows) {
                const name = row.event_name || 'Без названия';
                if (!byExercise[name]) byExercise[name] = [];
                byExercise[name].push(row);
            }
            root.innerHTML = '';
            Object.keys(byExercise).forEach((exerciseName) => {
                const card = document.createElement('div');
                card.className = 'profile-training-group';
                const title = document.createElement('div');
                title.className = 'profile-training-group-title';
                title.textContent = exerciseName;
                card.appendChild(title);
                byExercise[exerciseName].forEach((row) => {
                    const vid = document.createElement('video');
                    vid.className = 'training-user-video mb-2';
                    vid.controls = true;
                    vid.src = row.user_video;
                    card.appendChild(vid);
                });
                root.appendChild(card);
            });
        } catch (e) {
            root.innerHTML = `<p class="text-warning small mb-0">${e.message || 'Ошибка загрузки'}</p>`;
        }
    },

    updateStatus(msg) {
        const s = document.getElementById('status');
        if (s) s.textContent = msg || '';
    },

    updateCounter() {},

    updateDebug(poses) {
        if (!this.debug) return;
        const info = document.getElementById('info_debug');
        if (!info) return;
        let str = '';
        for (const pose of poses || []) {
            for (const kp of pose.keypoints || []) {
                str += `${kp.name}: x=${kp.x.toFixed(1)} y=${kp.y.toFixed(1)} s=${(kp.score || 0).toFixed(2)}<br>`;
            }
        }
        info.innerHTML = str;
    },
};

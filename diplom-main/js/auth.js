/**
 * API-backed sign-in (users table on server). Requires app to be served from the Node server.
 */
const auth = (function () {
    const SESSION_KEY = 'body_tracker_api_session';
    const SESSION_MS = 7 * 24 * 60 * 60 * 1000;
    const THEME_KEY = 'body_tracker_theme';

    let onAuthCallback = null;
    let serverAvailable = false;

    function getSessionRaw() {
        try {
            const raw = sessionStorage.getItem(SESSION_KEY);
            if (!raw) return null;
            const s = JSON.parse(raw);
            if (!s.expires || Date.now() > s.expires) {
                sessionStorage.removeItem(SESSION_KEY);
                return null;
            }
            return s;
        } catch {
            return null;
        }
    }

    function setSession(token, user) {
        sessionStorage.setItem(
            SESSION_KEY,
            JSON.stringify({ token, user, expires: Date.now() + SESSION_MS })
        );
    }

    function updateSessionUser(user) {
        const s = getSessionRaw();
        if (!s || !s.token) return;
        setSession(s.token, user);
    }

    function clearSession() {
        sessionStorage.removeItem(SESSION_KEY);
    }

    function getToken() {
        const s = getSessionRaw();
        return s ? s.token : null;
    }

    function getUser() {
        const s = getSessionRaw();
        return s ? s.user : null;
    }

    async function apiFetch(path, options = {}) {
        const token = getToken();
        const headers = { ...(options.headers || {}) };
        if (options.body && !headers['Content-Type'] && !(options.body instanceof FormData)) {
            headers['Content-Type'] = 'application/json';
        }
        if (token) headers.Authorization = `Bearer ${token}`;
        const res = await fetch(path, { ...options, headers });
        const text = await res.text();
        let data = null;
        try {
            data = text ? JSON.parse(text) : null;
        } catch {
            data = { error: text || res.statusText };
        }
        if (!res.ok) {
            const err = new Error(data && data.error ? data.error : res.statusText);
            err.status = res.status;
            err.data = data;
            throw err;
        }
        return data;
    }

    function showView(id) {
        document.querySelectorAll('[data-auth-view]').forEach((el) => {
            el.hidden = el.getAttribute('data-auth-view') !== id;
        });
        const icon = document.querySelector('.auth-register-icon');
        if (icon) icon.style.display = id === 'login' ? '' : 'none';
    }

    function setError(el, msg) {
        if (!el) return;
        el.textContent = msg || '';
        el.hidden = !msg;
    }

    function applyStoredTheme() {
        let theme = 'dark';
        try {
            theme = localStorage.getItem(THEME_KEY) || 'dark';
        } catch (_) {}
        const isLight = theme === 'light';
        document.body.classList.toggle('light-theme', isLight);
        const btn = document.getElementById('btn_theme_corner');
        if (btn) {
            btn.textContent = isLight ? 'Темная тема' : 'Светлая тема';
            btn.classList.toggle('btn-outline-dark', isLight);
            btn.classList.toggle('btn-outline-light', !isLight);
        }
    }

    function toggleThemeFromAuth() {
        const isLight = document.body.classList.contains('light-theme');
        const next = isLight ? 'dark' : 'light';
        try {
            localStorage.setItem(THEME_KEY, next);
        } catch (_) {}
        applyStoredTheme();
    }

    function finishAuth() {
        const overlay = document.getElementById('auth_overlay');
        if (overlay) overlay.hidden = true;
        document.body.classList.remove('auth-locked');
        if (typeof onAuthCallback === 'function') onAuthCallback();
    }

    function showServerDown(message) {
        const overlay = document.getElementById('auth_overlay');
        if (!overlay) return;
        overlay.hidden = false;
        document.body.classList.add('auth-locked');
        overlay.innerHTML =
            '<div class="auth-card">' +
            '<h2 class="auth-title">Нужен сервер</h2>' +
            '<p class="auth-note">' +
            (message ||
                'В папке проекта выполните <code class="auth-secret">npm run install:server</code> и <code class="auth-secret">npm start</code>, ' +
                'затем откройте <strong>http://localhost:3000</strong>.') +
            '</p></div>';
    }

    function mapApiError(msg) {
        const m = {
            'All fields are required': 'Заполните все поля',
            'Password must be at least 6 characters': 'Пароль не короче 6 символов',
            'Email already registered': 'Этот email уже зарегистрирован',
            'Registration failed': 'Ошибка регистрации',
            'Email/login and password required': 'Введите email (или логин) и пароль',
            'Invalid credentials': 'Неверный email/логин или пароль',
            'Login already registered': 'Этот логин уже занят',
            'Could not delete account': 'Не удалось удалить аккаунт',
        };
        return m[msg] || msg;
    }

    function bindForms() {
        const errReg = document.getElementById('auth_err_register');
        const errLogin = document.getElementById('auth_err_login');
        const formReg = document.getElementById('form_register');
        const formLogin = document.getElementById('form_login');
        const btnLogout = document.getElementById('auth_btn_logout');

        formReg?.addEventListener('submit', async (e) => {
            e.preventDefault();
            setError(errReg, '');
            const submitBtn = formReg.querySelector('button[type="submit"]');
            const body = {
                name: document.getElementById('reg_name').value.trim(),
                lastname: document.getElementById('reg_lastname').value.trim(),
                firstname: document.getElementById('reg_firstname').value.trim(),
                email: document.getElementById('reg_email').value.trim(),
                password: document.getElementById('reg_password').value,
            };
            const p2 = document.getElementById('reg_password2').value;
            if (body.password !== p2) {
                setError(errReg, 'Пароли не совпадают.');
                return;
            }
            if (submitBtn) submitBtn.disabled = true;
            try {
                const data = await apiFetch('/api/register', {
                    method: 'POST',
                    body: JSON.stringify(body),
                });
                if (!data || !data.token || !data.user) {
                    setError(errReg, 'Сервер вернул неполный ответ. Попробуйте войти по email и паролю.');
                    return;
                }
                try {
                    setSession(data.token, data.user);
                } catch (storeErr) {
                    const m = String(storeErr && storeErr.message ? storeErr.message : storeErr);
                    if (/QuotaExceeded|quota/i.test(m)) {
                        setError(errReg, 'Не удалось сохранить сессию: память браузера переполнена.');
                    } else if (/SecurityError|blocked|storage/i.test(m)) {
                        setError(
                            errReg,
                            'Регистрация выполнена на сервере, но браузер запретил сохранение сессии. Разрешите cookies/хранилище для этого сайта или войдите вручную.'
                        );
                    } else {
                        setError(errReg, 'Не удалось сохранить сессию в браузере. Попробуйте войти вручную.');
                    }
                    return;
                }
                finishAuth();
            } catch (err) {
                setError(errReg, mapApiError(err.message) || 'Ошибка регистрации');
            } finally {
                if (submitBtn) submitBtn.disabled = false;
            }
        });

        formLogin?.addEventListener('submit', async (e) => {
            e.preventDefault();
            setError(errLogin, '');
            const identity = document.getElementById('login_identity').value.trim();
            const password = document.getElementById('login_password').value;
            try {
                const data = await apiFetch('/api/login', {
                    method: 'POST',
                    body: JSON.stringify({ identity, password }),
                });
                setSession(data.token, data.user);
                finishAuth();
            } catch (err) {
                setError(errLogin, mapApiError(err.message) || 'Ошибка входа');
            }
        });

        btnLogout?.addEventListener('click', () => {
            clearSession();
            location.reload();
        });

        document.getElementById('link_show_login')?.addEventListener('click', (e) => {
            e.preventDefault();
            if (window.history && window.history.replaceState) {
                window.history.replaceState({}, '', '?');
            }
            showView('login');
        });
    }

    async function start() {
        applyStoredTheme();
        document.getElementById('btn_theme_corner')?.addEventListener('click', () => {
            toggleThemeFromAuth();
        });
        try {
            const r = await fetch('/api/health');
            serverAvailable = r.ok;
        } catch {
            serverAvailable = false;
        }

        if (!serverAvailable) {
            showServerDown();
            return;
        }

        bindForms();
        document.body.classList.add('auth-locked');
        const overlay = document.getElementById('auth_overlay');
        if (overlay) {
            overlay.hidden = false;
            if (!overlay.querySelector('.auth-card')) {
                location.reload();
                return;
            }
        }

        if (getSessionRaw() && getUser()) {
            finishAuth();
            return;
        }

        const qs = new URLSearchParams(window.location.search);
        if (qs.get('register') === '1') showView('register');
        else showView('login');
    }

    return {
        onAuthenticated(cb) {
            onAuthCallback = cb;
        },
        start,
        logout() {
            clearSession();
            location.reload();
        },
        getToken,
        getUser,
        updateUser: updateSessionUser,
        apiFetch,
        isServerAvailable() {
            return serverAvailable;
        },
        isSessionValid() {
            return !!(getSessionRaw() && getUser());
        },
    };
})();

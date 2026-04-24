const training = {
    _selectedTrainingId: 0,
    _eventByName: {},
    _activeName: 'Отжимания',

    async refreshEventCacheAndRef() {
        try {
            const events = await auth.apiFetch('/api/events');
            training._eventByName = {};
            for (const e of events) training._eventByName[e.name] = e;
            training.updateReferenceVideoUI();
        } catch (e) {
            console.warn(e);
        }
    },

    updateReferenceVideoUI() {
        const vid = document.getElementById('training_ref_video');
        const empty = document.getElementById('training_ref_empty');
        if (!vid) return;
        const ev = training._eventByName[training._activeName];
        const url = ev && ev.video && String(ev.video).trim();
        if (url) {
            if (vid.dataset.loadedUrl !== url) {
                vid.src = url;
                vid.dataset.loadedUrl = url;
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
    },

    async init() {
        const listEl = document.getElementById('training_results_list');
        const label = document.getElementById('selected_exercise_label');
        const setsInput = document.getElementById('training_sets_count');
        const buttons = Array.from(document.querySelectorAll('.exercise-btn'));
        if (!listEl || !buttons.length) return;
        if (setsInput) {
            setsInput.addEventListener('change', () => {
                const n = Number(setsInput.value);
                setsInput.value = String(Number.isInteger(n) && n > 0 ? n : 1);
            });
        }

        const events = await auth.apiFetch('/api/events');
        training._eventByName = {};
        for (const e of events) training._eventByName[e.name] = e;

        function setActive(name) {
            training._activeName = name;
            buttons.forEach((b) => b.classList.toggle('active', b.dataset.exName === name));
            const ev = training._eventByName[name];
            if (ev) {
                training._selectedTrainingId = Number(ev.id_trainies);
                label.textContent = `Выбрано упражнение: ${name}`;
            } else {
                training._selectedTrainingId = 0;
                label.textContent = 'Упражнение не найдено в базе.';
            }
            training.updateReferenceVideoUI();
            training.refreshResults();
        }

        buttons.forEach((b) => {
            b.addEventListener('click', () => setActive(b.dataset.exName));
        });
        setActive('Отжимания');
    },

    getSelectedTrainingId() {
        return training._selectedTrainingId;
    },

    getSetsCount() {
        const input = document.getElementById('training_sets_count');
        if (!input) return 1;
        const n = Number(input.value);
        if (!Number.isInteger(n) || n < 1) return 1;
        return n;
    },

    async refreshResults() {
        const listEl = document.getElementById('training_results_list');
        if (!listEl) return;
        listEl.innerHTML = '';
        let rows = [];
        try {
            rows = await auth.apiFetch('/api/results/mine');
        } catch {
            listEl.innerHTML = '<p class="text-muted small">Войдите, чтобы видеть сохраненные видео.</p>';
            return;
        }
        const selected = training.getSelectedTrainingId();
        const filtered = selected ? rows.filter((r) => Number(r.id_trainies) === selected) : rows;
        if (!filtered.length) {
            listEl.innerHTML = '<p class="text-muted small">Пока нет сохраненных записей.</p>';
            return;
        }
        filtered.forEach((row) => {
            const wrap = document.createElement('div');
            wrap.className = 'training-result-item';
            const title = document.createElement('div');
            title.className = 'training-result-title';
            title.textContent = row.event_name + ' (запись №' + row.user_result_no + ')';
            const v = document.createElement('video');
            v.className = 'training-user-video';
            v.controls = true;
            v.src = row.user_video;
            const report = document.createElement('div');
            report.className = 'training-result-report small mt-2';
            let reportText = '';
            try {
                const analysis = row.analysis_json ? JSON.parse(row.analysis_json) : null;
                if (analysis && Array.isArray(analysis.errors) && analysis.errors.length) {
                    reportText = `Ошибки: ${analysis.errors.slice(0, 4).join('; ')}`;
                } else if (analysis && analysis.summary) {
                    reportText = analysis.summary;
                }
            } catch (_) {}
            if (!reportText && row.sets_done) {
                reportText = `Подходов: ${row.sets_done}`;
            }
            if (reportText) {
                report.textContent = reportText;
            }
            const del = document.createElement('button');
            del.type = 'button';
            del.className = 'btn btn-sm btn-outline-danger training-delete-btn';
            del.textContent = 'Удалить запись';
            del.addEventListener('click', async () => {
                del.disabled = true;
                try {
                    await auth.apiFetch('/api/results/' + row.id, { method: 'DELETE' });
                    await training.refreshResults();
                } catch (e) {
                    del.disabled = false;
                    alert(e.message || 'Ошибка удаления');
                }
            });
            wrap.appendChild(title);
            wrap.appendChild(v);
            if (reportText) {
                wrap.appendChild(report);
            }
            wrap.appendChild(del);
            listEl.appendChild(wrap);
        });
    },
};

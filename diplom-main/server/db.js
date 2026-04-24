const fs = require('fs');
const path = require('path');

const dbPath = path.join(__dirname, 'data', 'app.db');
let db = null;

function persist() {
    if (!db) return;
    fs.mkdirSync(path.dirname(dbPath), { recursive: true });
    fs.writeFileSync(dbPath, Buffer.from(db.export()));
}

function run(sql, params = []) {
    if (params && params.length) db.run(sql, params);
    else db.run(sql);
    persist();
}

function get(sql, params = []) {
    const stmt = db.prepare(sql);
    if (params && params.length) stmt.bind(params);
    if (!stmt.step()) {
        stmt.free();
        return undefined;
    }
    const row = stmt.getAsObject();
    stmt.free();
    return row;
}

function all(sql, params = []) {
    const stmt = db.prepare(sql);
    if (params && params.length) stmt.bind(params);
    const rows = [];
    while (stmt.step()) {
        rows.push(stmt.getAsObject());
    }
    stmt.free();
    return rows;
}

function lastInsertRowid() {
    const r = db.exec('SELECT last_insert_rowid() AS id');
    if (!r.length || !r[0].values || !r[0].values.length) return 0;
    return Number(r[0].values[0][0]);
}

function seedEventsIfEmpty() {
    const defaults = [
        ['Отжимания', 'Базовое упражнение на грудь и трицепс.', ''],
        ['Приседания', 'Базовое упражнение на ноги и корпус.', ''],
        ['Пресс', 'Упражнения на мышцы пресса и кора.', ''],
    ];
    for (const [name, description, video] of defaults) {
        const row = get('SELECT id_trainies FROM events WHERE name = ?', [name]);
        if (!row) {
            run('INSERT INTO events (name, description, video) VALUES (?, ?, ?)', [
                name,
                description,
                video,
            ]);
        }
    }
    // Keep only supported training buttons in UI.
    run(
        "DELETE FROM events WHERE name NOT IN ('Отжимания','Приседания','Пресс')",
        []
    );
}

function ensureUniqueUserLogins() {
    const rows = all('SELECT id_users, name FROM users ORDER BY id_users ASC', []);
    const used = new Set();
    for (const row of rows) {
        const id = Number(row.id_users);
        const raw = String(row.name || '').trim();
        let base = raw || `user_${id}`;
        let candidate = base;
        let probe = 1;
        while (used.has(candidate.toLowerCase())) {
            candidate = `${base}_${probe}`;
            probe += 1;
        }
        used.add(candidate.toLowerCase());
        if (candidate !== raw) {
            db.run('UPDATE users SET name = ? WHERE id_users = ?', [candidate, id]);
        }
    }
}

async function initDatabase() {
    const initSqlJs = require('sql.js');
    const SQL = await initSqlJs({
        locateFile: (file) => path.join(__dirname, 'node_modules', 'sql.js', 'dist', file),
    });
    fs.mkdirSync(path.dirname(dbPath), { recursive: true });
    if (fs.existsSync(dbPath)) {
        db = new SQL.Database(new Uint8Array(fs.readFileSync(dbPath)));
    } else {
        db = new SQL.Database();
    }

    db.run('PRAGMA foreign_keys = ON');

    db.run(`
    CREATE TABLE IF NOT EXISTS users (
      id_users INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      lastname TEXT NOT NULL,
      firstname TEXT NOT NULL,
      email TEXT NOT NULL UNIQUE,
      password TEXT NOT NULL,
      avatar TEXT
    );
  `);

    db.run(`
    CREATE TABLE IF NOT EXISTS events (
      id_trainies INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      description TEXT,
      video TEXT
    );
  `);
    db.run('CREATE UNIQUE INDEX IF NOT EXISTS idx_events_name_unique ON events(name);');
    ensureUniqueUserLogins();
    db.run(
        'CREATE UNIQUE INDEX IF NOT EXISTS idx_users_login_unique ON users(name COLLATE NOCASE);'
    );

    db.run(`
    CREATE TABLE IF NOT EXISTS results (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      id_trainies INTEGER NOT NULL,
      id_users INTEGER NOT NULL,
      user_result_no INTEGER,
      user_video TEXT NOT NULL,
      FOREIGN KEY (id_trainies) REFERENCES events(id_trainies) ON DELETE CASCADE,
      FOREIGN KEY (id_users) REFERENCES users(id_users) ON DELETE CASCADE
    );
  `);

    db.run('CREATE INDEX IF NOT EXISTS idx_results_user ON results(id_users);');
    db.run('CREATE INDEX IF NOT EXISTS idx_results_training ON results(id_trainies);');

    // Migration for old DBs: add missing column user_result_no.
    const cols = all("PRAGMA table_info('results')", []);
    const hasUserNo = cols.some((c) => String(c.name) === 'user_result_no');
    if (!hasUserNo) {
        db.run('ALTER TABLE results ADD COLUMN user_result_no INTEGER;');
    }
    db.run('CREATE UNIQUE INDEX IF NOT EXISTS idx_results_user_no ON results(id_users, user_result_no);');
    const hasSetsDone = cols.some((c) => String(c.name) === 'sets_done');
    if (!hasSetsDone) {
        db.run('ALTER TABLE results ADD COLUMN sets_done INTEGER;');
    }
    const hasAnalysisJson = cols.some((c) => String(c.name) === 'analysis_json');
    if (!hasAnalysisJson) {
        db.run('ALTER TABLE results ADD COLUMN analysis_json TEXT;');
    }
    const hasTrimmedFrom = cols.some((c) => String(c.name) === 'trimmed_from_sec');
    if (!hasTrimmedFrom) {
        db.run('ALTER TABLE results ADD COLUMN trimmed_from_sec REAL;');
    }
    const hasTrimmedTo = cols.some((c) => String(c.name) === 'trimmed_to_sec');
    if (!hasTrimmedTo) {
        db.run('ALTER TABLE results ADD COLUMN trimmed_to_sec REAL;');
    }
    const userCols = all("PRAGMA table_info('users')", []);
    const hasAvatar = userCols.some((c) => String(c.name) === 'avatar');
    if (!hasAvatar) {
        db.run('ALTER TABLE users ADD COLUMN avatar TEXT;');
    }

    // Requested cleanup: delete old records with id 1 and 2.
    db.run('DELETE FROM results WHERE id IN (1, 2);');

    // Backfill unique per-user numbering for existing rows.
    const users = all('SELECT DISTINCT id_users FROM results ORDER BY id_users', []);
    for (const u of users) {
        const rows = all(
            'SELECT id FROM results WHERE id_users = ? ORDER BY id ASC',
            [Number(u.id_users)]
        );
        let n = 1;
        for (const row of rows) {
            db.run('UPDATE results SET user_result_no = ? WHERE id = ?', [n, Number(row.id)]);
            n++;
        }
    }

    seedEventsIfEmpty();
    persist();
}

module.exports = {
    initDatabase,
    run,
    get,
    all,
    lastInsertRowid,
    persist,
};

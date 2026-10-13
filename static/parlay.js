/* Parlay Lab v3 — rebuilt parlay builder.
   Loads priced legs from the existing props and game-bet boards, then does all
   parlay math client-side.

   v3 changes over v2:
   - Finder is a beam search over the whole slate (≈300 prop legs + game bets),
     not a truncated brute-force recursion, so results stream in live as
     controls change.
   - Simple finder: sort parlays by EV ("Best value") or model hit chance
     ("Most likely"), pick 2-5 legs. Same-game legs are allowed — the slip
     flags when the independence math blurs.
   - Kelly staking: the slip shows the Kelly fraction, with a one-click
     quarter-Kelly stake against a bankroll input.
   - Same-game legs get the Fréchet bounds the independence assumption
     really implies.
   - The slip persists for the day (localStorage + #s= URL hash), supports
     undo and reordering, and gets "best add" suggestions ranked by EV.
   - Team logos, per-leg game times and countdowns throughout. */
'use strict';

// ── Odds helpers (same conversions as the server) ─────────────────────
function americanToDecimal(a) {
    a = parseFloat(a);
    if (isNaN(a) || a === 0) return null;
    return a > 0 ? 1 + a / 100 : 1 + 100 / Math.abs(a);
}

function decimalToAmerican(d) {
    d = parseFloat(d);
    if (!d || d <= 1) return null;
    return d >= 2 ? Math.round((d - 1) * 100) : Math.round(-100 / (d - 1));
}

function fmtAmerican(a) {
    if (a === null || a === undefined || isNaN(a)) return '—';
    return a > 0 ? `+${Math.round(a)}` : `${Math.round(a)}`;
}

function pct(x, digits = 1) {
    return (x * 100).toFixed(digits) + '%';
}

function money(v) {
    const neg = v < 0;
    const s = Math.abs(v).toFixed(2);
    return (neg ? '-$' : '$') + s;
}

// EV of one bet at decimal odds d with (assumed) hit probability p.
function evOf(p, d) {
    return p * (d - 1) - (1 - p);
}

// Kelly fraction of bankroll for a bet at decimal odds d with hit prob p.
function kellyFrac(p, d) {
    if (!d || d <= 1) return 0;
    return Math.max(0, (p * d - 1) / (d - 1));
}

// ── Market families ───────────────────────────────────────────────────
const PROP_FAMILIES = {
    'Anytime Goal Scorer': 'ags',
    'Points': 'points',
    'Assists': 'assists',
    'Shots On Goal': 'sog',
    'Total Saves': 'saves',
};

const FAMILY_META = {
    ags:       { label: 'Anytime Goal',  accent: 'var(--accent-gold)',   dim: 'var(--accent-gold-dim)' },
    points:    { label: 'Points',        accent: 'var(--accent-purple)', dim: 'var(--accent-purple-dim)' },
    assists:   { label: 'Assists',       accent: 'var(--accent-blue)',   dim: 'var(--accent-blue-dim)' },
    sog:       { label: 'Shots on Goal', accent: 'var(--accent-green)',  dim: 'var(--accent-green-dim)' },
    saves:     { label: 'Goalie Saves',  accent: 'var(--accent-red)',    dim: 'var(--accent-red-dim)' },
    moneyline: { label: 'Moneyline',     accent: null,                   dim: null },
    total:     { label: 'Total',         accent: null,                   dim: null },
    puckline:  { label: 'Puck Line',     accent: null,                   dim: null },
    other:     { label: 'Other',         accent: null,                   dim: null },
};
const CHIP_ORDER = ['ags', 'points', 'assists', 'sog', 'saves', 'moneyline', 'total', 'puckline', 'other'];

function familyOfProp(market) {
    return PROP_FAMILIES[market] || 'other';
}

function familyOfGame(market) {
    const m = String(market || '');
    if (m.startsWith('Moneyline')) return 'moneyline';
    if (m.startsWith('Total')) return 'total';
    if (m.startsWith('Puck Line')) return 'puckline';
    return 'other';
}

// ── State ─────────────────────────────────────────────────────────────
// Leg: {key, source('props'|'games'), family, recommended, side, gameLabel,
//       gameKey, team, label, detail, american, decimal, modelProb, edge}
let _legs = [];
let _legByKey = new Map();
let _slip = [];              // leg keys, in insertion order
let _undo = [];              // previous slip states, most recent last
let _gameTimes = {};         // gameKey -> Date (game-bet feed start_time or props commence_time)
let _knownGames = new Set(); // gameKeys present on the served game-bet board
let _hiddenGames = new Set(); // props games the betting board already dropped as started
let _lastResults = [];       // finder results, for Show more
let _resultShown = 0;

const SETTINGS_KEY = 'pl3-settings';
const SLIP_KEY = 'pl3-slip';
const DEFAULTS = {
    markets: ['ags', 'points', 'assists', 'sog', 'saves', 'moneyline', 'total', 'puckline'],
    games: null,                       // null = all games; else a Set of gameKeys
    sides: 'rec',
    mode: 'ev',                        // 'ev' (best value) | 'chance' (most likely)
    size: 3,
    maxOdds: null,                     // null = any odds; else American cap (e.g. 200)
    stake: 10,
    bankroll: 500,
};
let _settings = { ...DEFAULTS, markets: new Set(DEFAULTS.markets) };

function saveSettings() {
    try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify({
            ..._settings,
            markets: [..._settings.markets],
            games: _settings.games ? [..._settings.games] : null,
        }));
    } catch (e) {}
}

function loadSettings() {
    try {
        let raw = localStorage.getItem(SETTINGS_KEY);
        if (!raw) raw = localStorage.getItem('pl2-settings'); // v2 migration: same shape minus the dead keys
        if (!raw) return;
        const s = JSON.parse(raw);
        if (s && typeof s === 'object') {
            const { markets, ...rest } = s;
            Object.assign(_settings, rest);
            delete _settings.hideStarted; // superseded: started games always hide now
            delete _settings.evOnly;      // superseded by the "Best value" sort
            delete _settings.target;      // superseded: finder now sorts by EV / hit chance
            delete _settings.tolerance;
            if (_settings.mode === 'target') _settings.mode = 'ev';
            if (Array.isArray(markets)) {
                _settings.markets = new Set(markets.filter(id => FAMILY_META[id]));
            }
            if (Array.isArray(s.games)) {
                _settings.games = new Set(s.games.filter(k => typeof k === 'string'));
            } else if (s.games === null) {
                _settings.games = null;
            }
        }
    } catch (e) {}
}

// ── Slip persistence ──────────────────────────────────────────────────
function todayStamp() {
    return new Date().toDateString();
}

function saveSlip() {
    try {
        localStorage.setItem(SLIP_KEY, JSON.stringify({ day: todayStamp(), keys: _slip }));
    } catch (e) {}
    // The hash is the shareable form of the same slip: refresh and it's back.
    try {
        const hash = _slip.length ? '#s=' + encodeURIComponent(_slip.join('|')) : '';
        history.replaceState(null, '', location.pathname + hash);
    } catch (e) {}
}

function slipKeysFromHash() {
    try {
        const m = /^#s=(.+)$/.exec(location.hash || '');
        return m ? decodeURIComponent(m[1]).split('|').filter(Boolean) : null;
    } catch (e) {
        return null;
    }
}

function slipKeysFromStorage() {
    try {
        const raw = localStorage.getItem(SLIP_KEY);
        if (!raw) return null;
        const s = JSON.parse(raw);
        return (s && s.day === todayStamp() && Array.isArray(s.keys)) ? s.keys.filter(Boolean) : null;
    } catch (e) {
        return null;
    }
}

// Restore after the leg tables exist; keys that haven't loaded yet (the next
// day's legs stream in later) are kept and resolve when they arrive.
function restoreSlip(preferred) {
    if (!preferred) return;
    _slip = [...new Set(preferred.filter(k => typeof k === 'string'))];
}

// ── Leg construction ──────────────────────────────────────────────────
// The pool spans two league days (today + tomorrow); the day is part of a
// leg's key so the same player-market appearing both days stays distinct.
function addPropLegs(payload, day) {
    const d = day || '';
    (payload.props || []).forEach(p => {
        const family = familyOfProp(p.market);
        const recSide = p.recommendation === 'Under' ? 'Under'
            : p.recommendation === 'Over' ? 'Over' : null;
        const gk = gameKey(p.away_abbr, p.home_abbr);
        if (p.commence_time) recordGameTime(gk, p.commence_time);
        const gameLabel = gameLabelFrom(abbrName(p.home_abbr), abbrName(p.away_abbr));
        const team = canonAbbr(p.player_team) || '';
        const base = {
            source: 'props',
            family: family,
            gameLabel: gameLabel,
            gameKey: gk,
            team: team,
            detail: p.player_team ? p.player_team.toUpperCase() : '',
        };

        const lineTxt = `${p.line > 0 ? '+' : ''}${p.line}`;
        // Over side.
        if (p.over_american != null || p.over_decimal != null) {
            const decimal = p.over_decimal || americanToDecimal(p.over_american);
            const implied = p.implied_over != null ? p.implied_over / 100 : 1 / decimal;
            _legs.push({
                ...base,
                key: `prop:${d}:${p.player}::${p.market}::${p.line}::Over`,
                recommended: recSide === 'Over',
                side: 'Over',
                label: `${p.player} — ${p.market} ${lineTxt} (Over)`,
                american: p.over_american != null ? p.over_american : decimalToAmerican(decimal),
                decimal: decimal,
                modelProb: p.prob_over / 100,
                edge: (p.prob_over / 100) - implied,
            });
        }
        // Under side (the feed keeps Under prices for most markets; AGS is Over-only).
        if (p.under_american != null || p.under_decimal != null) {
            const decimal = p.under_decimal || americanToDecimal(p.under_american);
            const probUnder = 1 - p.prob_over / 100;
            const implied = p.implied_under != null ? p.implied_under / 100 : 1 / decimal;
            _legs.push({
                ...base,
                key: `prop:${d}:${p.player}::${p.market}::${p.line}::Under`,
                recommended: recSide === 'Under',
                side: 'Under',
                label: `${p.player} — ${p.market} ${lineTxt} (Under)`,
                american: p.under_american != null ? p.under_american : decimalToAmerican(decimal),
                decimal: decimal,
                modelProb: probUnder,
                edge: probUnder - implied,
            });
        }
    });
}

function addGameLegs(payload, day) {
    const d = day || '';
    (payload.games || []).forEach(g => {
        const gk = gameKey(g.away, g.home);
        _knownGames.add(gk);
        if (g.start_time) recordGameTime(gk, g.start_time);
        (g.edges || []).forEach(e => {
            const american = e.odds;
            if (american == null) return;
            _legs.push({
                key: `game:${d}:${g.away}@${g.home}:${e.market}:${e.side || e.pick}`,
                source: 'games',
                family: familyOfGame(e.market),
                recommended: true,
                side: e.side || e.pick || '',
                gameLabel: gameLabelFrom(abbrName(g.home), abbrName(g.away)),
                gameKey: gk,
                team: canonAbbr(e.team) || '',
                label: `${e.pick} ${e.market || ''}`.trim(),
                detail: e.book || '',
                american: american,
                decimal: e.odds_decimal || americanToDecimal(american),
                modelProb: e.model_prob,   // already 0-1
                edge: e.edge || 0,
            });
        });
    });
}

function abbrName(abbr) {
    // "TOR MAPLE LEAFS" / "CBJ BLUE JACKETS" forms appear in the props feed;
    // keep the abbr (first token) only for display.
    return String(abbr || '').split(' ')[0].toUpperCase();
}

// SharpAPI props and the game-bet board abbreviate a few teams differently
// (LA/LAK, NJ/NJD, SJS/SJ) -- canonicalize so the two feeds agree on keys.
const ABBR_ALIAS = { NJ: 'NJD', LA: 'LAK', SJS: 'SJ', UTAH: 'UTA' };

function canonAbbr(abbr) {
    const a = abbrName(abbr);
    return ABBR_ALIAS[a] || a;
}

function gameLabelFrom(home, away) {
    return `${canonAbbr(away)} @ ${canonAbbr(home)}`;
}

function gameKey(away, home) {
    return `${canonAbbr(away)}@${canonAbbr(home)}`;
}

function recordGameTime(key, raw) {
    const t = new Date(raw);
    if (!isNaN(t)) _gameTimes[key] = t;
}

function logoSrc(team) {
    // Logo files are named after the canonical abbr (SJS.png, not SJ.png);
    // teams only ever differ in the game-board's short forms.
    const LOGO_ALIAS = { SJ: 'SJS' };
    const t = LOGO_ALIAS[team] || team;
    return t ? `/api/logos/${encodeURIComponent(t)}.png` : null;
}

// ── Times ─────────────────────────────────────────────────────────────
// Games off today's slate (tomorrow's, mainly) get their weekday in the label.
const BOOT_DAY = new Date().toDateString();

function fmtGameTime(d) {
    try {
        if (d.toDateString() !== BOOT_DAY) {
            return d.toLocaleTimeString([], { weekday: 'short', hour: 'numeric', minute: '2-digit' });
        }
        return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    } catch (e) {
        return '';
    }
}

function fmtCountdown(d) {
    const ms = d.getTime() - Date.now();
    if (isNaN(ms) || ms <= 0) return null;
    const mins = Math.floor(ms / 60000);
    if (mins < 60) return `in ${mins}m`;
    return `in ${Math.floor(mins / 60)}h ${String(mins % 60).padStart(2, '0')}m`;
}

function timeChip(gameKey2) {
    const d = _gameTimes[gameKey2];
    if (!d) return '';
    const count = fmtCountdown(d) || 'started';
    return `${fmtGameTime(d)} · ${count}`;
}

// ── Filtering ─────────────────────────────────────────────────────────
function gameStarted(gameKey2) {
    const t = _gameTimes[gameKey2];
    if (t) return t.getTime() <= Date.now();
    // A props game the game-bet board no longer lists (same slate date) was
    // dropped server-side as already started -- hide it too.
    return _hiddenGames.has(gameKey2);
}

function legAllowed(l) {
    if (!_settings.markets.has(l.family)) return false;
    if (gameStarted(l.gameKey)) return false; // started games drop automatically
    if (_settings.games && !_settings.games.has(l.gameKey)) return false;
    if (l.source === 'props' && _settings.sides !== 'both' && !l.recommended) return false;
    return true;
}

function filteredLegs() {
    const q = (document.getElementById('plSearch').value || '').trim().toLowerCase();
    const sort = document.getElementById('plSort').value;

    let out = _legs.filter(legAllowed);
    if (q) out = out.filter(l => l.label.toLowerCase().includes(q) || l.gameLabel.toLowerCase().includes(q));

    if (sort === 'prob') out.sort((a, b) => b.modelProb - a.modelProb);
    else if (sort === 'odds') out.sort((a, b) => b.decimal - a.decimal);
    else out.sort((a, b) => b.edge - a.edge);
    return out;
}

// ── Market chips ──────────────────────────────────────────────────────
function familyCounts() {
    const counts = {};
    _legs.forEach(l => {
        if (l.source === 'props' && _settings.sides !== 'both' && !l.recommended) return;
        counts[l.family] = (counts[l.family] || 0) + 1;
    });
    return counts;
}

function renderChips() {
    const host = document.getElementById('plChips');
    const counts = familyCounts();
    const present = CHIP_ORDER.filter(id => counts[id]);
    _settings.markets = new Set([..._settings.markets].filter(id => counts[id]));
    if (!present.length) { host.innerHTML = '<p class="empty-desc">No markets loaded yet.</p>'; return; }

    const allOn = present.every(id => _settings.markets.has(id));
    host.innerHTML =
        `<button type="button" class="pl-chip pl-chip-all" data-chip="__all" aria-pressed="${allOn}">
            <i class="fa-solid fa-layer-group"></i> All markets
        </button>` +
        present.map(id => {
            const meta = FAMILY_META[id];
            const style = meta.accent ? `style="--chip-accent:${meta.accent};--chip-accent-dim:${meta.dim}"` : '';
            return `<button type="button" class="pl-chip" ${style} data-chip="${id}" aria-pressed="${_settings.markets.has(id)}">
                ${escapeHtml(meta.label)} <span class="pl-chip-count">${counts[id]}</span>
            </button>`;
        }).join('');
}

// ── Game dropdown ─────────────────────────────────────────────────────
function gameList() {
    // Upcoming games with at least one visible leg under the current
    // market/side filters — started games are excluded, oldest start first.
    const counts = {};
    _legs.forEach(l => {
        if (gameStarted(l.gameKey)) return;
        if (!_settings.markets.has(l.family)) return;
        if (l.source === 'props' && _settings.sides !== 'both' && !l.recommended) return;
        counts[l.gameKey] = (counts[l.gameKey] || 0) + 1;
    });
    return Object.keys(counts)
        .map(k => ({ key: k, count: counts[k], time: _gameTimes[k] || null }))
        .sort((a, b) => (a.time ? a.time.getTime() : Infinity) - (b.time ? b.time.getTime() : Infinity)
            || a.key.localeCompare(b.key));
}

function gameChipLabel(key) {
    return key.replace('@', ' @ ');
}

function renderGamesDD() {
    const label = document.getElementById('plGamesLabel');
    const allBox = document.getElementById('plGamesAll');
    const items = document.getElementById('plGamesItems');
    const list = gameList();

    if (!list.length) {
        label.textContent = 'All games';
        allBox.checked = true;
        allBox.disabled = true;
        items.innerHTML = '<p class="empty-desc">No upcoming games left — every game has started.</p>';
        return;
    }

    allBox.disabled = false;
    allBox.checked = _settings.games === null;
    const sel = k => _settings.games === null || _settings.games.has(k);
    items.innerHTML = list.map(g =>
        `<label class="pl-dd-item"><input type="checkbox" data-game="${escapeAttr(g.key)}" ${sel(g.key) ? 'checked' : ''}>
            ${escapeHtml(gameChipLabel(g.key))}
            ${g.time ? `<span class="pl-chip-count">${fmtGameTime(g.time)}</span>` : ''}
        </label>`).join('');
    label.textContent = _settings.games === null
        ? `All games (${list.length})`
        : `${_settings.games.size} of ${list.length} games`;
}

// ── Leg pool rendering ────────────────────────────────────────────────
const MAX_POOL_ROWS = 80;

function renderPool() {
    const host = document.getElementById('plPool');
    const legs = filteredLegs();

    if (!_legs.length) return; // boot error already rendered
    if (!legs.length) {
        const noMarkets = !_settings.markets.size;
        const noGames = _settings.games && !_settings.games.size;
        const title = noMarkets ? 'No markets selected'
            : noGames ? 'No games selected' : 'No legs match';
        const desc = noMarkets ? 'Toggle at least one market chip above.'
            : noGames ? 'Select at least one game chip above.'
            : 'Loosen the search, check Both sides, or re-enable a market.';
        host.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-sliders empty-icon"></i>
                <p class="empty-title">${title}</p>
                <p class="empty-desc">${desc}</p>
            </div>`;
        return;
    }

    const shown = legs.slice(0, MAX_POOL_ROWS);
    host.innerHTML = shown.map(l => legRow(l)).join('') +
        (legs.length > shown.length
            ? `<p class="pl-pool-more">${legs.length - shown.length} more hidden — narrow the search</p>`
            : '');
}

function legLogoImg(l, cls) {
    const src = logoSrc(l.team);
    return `<span class="pl-logo-wrap">
        ${src ? `<img class="${cls} pl-logo" src="${escapeAttr(src)}" alt="" loading="lazy" onerror="this.remove()">` : ''}
        <span class="pl-logo-fallback" aria-hidden="true"><i class="fa-solid fa-shirt"></i></span>
    </span>`;
}

function legRow(l) {
    const inSlip = _slip.includes(l.key);
    const meta = FAMILY_META[l.family] || FAMILY_META.other;
    const style = meta.accent ? `style="--chip-accent:${meta.accent};--chip-accent-dim:${meta.dim}"` : '';
    const time = timeChip(l.gameKey);
    return `
        <div class="pl-leg${inSlip ? ' pl-leg-added' : ''}" data-key="${escapeAttr(l.key)}">
            ${legLogoImg(l, 'pl-leg-logo')}
            <div class="pl-leg-main">
                <span class="pl-leg-titleline">
                    <span class="pl-leg-badge" ${style}>${escapeHtml(meta.label)}</span>
                    <span class="pl-leg-label">${escapeHtml(l.label)}</span>
                </span>
                <span class="pl-leg-sub">${escapeHtml(l.gameLabel)}${l.detail ? ' · ' + escapeHtml(l.detail) : ''}${time ? ' · ' + escapeHtml(time) : ''}</span>
            </div>
            <div class="pl-leg-nums">
                <span class="pl-leg-prob">${pct(l.modelProb)}</span>
                <span class="pl-leg-odds">${fmtAmerican(l.american)}</span>
                <span class="pl-leg-edge ${l.edge > 0 ? 'pos' : 'neg'}">${l.edge >= 0 ? '+' : ''}${(l.edge * 100).toFixed(1)}%</span>
                <button class="props-sort-btn pl-add" type="button" data-add="${escapeAttr(l.key)}" aria-label="${inSlip ? 'Remove from' : 'Add to'} slip">
                    <i class="fa-solid ${inSlip ? 'fa-check' : 'fa-plus'}"></i>
                </button>
            </div>
        </div>`;
}

// ── Slip rendering + math ─────────────────────────────────────────────
function slipLegs() {
    return _slip.map(k => _legByKey.get(k)).filter(Boolean);
}

function pushUndo() {
    _undo.push([..._slip]);
    if (_undo.length > 20) _undo.shift();
}

function renderSlip() {
    const legs = slipLegs();
    const host = document.getElementById('slipLegs');

    host.innerHTML = legs.length
        ? legs.map((l, i) => `
            <div class="pl-slip-leg" data-key="${escapeAttr(l.key)}">
                ${legLogoImg(l, 'pl-slip-logo')}
                <span class="pl-slip-col">
                    <span class="pl-leg-label">${escapeHtml(l.label)}</span>
                    <span class="pl-leg-sub">${escapeHtml(l.gameLabel)} · ${escapeHtml(timeChip(l.gameKey) || '')} · ${fmtAmerican(l.american)}
                        <span class="pl-leg-edge ${l.edge > 0 ? 'pos' : 'neg'}">${l.edge >= 0 ? '+' : ''}${(l.edge * 100).toFixed(1)}%</span></span>
                </span>
                <span class="pl-slip-tools">
                    <button class="pl-reorder" type="button" data-move="-1" ${i === 0 ? 'disabled' : ''} aria-label="Move leg up"><i class="fa-solid fa-arrow-up"></i></button>
                    <button class="pl-reorder" type="button" data-move="1" ${i === legs.length - 1 ? 'disabled' : ''} aria-label="Move leg down"><i class="fa-solid fa-arrow-down"></i></button>
                    <button class="pl-remove" type="button" data-remove="${escapeAttr(l.key)}" aria-label="Remove leg"><i class="fa-solid fa-xmark"></i></button>
                </span>
            </div>`).join('')
        : '<p class="empty-desc">Add legs from the left — combined odds, hit chance, EV and Kelly all show up here.</p>';

    renderSlipTotals(legs);
    renderPool(); // reflect the added marks
    renderSuggestions();
    saveSlip();
}

function renderSlipTotals(legs) {
    const stake = Math.max(0, parseFloat(document.getElementById('plStake').value) || 0);
    const bankroll = Math.max(0, parseFloat(document.getElementById('plBankroll').value) || 0);
    const combinedDecimal = legs.reduce((acc, l) => acc * (l.decimal || 1), 1);
    const jointProb = legs.reduce((acc, l) => acc * l.modelProb, 1);
    const implied = combinedDecimal > 1 ? 1 / combinedDecimal : 1;
    const american = decimalToAmerican(combinedDecimal);
    const ev = legs.length ? evOf(jointProb, combinedDecimal) : 0;
    const payout = stake * combinedDecimal;

    document.getElementById('plCombinedOdds').textContent = legs.length
        ? fmtAmerican(american) : '—';
    document.getElementById('plJointProb').textContent = legs.length ? pct(jointProb) : '—';

    const fill = document.getElementById('plMeterFill');
    fill.style.width = legs.length ? `${Math.min(100, jointProb * 100).toFixed(1)}%` : '0%';
    fill.className = 'pl-meter-fill ' + (jointProb >= 0.2 ? '' : jointProb >= 0.05 ? 'mid' : 'low');

    document.getElementById('plImplied').textContent = legs.length ? pct(implied) : '—';

    const evEl = document.getElementById('plEV');
    evEl.textContent = legs.length ? `${(ev * 100).toFixed(1)}%` : '—';
    evEl.className = ev > 0 ? 'pl-ev pos' : 'pl-ev neg';

    document.getElementById('plPayout').textContent = legs.length
        ? `${money(payout)} / ${money(stake * (combinedDecimal - 1))}` : '—';

    // Kelly staking: the fraction of bankroll this parlay justifies at the
    // model's hit chance, shown full and quarter. Negative edge means no bet.
    const kEl = document.getElementById('plKelly');
    const useBtn = document.getElementById('plUseKelly');
    if (legs.length) {
        const f = kellyFrac(jointProb, combinedDecimal);
        if (f <= 0) {
            kEl.textContent = 'no edge — skip';
            kEl.className = 'pl-kelly-none';
            useBtn.hidden = true;
        } else {
            kEl.innerHTML = `${pct(f)} of bankroll · ${money(bankroll * f)} <small>1/4 Kelly = ${money(bankroll * f / 4)}</small>`;
            kEl.className = '';
            useBtn.hidden = false;
        }
    } else {
        kEl.textContent = '—';
        kEl.className = '';
        useBtn.hidden = true;
    }

    // Correlation honesty: the product above assumes independence. Any two
    // legs sharing a game break that — report the Frechet bounds instead of
    // pretending the single number is exact.
    const keys = legs.map(l => l.gameKey);
    const clash = new Set(keys).size !== keys.length;
    const note = document.getElementById('plCorrelationNote');
    note.hidden = !clash;
    if (clash && legs.length) {
        const lower = Math.max(0, legs.reduce((a, l) => a + l.modelProb, 0) - (legs.length - 1));
        const upper = Math.min(...legs.map(l => l.modelProb));
        document.getElementById('plCorrRange').textContent = `${pct(lower, 1)} – ${pct(upper, 1)}`;
    }
}

// ── Suggestions (best add / best swap, ranked by EV) ──────────────────
function renderSuggestions() {
    const host = document.getElementById('plSuggest');
    if (!host) return;
    const slip = slipLegs();
    if (!slip.length || !_legs.length) { host.innerHTML = ''; host.hidden = true; return; }

    const baseD = slip.reduce((a, l) => a * l.decimal, 1);
    const baseP = slip.reduce((a, l) => a * l.modelProb, 1);
    const baseEV = evOf(baseP, baseD);
    const slipGames = new Set(slip.map(l => l.gameKey));
    const slipKeys = new Set(_slip);

    const independent = _legs.filter(legAllowed)
        .filter(l => !slipKeys.has(l.key) && !slipGames.has(l.gameKey));

    // Adds: EV of the slip with this leg appended.
    const adds = independent
        .map(l => ({ leg: l, delta: evOf(baseP * l.modelProb, baseD * l.decimal) - baseEV }))
        .sort((a, b) => b.delta - a.delta)
        .slice(0, 3)
        .filter(s => s.delta > 0);

    // Swaps: for each slip leg, the best leg that could replace it.
    const swaps = [];
    slip.forEach((sl, i) => {
        const others = slip.filter((_, j) => j !== i);
        const openGames = new Set(others.map(l => l.gameKey));
        const candD = others.reduce((a, l) => a * l.decimal, 1);
        const candP = others.reduce((a, l) => a * l.modelProb, 1);
        const cand = independent
            .filter(l => l.gameKey !== sl.gameKey)
            .filter(l => !openGames.has(l.gameKey))
            .map(l => ({ leg: l, delta: evOf(candP * l.modelProb, candD * l.decimal) - baseEV }))
            .sort((a, b) => b.delta - a.delta)[0];
        if (cand) swaps.push({ out: sl, ...cand });
    });
    swaps.sort((a, b) => b.delta - a.delta);
    const topSwaps = swaps.slice(0, 2).filter(s => s.delta > 0);

    if (!adds.length && !topSwaps.length) { host.innerHTML = ''; host.hidden = true; return; }

    host.hidden = false;
    host.innerHTML =
        (adds.length
            ? `<p class="pl-suggest-title"><i class="fa-solid fa-lightbulb"></i> Best adds</p>` +
              adds.map(s => suggestRow('sugg-add', legMini(s.leg), s.delta, s.leg.key)).join('')
            : '') +
        (topSwaps.length
            ? `<p class="pl-suggest-title"><i class="fa-solid fa-right-left"></i> Best swaps</p>` +
              topSwaps.map(s => suggestRow('sugg-swap', `out: ${escapeHtml(shortLeg(s.out))} — in: ${legMini(s.leg)}`, s.delta, `${s.out.key}>>${s.leg.key}`)).join('')
            : '');
}

function shortLeg(l) {
    return l.label.length > 34 ? l.label.slice(0, 33) + '…' : l.label;
}

function legMini(l) {
    return `${escapeHtml(l.label)} <span class="pl-suggest-odds">${fmtAmerican(l.american)}</span>`;
}

function suggestRow(kind, inner, delta, arg) {
    return `
        <div class="pl-suggest-row">
            <span class="pl-suggest-col">${inner}</span>
            <span class="pl-suggest-nums">
                <span class="pl-leg-edge ${delta > 0 ? 'pos' : 'neg'}">${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)}% EV</span>
                <button class="props-sort-btn pl-add" type="button" data-${kind}="${escapeAttr(arg)}" aria-label="Apply suggestion">
                    <i class="fa-solid fa-plus"></i>
                </button>
            </span>
        </div>`;
}

// ── Finder controls ───────────────────────────────────────────────────
function syncSegButtons(hostId, attr, value) {
    document.querySelectorAll(`#${hostId} button`).forEach(b =>
        b.setAttribute('aria-pressed', String(b.dataset[attr] === String(value))));
}

// ── Auto-builder (beam search over legs) ──────────────────────────────
// Legs are ranked by the current sort, then a beam walk keeps the best
// partial combos per depth and extends them with deeper-ranked legs. That
// covers the whole slate at trivial cost, so results recompute live on every
// control change. Legs from the same game may share a parlay — real books
// sell those; the slip flags it instead of pretending the math is pure.
const BEAM_KEEP = 400;    // states kept per depth
const BEAM_LEG_CAP = 260; // ranked legs that feed the beam
const MAX_SHOW = 24;

function legModeScore(l, mode) {
    return mode === 'chance' ? l.modelProb : evOf(l.modelProb, l.decimal);
}

function buildPool() {
    const pool = _legs.filter(legAllowed);
    const capDec = _settings.maxOdds ? (americanToDecimal(_settings.maxOdds) || null) : null;
    pool.sort((a, b) => legModeScore(b, _settings.mode) - legModeScore(a, _settings.mode));
    if (capDec) {
        // Under the cap, short-odds legs rank last by EV — put every leg that
        // can still belong to a sub-cap parlay ahead of the dead longshots.
        const capList = [];
        const rest = [];
        for (const l of pool) ((l.decimal || 1) <= capDec ? capList : rest).push(l);
        return capList.concat(rest).slice(0, BEAM_LEG_CAP);
    }
    return pool.slice(0, BEAM_LEG_CAP);
}

function beamParlays(pool, size, mode, capDecimal = null) {
    // Combos are built with strictly increasing leg indices, so every combo
    // is visited exactly once. Odds only multiply upward, so once a combo
    // crosses the odds cap it can never come back down — drop it at each
    // depth, or the score-ranked beam would fill every slot with longshots
    // and the cap would filter the survivors down to nothing.
    const score = s => (mode === 'chance' ? s.prob : evOf(s.prob, s.decimal));
    const underCap = s => !capDecimal || s.decimal <= capDecimal;
    let states = pool.map((l, i) => ({
        decimal: l.decimal,
        prob: l.modelProb,
        idxs: [i],
        legs: [l],
    }));
    states = states.filter(underCap);
    states.sort((a, b) => score(b) - score(a));
    states = states.slice(0, BEAM_KEEP);

    for (let depth = 2; depth <= size; depth++) {
        const next = [];
        for (const s of states) {
            for (let i = s.idxs[s.idxs.length - 1] + 1; i < pool.length; i++) {
                const l = pool[i];
                const decimal = s.decimal * l.decimal;
                if (capDecimal && decimal > capDecimal) continue;
                next.push({
                    decimal,
                    prob: s.prob * l.modelProb,
                    idxs: s.idxs.concat(i),
                    legs: s.legs.concat(l),
                });
                if (next.length > 60000) break; // hard stop, safety only
            }
        }
        next.sort((a, b) => score(b) - score(a));
        states = next.slice(0, BEAM_KEEP);
    }
    return states;
}

// Diversity picker: repeatedly take the best-ranked combo whose legs are all
// under the per-leg usage cap, scoring candidates by how often their legs have
// already been picked. Top-ranked fresh combos win; reused legs only reappear
// when nothing fresher qualifies. Rather than return almost nothing when the
// top combos all share legs (a thin slate, or one mode of legs dominating),
// the cap relaxes stepwise until the list fills -- _lastCapUsed reports how
// permissive the final round had to be.
let _lastCapUsed = 1;

function pickDiverse(sorted, count, capStart = 1, capMax = 99) {
    const scan = sorted.slice(0, 4000); // worst-ranked tail isn't worth the scan
    const usage = new Map();            // leg key -> times already picked
    const out = [];
    let cap = capStart;
    while (out.length < count) {
        let best = -1, bestUse = Infinity;
        for (let i = 0; i < scan.length; i++) {
            if (out.includes(scan[i])) continue;
            let maxUse = 0;
            for (const l of scan[i].legs) {
                const u = usage.get(l.key) || 0;
                if (u >= cap) { maxUse = cap + 1; break; }
                maxUse = Math.max(maxUse, u);
            }
            if (maxUse <= cap && maxUse < bestUse) {
                bestUse = maxUse;
                best = i;
                if (maxUse === 0) break; // fully fresh combo, can't beat it
            }
        }
        if (best === -1) {
            cap++;
            if (cap > capMax) break; // combos genuinely exhausted
            continue;
        }
        scan[best].legs.forEach(l => usage.set(l.key, (usage.get(l.key) || 0) + 1));
        out.push(scan[best]);
    }
    _lastCapUsed = cap <= capMax ? Math.max(cap, capStart) : capStart;
    return out;
}

function buildParlays() {
    const host = document.getElementById('plAutoplay');
    _lastResults = [];
    _resultShown = 0;
    const pool = buildPool();
    const mode = _settings.mode;

    if (pool.length < 2) {
        host.innerHTML = `<p class="empty-desc">Only ${pool.length} usable leg — pick more markets or games at the top to build parlays.</p>`;
        return;
    }

    // A parlay needs 2+ legs; the leg count can't exceed the usable-pool size.
    const size = Math.min(_settings.size, pool.length);
    let notePrefix = '';
    if (size < _settings.size) {
        notePrefix = `Only ${size} usable legs — showing ${size}-leg parlays. `;
    }

    // Dedupe: the beam can reach the same leg set through different orders.
    const oddsCap = _settings.maxOdds ? (americanToDecimal(_settings.maxOdds) || null) : null;
    const seen = new Set();
    const scored = beamParlays(pool, size, mode, oddsCap)
        .filter(s => {
            const id = s.legs.map(l => l.key).sort().join('|');
            if (seen.has(id)) return false;
            seen.add(id);
            return true;
        })
        .map(s => ({
            legs: s.legs,
            combinedDecimal: s.decimal,
            jointProb: s.prob,
            ev: evOf(s.prob, s.decimal),
        }));

    // Rare reuse first: each prop may appear in at most one of the shown
    // parlays so the list reads as a menu of distinct bets — but a thin slate
    // relaxes the cap stepwise rather than coming back nearly empty.
    // Same-game combos are valued at the pessimistic end of their Fréchet
    // range: their naive EV assumes independence and comes out wildly high.
    const capList = oddsCap ? scored.filter(p => p.combinedDecimal <= oddsCap) : scored;
    if (!capList.length) {
        host.innerHTML = `<p class="empty-desc">${oddsCap
            ? `No ${size}-leg parlays at or under +${_settings.maxOdds} — raise the odds cap or drop a leg.`
            : `No qualifying combo — pick more markets or games at the top.`}</p>`;
        return;
    }
    let results, note;
    const floorEV = p => {
        const sameGame = new Set(p.legs.map(l => l.gameKey)).size !== p.legs.length;
        if (!sameGame) return p.ev;
        const lower = Math.max(0, p.legs.reduce((a, l) => a + l.modelProb, 0) - (p.legs.length - 1));
        return evOf(lower, p.combinedDecimal);
    };
    if (mode === 'ev') {
        results = pickDiverse(capList.sort((a, b) => floorEV(b) - floorEV(a) || b.jointProb - a.jointProb), MAX_SHOW);
        note = `${notePrefix}${results.length} best-value ${size}-leg parlays${oddsCap ? ` at or under +${_settings.maxOdds}` : ''}, biggest model edge first — EV leans on longshots, so watch the hit chance.`;
    } else {
        results = pickDiverse(capList.sort((a, b) => b.jointProb - a.jointProb || b.ev - a.ev), MAX_SHOW);
        note = `${notePrefix}${results.length} most likely ${size}-leg parlays${oddsCap ? ` at or under +${_settings.maxOdds}` : ''}.`;
    }
    if (_lastCapUsed > 1 && results.length > 2) note += ' A few props repeat across picks — the slate is short on legs.';

    setResultsNote(note);
    _lastResults = results;
    _resultShown = Math.min(MAX_SHOW, results.length);
    renderResults();
}

function renderResults() {
    const host = document.getElementById('plAutoplay');
    if (!_lastResults.length) {
        host.innerHTML = `<p class="empty-desc">No qualifying combo — pick more markets or games at the top.</p>`;
        return;
    }
    const shown = _lastResults.slice(0, _resultShown);
    host.innerHTML =
        (resultsNoteCache ? `<p class="pl-autoplay-note">${escapeHtml(resultsNoteCache)}</p>` : '') +
        shown.map(p => parlayRow(p)).join('') +
        (_lastResults.length > shown.length
            ? `<button type="button" class="pl-chip pl-more-btn" data-more="1"><i class="fa-solid fa-chevron-down"></i> Show ${Math.min(12, _lastResults.length - shown.length)} more</button>`
            : '');
}

let resultsNoteCache = '';

function setResultsNote(note) {
    resultsNoteCache = note;
}

function parlayRow(p) {
    const stake = Math.max(0, parseFloat(document.getElementById('plStake').value) || 0);
    const bankroll = Math.max(0, parseFloat(document.getElementById('plBankroll').value) || 0);
    const american = decimalToAmerican(p.combinedDecimal);
    // Plain-speak for the hit chance: longshots read better as "1 in N".
    const hit = p.jointProb > 0 && p.jointProb < 0.2
        ? `hits about 1 in ${Math.max(2, Math.round(1 / p.jointProb))}`
        : `hits ${pct(p.jointProb)}`;
    const legsHtml = p.legs.map((l, i) => {
        const src = logoSrc(l.team);
        return `<span class="pl-combo-leg">
            <span class="pl-combo-num">${i + 1}.</span>
            <span class="pl-logo-wrap">
                ${l.team ? `<img class="pl-combo-logo pl-logo" src="${escapeAttr(src)}" alt="" loading="lazy" onerror="this.remove()">` : ''}
                <span class="pl-logo-fallback" aria-hidden="true"><i class="fa-solid fa-shirt"></i></span>
            </span>
            <span class="pl-combo-leg-col">
                <span class="pl-combo-leg-row">
                    <span class="pl-leg-label">${escapeHtml(l.label)}</span>
                    <span class="pl-leg-odds">${fmtAmerican(l.american)}</span>
                </span>
                <span class="pl-leg-sub">${escapeHtml(l.gameLabel)} · ${escapeHtml(timeChip(l.gameKey) || '')}</span>
            </span>
        </span>`;
    }).join('');

    // Same-game legs: the products above multiply independent probabilities,
    // which same-game legs violate. Show the honest range instead of EV/Kelly
    // numbers that assume independence.
    const allFresh = new Set(p.legs.map(l => l.gameKey)).size === p.legs.length;
    let foot;
    if (!allFresh) {
        const lower = pct(Math.max(0, p.legs.reduce((a, l) => a + l.modelProb, 0) - (p.legs.length - 1)));
        const upper = pct(Math.min(...p.legs.map(l => l.modelProb)));
        foot = `<span class="pl-dist" title="Some legs share a game — hits can be more or less likely together, so the real hit chance falls somewhere in this range rather than the single number above">
            same-game · real hit chance ${lower}–${upper}</span>`;
    } else {
        const kelly = kellyFrac(p.jointProb, p.combinedDecimal);
        const kellyPill = kelly > 0
            ? `<span class="pl-dist pl-kelly-pill" title="A quarter-Kelly bet on this parlay, from your bankroll">1/4 Kelly: ${money(bankroll * kelly / 4)}</span>`
            : `<span class="pl-dist" title="The model gives this parlay no positive edge">no edge</span>`;
        foot = `<span class="pl-leg-edge ${p.ev > 0 ? 'pos' : 'neg'}" title="Expected value per $1 at the model's hit chance">EV ${(p.ev * 100).toFixed(1)}%</span>${kellyPill}`;
    }

    return `
        <div class="pl-combo">
            <div class="pl-combo-head">
                <span class="pl-combo-odds-big">${fmtAmerican(american)}</span>
                <span class="pl-combo-hit">${hit}</span>
                <span class="pl-combo-win">${stake > 0 ? money(stake) + ' → ' + money(stake * p.combinedDecimal) : ''}</span>
                <span class="pl-combo-actions">
                    <button class="props-sort-btn pl-add" type="button" data-load="${escapeAttr(p.legs.map(l => l.key).join('|'))}" title="Replace the slip with this parlay" aria-label="Load this parlay into the slip">
                        <i class="fa-solid fa-download"></i>
                    </button>
                    <button class="props-sort-btn pl-add" type="button" data-append="${escapeAttr(p.legs.map(l => l.key).join('|'))}" title="Append these legs to the slip" aria-label="Append this parlay into the slip">
                        <i class="fa-solid fa-plus"></i>
                    </button>
                </span>
            </div>
            <div class="pl-combo-body">${legsHtml}</div>
            <div class="pl-combo-foot">${foot}</div>
        </div>`;
}

let _buildTimer = null;
function scheduleBuild(delay = 200) {
    clearTimeout(_buildTimer);
    _buildTimer = setTimeout(buildParlays, delay);
}

// ── Events ────────────────────────────────────────────────────────────
document.addEventListener('click', e => {
    const chip = e.target.closest('[data-chip]');
    if (chip) {
        const id = chip.getAttribute('data-chip');
        const counts = familyCounts();
        const present = CHIP_ORDER.filter(fid => counts[fid]);
        if (id === '__all') {
            _settings.markets = new Set(present);
        } else if (chip.getAttribute('aria-pressed') === 'true') {
            _settings.markets.delete(id);
        } else {
            _settings.markets.add(id);
        }
        saveSettings();
        scheduleRefresh();
        scheduleBuild();
        return;
    }
    const add = e.target.closest('[data-add]');
    if (add) {
        const key = add.getAttribute('data-add');
        if (_slip.includes(key)) { pushUndo(); _slip = _slip.filter(k => k !== key); }
        else { pushUndo(); _slip.push(key); }
        renderSlip();
        return;
    }
    const remove = e.target.closest('[data-remove]');
    if (remove) {
        pushUndo();
        _slip = _slip.filter(k => k !== remove.getAttribute('data-remove'));
        renderSlip();
        return;
    }
    const move = e.target.closest('[data-move]');
    if (move) {
        const i = _slip.indexOf(move.closest('.pl-slip-leg').dataset.key);
        const j = i + parseInt(move.dataset.move, 10);
        if (i !== -1 && j >= 0 && j < _slip.length) {
            pushUndo();
            [_slip[i], _slip[j]] = [_slip[j], _slip[i]];
            renderSlip();
        }
        return;
    }
    const load = e.target.closest('[data-load]');
    if (load) {
        pushUndo();
        _slip = load.getAttribute('data-load').split('|');
        renderSlip();
        document.querySelector('.pl-hero').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        return;
    }
    const append = e.target.closest('[data-append]');
    if (append) {
        pushUndo();
        for (const k of append.getAttribute('data-append').split('|')) {
            if (!_slip.includes(k)) _slip.push(k);
        }
        renderSlip();
        return;
    }
    const suggAdd = e.target.closest('[data-sugg-add]');
    if (suggAdd) {
        pushUndo();
        const k = suggAdd.getAttribute('data-sugg-add');
        if (!_slip.includes(k)) _slip.push(k);
        renderSlip();
        return;
    }
    const suggSwap = e.target.closest('[data-sugg-swap]');
    if (suggSwap) {
        const [outKey, inKey] = suggSwap.getAttribute('data-sugg-swap').split('>>');
        const i = _slip.indexOf(outKey);
        if (i !== -1 && inKey && !_slip.includes(inKey)) {
            pushUndo();
            _slip.splice(i, 1, inKey);
            renderSlip();
        }
        return;
    }
    const modeBtn = e.target.closest('#plMode button');
    if (modeBtn) {
        _settings.mode = modeBtn.dataset.mode;
        syncSegButtons('plMode', 'mode', _settings.mode);
        saveSettings();
        buildParlays();
        return;
    }
    const sizeBtn = e.target.closest('#plSize button');
    if (sizeBtn) {
        _settings.size = parseInt(sizeBtn.dataset.size, 10) || 3;
        syncSegButtons('plSize', 'size', _settings.size);
        saveSettings();
        buildParlays();
        return;
    }
    const oddsBtn = e.target.closest('#plOddsCap button');
    if (oddsBtn) {
        const v = oddsBtn.dataset.odds;
        _settings.maxOdds = v === 'any' ? null : parseInt(v, 10);
        syncSegButtons('plOddsCap', 'odds', v);
        saveSettings();
        buildParlays();
        return;
    }
    if (e.target.closest('[data-more]')) {
        _resultShown = Math.min(_lastResults.length, _resultShown + 12);
        renderResults();
        return;
    }
    const undo = e.target.closest('#plUndo');
    if (undo) {
        if (_undo.length) {
            _slip = _undo.pop();
            renderSlip();
        }
        return;
    }
    const useKelly = e.target.closest('#plUseKelly');
    if (useKelly) {
        const legs = slipLegs();
        const d = legs.reduce((a, l) => a * l.decimal, 1);
        const p = legs.reduce((a, l) => a * l.modelProb, 1);
        const f = kellyFrac(p, d);
        const bankroll = Math.max(0, parseFloat(document.getElementById('plBankroll').value) || 0);
        if (f > 0) {
            const stake = document.getElementById('plStake');
            stake.value = (bankroll * f / 4).toFixed(2);
            _settings.stake = parseFloat(stake.value);
            saveSettings();
            renderSlipTotals(slipLegs());
        }
        return;
    }
});

function scheduleRefresh() {
    renderChips();
    renderGamesDD();
    renderSlip(); // re-renders the pool inside
}

document.getElementById('plSides').addEventListener('click', () => {
    const btn = document.getElementById('plSides');
    _settings.sides = _settings.sides === 'both' ? 'rec' : 'both';
    btn.setAttribute('aria-pressed', String(_settings.sides === 'both'));
    saveSettings();
    scheduleRefresh();
    scheduleBuild();
});

// Game dropdown: open/close, checkbox changes, close on outside click.
document.getElementById('plGamesBtn').addEventListener('click', () => {
    const panel = document.getElementById('plGamesPanel');
    panel.hidden = !panel.hidden;
    document.getElementById('plGamesBtn').setAttribute('aria-expanded', String(!panel.hidden));
});

document.addEventListener('click', e => {
    if (!e.target.closest('#plGamesDD')) {
        document.getElementById('plGamesPanel').hidden = true;
        document.getElementById('plGamesBtn').setAttribute('aria-expanded', 'false');
    }
});

document.addEventListener('change', e => {
    if (e.target.id === 'plGamesAll') {
        // Unchecking "All" selects every listed game; you then narrow from there.
        _settings.games = e.target.checked ? null : new Set(gameList().map(g => g.key));
        saveSettings();
        scheduleRefresh();
        scheduleBuild();
        return;
    }
    const gameBox = e.target.closest('[data-game]');
    if (gameBox) {
        const k = gameBox.getAttribute('data-game');
        const keys = gameList().map(g => g.key);
        if (!_settings.games) _settings.games = new Set(keys);
        if (gameBox.checked) _settings.games.add(k);
        else _settings.games.delete(k);
        if (_settings.games.size === keys.length) _settings.games = null; // all ticked == all
        saveSettings();
        scheduleRefresh();
        scheduleBuild();
    }
});

// Started games leave the pool automatically; recheck about once a minute.
setInterval(() => {
    if (_legs.length) {
        renderGamesDD();
        renderPool();
        renderSlipTotals(slipLegs());
        renderSuggestions();
    }
}, 60000);

document.getElementById('plSort').addEventListener('change', renderPool);
document.getElementById('plSearch').addEventListener('input', renderPool);

function stakeOrBankrollChanged(kind) {
    const el = document.getElementById(kind === 'stake' ? 'plStake' : 'plBankroll');
    const v = Math.max(0, parseFloat(el.value) || 0);
    if (kind === 'stake') _settings.stake = v; else _settings.bankroll = v;
    saveSettings();
    renderSlipTotals(slipLegs());
    if (kind === 'stake' && _lastResults.length) {
        renderResults(); // result cards show the stake → payout conversion
    }
}

document.getElementById('plStake').addEventListener('input', () => stakeOrBankrollChanged('stake'));
document.getElementById('plBankroll').addEventListener('input', () => stakeOrBankrollChanged('bankroll'));
document.getElementById('plBuild').addEventListener('click', buildParlays);
document.getElementById('plClear').addEventListener('click', () => { pushUndo(); _slip = []; renderSlip(); });

document.getElementById('plCopy').addEventListener('click', () => {
    const legs = slipLegs();
    if (!legs.length) return;
    const stake = Math.max(0, parseFloat(document.getElementById('plStake').value) || 0);
    const combinedDecimal = legs.reduce((acc, l) => acc * (l.decimal || 1), 1);
    const jointProb = legs.reduce((acc, l) => acc * l.modelProb, 1);
    const ev = evOf(jointProb, combinedDecimal);
    const text = [
        'Parlay Lab slip',
        ...legs.map(l => `${l.label} @ ${l.gameLabel} (${escapeHtml(timeChip(l.gameKey) || '')}) (${fmtAmerican(l.american)})`),
        `Combined: ${fmtAmerican(decimalToAmerican(combinedDecimal))} (${combinedDecimal.toFixed(2)})`,
        `Hit chance (indep. legs): ${pct(jointProb)} · EV ${(ev * 100).toFixed(1)}%`,
        stake ? `Stake: $${stake.toFixed(2)} → $${(stake * combinedDecimal).toFixed(2)}` : '',
    ].filter(Boolean).join('\n');
    const btn = document.getElementById('plCopy');
    try {
        navigator.clipboard.writeText(text).then(() => {
            btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied';
            setTimeout(() => { btn.innerHTML = '<i class="fa-solid fa-copy"></i> Copy slip'; }, 1500);
        }).catch(() => {});
    } catch (err) {}
});

// Same theme toggle behavior as index.html.
document.getElementById('themeToggle').addEventListener('click', () => {
    const root = document.documentElement;
    const next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('hdf-theme', next); } catch (e) {}
});

// ── Boot ──────────────────────────────────────────────────────────────
function restoreControls() {
    syncSegButtons('plMode', 'mode', _settings.mode);
    syncSegButtons('plSize', 'size', _settings.size);
    syncSegButtons('plOddsCap', 'odds', String(_settings.maxOdds === null || _settings.maxOdds === undefined ? 'any' : _settings.maxOdds));
    document.getElementById('plSides').setAttribute('aria-pressed', String(_settings.sides === 'both'));
    document.getElementById('plStake').value = _settings.stake;
    document.getElementById('plBankroll').value = _settings.bankroll;
}

async function boot() {
    loadSettings();
    restoreControls();
    document.getElementById('plLoadState').hidden = false;

    const [propsRes, gamesRes] = await Promise.allSettled([
        fetch('/api/player-props').then(r => r.json()),
        fetch('/api/betting-edge').then(r => r.json()),
    ]);

    if (propsRes.status === 'fulfilled' && propsRes.value && propsRes.value.props) {
        addPropLegs(propsRes.value);
    }
    if (gamesRes.status === 'fulfilled' && gamesRes.value && gamesRes.value.games) {
        addGameLegs(gamesRes.value);
    }
    _legByKey = new Map(_legs.map(l => [l.key, l]));

    // Props games absent from the game-bet board are games the server already
    // dropped as started (the betting feed strips them). Only trust that signal
    // when both boards agree on the slate date and the betting feed is non-empty.
    if (
        propsRes.status === 'fulfilled' && gamesRes.status === 'fulfilled' &&
        propsRes.value && gamesRes.value &&
        propsRes.value.date && propsRes.value.date === gamesRes.value.date &&
        _knownGames.size
    ) {
        _legs.forEach(l => {
            if (l.source === 'props' && !_gameTimes[l.gameKey] && !_knownGames.has(l.gameKey)) {
                _hiddenGames.add(l.gameKey);
            }
        });
    }

    if (!_legs.length) {
        document.getElementById('plLoadState').innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-inbox empty-icon"></i>
                <p class="empty-title">No legs available</p>
                <p class="empty-desc">The props and game-bet boards are empty — run the daily update first.</p>
            </div>`;
    } else {
        document.getElementById('plLoadState').hidden = true;
    }
    // Yesterday's persisted game selections are stale on a new slate — if none
    // of them exist today, start from "all games".
    if (_settings.games) {
        const today = new Set(_legs.map(l => l.gameKey));
        _settings.games = new Set([..._settings.games].filter(k => today.has(k)));
        if (!_settings.games.size) _settings.games = null;
    }
    // Slip restore: the #s= hash wins (it's the shareable form), else today's
    // localStorage copy.
    restoreSlip(slipKeysFromHash() || slipKeysFromStorage());
    renderGamesDD();
    renderChips();
    renderPool();
    buildParlays();
    renderSlip();

    // Today's slate is live — now pull in tomorrow's games in the background
    // (the props endpoint computes tomorrow's board live on first request,
    // which can take a minute or two, so the page must not wait on it).
    const day1 = propsRes.status === 'fulfilled' && propsRes.value && propsRes.value.date
        ? propsRes.value.date : null;
    if (day1) loadNextDay(day1);
}

async function loadNextDay(dayStr, retried) {
    const d = new Date(dayStr + 'T12:00:00');
    if (isNaN(d)) return;
    d.setDate(d.getDate() + 1);
    const next = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;

    const dayNote = document.getElementById('plDayNote');
    if (dayNote) {
        dayNote.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> ${d.toLocaleDateString([], { weekday: 'long' })}'s games are loading…`;
        dayNote.hidden = false;
    }

    const [propsRes, gamesRes] = await Promise.allSettled([
        fetch(`/api/player-props/${next}`).then(r => r.json()),
        fetch(`/api/betting-edge?date=${next}`).then(r => r.json()),
    ]);

    const before = _legs.length;
    if (propsRes.status === 'fulfilled' && propsRes.value && propsRes.value.props) {
        addPropLegs(propsRes.value, next);
    }
    if (gamesRes.status === 'fulfilled' && gamesRes.value && gamesRes.value.games) {
        addGameLegs(gamesRes.value, next);
    }

    if (dayNote) dayNote.hidden = true;
    if (_legs.length === before) {
        // Books often post the next day's lines late (SharpAPI's snapshot can
        // lag the slate) — say so instead of silently merging nothing, and
        // try once more in case prices land while the page is open.
        if (!retried) {
            if (dayNote) {
                dayNote.innerHTML = 'No prices posted for the next day yet — tonight\'s odds refresh fills these games in. One retry is scheduled.';
                dayNote.hidden = false;
            }
            setTimeout(() => loadNextDay(dayStr, true), 25 * 60 * 1000);
        }
        return;
    }

    _legByKey = new Map(_legs.map(l => [l.key, l]));
    renderGamesDD();
    renderChips();
    buildParlays();
    renderSlip();
}

// Minimal escaping so book/feed strings can't inject markup.
function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[c]);
}

function escapeAttr(s) {
    return escapeHtml(s);
}

boot();
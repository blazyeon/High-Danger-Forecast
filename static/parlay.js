/* Parlay Lab v2 — unlisted parlay builder.
   Loads priced legs from the existing props and game-bet boards, then does all
   parlay math client-side: combined decimal odds, joint hit probability under
   independence, and EV vs the book's combined implied probability.

   v2: market chips drive both the pool and the finder; both prop sides can be
   exposed as legs; the finder can aim at a user-set target odds (within a
   tolerance band) instead of just maximizing hit chance; settings persist in
   localStorage. */
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
//       gameKey, label, detail, american, decimal, modelProb, edge}
let _legs = [];
let _slip = [];              // leg keys, in insertion order
let _gameTimes = {};         // gameKey -> Date (game-bet feed start_time or props commence_time)
let _knownGames = new Set(); // gameKeys present on the served game-bet board
let _hiddenGames = new Set(); // props games the betting board already dropped as started

const SETTINGS_KEY = 'pl2-settings';
const DEFAULTS = {
    markets: ['ags', 'points', 'assists', 'sog', 'saves', 'moneyline', 'total', 'puckline'],
    games: null,                       // null = all games; else a Set of gameKeys
    sides: 'rec',
    mode: 'target',
    size: 3,
    target: 350,                       // American odds the finder aims at
    tolerance: 0.2,                    // +/-% band around the target (on decimal odds)
    evOnly: false,
    stake: 10,
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
        const raw = localStorage.getItem(SETTINGS_KEY);
        if (!raw) return;
        const s = JSON.parse(raw);
        if (s && typeof s === 'object') {
            const { markets, ...rest } = s;
            Object.assign(_settings, rest);
            delete _settings.hideStarted; // superseded: started games always hide now
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

// ── Leg construction ──────────────────────────────────────────────────
function addPropLegs(payload) {
    (payload.props || []).forEach(p => {
        const family = familyOfProp(p.market);
        const recSide = p.recommendation === 'Under' ? 'Under'
            : p.recommendation === 'Over' ? 'Over' : null;
        const gk = gameKey(p.away_abbr, p.home_abbr);
        if (p.commence_time) recordGameTime(gk, p.commence_time);
        const gameLabel = gameLabelFrom(abbrName(p.home_abbr), abbrName(p.away_abbr));
        const base = {
            source: 'props',
            family: family,
            gameLabel: gameLabel,
            gameKey: gk,
            detail: p.player_team ? p.player_team.toUpperCase() : '',
        };

        const lineTxt = `${p.line > 0 ? '+' : ''}${p.line}`;
        // Over side.
        if (p.over_american != null || p.over_decimal != null) {
            const decimal = p.over_decimal || americanToDecimal(p.over_american);
            const implied = p.implied_over != null ? p.implied_over / 100 : 1 / decimal;
            _legs.push({
                ...base,
                key: `prop:${p.player}::${p.market}::${p.line}::Over`,
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
                key: `prop:${p.player}::${p.market}::${p.line}::Under`,
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

function addGameLegs(payload) {
    (payload.games || []).forEach(g => {
        const gk = gameKey(g.away, g.home);
        _knownGames.add(gk);
        if (g.start_time) recordGameTime(gk, g.start_time);
        (g.edges || []).forEach(e => {
            const american = e.odds;
            if (american == null) return;
            _legs.push({
                key: `game:${g.away}@${g.home}:${e.market}:${e.side || e.pick}`,
                source: 'games',
                family: familyOfGame(e.market),
                recommended: true,
                side: e.side || e.pick || '',
                gameLabel: gameLabelFrom(abbrName(g.home), abbrName(g.away)),
                gameKey: gk,
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

function fmtGameTime(d) {
    try {
        return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    } catch (e) {
        return '';
    }
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

function legRow(l) {
    const inSlip = _slip.includes(l.key);
    const meta = FAMILY_META[l.family] || FAMILY_META.other;
    const style = meta.accent ? `style="--chip-accent:${meta.accent};--chip-accent-dim:${meta.dim}"` : '';
    return `
        <div class="pl-leg${inSlip ? ' pl-leg-added' : ''}" data-key="${escapeAttr(l.key)}">
            <div class="pl-leg-main">
                <span class="pl-leg-titleline">
                    <span class="pl-leg-badge" ${style}>${escapeHtml(meta.label)}</span>
                    <span class="pl-leg-label">${escapeHtml(l.label)}</span>
                </span>
                <span class="pl-leg-sub">${escapeHtml(l.gameLabel)}${l.detail ? ' · ' + escapeHtml(l.detail) : ''}</span>
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
    return _slip.map(k => _legs.find(l => l.key === k)).filter(Boolean);
}

function renderSlip() {
    const legs = slipLegs();
    const host = document.getElementById('slipLegs');

    host.innerHTML = legs.length
        ? legs.map(l => `
            <div class="pl-slip-leg" data-key="${escapeAttr(l.key)}">
                <span class="pl-leg-label">${escapeHtml(l.label)}</span>
                <span class="pl-leg-sub">${escapeHtml(l.gameLabel)} · ${fmtAmerican(l.american)}</span>
                <button class="pl-remove" type="button" data-remove="${escapeAttr(l.key)}" aria-label="Remove leg">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            </div>`).join('')
        : '<p class="empty-desc">Add legs from the left — combined odds and hit chance show up here.</p>';

    renderSlipTotals(legs);
    renderPool(); // reflect the added marks
}

function renderSlipTotals(legs) {
    const stake = Math.max(0, parseFloat(document.getElementById('plStake').value) || 0);
    const combinedDecimal = legs.reduce((acc, l) => acc * (l.decimal || 1), 1);
    const jointProb = legs.reduce((acc, l) => acc * l.modelProb, 1);
    const implied = combinedDecimal > 1 ? 1 / combinedDecimal : 1;
    const american = decimalToAmerican(combinedDecimal);
    const ev = legs.length ? jointProb * (combinedDecimal - 1) - (1 - jointProb) : 0;
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

    // Correlation note: any two legs from the same game.
    const keys = legs.map(l => l.gameKey);
    const clash = new Set(keys).size !== keys.length;
    document.getElementById('plCorrelationNote').hidden = !clash;
}

// ── Finder controls ───────────────────────────────────────────────────
const RANGE_MIN = 1.3, RANGE_MAX = 12;

function targetDecimal() {
    // The number input wins when valid; otherwise the slider drives.
    const num = parseFloat(document.getElementById('plTargetInput').value);
    if (!isNaN(num) && num !== 0 && americanToDecimal(num)) return americanToDecimal(num);
    return slideToDecimal(parseFloat(document.getElementById('plTargetRange').value) || 50);
}

function slideToDecimal(pos) {
    const t = Math.min(1, Math.max(0, pos / 100));
    return Math.exp(Math.log(RANGE_MIN) + t * (Math.log(RANGE_MAX) - Math.log(RANGE_MIN)));
}

function decimalToSlide(d) {
    const t = (Math.log(Math.min(RANGE_MAX, Math.max(RANGE_MIN, d))) - Math.log(RANGE_MIN))
        / (Math.log(RANGE_MAX) - Math.log(RANGE_MIN));
    return t * 100;
}

function syncTargetUI(fromRange) {
    const num = document.getElementById('plTargetInput');
    const range = document.getElementById('plTargetRange');
    if (fromRange) {
        num.value = fmtAmerican(decimalToAmerican(slideToDecimal(parseFloat(range.value)))).replace('+', '');
    } else {
        range.value = decimalToSlide(targetDecimal());
    }
    document.getElementById('plTargetDec').textContent = targetDecimal().toFixed(2);
    document.getElementById('plTargetWrap').hidden = _settings.mode !== 'target';
    document.getElementById('plTol').style.display = _settings.mode === 'target' ? '' : 'none';
}

function syncSegButtons(hostId, attr, value) {
    document.querySelectorAll(`#${hostId} button`).forEach(b =>
        b.setAttribute('aria-pressed', String(b.dataset[attr] === String(value))));
}

// ── Auto-builder ──────────────────────────────────────────────────────
const BUILD_POOL_CAP = 30;
const MAX_SHOW = 12;

function builderPool() {
    const pool = _legs.filter(legAllowed).sort((a, b) => b.edge - a.edge);
    return pool.slice(0, BUILD_POOL_CAP);
}

function comboScore(legs) {
    const combinedDecimal = legs.reduce((a, l) => a * l.decimal, 1);
    const jointProb = legs.reduce((a, l) => a * l.modelProb, 1);
    const ev = jointProb * (combinedDecimal - 1) - (1 - jointProb);
    return { legs, combinedDecimal, jointProb, ev };
}

function buildParlays() {
    const size = _settings.size;
    const pool = builderPool();
    const host = document.getElementById('plAutoplay');

    if (pool.length < size) {
        host.innerHTML = `<p class="empty-desc">Only ${pool.length} legs in the selected markets — check more markets or lower the leg count.</p>`;
        return;
    }

    if (!_settings.markets.size) {
        host.innerHTML = `<p class="empty-desc">No markets selected — toggle a market chip above.</p>`;
        return;
    }

    const combos = [];
    const idx = [];
    let checked = 0;
    (function rec(start, depth) {
        if (depth === size) { combos.push([...idx]); checked++; return; }
        for (let i = start; i < pool.length; i++) {
            idx[depth] = i;
            rec(i + 1, depth + 1);
            if (combos.length > 400000) return; // hard stop
        }
    })(0, 0);

    const target = _settings.mode === 'target' ? targetDecimal() : null;
    const games = new Set();

    const scored = combos
        .filter(idxs => {
            games.clear();
            for (const i of idxs) {
                const gk = pool[i].gameKey;
                if (games.has(gk)) return false; // independent legs only
                games.add(gk);
            }
            return true;
        })
        .map(idxs => comboScore(idxs.map(i => pool[i])))
        .map(p => ({
            ...p,
            ratio: target ? p.combinedDecimal / target : 1,
            dist: target ? Math.abs(p.combinedDecimal / target - 1) : 0,
        }))
        .filter(p => !_settings.evOnly || p.ev > 0);

    let results, note;
    if (_settings.mode === 'target') {
        const within = scored.filter(p => p.dist <= _settings.tolerance);
        if (within.length) {
            // Rank: most likely to hit first. Each row shows how far off the
            // target it lands.
            results = within.sort((a, b) => b.jointProb - a.jointProb || a.dist - b.dist).slice(0, MAX_SHOW);
            note = `${within.length} of ${checked.toLocaleString()} independent combos landed within ${Math.round(_settings.tolerance * 100)}% of ${fmtAmerican(decimalToAmerican(target))}.`;
        } else {
            // Nothing near the target — show the closest shots instead.
            // Closest first, most likely among equals.
            results = scored.sort((a, b) => a.dist - b.dist || b.jointProb - a.jointProb).slice(0, MAX_SHOW);
            note = `Nothing within ${Math.round(_settings.tolerance * 100)}% of ${fmtAmerican(decimalToAmerican(target))} — closest combos below.`;
        }
    } else {
        // Safest: rank by hit chance, best odds-breaker last.
        results = scored.sort((a, b) => b.jointProb - a.jointProb || b.ev - a.ev).slice(0, MAX_SHOW);
        note = `Top ${results.length} of ${checked.toLocaleString()} independent combos by hit chance.`;
    }

    host.innerHTML =
        `<p class="pl-autoplay-note">${escapeHtml(note)}</p>` +
        (results.length
            ? results.map(p => parlayRow(p)).join('')
            : '<p class="empty-desc">No qualifying combo — widen the tolerance, drop the +EV filter, or change markets.</p>');
}

function parlayRow(p) {
    const legsHtml = p.legs.map(l => {
        const meta = FAMILY_META[l.family] || FAMILY_META.other;
        const style = meta.accent ? `style="--chip-accent:${meta.accent}"` : '';
        return `<span class="pl-combo-leg"><span class="pl-fam-dot" ${style}></span>
            <span class="pl-leg-label">${escapeHtml(l.label)}</span>
            <span class="pl-leg-odds">${fmtAmerican(l.american)}</span></span>`;
    }).join('');
    const targetPill = _settings.mode === 'target'
        ? `<span class="pl-dist">${(p.ratio >= 1 ? '+' : '') + ((p.ratio - 1) * 100).toFixed(0)}% vs target</span>`
        : '';
    const american = decimalToAmerican(p.combinedDecimal);
    return `
        <div class="pl-combo">
            <div class="pl-combo-body">${legsHtml}</div>
            <div class="pl-combo-nums">
                ${targetPill}
                <span class="pl-leg-odds pl-combo-odds">${fmtAmerican(american)} <small>(${p.combinedDecimal.toFixed(2)})</small></span>
                <span class="pl-leg-prob">${pct(p.jointProb)}</span>
                <span class="pl-leg-edge ${p.ev > 0 ? 'pos' : 'neg'}">${(p.ev * 100).toFixed(1)}%</span>
                <button class="props-sort-btn pl-add" type="button" data-load="${escapeAttr(p.legs.map(l => l.key).join('|'))}" aria-label="Load this parlay into the slip">
                    <i class="fa-solid fa-download"></i>
                </button>
            </div>
        </div>`;
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
        renderChips();
        renderGamesDD();
        renderPool();
        return;
    }
    const add = e.target.closest('[data-add]');
    if (add) {
        const key = add.getAttribute('data-add');
        if (_slip.includes(key)) _slip = _slip.filter(k => k !== key);
        else _slip.push(key);
        renderSlip();
        return;
    }
    const remove = e.target.closest('[data-remove]');
    if (remove) {
        _slip = _slip.filter(k => k !== remove.getAttribute('data-remove'));
        renderSlip();
        return;
    }
    const load = e.target.closest('[data-load]');
    if (load) {
        _slip = load.getAttribute('data-load').split('|');
        renderSlip();
        document.querySelector('.pl-hero').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        return;
    }
    const preset = e.target.closest('[data-american]');
    if (preset) {
        const a = parseInt(preset.getAttribute('data-american'), 10);
        document.getElementById('plTargetInput').value = a;
        syncTargetUI(false);
        _settings.target = a;
        saveSettings();
        return;
    }
    const modeBtn = e.target.closest('#plMode button');
    if (modeBtn) {
        _settings.mode = modeBtn.dataset.mode;
        syncSegButtons('plMode', 'mode', _settings.mode);
        syncTargetUI(false);
        saveSettings();
        return;
    }
    const sizeBtn = e.target.closest('#plSize button');
    if (sizeBtn) {
        _settings.size = parseInt(sizeBtn.dataset.size, 10) || 3;
        syncSegButtons('plSize', 'size', _settings.size);
        saveSettings();
        return;
    }
    const tolBtn = e.target.closest('#plTol button');
    if (tolBtn) {
        _settings.tolerance = parseFloat(tolBtn.dataset.tol) || 0.2;
        syncSegButtons('plTol', 'tol', _settings.tolerance);
        saveSettings();
        return;
    }
});

document.getElementById('plSides').addEventListener('click', () => {
    const btn = document.getElementById('plSides');
    _settings.sides = _settings.sides === 'both' ? 'rec' : 'both';
    btn.setAttribute('aria-pressed', String(_settings.sides === 'both'));
    saveSettings();
    renderChips();
    renderGamesDD();
    renderPool();
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
        renderGamesDD();
        renderPool();
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
        renderGamesDD();
        renderPool();
    }
});

// Started games leave the pool automatically; recheck about once a minute.
setInterval(() => {
    if (_legs.length) {
        renderGamesDD();
        renderPool();
    }
}, 60000);

document.getElementById('plEvOnly').addEventListener('click', () => {
    const btn = document.getElementById('plEvOnly');
    _settings.evOnly = !_settings.evOnly;
    btn.setAttribute('aria-pressed', String(_settings.evOnly));
    saveSettings();
});

document.getElementById('plSort').addEventListener('change', renderPool);
document.getElementById('plSearch').addEventListener('input', renderPool);
document.getElementById('plTargetRange').addEventListener('input', () => syncTargetUI(true));
document.getElementById('plTargetInput').addEventListener('input', () => syncTargetUI(false));
document.getElementById('plStake').addEventListener('input', () => {
    _settings.stake = Math.max(0, parseFloat(document.getElementById('plStake').value) || 0);
    saveSettings();
    renderSlipTotals(slipLegs());
});
document.getElementById('plBuild').addEventListener('click', buildParlays);
document.getElementById('plClear').addEventListener('click', () => { _slip = []; renderSlip(); });

document.getElementById('plCopy').addEventListener('click', () => {
    const legs = slipLegs();
    if (!legs.length) return;
    const stake = Math.max(0, parseFloat(document.getElementById('plStake').value) || 0);
    const combinedDecimal = legs.reduce((acc, l) => acc * (l.decimal || 1), 1);
    const text = [
        'Parlay Lab slip',
        ...legs.map(l => `${l.label} @ ${l.gameLabel} (${fmtAmerican(l.american)})`),
        `Combined: ${fmtAmerican(decimalToAmerican(combinedDecimal))} (${combinedDecimal.toFixed(2)})`,
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
function renderPresets() {
    const host = document.getElementById('plPresets');
    host.innerHTML = [130, 200, 350, 600, 1200].map(a =>
        `<button type="button" class="pl-chip pl-preset" data-american="${a}">${fmtAmerican(a)}</button>`).join('');
}

function restoreControls() {
    syncSegButtons('plMode', 'mode', _settings.mode);
    syncSegButtons('plSize', 'size', _settings.size);
    syncSegButtons('plTol', 'tol', _settings.tolerance);
    document.getElementById('plSides').setAttribute('aria-pressed', String(_settings.sides === 'both'));
    document.getElementById('plEvOnly').setAttribute('aria-pressed', String(_settings.evOnly));
    document.getElementById('plTargetInput').value = _settings.target;
    syncTargetUI(false);
    document.getElementById('plStake').value = _settings.stake;
}

async function boot() {
    loadSettings();
    renderPresets();
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
    renderGamesDD();
    renderChips();
    renderPool();
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
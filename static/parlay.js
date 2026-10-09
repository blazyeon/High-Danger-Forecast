/* Parlay Lab — unlisted parlay builder.
   Loads priced legs from the existing props and game-bet boards, then does all
   parlay math client-side: combined decimal odds, joint hit probability under
   independence, and EV vs the book's combined implied probability. */
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

// ── State ─────────────────────────────────────────────────────────────
let _legs = [];              // priced legs: {key, source, gameLabel, gameKey, label, detail, american, decimal, modelProb, edge}
let _slip = [];              // leg keys, in insertion order

// ── Leg construction ──────────────────────────────────────────────────
function addPropLegs(payload) {
    (payload.props || []).forEach(p => {
        let american, decimal, modelProb;
        if (p.recommendation === 'Under') {
            // Goalie saves keeps both sides; the Under is priced against the
            // complement of the model's over probability.
            american = p.under_american;
            decimal = p.under_decimal;
            modelProb = 1 - p.prob_over / 100;
        } else if (p.recommendation === 'Over') {
            american = p.over_american;
            decimal = p.over_decimal;
            modelProb = p.prob_over / 100;
        } else {
            return; // Pass — the model has no position on it.
        }
        if (american == null && decimal == null) return;
        if (decimal == null) decimal = americanToDecimal(american);

        const gameLabel = gameLabelFrom(abbrName(p.home_abbr), abbrName(p.away_abbr));
        _legs.push({
            key: `prop:${p.player}:${p.market}:${p.line}`,
            source: 'props',
            gameLabel: gameLabel,
            gameKey: gameKey(p.away_abbr, p.home_abbr),
            label: `${p.player} — ${p.market} ${p.line > 0 ? '+' : ''}${p.line}`,
            detail: p.player_team ? p.player_team.toUpperCase() : '',
            american: american != null ? american : decimalToAmerican(decimal),
            decimal: decimal,
            modelProb: modelProb,
            edge: p.edge || 0,
        });
    });
}

function addGameLegs(payload) {
    (payload.games || []).forEach(g => {
        (g.edges || []).forEach(e => {
            const american = e.odds;
            if (american == null) return;
            _legs.push({
                key: `game:${g.away}@${g.home}:${e.market}:${e.side || e.pick}`,
                source: 'games',
                gameLabel: gameLabelFrom(abbrName(g.home), abbrName(g.away)),
                gameKey: gameKey(g.away, g.home),
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

function gameLabelFrom(home, away) {
    return `${away} @ ${home}`;
}

function gameKey(away, home) {
    return `${abbrName(away)}@${abbrName(home)}`;
}

// ── Leg pool rendering ────────────────────────────────────────────────
function currentSource() {
    return document.getElementById('plSource').value || 'all';
}

function filteredLegs() {
    const src = currentSource();
    const q = (document.getElementById('plSearch').value || '').trim().toLowerCase();
    const sort = document.getElementById('plSort').value;

    let out = _legs.filter(l => src === 'all' || l.source === src);
    if (q) out = out.filter(l => l.label.toLowerCase().includes(q) || l.gameLabel.toLowerCase().includes(q));

    if (sort === 'prob') out.sort((a, b) => b.modelProb - a.modelProb);
    else if (sort === 'odds') out.sort((a, b) => b.decimal - a.decimal);
    else out.sort((a, b) => b.edge - a.edge);
    return out;
}

const MAX_POOL_ROWS = 80;

function renderPool() {
    const host = document.getElementById('plPool');
    const legs = filteredLegs();

    if (!legs.length) {
        host.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-inbox empty-icon"></i>
                <p class="empty-title">No legs match</p>
                <p class="empty-desc">Loosen the search or check back after the daily update.</p>
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
    return `
        <div class="pl-leg${inSlip ? ' pl-leg-added' : ''}" data-key="${escapeAttr(l.key)}">
            <div class="pl-leg-main">
                <span class="pl-leg-label">${escapeHtml(l.label)}</span>
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
        : '<p class="empty-desc">Add legs from the left — combined odds and EV show up here.</p>';

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
        ? `${fmtAmerican(american)} (${combinedDecimal.toFixed(2)})` : '—';
    document.getElementById('plJointProb').textContent = legs.length ? pct(jointProb) : '—';
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

// ── Auto-builder ──────────────────────────────────────────────────────
const BUILD_POOL_CAP = 36;
const MAX_SHOW = 12;

function buildParlays() {
    const size = parseInt(document.getElementById('plBuildSize').value, 10) || 2;
    // Candidates: legs the model actually likes, ranked by per-leg edge.
    const pool = [..._legs].sort((a, b) => b.edge - a.edge).slice(0, BUILD_POOL_CAP);

    const host = document.getElementById('plAutoplay');
    if (pool.length < size) {
        host.innerHTML = `<p class="empty-desc">Only ${pool.length} priced legs available — need at least ${size}.</p>`;
        return;
    }

    const combos = [];
    const idx = [];
    (function rec(start, depth) {
        if (depth === size) { combos.push([...idx]); return; }
        for (let i = start; i < pool.length; i++) {
            idx[depth] = i;
            rec(i + 1, depth + 1);
            if (combos.length > 250000) break; // hard stop, never hit at cap 36
        }
    })(0, 0);

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
        .map(idxs => {
            const legs = idxs.map(i => pool[i]);
            const combinedDecimal = legs.reduce((a, l) => a * l.decimal, 1);
            const jointProb = legs.reduce((a, l) => a * l.modelProb, 1);
            const ev = jointProb * (combinedDecimal - 1) - (1 - jointProb);
            return { legs, combinedDecimal, jointProb, ev };
        })
        .sort((a, b) => b.ev - a.ev)
        .slice(0, MAX_SHOW);

    host.innerHTML = scored.length
        ? scored.map(p => parlayRow(p)).join('')
        : '<p class="empty-desc">No independent combo among the top legs — raise the leg count or widen the pool.</p>';
}

function parlayRow(p) {
    const legsHtml = p.legs.map(l => `
        <span class="pl-combo-leg"><span class="pl-leg-label">${escapeHtml(l.label)}</span>
        <span class="pl-leg-odds">${fmtAmerican(l.american)}</span></span>`).join('');
    const ev = p.ev;
    return `
        <div class="pl-combo">
            <div class="pl-combo-legs">${legsHtml}</div>
            <div class="pl-combo-nums">
                <span class="pl-leg-odds pl-combo-odds">${decimalToAmerican(p.combinedDecimal) != null ? fmtAmerican(decimalToAmerican(p.combinedDecimal)) : ''} (${p.combinedDecimal.toFixed(2)})</span>
                <span class="pl-leg-prob">${pct(p.jointProb)}</span>
                <span class="pl-leg-edge ${ev > 0 ? 'pos' : 'neg'}">${(ev * 100).toFixed(1)}%</span>
                <button class="props-sort-btn pl-add" type="button" data-load="${escapeAttr(p.legs.map(l => l.key).join('|'))}" aria-label="Load this parlay into the slip">
                    <i class="fa-solid fa-download"></i>
                </button>
            </div>
        </div>`;
}

// ── Events ────────────────────────────────────────────────────────────
document.addEventListener('click', e => {
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
    }
});

document.getElementById('plSource').addEventListener('change', renderPool);
document.getElementById('plSort').addEventListener('change', renderPool);
document.getElementById('plSearch').addEventListener('input', renderPool);
document.getElementById('plStake').addEventListener('input', () => renderSlipTotals(slipLegs()));
document.getElementById('plBuild').addEventListener('click', buildParlays);
document.getElementById('plClear').addEventListener('click', () => { _slip = []; renderSlip(); });

// Same theme toggle behavior as index.html.
document.getElementById('themeToggle').addEventListener('click', () => {
    const root = document.documentElement;
    const next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('hdf-theme', next); } catch (e) {}
});

// ── Boot ──────────────────────────────────────────────────────────────
async function boot() {
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
/**
 * NHL Game Predictor — Frontend v3.3
 * Premium dark SPA backed by the Flask API.
 */

// ── Division order / labels for matchup selectors ────────────────
const DIVISION_ORDER = ['Atlantic', 'Metropolitan', 'Central', 'Pacific'];
const DIVISION_LABELS = { Atlantic: 'Atlantic', Metropolitan: 'Metro', Central: 'Central', Pacific: 'Pacific' };
const FORWARD_POSITIONS = ['C', 'L', 'R', 'LW', 'RW', 'W'];

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Minimal offline fallback (used only if /api/teams fails) ─────
const TEAMS_FALLBACK = {
    Atlantic: [
        { abbr:"BOS", full_name:"Boston Bruins" },
        { abbr:"BUF", full_name:"Buffalo Sabres" },
        { abbr:"DET", full_name:"Detroit Red Wings" },
        { abbr:"FLA", full_name:"Florida Panthers" },
        { abbr:"MTL", full_name:"Montreal Canadiens" },
        { abbr:"OTT", full_name:"Ottawa Senators" },
        { abbr:"TBL", full_name:"Tampa Bay Lightning" },
        { abbr:"TOR", full_name:"Toronto Maple Leafs" },
    ],
    Metropolitan: [
        { abbr:"CAR", full_name:"Carolina Hurricanes" },
        { abbr:"CBJ", full_name:"Columbus Blue Jackets" },
        { abbr:"NJD", full_name:"New Jersey Devils" },
        { abbr:"NYI", full_name:"New York Islanders" },
        { abbr:"NYR", full_name:"New York Rangers" },
        { abbr:"PHI", full_name:"Philadelphia Flyers" },
        { abbr:"PIT", full_name:"Pittsburgh Penguins" },
        { abbr:"WSH", full_name:"Washington Capitals" },
    ],
    Central: [
        { abbr:"CHI", full_name:"Chicago Blackhawks" },
        { abbr:"COL", full_name:"Colorado Avalanche" },
        { abbr:"DAL", full_name:"Dallas Stars" },
        { abbr:"MIN", full_name:"Minnesota Wild" },
        { abbr:"NSH", full_name:"Nashville Predators" },
        { abbr:"STL", full_name:"St. Louis Blues" },
        { abbr:"UTA", full_name:"Utah Mammoth" },
        { abbr:"WPG", full_name:"Winnipeg Jets" },
    ],
    Pacific: [
        { abbr:"ANA", full_name:"Anaheim Ducks" },
        { abbr:"CGY", full_name:"Calgary Flames" },
        { abbr:"EDM", full_name:"Edmonton Oilers" },
        { abbr:"LAK", full_name:"Los Angeles Kings" },
        { abbr:"SJS", full_name:"San Jose Sharks" },
        { abbr:"SEA", full_name:"Seattle Kraken" },
        { abbr:"VAN", full_name:"Vancouver Canucks" },
        { abbr:"VGK", full_name:"Vegas Golden Knights" },
    ],
};

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initDateDefaults();
    initDatePickers();
    initSettingsToggle();
    populateTeams();
    loadTeams();
    loadSeasons();
    setupEventListeners();
    document.getElementById('modalClose').addEventListener('click', () => {
        document.getElementById('gameModal').style.display = 'none';
    });
    document.getElementById('gameModal').addEventListener('click', (e) => {
        if (e.target.id === 'gameModal') document.getElementById('gameModal').style.display = 'none';
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && document.getElementById('gameModal').style.display === 'flex') {
            document.getElementById('gameModal').style.display = 'none';
        }
    });
    loadAppState();
    updateLogos();
    updatePredictBtn();
});

function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
            if (btn.dataset.tab === 'elo') runElo();
            if (btn.dataset.tab === 'betting-edge') runBettingEdge();
            if (btn.dataset.tab === 'props') runProps();
        });
    });
}

function initDateDefaults() {
    const today = new Date().toISOString().split('T')[0];
    const lookupDate = document.getElementById('lookupDate');
    if (lookupDate) lookupDate.value = today;
    const propsDate = document.getElementById('propsDate');
    if (propsDate) propsDate.value = today;
    const bettingEdgeDate = document.getElementById('bettingEdgeDate');
    if (bettingEdgeDate) bettingEdgeDate.value = today;
}

// ── Custom Calendar Date Picker ─────────────────────────────────────
class DatePicker {
    constructor(input) {
        this.input = input;
        this.viewDate = new Date();
        this.selectedDate = null;
        this.isOpen = false;
        this.container = null;
        this.popup = null;
        this.trigger = null;

        this._parseInput();
        this._build();
        this._bind();
    }

    _parseInput() {
        const val = this.input.value;
        if (val) {
            const [y, m, d] = val.split('-').map(Number);
            this.selectedDate = new Date(y, m - 1, d);
            this.viewDate = new Date(this.selectedDate);
        }
    }

    _build() {
        this.container = document.createElement('div');
        this.container.className = 'date-picker';
        this.input.parentNode.insertBefore(this.container, this.input);
        this.container.appendChild(this.input);

        // Switch to text so the browser's native calendar never appears.
        this.input.type = 'text';
        this.input.className = (this.input.className + ' date-picker-input').trim();
        this.input.placeholder = 'YYYY-MM-DD';
        this.input.inputMode = 'numeric';
        this.input.autocomplete = 'off';

        // Visible calendar trigger button.
        this.trigger = document.createElement('button');
        this.trigger.type = 'button';
        this.trigger.className = 'date-picker-trigger';
        this.trigger.setAttribute('aria-label', 'Open calendar');
        this.trigger.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
                <line x1="16" y1="2" x2="16" y2="6"/>
                <line x1="8" y1="2" x2="8" y2="6"/>
                <line x1="3" y1="10" x2="21" y2="10"/>
            </svg>
        `;
        this.container.appendChild(this.trigger);

        // Append popup to <body> to avoid clipping by parent containers.
        this.popup = document.createElement('div');
        this.popup.className = 'date-picker-popup';
        this.popup.setAttribute('role', 'dialog');
        this.popup.setAttribute('aria-modal', 'true');
        document.body.appendChild(this.popup);

        this._render();
    }

    _render() {
        const year = this.viewDate.getFullYear();
        const month = this.viewDate.getMonth();
        const firstDay = new Date(year, month, 1).getDay();
        const daysInMonth = new Date(year, month + 1, 0).getDate();
        const today = new Date();
        const todayYMD = this._ymd(today);

        const monthNames = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ];
        const dayLabels = ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'];

        let daysHtml = '';
        for (let i = 0; i < firstDay; i++) {
            daysHtml += '<div class="dp-day dp-empty"></div>';
        }
        for (let d = 1; d <= daysInMonth; d++) {
            const date = new Date(year, month, d);
            const ymd = this._ymd(date);
            const isSelected = this.selectedDate && this._ymd(this.selectedDate) === ymd;
            const isToday = ymd === todayYMD;
            let cls = 'dp-day';
            if (isSelected) cls += ' selected';
            if (isToday) cls += ' today';
            daysHtml += `<button type="button" class="${cls}" data-ymd="${ymd}" data-year="${year}" data-month="${month}" data-day="${d}">${d}</button>`;
        }

        this.popup.innerHTML = `
            <div class="dp-header">
                <button type="button" class="dp-nav dp-prev" aria-label="Previous month">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>
                </button>
                <div class="dp-title">${monthNames[month]} ${year}</div>
                <button type="button" class="dp-nav dp-next" aria-label="Next month">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
                </button>
            </div>
            <div class="dp-weekdays">
                ${dayLabels.map(l => `<div class="dp-weekday">${l}</div>`).join('')}
            </div>
            <div class="dp-days">${daysHtml}</div>
            <div class="dp-footer">
                <button type="button" class="dp-today">Today</button>
            </div>
        `;
    }

    _bind() {
        // Open on input click / focus.
        this.input.addEventListener('click', (e) => {
            e.stopPropagation();
            this.open();
        });
        this.input.addEventListener('focus', () => this.open());

        // Open on trigger click.
        this.trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            this.open();
        });

        // Allow manual typing; validate on blur.
        this.input.addEventListener('input', () => this._onManualInput());
        this.input.addEventListener('blur', () => this._validateInput());

        // Calendar navigation / day selection via event delegation on the popup.
        this.popup.addEventListener('click', (e) => {
            const target = e.target;
            if (target.closest('.dp-prev')) {
                e.stopPropagation();
                this._changeMonth(-1);
                return;
            }
            if (target.closest('.dp-next')) {
                e.stopPropagation();
                this._changeMonth(1);
                return;
            }
            if (target.closest('.dp-today')) {
                e.stopPropagation();
                this._selectToday();
                return;
            }
            const dayBtn = target.closest('.dp-day[data-day]');
            if (dayBtn) {
                e.stopPropagation();
                const year = parseInt(dayBtn.dataset.year, 10);
                const month = parseInt(dayBtn.dataset.month, 10);
                const day = parseInt(dayBtn.dataset.day, 10);
                this._selectDate(new Date(year, month, day));
            }
        });

        // Close on Escape and outside click.
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) this.close();
        });
        this._outsideClick = (e) => {
            if (!this.isOpen) return;
            if (this.container.contains(e.target) || this.popup.contains(e.target)) return;
            this.close();
        };
        document.addEventListener('click', this._outsideClick);

        // Reposition on resize/orientation change with a small debounce.
        let resizeTimer;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => {
                if (this.isOpen) this._position();
            }, 100);
        });
    }

    _onManualInput() {
        const val = this.input.value;
        if (/^\d{4}-\d{2}-\d{2}$/.test(val)) {
            const [y, m, d] = val.split('-').map(Number);
            const date = new Date(y, m - 1, d);
            if (date.getFullYear() === y && date.getMonth() === m - 1 && date.getDate() === d) {
                this.selectedDate = date;
                this.viewDate = new Date(date);
                this._render();
            }
        }
    }

    _validateInput() {
        const val = this.input.value;
        if (!val) return;
        if (!/^\d{4}-\d{2}-\d{2}$/.test(val)) {
            this.input.value = this.selectedDate ? this._ymd(this.selectedDate) : '';
            return;
        }
        const [y, m, d] = val.split('-').map(Number);
        const date = new Date(y, m - 1, d);
        if (date.getFullYear() !== y || date.getMonth() !== m - 1 || date.getDate() !== d) {
            this.input.value = this.selectedDate ? this._ymd(this.selectedDate) : '';
        } else {
            this.selectedDate = date;
            this.viewDate = new Date(date);
            this.input.dispatchEvent(new Event('change', { bubbles: true }));
            this._render();
        }
    }

    _changeMonth(delta) {
        const next = new Date(this.viewDate);
        // Set day to 1 first so selecting the 31st does not skip months
        // (e.g. Jan 31 + 1 month would otherwise roll to March).
        next.setDate(1);
        next.setMonth(next.getMonth() + delta);
        this.viewDate = next;
        this._render();
    }

    _selectToday() {
        const now = new Date();
        this._selectDate(now);
        this.viewDate = new Date(now);
    }

    _selectDate(date) {
        this.selectedDate = date;
        this.input.value = this._ymd(date);
        this.input.dispatchEvent(new Event('input', { bubbles: true }));
        this.input.dispatchEvent(new Event('change', { bubbles: true }));
        this.close();
    }

    _ymd(date) {
        const y = date.getFullYear();
        const m = String(date.getMonth() + 1).padStart(2, '0');
        const d = String(date.getDate()).padStart(2, '0');
        return `${y}-${m}-${d}`;
    }

    open() {
        if (this.isOpen) return;
        this._parseInput();
        if (this.selectedDate) {
            this.viewDate = new Date(this.selectedDate);
        }
        this._render();
        this.popup.classList.add('open');
        this.isOpen = true;
        this._position();
    }

    close() {
        this.popup.classList.remove('open');
        this.isOpen = false;
    }

    _position() {
        const rect = this.container.getBoundingClientRect();
        const popupRect = this.popup.getBoundingClientRect();
        const width = popupRect.width || 280;
        const height = popupRect.height || 320;
        const margin = 8;

        let left = rect.left + window.scrollX;
        let top = rect.bottom + window.scrollY + margin;

        // Flip above if not enough room below.
        if (top + height > window.innerHeight + window.scrollY && rect.top - height - margin >= window.scrollY) {
            top = rect.top + window.scrollY - height - margin;
        }

        // Keep inside viewport horizontally.
        if (left + width > window.innerWidth + window.scrollX) {
            left = Math.max(8, window.innerWidth + window.scrollX - width - margin);
        }

        this.popup.style.position = 'absolute';
        this.popup.style.left = `${left}px`;
        this.popup.style.top = `${top}px`;
    }
}

function initDatePickers() {
    document.querySelectorAll('input[type="date"], input[type="text"].date-input').forEach(input => {
        if (!input.classList.contains('date-picker-input')) {
            new DatePicker(input);
        }
    });
}

function initSettingsToggle() {
    const toggle = document.getElementById('settingsToggle');
    const body = document.getElementById('settingsBody');
    toggle.addEventListener('click', () => {
        const open = body.style.display === 'block';
        body.style.display = open ? 'none' : 'block';
        toggle.setAttribute('aria-expanded', String(!open));
    });
}

function populateTeams() {
    const homeSel = document.getElementById('homeTeam');
    const awaySel = document.getElementById('awayTeam');
    if (!homeSel || !awaySel) return;
    homeSel.innerHTML = '';
    awaySel.innerHTML = '';
    homeSel.add(new Option('Select Team', ''));
    awaySel.add(new Option('Select Team', ''));
    const divisions = getTeamsData();
    for (const div of DIVISION_ORDER) {
        const teams = divisions[div];
        if (!teams) continue;
        const label = DIVISION_LABELS[div] || div;
        const homeGroup = document.createElement('optgroup');
        homeGroup.label = label;
        const awayGroup = document.createElement('optgroup');
        awayGroup.label = label;
        teams.forEach(t => {
            const abbr = t.abbr || t;
            const name = t.full_name || t.name || abbr;
            homeGroup.appendChild(new Option(`${name} (${abbr})`, abbr));
            awayGroup.appendChild(new Option(`${name} (${abbr})`, abbr));
        });
        homeSel.appendChild(homeGroup);
        awaySel.appendChild(awayGroup);
    }
    homeSel.value = '';
    awaySel.value = '';
}

function currentNHLSeasonKey() {
    const now = new Date();
    const year = now.getFullYear();
    const month = now.getMonth() + 1; // 1-12
    // NHL season spans two calendar years; start in October.
    const startYear = month >= 10 ? year : year - 1;
    return `${startYear}${startYear + 1}`;
}

async function populateGoalies() {
    const home = document.getElementById('homeTeam').value;
    const away = document.getElementById('awayTeam').value;
    const hRow = document.getElementById('homeGoalieRow');
    const aRow = document.getElementById('awayGoalieRow');
    const hSel = document.getElementById('homeGoalie');
    const aSel = document.getElementById('awayGoalie');
    const dateStr = new Date().toISOString().split('T')[0];

    // Remember user selections so toggling B2B or refetching doesn't wipe them.
    const prevHomeGoalie = hSel.value;
    const prevAwayGoalie = aSel.value;

    hSel.innerHTML = '';
    aSel.innerHTML = '';
    hRow.style.display = home ? 'block' : 'none';
    aRow.style.display = away ? 'block' : 'none';
    if (!home && !away) return;

    const homeB2B = document.getElementById('homeB2B')?.checked || false;
    const awayB2B = document.getElementById('awayB2B')?.checked || false;

    const fill = (sel, list, previous) => {
        if (!list || !list.length) return false;
        sel.innerHTML = '';
        list.forEach(g => sel.add(new Option(g, g)));
        // Restore the previous selection if it's still valid, otherwise keep first option.
        if (previous && list.includes(previous)) {
            sel.value = previous;
        } else {
            sel.selectedIndex = 0;
        }
        return true;
    };

    if (home) {
        let live = [];
        try {
            const params = away ? `?opponent=${away}&b2b=${homeB2B ? 1 : 0}` : '';
            live = (await safeFetchJson(`/api/goalies/${home}/${dateStr}${params}`)).goalies || [];
        }
        catch (e) { console.warn(`Goalie API failed for ${home}:`, e); }
        if (!fill(hSel, live, prevHomeGoalie)) hRow.style.display = 'none';
    }
    if (away) {
        let live = [];
        try {
            const params = home ? `?opponent=${home}&b2b=${awayB2B ? 1 : 0}` : '';
            live = (await safeFetchJson(`/api/goalies/${away}/${dateStr}${params}`)).goalies || [];
        }
        catch (e) { console.warn(`Goalie API failed for ${away}:`, e); }
        if (!fill(aSel, live, prevAwayGoalie)) aRow.style.display = 'none';
    }
}

function updateLogos() {
    const home = document.getElementById('homeTeam').value;
    const away = document.getElementById('awayTeam').value;
    const hLogo = document.getElementById('homeLogo');
    const aLogo = document.getElementById('awayLogo');

    if (home) { hLogo.src = `/api/logos/${home}.png`; hLogo.style.display = 'block'; hLogo.onerror = () => hLogo.style.display = 'none'; }
    else hLogo.style.display = 'none';
    if (away) { aLogo.src = `/api/logos/${away}.png`; aLogo.style.display = 'block'; aLogo.onerror = () => aLogo.style.display = 'none'; }
    else aLogo.style.display = 'none';
}

function updatePredictBtn() {
    const home = document.getElementById('homeTeam').value;
    const away = document.getElementById('awayTeam').value;
    document.getElementById('predictBtn').disabled = !(home && away);
}

function setupEventListeners() {
    document.getElementById('homeTeam').addEventListener('change', () => { updateLogos(); updatePredictBtn(); populateGoalies(); });
    document.getElementById('awayTeam').addEventListener('change', () => { updateLogos(); updatePredictBtn(); populateGoalies(); });
    document.getElementById('homeB2B')?.addEventListener('change', populateGoalies);
    document.getElementById('awayB2B')?.addEventListener('change', populateGoalies);
    document.getElementById('predictBtn').addEventListener('click', runPrediction);
    document.getElementById('lookupBtn').addEventListener('click', runLookup);
    document.getElementById('statsBtn').addEventListener('click', runStats);
    document.getElementById('propsBtn').addEventListener('click', runProps);
    document.getElementById('bettingEdgeBtn').addEventListener('click', runBettingEdge);
    document.getElementById('eloBtn').addEventListener('click', runElo);
}

// ── Simulation Log ────────────────────────────────────────────────
let simLogs = [];

function logStep(step, detail) {
    simLogs.push({ step, detail, ts: new Date().toLocaleTimeString() });
}

function renderLog() {
    const container = document.getElementById('simLog');
    if (!container) return;
    let html = '<div class="log-header"><span class="log-title">Simulation Log</span><span class="log-count">' + simLogs.length + ' steps</span></div>';
    html += '<div class="log-body">';
    simLogs.forEach((entry, i) => {
        html += `<div class="log-entry ${i % 2 === 0 ? 'even' : ''}">
            <span class="log-ts">${entry.ts}</span>
            <span class="log-step">${entry.step}</span>
            <span class="log-detail">${entry.detail}</span>
        </div>`;
    });
    html += '</div>';
    container.innerHTML = html;
    container.style.display = 'block';
}

// ── Prediction ──────────────────────────────────────────────────
async function runPrediction() {
    const home = document.getElementById('homeTeam').value;
    const away = document.getElementById('awayTeam').value;
    if (!home || !away) return;

    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
    const content = document.getElementById('resultsContent');
    content.innerHTML = '<div class="loading"><div class="spinner"></div><span>Running Monte Carlo simulation...</span></div>';
    document.getElementById('simLog').style.display = 'none';
    simLogs = [];

    const homeB2B = document.getElementById('homeB2B')?.checked || false;
    const awayB2B = document.getElementById('awayB2B')?.checked || false;
    const homeGoalie = document.getElementById('homeGoalie').value;
    const awayGoalie = document.getElementById('awayGoalie').value;

    let data;
    try {
        const body = {
            home_team: home,
            away_team: away,
            simulations: parseInt(document.getElementById('sims').value) || 10000,
            trend_games: parseInt(document.getElementById('trendGames').value) || 25,
            nst_window: parseInt(document.getElementById('nstWindow').value) || 14,
            season_type: 2,
            home_goalie: homeGoalie || null,
            away_goalie: awayGoalie || null,
            home_b2b: homeB2B,
            away_b2b: awayB2B,
            date: new Date().toISOString().split('T')[0],
        };
        data = await safeFetchJson('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (data.error) { content.innerHTML = `<div class="error-box">${escapeHtml(data.error)}</div>`; btn.disabled = false; return; }

        logStep('INIT', `Matchup: ${getTeamName(home)} (HOME) vs ${getTeamName(away)} (AWAY)`);
        logStep('ELO', `${home} rating: ${Math.round(data.home_elo_adj || 1500)} | ${away} rating: ${Math.round(data.away_elo_adj || 1500)}`);
        logStep('PROB', `Win prob: ${data.home_win_pct}% home / ${data.away_win_pct}% away`);
        const actualHomeGoalie = data.home_goalie || homeGoalie || 'N/A';
        const actualAwayGoalie = data.away_goalie || awayGoalie || 'N/A';
        logStep('GOALIE', `Home goalie: ${actualHomeGoalie} | Away goalie: ${actualAwayGoalie}`);
        if (homeB2B) logStep('B2B', `${home} flagged as back-to-back (~14% fatigue penalty)`);
        if (awayB2B) logStep('B2B', `${away} flagged as back-to-back (~14% fatigue penalty)`);
        logStep('SIM', `Ran ${data.sims || body.simulations} Monte Carlo iterations`);
        logStep('ENSEMBLE', 'Blended Elo (25%) + simulation (50%) + ML (25%) outcomes');
        logStep('DONE', `Prediction complete. Confidence: ${data.confidence}`);
    } catch (e) {
        console.error('Prediction failed:', e);
        content.innerHTML = `<div class="error-box">Prediction failed: ${escapeHtml(e.message)}</div>`;
        btn.disabled = false;
        return;
    }

    renderResults(data, home, away);
    renderLog();
    btn.disabled = false;
}

function renderResults(sim, homeAbbr, awayAbbr) {
    const homeName = getTeamName(homeAbbr);
    const awayName = getTeamName(awayAbbr);
    const homeWin = parseFloat(sim.home_win_pct) > parseFloat(sim.away_win_pct);

    let html = '';

    // Banner with explicit HOME / AWAY labels
    html += `<div class="result-banner">`;
    html += `<div class="result-team-block">
        <div class="result-badge home">HOME</div>
        <img class="result-team-logo" src="/api/logos/${homeAbbr}.png" alt="${homeName}" onerror="this.style.display='none'">
        <div class="result-team-name" title="${escapeHtml(homeName)}">${homeName}</div>
        <div class="result-team-abbr">${homeAbbr}</div>
    </div>`;
    html += `<div class="result-center">
        <div class="result-prediction-label">Prediction</div>
        <div class="result-prediction-value ${homeWin ? 'home-win' : 'away-win'}">${homeWin ? 'HOME WIN' : 'AWAY WIN'}</div>
    </div>`;
    html += `<div class="result-team-block">
        <div class="result-badge away">AWAY</div>
        <img class="result-team-logo" src="/api/logos/${awayAbbr}.png" alt="${awayName}" onerror="this.style.display='none'">
        <div class="result-team-name" title="${escapeHtml(awayName)}">${awayName}</div>
        <div class="result-team-abbr">${awayAbbr}</div>
    </div>`;
    html += `</div>`;

    // Probability bar
    const hPct = parseFloat(sim.home_win_pct);
    const aPct = parseFloat(sim.away_win_pct);
    html += `<div class="prob-section">
        <div class="prob-header">
            <span class="prob-team">${homeName} <span class="prob-pct home">${hPct.toFixed(1)}%</span></span>
            <span class="prob-team">${awayName} <span class="prob-pct away">${aPct.toFixed(1)}%</span></span>
        </div>
        <div class="prob-bar-track">
            <div class="prob-bar-home" style="width:${hPct.toFixed(1)}%">${hPct > 18 ? hPct.toFixed(0) + '%' : ''}</div>
            <div class="prob-bar-away" style="width:${aPct.toFixed(1)}%">${aPct > 18 ? aPct.toFixed(0) + '%' : ''}</div>
        </div>
    </div>`;

    // Stats grid
    html += `<div class="stats-grid">`;
    const stats = [
        { label: 'Home Elo', value: Math.round(sim.home_elo_adj || 1500) },
        { label: 'Away Elo', value: Math.round(sim.away_elo_adj || 1500) },
        { label: 'Exp Home G', value: parseFloat(sim.exp_home_goals).toFixed(2) },
        { label: 'Exp Away G', value: parseFloat(sim.exp_away_goals).toFixed(2) },
        { label: 'Reg Home Win', value: (hPct * (sim.regulation_games_pct || 100) / 100).toFixed(1) + '%' },
        { label: 'Reg Away Win', value: (aPct * (sim.regulation_games_pct || 100) / 100).toFixed(1) + '%' },
        { label: 'OT %', value: (sim.ot_games_pct || 16).toFixed(1) + '%' },
        { label: 'Most Likely Score', value: `${sim.mode_home_goals}-${sim.mode_away_goals}`, cls: 'gold' },
    ];
    stats.forEach(s => {
        html += `<div class="stat-card"><div class="stat-value ${s.cls || ''}">${s.value}</div><div class="stat-label">${s.label}</div></div>`;
    });
    html += `</div>`;

    // Injuries
    const homeInj = sim.home_injuries || {};
    const awayInj = sim.away_injuries || {};
    const homePlayers = homeInj.players || [];
    const awayPlayers = awayInj.players || [];
    const homeTopDefense = homeInj.top_defensive_injuries || [];
    const awayTopDefense = awayInj.top_defensive_injuries || [];
    const hasInjuries = homePlayers.length || awayPlayers.length || homeTopDefense.length || awayTopDefense.length;
    if (hasInjuries) {
        html += `<hr class="section-divider"><div class="section-title"><i class="fa-solid fa-user-injured"></i> Injury Impact</div>`;

        const mergeInjuries = (players, topDefense) => {
            const byName = {};
            players.forEach(p => { byName[p.name] = { ...p }; });
            topDefense.forEach(d => {
                if (byName[d.name]) {
                    byName[d.name].defensive_score = d.defensive_score;
                    byName[d.name].defensive_percentile = d.defensive_percentile;
                    byName[d.name].position = d.position || byName[d.name].position;
                } else {
                    byName[d.name] = {
                        name: d.name,
                        points: 0,
                        contribution_pct: 0,
                        impact_pct: 0,
                        position: d.position || 'D',
                        defensive_score: d.defensive_score,
                        defensive_percentile: d.defensive_percentile,
                        status: d.status,
                    };
                }
            });
            return Object.values(byName).sort((a, b) => {
                const aOff = Math.abs(a.impact_pct || 0);
                const bOff = Math.abs(b.impact_pct || 0);
                const aDef = ((a.defensive_score || 0) > 0.5) ? (a.defensive_score / 5) : 0;
                const bDef = ((b.defensive_score || 0) > 0.5) ? (b.defensive_score / 5) : 0;
                return (bOff + bDef) - (aOff + aDef);
            });
        };

        const renderTeamInjuries = (teamName, teamAbbr, injData, players, topDefense) => {
            const totalImpact = (injData.offense_impact || 0) * 100;
            const defenseImpact = (injData.defense_impact || 0) * 100;
            const merged = mergeInjuries(players, topDefense);
            html += `<div class="injury-team"><div class="injury-team-header">`;
            html += `<img class="injury-team-logo" src="/api/logos/${teamAbbr}.png" alt="${teamName}" onerror="this.style.display='none'">`;
            html += `<span>${teamName}</span><div class="injury-totals">`;
            html += `<span class="injury-total ${totalImpact < 0 ? 'negative' : ''}">${totalImpact.toFixed(1)}% offense</span>`;
            if (defenseImpact > 0.5) {
                html += `<span class="injury-total defense">+${defenseImpact.toFixed(1)}% opp goals</span>`;
            }
            html += `</div></div>`;
            if (merged.length) {
                html += `<div class="injury-list">`;
                merged.forEach(p => {
                    const contrib = ((p.contribution_pct || 0) * 100).toFixed(1);
                    const status = (p.status || 'injured').toUpperCase();
                    const pos = (p.position || '').toUpperCase();
                    const isDefense = pos.startsWith('D');
                    const isForward = FORWARD_POSITIONS.includes(pos);
                    const dScore = p.defensive_score || 0;
                    const defenseNote = (dScore > 0.5)
                        ? `<span class="injury-contrib defense" title="Defensive score ${dScore.toFixed(2)}"><i class="fa-solid fa-shield-halved"></i> ${dScore.toFixed(1)}</span>`
                        : '';
                    const offenseNote = (p.contribution_pct > 0)
                        ? `<span class="injury-contrib">${contrib}% team pts</span>`
                        : '';
                    const positionBadge = isDefense
                        ? `<span class="injury-badge defense">D</span>`
                        : (isForward ? `<span class="injury-badge forward">F</span>` : '');
                    html += `<div class="injury-row"><div class="injury-name">${positionBadge}${p.name}</div><div class="injury-meta"><span class="injury-status ${status === 'DTD' ? 'dtd' : ''}">${status}</span>${offenseNote}${defenseNote}</div></div>`;
                });
                html += `</div>`;
            } else {
                html += `<div class="injury-none">No significant injuries</div>`;
            }
            html += `</div>`;
        };

        renderTeamInjuries(homeName, homeAbbr, homeInj, homePlayers, homeTopDefense);
        renderTeamInjuries(awayName, awayAbbr, awayInj, awayPlayers, awayTopDefense);
    }

    // Totals
    if (sim.totals_distribution) {
        html += `<hr class="section-divider"><div class="section-title">Total Goals Distribution</div><div class="ou-section">`;
        const totals = Object.entries(sim.totals_distribution)
            .map(([k,v]) => ({total:parseInt(k), count:v}))
            .sort((a,b) => a.total - b.total);
        const totalSims = totals.reduce((sum, t) => sum + t.count, 0) || (sim.sims || 10000);
        const maxPct = Math.max(...totals.map(t => t.count / totalSims));
        totals.forEach(t => {
            const pct = t.count / totalSims;
            const width = maxPct > 0 ? (pct / maxPct * 100) : 0;
            html += `<div class="ou-row"><div class="ou-total">${t.total}</div><div class="ou-bar-track"><div class="ou-bar-fill" style="width:${width.toFixed(1)}%"></div></div><div class="ou-pct">${(pct * 100).toFixed(1)}%</div></div>`;
        });
        html += `</div>`;
    }

    document.getElementById('resultsContent').innerHTML = html;
}

function getTeamName(abbr) {
    const divisions = getTeamsData();
    for (const div of Object.values(divisions)) {
        const t = div.find(x => (x.abbr || x) === abbr);
        if (t) return t.full_name || t.name || abbr;
    }
    return abbr;
}

async function safeFetchJson(url, opts={}) {
    const resp = await fetch(url, opts);
    const ct = (resp.headers.get('content-type') || '').toLowerCase();
    if (!resp.ok || !ct.includes('application/json')) {
        const text = await resp.text().catch(() => '');
        throw new Error(`${url} returned ${resp.status} (${ct.split(';')[0] || 'unknown'}): ${text.slice(0, 160)}`);
    }
    return resp.json();
}

// ── Schedule Tab ─────────────────────────────────────────────────
async function runLookup() {
    const container = document.getElementById('lookupResults');
    const date = document.getElementById('lookupDate').value;
    if (!date) { container.innerHTML = '<div class="error-box">Pick a date first.</div>'; return; }

    container.innerHTML = '<div class="loading"><div class="spinner"></div><span>Fetching NHL schedule...</span></div>';

    let games = [];
    let apiWorked = false;

    try {
        const data = await safeFetchJson(`/api/lookup?date=${encodeURIComponent(date)}`);
        if (Array.isArray(data.games)) {
            games = data.games;
            apiWorked = games.length > 0;
        }
    } catch (e) {
        console.warn('Schedule backend fetch failed:', e.message);
    }

    if (!games.length) {
        container.innerHTML = `<div class="empty-state"><div class="empty-icon"><i class="fa-solid fa-calendar-xmark"></i></div><h3 class="empty-title">No Games</h3><p class="empty-desc">No NHL games scheduled for ${date}.</p></div>`;
        return;
    }

    let html = '';

    games.forEach(g => {
        const state = (g.state || 'FUT').toString().toUpperCase();
        let stateLabel = 'Scheduled';
        let stateClass = ' upcoming';
        let scoreHtml = '';

        if (state === 'LIVE' || state === 'CRIT') {
            stateLabel = '<i class="fa-solid fa-circle live-dot"></i> Live';
            stateClass = '';
            const hScore = g.home_score ?? '-';
            const aScore = g.away_score ?? '-';
            scoreHtml = `<div class="game-card-score">${aScore} – ${hScore}</div>`;
        } else if (state === 'OFF' || state === 'FINAL') {
            stateLabel = 'Final';
            stateClass = '';
            const hScore = g.home_score ?? '-';
            const aScore = g.away_score ?? '-';
            scoreHtml = `<div class="game-card-score">${aScore} – ${hScore}</div>`;
        }

        const homeAbbr = g.home || 'TBD';
        const awayAbbr = g.away || 'TBD';
        const homeName = g.home_name || getTeamName(homeAbbr);
        const awayName = g.away_name || getTeamName(awayAbbr);
        const time = g.startTime ? new Date(g.startTime).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', timeZoneName: 'short'}) : '';

        html += `<div class="game-card" data-gameid="${g.id}" data-home="${homeAbbr}" data-away="${awayAbbr}">
            <div class="game-card-left">
                <div class="game-card-teams">
                    <img class="game-card-logo" src="/api/logos/${awayAbbr}.png" alt="${awayAbbr}" onerror="this.style.display='none'">
                    <span class="team-name">${awayName}</span>
                    <span class="vs-sep">@</span>
                    <img class="game-card-logo" src="/api/logos/${homeAbbr}.png" alt="${homeAbbr}" onerror="this.style.display='none'">
                    <span class="team-name">${homeName}</span>
                </div>
                <div class="game-card-time">${time}</div>
            </div>
            <div class="game-card-right">
                ${scoreHtml}
                <span class="game-status${stateClass}">${stateLabel}</span>
            </div>
        </div>`;
    });
    container.innerHTML = html;

    container.querySelectorAll('.game-card').forEach(card => {
        card.addEventListener('click', () => showGameDetail(card.dataset.gameid, card.dataset.home, card.dataset.away));
    });
}

async function showGameDetail(gameId, homeAbbrFallback, awayAbbrFallback) {
    const modal = document.getElementById('gameModal');
    const body = document.getElementById('modalBody');
    modal.style.display = 'flex';
    body.innerHTML = '<div class="loading"><div class="spinner"></div><span>Loading game details...</span></div>';

    try {
        const data = await safeFetchJson(`/api/boxscore/${gameId}`);

        const home = data.home_team || {};
        const away = data.away_team || {};
        const homeAbbr = home.abbrev || homeAbbrFallback || 'HOME';
        const awayAbbr = away.abbrev || awayAbbrFallback || 'AWAY';
        const homeName = home.name || getTeamName(homeAbbr);
        const awayName = away.name || getTeamName(awayAbbr);
        const hScore = home.score ?? 0;
        const aScore = away.score ?? 0;
        const state = (data.state || '').toString().toUpperCase();
        const period = data.period ? `P${data.period}` : '';
        const clock = data.clock ? ` – ${data.clock}` : '';
        let statusText = state === 'LIVE' || state === 'CRIT' ? `<i class="fa-solid fa-circle live-dot"></i> LIVE ${period}${clock}` : (state === 'OFF' || state === 'FINAL' ? 'FINAL' : 'UPCOMING');
        if ((state === 'OFF' || state === 'FINAL') && data.last_period_type && data.last_period_type.toUpperCase() !== 'REG') {
            statusText += ` (${data.last_period_type.toUpperCase()})`;
        }

        let html = `<div class="modal-header">
            <div class="modal-teams">
                <div class="modal-team">
                    <img src="/api/logos/${awayAbbr}.png" alt="${awayName}" onerror="this.style.display='none'">
                    <div class="modal-team-name">${awayName}</div>
                    <div class="modal-team-abbr">${awayAbbr}</div>
                </div>
                <div class="modal-score">
                    <div class="modal-score-value">${aScore} – ${hScore}</div>
                    <div class="modal-status">${statusText}</div>
                </div>
                <div class="modal-team">
                    <img src="/api/logos/${homeAbbr}.png" alt="${homeName}" onerror="this.style.display='none'">
                    <div class="modal-team-name">${homeName}</div>
                    <div class="modal-team-abbr">${homeAbbr}</div>
                </div>
            </div>
        </div>`;

        // Team stats comparison
        function ppPct(pp) {
            if (!pp || pp === '-') return null;
            const [g, o] = String(pp).split('/').map(Number);
            if (!o || isNaN(g)) return 0;
            return (g / o) * 100;
        }
        function statCompare(label, awayVal, homeVal, awaySub, homeSub, awayNum, homeNum) {
            let a = parseFloat(awayNum);
            let h = parseFloat(homeNum);
            if (isNaN(a)) a = 0;
            if (isNaN(h)) h = 0;
            const total = a + h;
            let awayWidth, homeWidth;
            if (total === 0) {
                awayWidth = 50;
                homeWidth = 50;
            } else {
                awayWidth = (a / total) * 100;
                homeWidth = (h / total) * 100;
            }
            return `<div class="stat-compare">
                <div class="stat-compare-values">
                    <div class="stat-compare-team left">
                        <div class="stat-compare-num">${awayVal}</div>
                        ${awaySub ? `<div class="stat-compare-sub">${awaySub}</div>` : ''}
                    </div>
                    <div class="stat-compare-label">${label}</div>
                    <div class="stat-compare-team right">
                        <div class="stat-compare-num">${homeVal}</div>
                        ${homeSub ? `<div class="stat-compare-sub">${homeSub}</div>` : ''}
                    </div>
                </div>
                <div class="stat-compare-bars">
                    <div class="stat-compare-bar-left" style="width:${awayWidth.toFixed(1)}%"></div>
                    <div class="stat-compare-bar-right" style="width:${homeWidth.toFixed(1)}%"></div>
                </div>
            </div>`;
        }

        html += `<div class="modal-section"><div class="modal-section-title">Game Stats</div>`;
        html += `<div class="stat-compare-header">
            <div class="stat-compare-header-team">
                <img src="/api/logos/${awayAbbr}.png" alt="${awayName}" onerror="this.style.display='none'">
                <span>${awayAbbr}</span>
            </div>
            <div class="stat-compare-header-team right">
                <span>${homeAbbr}</span>
                <img src="/api/logos/${homeAbbr}.png" alt="${homeName}" onerror="this.style.display='none'">
            </div>
        </div>`;
        html += statCompare('Shots On Goal', away.sog ?? '-', home.sog ?? '-', '', '', away.sog, home.sog);
        html += statCompare('Face-off %', fmtPct(away.faceoff_pct), fmtPct(home.faceoff_pct), away.faceoff_record ?? '-', home.faceoff_record ?? '-', away.faceoff_pct, home.faceoff_pct);
        html += statCompare('Power Play %', fmtPct(ppPct(away.power_play)), fmtPct(ppPct(home.power_play)), away.power_play ?? '-', home.power_play ?? '-', ppPct(away.power_play), ppPct(home.power_play));
        html += statCompare('Penalty Minutes', away.pim ?? '-', home.pim ?? '-', '', '', away.pim, home.pim);
        html += statCompare('Hits', away.hits ?? '-', home.hits ?? '-', '', '', away.hits, home.hits);
        html += statCompare('Blocked Shots', away.blocked_shots ?? '-', home.blocked_shots ?? '-', '', '', away.blocked_shots, home.blocked_shots);
        html += statCompare('Giveaways', away.giveaways ?? '-', home.giveaways ?? '-', '', '', away.giveaways, home.giveaways);
        html += statCompare('Takeaways', away.takeaways ?? '-', home.takeaways ?? '-', '', '', away.takeaways, home.takeaways);
        html += `</div>`;

        // Goalies
        const awayGoalies = data.away_goalies || [];
        const homeGoalies = data.home_goalies || [];

        if (awayGoalies.length || homeGoalies.length) {
            html += `<div class="modal-section"><div class="modal-section-title">Goalies</div></div>`;
            html += `<div class="modal-goalie-grid">`;

            const renderGoalie = (g) => {
                const stats = [];
                if (g.saves !== null && g.saves !== undefined) stats.push(`${g.saves} saves`);
                if (g.shots_against !== null && g.shots_against !== undefined) stats.push(`on ${g.shots_against} shots`);
                if (g.goals_against !== null && g.goals_against !== undefined) stats.push(`${g.goals_against} GA`);
                if (g.save_pct !== null && g.save_pct !== undefined) stats.push(`${(g.save_pct * 100).toFixed(1)}%`);
                if (g.toi) stats.push(g.toi);
                const starterBadge = g.starter ? `<span class="goalie-starter">Starter</span>` : '';
                return `<div class="modal-goalie">
                    <div class="modal-goalie-name">${g.name || 'Unknown'} ${starterBadge}</div>
                    <div class="modal-goalie-stats">${stats.join(' • ')}</div>
                </div>`;
            };

            html += `<div class="modal-roster-col"><div class="modal-roster-header">${awayName}</div>`;
            if (awayGoalies.length) awayGoalies.forEach(g => { html += renderGoalie(g); });
            else html += `<div class="modal-goalie-empty">No goalie data</div>`;
            html += `</div>`;

            html += `<div class="modal-roster-col"><div class="modal-roster-header">${homeName}</div>`;
            if (homeGoalies.length) homeGoalies.forEach(g => { html += renderGoalie(g); });
            else html += `<div class="modal-goalie-empty">No goalie data</div>`;
            html += `</div>`;

            html += `</div>`;
        }

        // Rosters
        const awayRoster = data.away_roster || [];
        const homeRoster = data.home_roster || [];

        if (awayRoster.length || homeRoster.length) {
            html += `<div class="modal-section"><div class="modal-section-title">Rosters</div></div>`;
            html += `<div class="modal-roster-grid">`;

            html += `<div class="modal-roster-col"><div class="modal-roster-header">${awayName}</div>`;
            awayRoster.forEach(p => {
                const pos = p.position || '?';
                const name = p.name || 'Unknown';
                const stats = [];
                if (typeof p.goals === 'number') stats.push(`G:${p.goals}`);
                if (typeof p.assists === 'number') stats.push(`A:${p.assists}`);
                if (typeof p.sog === 'number') stats.push(`SOG:${p.sog}`);
                if (typeof p.toi === 'string') stats.push(p.toi);
                html += `<div class="modal-player"><span class="modal-player-pos">${pos}</span> <span class="modal-player-name">${name}</span> <span class="modal-player-stats">${stats.join(' | ')}</span></div>`;
            });
            html += `</div>`;

            html += `<div class="modal-roster-col"><div class="modal-roster-header">${homeName}</div>`;
            homeRoster.forEach(p => {
                const pos = p.position || '?';
                const name = p.name || 'Unknown';
                const stats = [];
                if (typeof p.goals === 'number') stats.push(`G:${p.goals}`);
                if (typeof p.assists === 'number') stats.push(`A:${p.assists}`);
                if (typeof p.sog === 'number') stats.push(`SOG:${p.sog}`);
                if (typeof p.toi === 'string') stats.push(p.toi);
                html += `<div class="modal-player"><span class="modal-player-pos">${pos}</span> <span class="modal-player-name">${name}</span> <span class="modal-player-stats">${stats.join(' | ')}</span></div>`;
            });
            html += `</div>`;

            html += `</div>`;
        }

        body.innerHTML = html;
    } catch (e) {
        console.error(e);
        body.innerHTML = `<div class="error-box">Could not load game details.<br><small>${escapeHtml(e.message)}</small></div>`;
    }
}

// ── Analytics Tab (PBP Advanced Stats) ────────────────────────────
async function loadStatsPayload(type) {
    const season = document.getElementById('statsSeason')?.value || '20252026';
    const endpoint = `/api/stats/${type}?season=${season}&stype=2`;
    return safeFetchJson(endpoint, { cache: 'no-store' });
}

function fmtNum(v, digits=1) {
    if (v === null || v === undefined || v === '') return '-';
    const n = parseFloat(v);
    if (isNaN(n)) return v;
    return n.toFixed(digits);
}

function fmtPct(v) {
    if (v === null || v === undefined || v === '') return '-';
    const n = parseFloat(v);
    if (isNaN(n)) return v;
    // API may return 0-1 or 0-100
    const pct = n > 0 && n <= 1 ? n * 100 : n;
    return pct.toFixed(1) + '%';
}

async function runStats() {
    const container = document.getElementById('statsResults');
    const type = document.getElementById('statsType').value;
    const season = document.getElementById('statsSeason')?.value || '';
    const seasonLabel = season ? `${season.slice(0, 4)}-${season.slice(4)}` : 'selected season';
    container.innerHTML = '<div class="loading"><div class="spinner"></div><span>Loading advanced stats...</span></div>';

    let payload;
    try {
        payload = await loadStatsPayload(type);
    } catch (e) {
        console.error('Stats load failed:', e);
        container.innerHTML = `<div class="error-box">Could not load advanced stats for ${seasonLabel}.<br><small>${escapeHtml(e.message)}</small></div>`;
        return;
    }
    const { data, meta } = payload;

    const source = meta?.source || 'unknown';
    const updatedAt = meta?.updated_at
        ? new Date(meta.updated_at).toLocaleString()
        : null;

    if (!data || !data.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon"><i class="fa-solid fa-chart-simple"></i></div><h3 class="empty-title">No Data</h3><p class="empty-desc">Advanced stats are not available yet. Run <code>python update_pbp_stats.py</code> to populate them.</p></div>';
        return;
    }

    let html = '';
    let notice = '';

    if (type === 'teams') {
        const sorted = [...data].sort(
            (a, b) => (parseFloat(b.xgf_pct) || 0) - (parseFloat(a.xgf_pct) || 0)
        );
        const cols = [
            { key: 'team', label: 'Team', fmt: v => `<div class="stats-team-cell"><img class="stats-team-logo" src="/api/logos/${v}.png" alt="${v}" onerror="this.style.display='none'"><strong>${v}</strong></div>` },
            { key: 'gp', label: 'GP' },
            { key: 'gf', label: 'GF' },
            { key: 'ga', label: 'GA' },
            { key: 'cf_pct', label: 'CF%', fmt: v => fmtPct(v) },
            { key: 'xgf_pct', label: 'xGF%', fmt: v => fmtPct(v) },
            { key: 'hdcf_pct', label: 'HDCF%', fmt: v => fmtPct(v) },
            { key: 'sv_pct', label: 'SV%', fmt: v => fmtPct(v) },
            { key: 'sh_pct', label: 'SH%', fmt: v => fmtPct(v) },
            { key: 'pdo', label: 'PDO', fmt: v => fmtNum(v, 1) },
        ];
        html = _buildStatsTable(cols, sorted, { sortable: true });
        notice = '<i class="fa-solid fa-chart-simple"></i> Team advanced metrics from PBP data (CF%, xGF%, HDCF%, PDO).';
    } else if (type === 'skaters') {
        const toolbar = _buildStatsToolbar(type, data);
        let sorted = [...data].sort(
            (a, b) => (parseInt(b.points) || 0) - (parseInt(a.points) || 0)
        );
        const cols = [
            { key: 'name', label: 'Player', fmt: v => `<strong class="player-name">${escapeHtml(v)}</strong>` },
            { key: 'team', label: 'Team' },
            { key: 'position', label: 'Pos' },
            { key: 'gp', label: 'GP' },
            { key: 'goals', label: 'G' },
            { key: 'assists', label: 'A' },
            { key: 'points', label: 'Pts', fmt: v => `<strong>${v}</strong>` },
            { key: 'ppg', label: 'PPG', fmt: v => fmtNum(v, 2) },
            { key: 'shots', label: 'SOG' },
            { key: 'sh_pct', label: 'SH%', fmt: v => fmtPct(v) },
            { key: 'plus_minus', label: '+/-' },
            { key: 'toi_pg', label: 'TOI/GP', fmt: v => _fmtToi(v) },
        ];
        html = toolbar + _buildStatsTable(cols, sorted, { sortable: true, type });
        notice = '<i class="fa-solid fa-chart-simple"></i> Skater stats sorted by points. Use filters to narrow by team, position, or games played.';
    } else if (type === 'goalies') {
        const toolbar = _buildStatsToolbar(type, data);
        let sorted = [...data].sort(
            (a, b) => (parseFloat(b.gsax) || 0) - (parseFloat(a.gsax) || 0)
        );
        const cols = [
            { key: 'name', label: 'Player', fmt: v => `<strong class="player-name">${escapeHtml(v)}</strong>` },
            { key: 'team', label: 'Team' },
            { key: 'gp', label: 'GP' },
            { key: 'gs', label: 'GS' },
            { key: 'w', label: 'W' },
            { key: 'l', label: 'L' },
            { key: 'otl', label: 'OTL' },
            { key: 'ga', label: 'GA' },
            { key: 'gaa', label: 'GAA', fmt: v => fmtNum(v, 2) },
            { key: 'sa', label: 'SA' },
            { key: 'sv_pct', label: 'SV%', fmt: v => fmtPct(v) },
            { key: 'gsax', label: 'GSAx', fmt: v => fmtNum(v, 1) },
        ];
        html = toolbar + _buildStatsTable(cols, sorted, { sortable: true, type });
        notice = '<i class="fa-solid fa-chart-simple"></i> Goalie stats sorted by GSAx. Use filters to narrow by team or games played.';
    }

    const sourceBadge = source === 'cache'
        ? '<span class="stats-source cached">Cached</span>'
        : `<span class="stats-source computed">${source}</span>`;
    const updatedText = updatedAt ? `Last updated: ${updatedAt}` : 'Live computation';
    html += `<div class="cors-notice" style="margin-top:12px">${notice} ${sourceBadge} • ${updatedText}</div>`;
    container.innerHTML = html;
    _attachStatsSortListeners();
}

let _currentStatsRows = [];
let _currentStatsCols = [];
let _currentStatsSort = { key: null, dir: 'desc' };
let _currentStatsType = null;

function _sortStatsBy(key) {
    const prev = _currentStatsSort;
    let dir = 'desc';
    if (prev.key === key) {
        dir = prev.dir === 'desc' ? 'asc' : 'desc';
    }
    _currentStatsSort = { key, dir };

    const col = _currentStatsCols.find(c => c.key === key);
    const sortable = col && col.key !== 'team' && col.key !== 'name';

    let rows = [..._currentStatsRows];
    if (sortable) {
        rows.sort((a, b) => {
            const av = parseFloat(a[key]);
            const bv = parseFloat(b[key]);
            const aNum = isNaN(av) ? -Infinity : av;
            const bNum = isNaN(bv) ? -Infinity : bv;
            const delta = bNum - aNum;
            return dir === 'desc' ? delta : -delta;
        });
    }

    const container = document.getElementById('statsResults');
    const notice = container.querySelector('.cors-notice');
    const tableHtml = _buildStatsTable(_currentStatsCols, rows, { sortable: true, activeKey: key, activeDir: dir, type: _currentStatsType });
    const tableWrap = container.querySelector('.table-wrap');
    if (tableWrap) {
        tableWrap.outerHTML = tableHtml;
    } else {
        const insertBefore = notice || container.lastElementChild;
        if (insertBefore) {
            insertBefore.insertAdjacentHTML('beforebegin', tableHtml);
        } else {
            container.insertAdjacentHTML('beforeend', tableHtml);
        }
    }
    if (notice) container.appendChild(notice);
    // Re-apply any active player filters after sorting.
    applyStatsFilters();
}

function _fmtToi(v) {
    if (v === null || v === undefined || v === '') return '-';
    const n = parseFloat(v);
    if (isNaN(n)) return v;
    const m = Math.floor(n);
    const s = Math.round((n - m) * 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}

function _buildStatsToolbar(type, rows) {
    if (type !== 'skaters' && type !== 'goalies') return '';
    const teams = [...new Set(rows.map(r => r.team).filter(Boolean))].sort();
    const teamOptions = teams.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
    let html = '<div class="stats-toolbar">';
    html += `<input type="text" id="statsSearch" class="param-input" placeholder="Search player..." oninput="applyStatsFilters()">`;
    html += `<select id="statsTeamFilter" class="team-select stats-filter" onchange="applyStatsFilters()"><option value="">All Teams</option>${teamOptions}</select>`;
    if (type === 'skaters') {
        html += `<select id="statsPositionFilter" class="team-select stats-filter" onchange="applyStatsFilters()"><option value="">All Positions</option><option value="C">C</option><option value="LW">LW</option><option value="RW">RW</option><option value="D">D</option></select>`;
    }
    html += `<input type="number" id="statsGpFilter" class="param-input stats-filter" min="0" placeholder="Min GP" oninput="applyStatsFilters()">`;
    html += '</div>';
    return html;
}

function _buildStatsTable(cols, rows, opts = {}) {
    _currentStatsRows = rows;
    _currentStatsCols = cols;
    _currentStatsType = opts.type || _currentStatsType;
    const { sortable = false, activeKey, activeDir, type } = opts;
    let html = '<div class="table-wrap"><table class="data-table"><thead><tr>';
    cols.forEach(c => {
        const canSort = c.key !== 'team' && c.key !== 'name';
        if (sortable && canSort) {
            const isActive = activeKey === c.key;
            const arrow = isActive ? (activeDir === 'desc' ? ' ▼' : ' ▲') : ' ⇅';
            html += `<th class="sortable" data-key="${c.key}" title="Click to sort">${c.label}${arrow}</th>`;
        } else {
            html += `<th>${c.label}</th>`;
        }
    });
    html += '</tr></thead><tbody>';
    rows.forEach((row, idx) => {
        const playerName = escapeHtml(row.name || '');
        const team = escapeHtml(row.team || '');
        const position = escapeHtml(row.position || '');
        const gp = parseInt(row.gp, 10) || 0;
        html += `<tr class="stats-row" data-player="${playerName}" data-team="${team}" data-position="${position}" data-gp="${gp}" data-type="${type || ''}" data-idx="${idx}" title="Click for details">`;
        cols.forEach(c => {
            const raw = row[c.key];
            html += `<td>${c.fmt ? c.fmt(raw) : (raw ?? '-')}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table></div>';
    return html;
}

function applyStatsFilters() {
    const query = (document.getElementById('statsSearch')?.value || '').toLowerCase();
    const team = (document.getElementById('statsTeamFilter')?.value || '').toLowerCase();
    const position = (document.getElementById('statsPositionFilter')?.value || '').toLowerCase();
    const minGp = parseInt(document.getElementById('statsGpFilter')?.value, 10);
    document.querySelectorAll('.stats-row').forEach(row => {
        const name = (row.dataset.player || '').toLowerCase();
        const rowTeam = (row.dataset.team || '').toLowerCase();
        const rowPos = (row.dataset.position || '').toLowerCase();
        const rowGp = parseInt(row.dataset.gp, 10) || 0;
        const nameMatch = !query || name.includes(query);
        const teamMatch = !team || rowTeam === team;
        const posMatch = !position || rowPos === position;
        const gpMatch = isNaN(minGp) || rowGp >= minGp;
        row.style.display = (nameMatch && teamMatch && posMatch && gpMatch) ? '' : 'none';
    });
}

function filterStatsRows(query) {
    const searchInput = document.getElementById('statsSearch');
    if (searchInput) searchInput.value = query;
    applyStatsFilters();
}

function _attachStatsSortListeners() {
    const container = document.getElementById('statsResults');
    if (!container) return;
    container.addEventListener('click', (e) => {
        const th = e.target.closest('th.sortable');
        if (th) {
            _sortStatsBy(th.dataset.key);
            return;
        }
        const row = e.target.closest('.stats-row');
        if (row) {
            const idx = parseInt(row.dataset.idx, 10);
            const type = row.dataset.type;
            if (!isNaN(idx) && _currentStatsRows[idx]) {
                showPlayerDetail(_currentStatsRows[idx], type);
            }
        }
    });
}

function showPlayerDetail(player, type) {
    const modal = document.getElementById('gameModal');
    const body = document.getElementById('modalBody');
    modal.style.display = 'flex';
    const season = document.getElementById('statsSeason')?.value || '';
    const seasonLabel = season ? `${season.slice(0, 4)}-${season.slice(4)}` : '';

    let rows = '';
    const addRow = (label, val) => {
        rows += `<div class="modal-player"><span class="modal-player-pos">${label}</span> <span class="modal-player-name">${val ?? '-'}</span></div>`;
    };

    addRow('Season', seasonLabel);
    if (type === 'skaters') {
        addRow('Team', player.team);
        addRow('Position', player.position);
        addRow('GP', player.gp);
        addRow('Goals', player.goals);
        addRow('Assists', player.assists);
        addRow('Points', player.points);
        addRow('Points/GP', fmtNum(player.ppg, 2));
        addRow('Shots', player.shots);
        addRow('Shooting %', fmtPct(player.sh_pct));
        addRow('Plus/Minus', player.plus_minus);
        addRow('PIM', player.pim);
        addRow('TOI/GP', _fmtToi(player.toi_pg));
        addRow('Faceoff %', fmtPct(player.faceoff_pct));
        addRow('PP Goals', player.power_play_goals);
        addRow('SH Goals', player.short_handed_goals);
        addRow('GW Goals', player.game_winning_goals);
        addRow('xGF/GP', fmtNum(player.xgf_pg, 2));
    } else {
        addRow('Team', player.team);
        addRow('GP', player.gp);
        addRow('GS', player.gs);
        addRow('Record', `${player.w || 0}-${player.l || 0}-${player.otl || 0}`);
        addRow('GA', player.ga);
        addRow('GAA', fmtNum(player.gaa, 2));
        addRow('SA', player.sa);
        addRow('SV', player.sv);
        addRow('SV%', fmtPct(player.sv_pct));
        addRow('GSAx', fmtNum(player.gsax, 1));
        addRow('GSAx/60', fmtNum(player.gsax_per_60, 2));
        addRow('Shutouts', player.shutouts);
    }

    body.innerHTML = `<div class="modal-header"><div class="modal-team-name">${escapeHtml(player.name)}</div><div class="modal-team-abbr">${type === 'skaters' ? 'Skater' : 'Goalie'} Details</div></div><div class="modal-section"><div class="modal-section-title">Season Stats</div>${rows}</div>`;
}

async function runElo() {
    const container = document.getElementById('eloResults');
    container.innerHTML = '<div class="loading"><div class="spinner"></div><span>Loading Elo leaderboard...</span></div>';

    try {
        const data = await safeFetchJson('/api/elo-leaderboard');
        const teams = data.teams || [];
        if (!teams.length) {
            container.innerHTML = '<div class="empty-state"><div class="empty-icon"><i class="fa-solid fa-trophy"></i></div><h3 class="empty-title">No Elo Data</h3><p class="empty-desc">Team Elo ratings are not available yet. Run <code>python update_elo_ratings.py --current-season --reset</code> to populate them.</p></div>';
            return;
        }

        const maxRating = Math.max(...teams.map(t => t.rating || 0));
        const minRating = Math.min(...teams.map(t => t.rating || 0));
        const range = Math.max(1, maxRating - minRating);

        let html = '<div class="table-wrap"><table class="data-table elo-table"><thead><tr>';
        html += '<th>#</th><th>Team</th><th>Rating</th><th>Games</th><th>Strength</th>';
        html += '</tr></thead><tbody>';

        teams.forEach((t, i) => {
            const abbr = t.team_abbr;
            const name = getTeamName(abbr);
            const rating = t.rating || 0;
            const gp = t.games_played || 0;
            const width = ((rating - minRating) / range * 100).toFixed(1);
            const rankClass = i < 3 ? 'gold' : '';
            html += `<tr>
                <td><strong class="${rankClass}">${i + 1}</strong></td>
                <td><strong>${abbr}</strong> <span class="muted">${name}</span></td>
                <td><strong class="${rankClass}">${Math.round(rating)}</strong></td>
                <td>${gp}</td>
                <td><div class="ou-bar-track" style="height:10px"><div class="ou-bar-fill" style="width:${width}%"></div></div></td>
            </tr>`;
        });

        html += '</tbody></table></div>';
        html += `<div class="cors-notice" style="margin-top:12px"><i class="fa-solid fa-trophy"></i> Elo ratings for season ${escapeHtml(data.season || 'current')}. League average is 1500; top teams are typically 1600+.</div>`;
        container.innerHTML = html;
    } catch (e) {
        console.error('Elo leaderboard failed:', e);
        container.innerHTML = `<div class="error-box">Could not load Elo leaderboard: ${escapeHtml(e.message)}</div>`;
    }
}

async function runProps() {
    const container = document.getElementById('propsResults');
    container.innerHTML = '<div class="loading"><div class="spinner"></div><span>Loading props...</span></div>';

    const date = document.getElementById('propsDate')?.value || new Date().toISOString().split('T')[0];
    const markets = ["player_points", "player_assists", "player_goals", "player_shots_on_goal"];

    const demoUrl = '/static/data/demo_props.json';
    async function loadDemo(reason) {
        console.warn(reason + ', using demo data.');
        const demo = await safeFetchJson(demoUrl);
        _lastPropsData = demo.props || [];
        resetPropsFilters();
        _propsIsDemo = true;
        _propsDemoReason = reason;
        renderProps(_lastPropsData);
    }

    try {
        const url = `/api/player-props/${date}?regions=us&markets=${markets.join(',')}`;
        const data = await safeFetchJson(url);
        if (data.error) {
            await loadDemo('Props API returned error: ' + data.error);
            return;
        }
        const liveProps = data.props || [];
        if (liveProps.length === 0) {
            await loadDemo('No live props for ' + date);
            return;
        }
        _lastPropsData = liveProps;
        resetPropsFilters();
        _propsIsDemo = false;
        _propsDemoReason = null;
        renderProps(_lastPropsData);
    } catch (e) {
        try {
            await loadDemo('Props API unavailable: ' + e.message);
        } catch (demoErr) {
            _lastPropsData = [];
            _propsIsDemo = false;
            container.innerHTML = `<div class="error-box">Could not load props: ${escapeHtml(e.message)}. Demo data also failed to load: ${escapeHtml(demoErr.message)}.</div>`;
            console.error('Props load failed:', e, demoErr);
        }
    }
}

function resetPropsFilters() {
    _propsSort = 'edge';
    _propsMarketFilter = {
        'Points': true,
        'Goals': true,
        'Assists': false,
        'Shots': true,
    };
    _propsSideFilter = 'Over';
}

function setPropsSort(sort) {
    if (!_lastPropsData) return;
    _propsSort = sort;
    const container = document.getElementById('propsResults');
    if (container) renderProps(_lastPropsData);
}

function setPropsMarketFilter(market, active) {
    if (!_lastPropsData) return;
    _propsMarketFilter[market] = active;
    const container = document.getElementById('propsResults');
    if (container) renderProps(_lastPropsData);
}

function setPropsSideFilter(side) {
    if (!_lastPropsData) return;
    _propsSideFilter = side || 'Over';
    const container = document.getElementById('propsResults');
    if (container) renderProps(_lastPropsData);
}

function _canonicalMarketName(prop) {
    const raw = String(prop.market || '').replace(/^Player\s+/i, '').trim();
    if (/shots/i.test(raw)) return 'Shots';
    if (/points/i.test(raw)) return 'Points';
    if (/goals/i.test(raw)) return 'Goals';
    if (/assists/i.test(raw)) return 'Assists';
    return raw;
}

function renderProps(props) {
    const container = document.getElementById('propsResults');
    if (!props || props.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon"><i class="fa-solid fa-dice"></i></div><h3 class="empty-title">No props available</h3><p class="empty-text">Try a different date or check that ODDS_API_KEY is set.</p></div>';
        return;
    }

    const marketOrder = ['Points', 'Goals', 'Assists', 'Shots'];

    let rows = props.map(p => {
        const rec = p.recommendation || 'Pass';
        const isOver = rec === 'Over';
        const edge = p.edge != null ? parseFloat(p.edge) : 0;
        const recPrice = isOver ? p.over_american : p.under_american;
        const recDecimal = isOver ? p.over_decimal : p.under_decimal;
        const impliedKey = isOver ? 'implied_over' : 'implied_under';
        const implied = p[impliedKey] != null ? parseFloat(p[impliedKey]) : null;
        return {
            ...p,
            edge,
            rec,
            isOver,
            recPrice,
            recDecimal,
            probOver: parseFloat(p.prob_over) || 0,
            impliedProb: implied,
            canonicalMarket: _canonicalMarketName(p),
        };
    });

    // Apply market filters. Goals market is hidden when Under is selected.
    rows = rows.filter(p => {
        if (p.canonicalMarket === 'Goals') {
            return p.isOver && _propsSideFilter !== 'Under' && _propsMarketFilter['Goals'];
        }
        if (!_propsMarketFilter[p.canonicalMarket]) return false;
        if (_propsSideFilter === 'both') return true;
        return p.rec === _propsSideFilter;
    });

    if (_propsSort === 'odds') {
        rows.sort((a, b) => (parseFloat(b.recDecimal) || 0) - (parseFloat(a.recDecimal) || 0));
    } else {
        rows.sort((a, b) => b.edge - a.edge);
    }

    let html = '';
    if (_propsIsDemo) {
        html += `<div class="demo-notice">
            <i class="fa-solid fa-tower-broadcast"></i> Showing sample props because live odds are unavailable${_propsDemoReason ? ': ' + escapeHtml(_propsDemoReason) : ''}.
            Set <code>ODDS_API_KEY</code> for real lines.
        </div>`;
    }

    // Market filter toggles
    html += `<div class="props-toolbar props-filterbar">
        <span class="props-toolbar-label">Markets:</span>`;
    marketOrder.forEach(m => {
        const active = !!_propsMarketFilter[m];
        html += `<button class="props-market-btn ${active ? 'active' : ''}" onclick="setPropsMarketFilter('${m}', ${!active})">${m}</button>`;
    });

    // Side filter toggles. Goals market is always Over-only regardless of this filter.
    html += `<span class="props-toolbar-label props-side-label">Side:</span>`;
    const sides = [
        { key: 'Over', label: 'Over' },
        { key: 'Under', label: 'Under' },
        { key: 'both', label: 'Both' },
    ];
    sides.forEach(s => {
        const active = _propsSideFilter === s.key;
        html += `<button class="props-side-btn ${active ? 'active' : ''}" onclick="setPropsSideFilter('${s.key}')">${s.label}</button>`;
    });

    // Sort toggles
    html += `<span class="props-toolbar-label props-sort-label">Sort:</span>
        <button class="props-sort-btn ${_propsSort === 'edge' ? 'active' : ''}" onclick="setPropsSort('edge')">Best Edge</button>
        <button class="props-sort-btn ${_propsSort === 'odds' ? 'active' : ''}" onclick="setPropsSort('odds')">Best Odds</button>
    </div>`;

    if (rows.length === 0) {
        html += '<div class="empty-state"><div class="empty-icon"><i class="fa-solid fa-magnifying-glass"></i></div><h3 class="empty-title">No props match your filters</h3><p class="empty-text">Try enabling more markets or switching the Over/Under side.</p></div>';
        container.innerHTML = html;
        return;
    }

    html += '<div class="props-table">';
    rows.forEach(p => {
        const edge = p.edge;
        const edgePct = (edge * 100).toFixed(1);
        const edgeSign = edge >= 0 ? '+' : '';
        const recClass = p.isOver ? 'prop-rec-over' : 'prop-rec-under';
        const rowClass = edge >= 0.05 ? 'edge-strong' : edge >= 0.02 ? 'edge-good' : 'edge-slight';
        const price = p.recPrice != null ? formatAmerican(p.recPrice) : '-';
        const prob = p.probOver.toFixed(1);
        const implied = p.impliedProb != null ? p.impliedProb.toFixed(1) : null;
        const teamAbbr = p.player_team || '';
        const matchup = p.matchup || '—';

        html += `<div class="props-row ${rowClass}">
            <div class="props-cell props-player">
                <div class="props-player-header">
                    ${teamAbbr ? `<img class="props-team-logo" src="/api/logos/${teamAbbr}.png" alt="${teamAbbr}" onerror="this.style.display='none'">` : ''}
                    <div class="props-name">${escapeHtml(p.player)}</div>
                </div>
                <div class="props-game">${escapeHtml(matchup)}</div>
            </div>
            <div class="props-cell props-market">
                <div class="props-market-name">${escapeHtml(p.market)}</div>
                <div class="props-line">${p.isOver ? 'Over' : 'Under'} ${parseFloat(p.line).toFixed(1)}</div>
            </div>
            <div class="props-cell props-rec-price">
                <div class="props-rec-badge ${recClass}">${p.isOver ? 'Over' : 'Under'}</div>
                <div class="props-price-val">${escapeHtml(price)}</div>
            </div>
            <div class="props-cell props-prob">
                <div class="props-probs">
                    <div class="props-prob-bar">
                        <div class="props-prob-track"><div class="props-prob-fill props-implied-fill" style="width:${implied || 0}%"></div></div>
                        <span class="props-prob-text">Book ${implied != null ? implied + '%' : '—'}</span>
                    </div>
                    <div class="props-prob-bar">
                        <div class="props-prob-track"><div class="props-prob-fill props-model-fill" style="width:${prob}%"></div></div>
                        <span class="props-prob-text">Model ${prob}%</span>
                    </div>
                </div>
            </div>
            <div class="props-cell props-edge">
                <div class="props-edge-badge">${edgeSign}${edgePct}%</div>
            </div>
        </div>`;
    });
    html += '</div>';

    container.innerHTML = html;
}

function formatAmerican(n) {
    if (n === null || n === undefined || n === '') return '-';
    let v = parseFloat(n);
    if (!isFinite(v) || isNaN(v)) return '-';
    // If it looks like an American price (integer and |v| ≥ 100), return as-is.
    if (Number.isInteger(v) && Math.abs(v) >= 100) {
        return v > 0 ? `+${v}` : `${v}`;
    }
    // Otherwise treat it as decimal odds (typical range ~1.2 – 3.5).
    if (v <= 1.0) return '-';
    const american = v >= 2.0 ? Math.round((v - 1.0) * 100.0) : Math.round(-100.0 / (v - 1.0));
    return american > 0 ? `+${american}` : `${american}`;
}

// ── Betting Edge Tab ─────────────────────────────────────────────
let _lastBettingEdgeData = null;
let _bettingEdgeSort = 'edge';
let _bettingEdgeIsDemo = false;
let _bettingEdgeDemoReason = null;

// ── Player Props Tab ─────────────────────────────────────────────
let _lastPropsData = null;
let _propsSort = 'edge';
let _propsIsDemo = false;
let _propsDemoReason = null;

// Default filters: show Over props for Points, Goals, and Shots; Assists off.
let _propsMarketFilter = {
    'Points': true,
    'Goals': true,
    'Assists': false,
    'Shots': true,
};
let _propsSideFilter = 'Over'; // 'Over' | 'Under' | 'both'

async function runBettingEdge() {
    const container = document.getElementById('bettingEdgeResults');
    if (!container) return;
    container.innerHTML = '<div class="loading"><div class="spinner"></div><span>Crunching model probabilities and odds...</span></div>';

    const date = document.getElementById('bettingEdgeDate')?.value || new Date().toISOString().split('T')[0];

    async function loadDemo(reason) {
        console.warn(reason + ', using demo betting edge cache.');
        try {
            const demo = await safeFetchJson('/static/data/betting_edge_cache.json');
            _lastBettingEdgeData = demo;
            _bettingEdgeSort = 'edge';
            _bettingEdgeIsDemo = true;
            _bettingEdgeDemoReason = reason;
            renderBettingEdge(demo, container);
        } catch (demoErr) {
            _lastBettingEdgeData = null;
            container.innerHTML = `<div class="error-box">Could not load betting edge: ${escapeHtml(e.message)}. Demo cache also failed to load: ${escapeHtml(demoErr.message)}.</div>`;
            console.error('Betting edge demo load failed:', e, demoErr);
        }
    }

    try {
        const data = await safeFetchJson(`/api/betting-edge?date=${encodeURIComponent(date)}`);
        if (data.error) {
            await loadDemo('Betting Edge API returned error: ' + data.error);
            return;
        }
        _lastBettingEdgeData = data;
        _bettingEdgeSort = 'edge';
        _bettingEdgeIsDemo = false;
        _bettingEdgeDemoReason = null;
        renderBettingEdge(data, container);
    } catch (e) {
        await loadDemo('Betting Edge API unavailable: ' + e.message);
    }
}

function setBettingEdgeSort(sort) {
    if (!_lastBettingEdgeData) return;
    _bettingEdgeSort = sort;
    const container = document.getElementById('bettingEdgeResults');
    if (container) renderBettingEdge(_lastBettingEdgeData, container);
}

function renderBettingEdge(data, container) {
    const games = data.games || [];
    if (!games.length) {
        container.innerHTML = `<div class="empty-state">
            <div class="empty-icon"><i class="fa-solid fa-bullseye"></i></div>
            <h3 class="empty-title">No value bets</h3>
            <p class="empty-desc">No market edges above the 3% threshold for ${escapeHtml(data.date)}. Try a different date or check back after the next odds update.</p>
        </div>`;
        return;
    }

    const rows = [];
    games.forEach(g => {
        const homeAbbr = g.home;
        const awayAbbr = g.away;
        const homeName = g.home_name || getTeamName(homeAbbr) || homeAbbr;
        const awayName = g.away_name || getTeamName(awayAbbr) || awayAbbr;
        g.edges.forEach(e => {
            const oddsDecimal = parseFloat(e.odds_decimal) || 0;
            rows.push({
                homeAbbr, awayAbbr, homeName, awayName,
                edge: parseFloat(e.edge) || 0,
                oddsDecimal,
                oddsAmerican: e.odds,
                e,
            });
        });
    });

    if (_bettingEdgeSort === 'odds') {
        rows.sort((a, b) => b.oddsDecimal - a.oddsDecimal);
    } else {
        rows.sort((a, b) => b.edge - a.edge);
    }

    let html = '';
    if (data.warning) {
        html += `<div class="betting-edge-warning"><i class="fa-solid fa-triangle-exclamation"></i> ${escapeHtml(data.warning)}</div>`;
    }
    if (_bettingEdgeIsDemo) {
        html += `<div class="demo-notice">
            <i class="fa-solid fa-tower-broadcast"></i> Showing sample value bets because live edges are unavailable${_bettingEdgeDemoReason ? ': ' + escapeHtml(_bettingEdgeDemoReason) : ''}.
            Set <code>ODDS_API_KEY</code> for real odds.
        </div>`;
    }

    html += `<div class="betting-edge-toolbar">
        <span class="be-toolbar-label">Sort by:</span>
        <button class="be-sort-btn ${_bettingEdgeSort === 'edge' ? 'active' : ''}" onclick="setBettingEdgeSort('edge')">Best Edge</button>
        <button class="be-sort-btn ${_bettingEdgeSort === 'odds' ? 'active' : ''}" onclick="setBettingEdgeSort('odds')">Best Odds</button>
    </div>`;

    html += '<div class="betting-edge-table">';
    rows.forEach(r => {
        const e = r.e;
        const edge = r.edge;
        const edgePct = (edge * 100).toFixed(1);
        const edgeSign = edge > 0 ? '+' : '';
        const implied = (parseFloat(e.implied_prob) * 100).toFixed(1);
        const model = (parseFloat(e.model_prob) * 100).toFixed(1);
        const odds = e.odds != null ? formatAmerican(e.odds) : '-';
        const pick = e.pick || e.side || '-';
        const teamTag = e.team || (e.market.toLowerCase().startsWith('total') ? 'Total' : (r.homeName === pick || pick.includes(r.homeName) ? r.homeAbbr : r.awayAbbr));
        const cardClass = edge >= 0.05 ? 'edge-strong' : edge >= 0.03 ? 'edge-good' : 'edge-slight';

        html += `<div class="be-card ${cardClass}">
            <div class="be-card-top">
                <div class="be-matchup">
                    <div class="be-team-pill">
                        <img class="be-row-logo" src="/api/logos/${r.awayAbbr}.png" alt="${r.awayAbbr}" onerror="this.style.display='none'">
                        <span class="be-row-abbr">${r.awayAbbr}</span>
                        <span class="be-at">@</span>
                        <img class="be-row-logo" src="/api/logos/${r.homeAbbr}.png" alt="${r.homeAbbr}" onerror="this.style.display='none'">
                        <span class="be-row-abbr">${r.homeAbbr}</span>
                    </div>
                </div>
                <div class="be-edge-badge">${edgeSign}${edgePct}%</div>
            </div>
            <div class="be-card-top">
                <div class="be-play-meta">
                    <div class="be-play-market">${escapeHtml(e.market)}</div>
                    <div class="be-play-pick">${escapeHtml(pick)} <span class="be-team-tag">${escapeHtml(teamTag)}</span></div>
                </div>
                <div class="be-odds-price">
                    <div class="be-price-val">${escapeHtml(odds)}</div>
                    <div class="be-price-label">odds</div>
                </div>
            </div>
            <div class="be-probs">
                <div class="be-prob-bar">
                    <div class="be-prob-track"><div class="be-prob-fill" style="width:${implied}%"></div></div>
                    <span class="be-prob-text">Implied ${implied}%</span>
                </div>
                <div class="be-prob-bar">
                    <div class="be-prob-track"><div class="be-prob-fill be-model-fill" style="width:${model}%"></div></div>
                    <span class="be-prob-text">Model ${model}%</span>
                </div>
            </div>
        </div>`;
    });
    html += '</div>';

    container.innerHTML = html;
}

// ── Real API stubs (unused in demo) ───────────────────────────────
async function loadAppState() {
    const dot = document.querySelector('.status-dot');
    const txt = document.querySelector('.status-label');
    try {
        const data = await safeFetchJson('/api/state');
        if (data.state?.is_fallback) { dot.className='status-dot error'; txt.textContent='Fallback'; }
        else if (data.state?.ml_model_trained) { dot.className='status-dot ready'; txt.textContent='Ready'; }
        else { dot.className='status-dot loading'; txt.textContent='Loading...'; }
    } catch (e) {
        console.error('State load failed:', e);
        dot.className='status-dot error';
        txt.textContent='Offline';
    }
}

async function loadTeams() {
    try {
        const data = await safeFetchJson('/api/teams');
        window.TEAMS_API = data.divisions || {};
        populateTeams();
    } catch (e) { console.error('Teams load failed:', e); }
}

function getTeamsData() {
    if (typeof window !== 'undefined' && window.TEAMS_API) return window.TEAMS_API;
    return TEAMS_FALLBACK;
}

async function loadSeasons() {
    try {
        const data = await safeFetchJson('/api/seasons');
        const sel = document.getElementById('statsSeason');
        const currentSeason = currentNHLSeasonKey();
        sel.innerHTML = '';
        (data.seasons || []).forEach(s => {
            const opt = new Option(s.label, s.key);
            if (s.has_data === false) {
                opt.text = `${s.label} (no data)`;
                opt.disabled = true;
            }
            sel.add(opt);
        });
        if (data.seasons?.some(s => s.key === currentSeason && s.has_data !== false)) {
            sel.value = currentSeason;
        } else if (data.seasons?.length) {
            // Fall back to the most recent season that actually has data.
            const latest = [...data.seasons].reverse().find(s => s.has_data !== false);
            if (latest) sel.value = latest.key;
        }
    } catch (e) { console.error('Seasons load failed:', e); }
}

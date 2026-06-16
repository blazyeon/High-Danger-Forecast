/**
 * NHL Game Predictor — Frontend v3.2
 * Premium dark SPA with offline demo mode, simulation logs, and live API integration.
 */

// ── Demo Mode Detection ───────────────────────────────────────────
const USE_DEMO = false; // Set true for offline standalone mode

// ── Team Data ──────────────────────────────────────────────────────
const TEAMS = {
    Atlantic: [
        { abbr:"BOS", name:"Boston Bruins" },
        { abbr:"BUF", name:"Buffalo Sabres" },
        { abbr:"DET", name:"Detroit Red Wings" },
        { abbr:"FLA", name:"Florida Panthers" },
        { abbr:"MTL", name:"Montreal Canadiens" },
        { abbr:"OTT", name:"Ottawa Senators" },
        { abbr:"TBL", name:"Tampa Bay Lightning" },
        { abbr:"TOR", name:"Toronto Maple Leafs" },
    ],
    Metropolitan: [
        { abbr:"CAR", name:"Carolina Hurricanes" },
        { abbr:"CBJ", name:"Columbus Blue Jackets" },
        { abbr:"NJD", name:"New Jersey Devils" },
        { abbr:"NYI", name:"New York Islanders" },
        { abbr:"NYR", name:"New York Rangers" },
        { abbr:"PHI", name:"Philadelphia Flyers" },
        { abbr:"PIT", name:"Pittsburgh Penguins" },
        { abbr:"WSH", name:"Washington Capitals" },
    ],
    Central: [
        { abbr:"CHI", name:"Chicago Blackhawks" },
        { abbr:"COL", name:"Colorado Avalanche" },
        { abbr:"DAL", name:"Dallas Stars" },
        { abbr:"MIN", name:"Minnesota Wild" },
        { abbr:"NSH", name:"Nashville Predators" },
        { abbr:"STL", name:"St. Louis Blues" },
        { abbr:"UTA", name:"Utah Mammoth" },
        { abbr:"WPG", name:"Winnipeg Jets" },
    ],
    Pacific: [
        { abbr:"ANA", name:"Anaheim Ducks" },
        { abbr:"CGY", name:"Calgary Flames" },
        { abbr:"EDM", name:"Edmonton Oilers" },
        { abbr:"LAK", name:"Los Angeles Kings" },
        { abbr:"SJS", name:"San Jose Sharks" },
        { abbr:"SEA", name:"Seattle Kraken" },
        { abbr:"VAN", name:"Vancouver Canucks" },
        { abbr:"VGK", name:"Vegas Golden Knights" },
    ],
};

// Realistic Elo ratings (end-of-2024-25 season approximation)
// Best teams ~1650, worst ~1380. League average = 1500.
const TEAM_ELO = {
    COL: 1650, WPG: 1640, DAL: 1600, TOR: 1590, FLA: 1580,
    CAR: 1570, EDM: 1560, NYR: 1550, BOS: 1540, VGK: 1535,
    TBL: 1530, LAK: 1520, MIN: 1515, NJD: 1510, WSH: 1505,
    NSH: 1500, CGY: 1490, STL: 1485, PIT: 1480, PHI: 1475,
    SEA: 1470, DET: 1460, BUF: 1450, OTT: 1445, NYI: 1440,
    MTL: 1430, CBJ: 1420, CHI: 1410, UTA: 1400, ANA: 1390,
    VAN: 1380, SJS: 1360,
};

const HOME_ADVANTAGE_ELO = 45; // ~56% win prob for evenly matched teams at home

// Expected goals model: base 3.2 + (elo - 1500) * 0.003
// So a 1650 team scores ~3.65, a 1380 team scores ~2.54
function getBaseXg(elo) {
    return Math.max(1.8, 3.2 + (elo - 1500) * 0.0035);
}

// Fallback goalie names used only when the live roster API is unreachable.
// The UI normally populates this from /api/goalies/<team>/<date>.
const GOALIE_POOL = {
    ANA: ["L. Dostal", "J. Gibson"],
    BOS: ["J. Swayman", "J. Korpisalo"],
    BUF: ["U. Luukkonen", "D. Levi"],
    CGY: ["D. Vladar", "D. Wolf"],
    CAR: ["F. Andersen", "P. Kochetkov"],
    CHI: ["P. Mrazek", "A. Soderblom"],
    CBJ: ["E. Merzlikins", "D. Tarasov"],
    COL: ["J. Blackwood", "S. Wedgewood"],
    DAL: ["J. Oettinger", "C. Wedgewood"],
    DET: ["C. Talbot", "A. Lyon"],
    EDM: ["S. Skinner", "C. Pickard"],
    FLA: ["S. Bobrovsky", "A. Lyon"],
    LAK: ["D. Kuemper", "E. Portillo"],
    MIN: ["M. Fleury", "J. Wallstedt"],
    MTL: ["S. Montembeault", "C. Primeau"],
    NSH: ["J. Saros", "Y. Askarov"],
    NJD: ["J. Allen", "J. Markstrom"],
    NYI: ["I. Sorokin", "S. Varlamov"],
    NYR: ["I. Shesterkin", "J. Quick"],
    OTT: ["L. Ullmark", "A. Forsberg"],
    PHI: ["S. Ersson", "A. Kolosov"],
    PIT: ["T. Jarry", "A. Nedeljkovic"],
    SEA: ["J. Daccord", "P. Grubauer"],
    SJS: ["V. Vanecek", "Y. Askarov"],
    STL: ["J. Binnington", "J. Hofer"],
    TBL: ["A. Vasilevskiy", "J. Hlinka"],
    TOR: ["J. Woll", "A. Stolarz"],
    UTA: ["K. Vejmelka", "C. Ingram"],
    VAN: ["T. Demko", "K. Lankinen"],
    VGK: ["A. Hill", "I. Samsonov"],
    WSH: ["C. Lindgren", "L. Thompson"],
    WPG: ["C. Hellebuyck", "E. Comrie"],
};

function getTeamElo(abbr) {
    return TEAM_ELO[abbr] || 1500;
}

function computeWinProb(homeElo, awayElo) {
    const diff = (homeElo + HOME_ADVANTAGE_ELO) - awayElo;
    return 1.0 / (1.0 + Math.pow(10, -diff / 400.0));
}

function getMockResult(home, away, opts={}) {
    const homeElo = getTeamElo(home);
    const awayElo = getTeamElo(away);

    // Base win probability from Elo
    const baseHomeWinProb = computeWinProb(homeElo, awayElo);

    // Monte Carlo adds some variance (±8%)
    const simVariance = (Math.random() - 0.5) * 0.16;
    let simHomeWinProb = Math.max(0.05, Math.min(0.95, baseHomeWinProb + simVariance));

    // Ensemble: 35% Elo + 65% simulation (matches backend logic)
    const ensembleHomeWinProb = 0.35 * baseHomeWinProb + 0.65 * simHomeWinProb;
    const clampedHomeProb = Math.max(0.05, Math.min(0.95, ensembleHomeWinProb));

    const homeWinPct = (clampedHomeProb * 100);
    const awayWinPct = ((1 - clampedHomeProb) * 100);

    // Expected goals
    let homeXg = getBaseXg(homeElo) * 1.05; // home ice boost
    let awayXg = getBaseXg(awayElo) * 0.97; // road penalty

    // Confidence based on rating gap (bigger gap = higher confidence)
    const eloGap = Math.abs(homeElo - awayElo);
    const confidence = Math.min(0.95, 0.55 + eloGap / 800);

    // Most likely score = rounded expected goals
    const modeHome = Math.round(homeXg);
    const modeAway = Math.round(awayXg);

    // Totals distribution centered around (homeXg + awayXg)
    const totalMean = homeXg + awayXg;
    const totals = {};
    for (let t = 3; t <= 10; t++) {
        const dist = Math.exp(-Math.pow(t - totalMean, 2) / 4);
        totals[t] = Math.round(dist * 1000);
    }

    // Back-to-back adjustments
    const b2bFactor = 0.92;
    let finalHomeWin = homeWinPct;
    let finalAwayWin = awayWinPct;
    let finalHomeXg = homeXg;
    let finalAwayXg = awayXg;

    if (opts.homeB2B) {
        finalHomeWin = homeWinPct * b2bFactor;
        finalAwayWin = 100 - finalHomeWin;
        finalHomeXg = homeXg * 0.95;
    }
    if (opts.awayB2B) {
        finalAwayWin = awayWinPct * b2bFactor;
        finalHomeWin = 100 - finalAwayWin;
        finalAwayXg = awayXg * 0.95;
    }
    // If both are B2B, apply both but cap at reasonable bounds
    if (opts.homeB2B && opts.awayB2B) {
        finalHomeWin = homeWinPct * b2bFactor;
        finalAwayWin = awayWinPct * b2bFactor;
        const total = finalHomeWin + finalAwayWin;
        finalHomeWin = (finalHomeWin / total) * 100;
        finalAwayWin = (finalAwayWin / total) * 100;
        finalHomeXg = homeXg * 0.95;
        finalAwayXg = awayXg * 0.95;
    }

    return {
        home_win_pct: finalHomeWin.toFixed(1),
        away_win_pct: finalAwayWin.toFixed(1),
        home_elo_adj: Math.round(homeElo + HOME_ADVANTAGE_ELO),
        away_elo_adj: Math.round(awayElo),
        exp_home_goals: finalHomeXg.toFixed(2),
        exp_away_goals: finalAwayXg.toFixed(2),
        home_win_2plus_pct: (finalHomeWin * 0.45).toFixed(1),
        away_win_2plus_pct: (finalAwayWin * 0.38).toFixed(1),
        mode_home_goals: modeHome,
        mode_away_goals: modeAway,
        totals_distribution: totals,
        reg_games_pct: (74 + (eloGap / 100)).toFixed(1),
        ot_games_pct: (16 - (eloGap / 200)).toFixed(1),
        so_games_pct: (10 - (eloGap / 300)).toFixed(1),
        confidence: confidence.toFixed(2),
        breakdown: {
            HOME: {"Base xGF": getBaseXg(homeElo).toFixed(2), "Home Ice": "+5%", "Elo": homeElo},
            AWAY: {"Base xGF": getBaseXg(awayElo).toFixed(2), "Road Pen": "-3%", "Elo": awayElo}
        }
    };
}

// ── Init ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initDateDefaults();
    initDatePickers();
    initSettingsToggle();
    populateTeams();
    initSeasons();
    setupEventListeners();
    document.getElementById('modalClose').addEventListener('click', () => {
        document.getElementById('gameModal').style.display = 'none';
    });
    document.getElementById('gameModal').addEventListener('click', (e) => {
        if (e.target.id === 'gameModal') document.getElementById('gameModal').style.display = 'none';
    });
    if (USE_DEMO) {
        document.getElementById('statusPill').innerHTML =
            '<span class="status-dot ready"></span><span class="status-label">Demo Mode</span>';
        const sp = document.querySelector('.status-pill');
        if (sp) { sp.style.background = 'rgba(255,209,102,0.08)'; sp.style.borderColor = 'rgba(255,209,102,0.18)'; sp.style.color = 'var(--accent-gold)'; }
        document.getElementById('homeTeam').value = 'TOR';
        document.getElementById('awayTeam').value = 'NYR';
        updateLogos();
        updatePredictBtn();
        populateGoalies();
    } else {
        loadAppState();
        loadTeams();
        loadSeasons();
    }
});

function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
            if (btn.dataset.tab === 'elo') runElo();
        });
    });
}

function initDateDefaults() {
    const today = new Date().toISOString().split('T')[0];
    const lookupDate = document.getElementById('lookupDate');
    if (lookupDate) lookupDate.value = today;
    const propsDate = document.getElementById('propsDate');
    if (propsDate) propsDate.value = today;
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
        this.input.classList.add('date-picker-input');

        this.container = document.createElement('div');
        this.container.className = 'date-picker';
        this.input.parentNode.insertBefore(this.container, this.input);
        this.container.appendChild(this.input);

        this.popup = document.createElement('div');
        this.popup.className = 'date-picker-popup';
        this.popup.setAttribute('role', 'dialog');
        this.popup.setAttribute('aria-modal', 'true');
        this.container.appendChild(this.popup);

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
            daysHtml += `<button type="button" class="${cls}" data-ymd="${ymd}" data-day="${d}">${d}</button>`;
        }

        this.popup.innerHTML = `
            <div class="dp-header">
                <button type="button" class="dp-nav dp-prev" aria-label="Previous month"><i class="fa-solid fa-chevron-left"></i></button>
                <div class="dp-title">${monthNames[month]} ${year}</div>
                <button type="button" class="dp-nav dp-next" aria-label="Next month"><i class="fa-solid fa-chevron-right"></i></button>
            </div>
            <div class="dp-weekdays">
                ${dayLabels.map(l => `<div class="dp-weekday">${l}</div>`).join('')}
            </div>
            <div class="dp-days">${daysHtml}</div>
            <div class="dp-footer">
                <button type="button" class="dp-today">Today</button>
            </div>
        `;

        this.popup.querySelector('.dp-prev').addEventListener('click', () => this._changeMonth(-1));
        this.popup.querySelector('.dp-next').addEventListener('click', () => this._changeMonth(1));
        this.popup.querySelector('.dp-today').addEventListener('click', () => this._selectToday());
        this.popup.querySelectorAll('.dp-day').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const day = parseInt(e.currentTarget.dataset.day, 10);
                this._selectDate(new Date(year, month, day));
            });
        });
    }

    _bind() {
        this.input.addEventListener('click', (e) => {
            e.preventDefault();
            this.open();
        });

        this.input.addEventListener('focus', (e) => {
            e.preventDefault();
            this.open();
        });

        this.input.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.close();
            }
        });

        document.addEventListener('click', (e) => {
            if (this.isOpen && !this.container.contains(e.target)) {
                this.close();
            }
        });
    }

    _changeMonth(delta) {
        this.viewDate.setMonth(this.viewDate.getMonth() + delta);
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
        const popupHeight = this.popup.offsetHeight || 360;
        const spaceBelow = window.innerHeight - rect.bottom;
        const spaceAbove = rect.top;

        this.popup.style.left = '0';
        if (spaceBelow < popupHeight && spaceAbove > popupHeight) {
            this.popup.style.top = 'auto';
            this.popup.style.bottom = '100%';
            this.popup.style.marginBottom = '8px';
            this.popup.style.marginTop = '0';
        } else {
            this.popup.style.top = '100%';
            this.popup.style.bottom = 'auto';
            this.popup.style.marginTop = '8px';
            this.popup.style.marginBottom = '0';
        }
    }
}

function initDatePickers() {
    document.querySelectorAll('input[type="date"]').forEach(input => {
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
    for (const [div, teams] of Object.entries(TEAMS)) {
        const homeGroup = document.createElement('optgroup');
        homeGroup.label = div;
        const awayGroup = document.createElement('optgroup');
        awayGroup.label = div;
        teams.forEach(t => {
            homeGroup.appendChild(new Option(`${t.name} (${t.abbr})`, t.abbr));
            awayGroup.appendChild(new Option(`${t.name} (${t.abbr})`, t.abbr));
        });
        homeSel.appendChild(homeGroup);
        awaySel.appendChild(awayGroup);
    }
}

function initSeasons() {
    const sel = document.getElementById('statsSeason');
    if (!sel) return;
    sel.innerHTML = '';
    const seasons = [
        { key: '20252026', label: '2025-26' },
        { key: '20242025', label: '2024-25' },
        { key: '20232024', label: '2023-24' },
        { key: '20222023', label: '2022-23' },
        { key: '20212022', label: '2021-22' },
        { key: '20202021', label: '2020-21' },
        { key: '20192020', label: '2019-20' },
    ];
    seasons.forEach(s => sel.add(new Option(s.label, s.key)));
    sel.value = '20252026';
}

async function populateGoalies() {
    const home = document.getElementById('homeTeam').value;
    const away = document.getElementById('awayTeam').value;
    const hRow = document.getElementById('homeGoalieRow');
    const aRow = document.getElementById('awayGoalieRow');
    const hSel = document.getElementById('homeGoalie');
    const aSel = document.getElementById('awayGoalie');
    const dateStr = new Date().toISOString().split('T')[0];

    hSel.innerHTML = '<option value="">Auto-select</option>';
    aSel.innerHTML = '<option value="">Auto-select</option>';
    hRow.style.display = home ? 'block' : 'none';
    aRow.style.display = away ? 'block' : 'none';
    if (!home && !away) return;

    const fill = (sel, list) => {
        if (!list || !list.length) return false;
        list.forEach(g => sel.add(new Option(g, g)));
        return true;
    };

    if (home) {
        let live = [];
        if (!USE_DEMO) {
            try { live = (await safeFetchJson(`/api/goalies/${home}/${dateStr}`)).goalies || []; }
            catch (e) { console.warn(`Goalie API failed for ${home}:`, e); }
        }
        if (!fill(hSel, live.length ? live : GOALIE_POOL[home])) hRow.style.display = 'none';
    }
    if (away) {
        let live = [];
        if (!USE_DEMO) {
            try { live = (await safeFetchJson(`/api/goalies/${away}/${dateStr}`)).goalies || []; }
            catch (e) { console.warn(`Goalie API failed for ${away}:`, e); }
        }
        if (!fill(aSel, live.length ? live : GOALIE_POOL[away])) aRow.style.display = 'none';
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
    document.getElementById('predictBtn').addEventListener('click', runPrediction);
    document.getElementById('lookupBtn').addEventListener('click', runLookup);
    document.getElementById('statsBtn').addEventListener('click', runStats);
    document.getElementById('propsBtn').addEventListener('click', runProps);
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
    if (USE_DEMO) {
        const hElo = getTeamElo(home);
        const aElo = getTeamElo(away);
        const baseProb = computeWinProb(hElo, aElo);

        logStep('INIT', `Matchup: ${getTeamName(home)} (HOME) vs ${getTeamName(away)} (AWAY)`);
        await delay(120);
        logStep('ELO', `${home} rating: ${hElo} | ${away} rating: ${aElo} | Home adv: +${HOME_ADVANTAGE_ELO}`);
        await delay(150);
        logStep('PROB', `Base Elo win prob: ${(baseProb*100).toFixed(1)}% home / ${((1-baseProb)*100).toFixed(1)}% away`);
        await delay(100);
        logStep('GOALIE', `Home goalie: ${homeGoalie || 'Auto-select'} | Away goalie: ${awayGoalie || 'Auto-select'}`);
        await delay(100);
        if (homeB2B) logStep('B2B', `${home} flagged as back-to-back (~8% fatigue penalty)`);
        if (awayB2B) logStep('B2B', `${away} flagged as back-to-back (~8% fatigue penalty)`);
        await delay(100);
        logStep('SIM', `Running ${document.getElementById('sims').value || 10000} Monte Carlo iterations`);
        await delay(250);
        logStep('ENSEMBLE', `Blending Elo (35%) + simulation (65%) outcomes`);
        await delay(100);

        data = getMockResult(home, away, { homeB2B, awayB2B });
        logStep('DONE', `Prediction complete. Confidence: ${data.confidence}`);
    } else {
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
            if (data.error) { content.innerHTML = `<div class="error-box">${data.error}</div>`; btn.disabled = false; return; }
        } catch (e) {
            console.error('Prediction failed:', e);
            content.innerHTML = `<div class="error-box">Prediction failed: ${e.message}</div>`;
            btn.disabled = false;
            return;
        }
    }

    renderResults(data, home, away);
    renderLog();
    btn.disabled = false;
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

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
        <div class="result-team-name">${homeName}</div>
        <div class="result-team-abbr">${homeAbbr}</div>
    </div>`;
    html += `<div class="result-center">
        <div class="result-prediction-label">Prediction</div>
        <div class="result-prediction-value ${homeWin ? 'home-win' : 'away-win'}">${homeWin ? 'HOME WIN' : 'AWAY WIN'}</div>
    </div>`;
    html += `<div class="result-team-block">
        <div class="result-badge away">AWAY</div>
        <img class="result-team-logo" src="/api/logos/${awayAbbr}.png" alt="${awayName}" onerror="this.style.display='none'">
        <div class="result-team-name">${awayName}</div>
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
        { label: 'Reg Home Win', value: (hPct * 0.7).toFixed(1) + '%' },
        { label: 'Reg Away Win', value: (aPct * 0.7).toFixed(1) + '%' },
        { label: 'OT %', value: (sim.ot_games_pct || 16).toFixed(1) + '%' },
        { label: 'Most Likely', value: `${sim.mode_home_goals}-${sim.mode_away_goals}`, cls: 'gold' },
    ];
    stats.forEach(s => {
        html += `<div class="stat-card"><div class="stat-value ${s.cls || ''}">${s.value}</div><div class="stat-label">${s.label}</div></div>`;
    });
    html += `</div>`;

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
    for (const div of Object.values(TEAMS)) {
        const t = div.find(x => x.abbr === abbr);
        if (t) return t.name;
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
        container.innerHTML = `<div class="empty-state"><div class="empty-icon">📅</div><h3 class="empty-title">No Games</h3><p class="empty-desc">No NHL games scheduled for ${date}.</p></div>`;
        return;
    }

    let html = '';

    games.forEach(g => {
        const state = (g.state || 'FUT').toString().toUpperCase();
        let stateLabel = 'Scheduled';
        let stateClass = ' upcoming';
        let scoreHtml = '';

        if (state === 'LIVE' || state === 'CRIT') {
            stateLabel = '🔴 Live';
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
        let statusText = state === 'LIVE' || state === 'CRIT' ? `🔴 LIVE ${period}${clock}` : (state === 'OFF' || state === 'FINAL' ? 'FINAL' : 'UPCOMING');
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
        body.innerHTML = `<div class="error-box">Could not load game details.<br><small>${e.message}</small></div>`;
    }
}

// ── Analytics Tab (NST-Style Advanced Stats) ────────────────────
// Fallback hardcoded data used when JSON files are not yet scraped
const FALLBACK_TEAM_STATS = [
    { team:'WPG', gp:82, w:56, l:20, otl:6, pts:118, cf:54.2, ff:53.8, sf:52.9, xgf:55.1, scf:54.3, hdsf:53.2, sh:9.2, sv:0.918, pdo:101.0 },
    { team:'TOR', gp:82, w:52, l:24, otl:6, pts:110, cf:53.1, ff:52.5, sf:51.8, xgf:53.2, scf:52.8, hdsf:51.5, sh:10.1, sv:0.912, pdo:101.3 },
    { team:'DAL', gp:82, w:51, l:26, otl:5, pts:107, cf:52.8, ff:52.2, sf:51.5, xgf:52.9, scf:52.1, hdsf:50.8, sh:9.8, sv:0.915, pdo:101.3 },
    { team:'FLA', gp:82, w:50, l:26, otl:6, pts:106, cf:52.5, ff:51.9, sf:51.2, xgf:52.4, scf:51.7, hdsf:50.2, sh:9.5, sv:0.914, pdo:100.9 },
    { team:'CAR', gp:82, w:49, l:27, otl:6, pts:104, cf:55.8, ff:55.2, sf:54.6, xgf:56.2, scf:55.4, hdsf:54.1, sh:8.9, sv:0.916, pdo:100.5 },
    { team:'COL', gp:82, w:49, l:28, otl:5, pts:103, cf:53.5, ff:52.9, sf:52.1, xgf:53.8, scf:53.0, hdsf:51.9, sh:10.5, sv:0.910, pdo:101.5 },
    { team:'EDM', gp:82, w:48, l:28, otl:6, pts:102, cf:53.2, ff:52.6, sf:51.8, xgf:53.5, scf:52.7, hdsf:51.4, sh:10.8, sv:0.908, pdo:101.6 },
    { team:'NYR', gp:82, w:47, l:29, otl:6, pts:100, cf:50.5, ff:50.1, sf:49.6, xgf:50.8, scf:50.2, hdsf:49.5, sh:10.2, sv:0.911, pdo:101.3 },
    { team:'VGK', gp:82, w:46, l:30, otl:6, pts:98,  cf:51.2, ff:50.6, sf:49.9, xgf:51.0, scf:50.4, hdsf:49.2, sh:9.6, sv:0.913, pdo:100.9 },
    { team:'BOS', gp:82, w:45, l:30, otl:7, pts:97,  cf:51.5, ff:50.9, sf:50.2, xgf:51.2, scf:50.6, hdsf:49.5, sh:9.2, sv:0.917, pdo:100.9 },
    { team:'TBL', gp:82, w:45, l:31, otl:6, pts:96,  cf:52.1, ff:51.5, sf:50.8, xgf:52.0, scf:51.3, hdsf:50.1, sh:9.4, sv:0.912, pdo:100.6 },
    { team:'LAK', gp:82, w:44, l:32, otl:6, pts:94,  cf:52.5, ff:51.9, sf:51.2, xgf:52.1, scf:51.5, hdsf:50.3, sh:8.8, sv:0.916, pdo:100.4 },
    { team:'MIN', gp:82, w:43, l:32, otl:7, pts:93,  cf:50.8, ff:50.2, sf:49.6, xgf:50.5, scf:49.9, hdsf:48.8, sh:9.1, sv:0.914, pdo:100.5 },
    { team:'WSH', gp:82, w:42, l:33, otl:7, pts:91,  cf:49.5, ff:49.0, sf:48.5, xgf:49.2, scf:48.6, hdsf:47.8, sh:9.5, sv:0.911, pdo:100.6 },
    { team:'NJD', gp:82, w:41, l:34, otl:7, pts:89,  cf:50.2, ff:49.6, sf:49.0, xgf:49.8, scf:49.2, hdsf:48.2, sh:9.8, sv:0.909, pdo:100.7 },
    { team:'NSH', gp:82, w:40, l:35, otl:7, pts:87,  cf:49.2, ff:48.6, sf:48.0, xgf:48.8, scf:48.2, hdsf:47.2, sh:9.3, sv:0.910, pdo:100.3 },
    { team:'CGY', gp:82, w:39, l:36, otl:7, pts:85,  cf:48.8, ff:48.2, sf:47.6, xgf:48.2, scf:47.6, hdsf:46.8, sh:9.0, sv:0.911, pdo:100.1 },
    { team:'STL', gp:82, w:38, l:37, otl:7, pts:83,  cf:48.5, ff:47.9, sf:47.3, xgf:47.9, scf:47.3, hdsf:46.5, sh:8.8, sv:0.912, pdo:100.0 },
    { team:'PIT', gp:82, w:37, l:38, otl:7, pts:81,  cf:48.2, ff:47.6, sf:47.0, xgf:47.5, scf:46.9, hdsf:46.0, sh:8.9, sv:0.909, pdo:99.8 },
    { team:'PHI', gp:82, w:36, l:39, otl:7, pts:79,  cf:47.5, ff:47.0, sf:46.4, xgf:46.8, scf:46.2, hdsf:45.4, sh:8.6, sv:0.910, pdo:99.6 },
    { team:'SEA', gp:82, w:36, l:40, otl:6, pts:78,  cf:47.8, ff:47.2, sf:46.6, xgf:47.0, scf:46.4, hdsf:45.6, sh:8.4, sv:0.911, pdo:99.5 },
    { team:'DET', gp:82, w:35, l:40, otl:7, pts:77,  cf:47.2, ff:46.6, sf:46.0, xgf:46.5, scf:45.9, hdsf:45.0, sh:8.8, sv:0.908, pdo:99.6 },
    { team:'BUF', gp:82, w:34, l:41, otl:7, pts:75,  cf:46.8, ff:46.2, sf:45.6, xgf:46.0, scf:45.4, hdsf:44.5, sh:9.1, sv:0.906, pdo:99.7 },
    { team:'OTT', gp:82, w:34, l:42, otl:6, pts:74,  cf:46.5, ff:45.9, sf:45.3, xgf:45.7, scf:45.1, hdsf:44.2, sh:9.0, sv:0.907, pdo:99.7 },
    { team:'NYI', gp:82, w:33, l:42, otl:7, pts:73,  cf:46.2, ff:45.6, sf:45.0, xgf:45.4, scf:44.8, hdsf:43.9, sh:8.5, sv:0.909, pdo:99.4 },
    { team:'MTL', gp:82, w:32, l:43, otl:7, pts:71,  cf:45.5, ff:44.9, sf:44.3, xgf:44.6, scf:44.0, hdsf:43.2, sh:8.6, sv:0.907, pdo:99.3 },
    { team:'CBJ', gp:82, w:31, l:44, otl:7, pts:69,  cf:45.2, ff:44.6, sf:44.0, xgf:44.2, scf:43.6, hdsf:42.8, sh:8.4, sv:0.908, pdo:99.2 },
    { team:'CHI', gp:82, w:30, l:45, otl:7, pts:67,  cf:44.5, ff:43.9, sf:43.3, xgf:43.5, scf:42.9, hdsf:42.0, sh:8.2, sv:0.906, pdo:98.8 },
    { team:'UTA', gp:82, w:29, l:46, otl:7, pts:65,  cf:44.2, ff:43.6, sf:43.0, xgf:43.2, scf:42.6, hdsf:41.8, sh:8.3, sv:0.905, pdo:98.8 },
    { team:'ANA', gp:82, w:28, l:47, otl:7, pts:63,  cf:43.5, ff:42.9, sf:42.3, xgf:42.4, scf:41.8, hdsf:41.0, sh:8.1, sv:0.905, pdo:98.6 },
    { team:'VAN', gp:82, w:27, l:48, otl:7, pts:61,  cf:43.2, ff:42.6, sf:42.0, xgf:42.0, scf:41.4, hdsf:40.6, sh:7.9, sv:0.904, pdo:98.3 },
    { team:'SJS', gp:82, w:24, l:51, otl:7, pts:55,  cf:42.0, ff:41.4, sf:40.8, xgf:40.5, scf:39.9, hdsf:39.1, sh:7.6, sv:0.902, pdo:97.8 },
];

const FALLBACK_SKATERS = [
    { name:'C. McDavid', team:'EDM', gp:82, g:52, a:78, pts:130, cf:58.2, xgf:59.5, hdcf:62.1, pdo:102.5 },
    { name:'N. MacKinnon', team:'COL', gp:80, g:49, a:72, pts:121, cf:57.8, xgf:58.9, hdcf:60.8, pdo:101.8 },
    { name:'A. Matthews', team:'TOR', gp:82, g:69, a:38, pts:107, cf:56.2, xgf:57.1, hdcf:58.5, pdo:102.2 },
    { name:'D. Pastrnak', team:'BOS', gp:82, g:47, a:55, pts:102, cf:55.8, xgf:56.5, hdcf:57.2, pdo:101.5 },
    { name:'M. Tkachuk', team:'FLA', gp:80, g:42, a:58, pts:100, cf:55.2, xgf:55.8, hdcf:56.9, pdo:101.2 },
    { name:'J. Robertson', team:'DAL', gp:82, g:46, a:48, pts:94,  cf:54.5, xgf:55.1, hdcf:55.8, pdo:100.8 },
    { name:'S. Crosby', team:'PIT', gp:82, g:42, a:52, pts:94,  cf:54.1, xgf:54.6, hdcf:55.2, pdo:100.5 },
    { name:'A. Panarin', team:'NYR', gp:82, g:35, a:58, pts:93,  cf:53.8, xgf:54.3, hdcf:54.9, pdo:100.9 },
    { name:'B. Point', team:'TBL', gp:82, g:48, a:44, pts:92,  cf:53.5, xgf:54.0, hdcf:54.6, pdo:100.7 },
    { name:'S. Reinhart', team:'FLA', gp:82, g:55, a:35, pts:90,  cf:53.2, xgf:53.7, hdcf:54.2, pdo:101.1 },
];

const FALLBACK_GOALIES = [
    { name:'C. Hellebuyck', team:'WPG', gp:62, w:45, sv:0.925, gaa:2.12, gsax:28.5, hdsv:0.852 },
    { name:'J. Oettinger', team:'DAL', gp:58, w:38, sv:0.918, gaa:2.28, gsax:22.1, hdsv:0.841 },
    { name:'S. Bobrovsky', team:'FLA', gp:60, w:39, sv:0.916, gaa:2.32, gsax:20.8, hdsv:0.838 },
    { name:'A. Vasilevskiy', team:'TBL', gp:56, w:36, sv:0.915, gaa:2.35, gsax:19.5, hdsv:0.835 },
    { name:'I. Shesterkin', team:'NYR', gp:58, w:35, sv:0.914, gaa:2.38, gsax:18.2, hdsv:0.833 },
    { name:'J. Swayman', team:'BOS', gp:50, w:32, sv:0.913, gaa:2.42, gsax:16.8, hdsv:0.830 },
    { name:'F. Andersen', team:'CAR', gp:52, w:34, sv:0.912, gaa:2.40, gsax:17.5, hdsv:0.831 },
    { name:'D. Kuemper', team:'LAK', gp:54, w:33, sv:0.911, gaa:2.45, gsax:15.2, hdsv:0.828 },
    { name:'J. Woll', team:'TOR', gp:48, w:31, sv:0.910, gaa:2.48, gsax:14.5, hdsv:0.825 },
    { name:'J. Blackwood', team:'COL', gp:50, w:30, sv:0.909, gaa:2.50, gsax:13.8, hdsv:0.824 },
];

async function loadStatsPayload(type) {
    const season = document.getElementById('statsSeason')?.value || '20252026';
    const endpoint = USE_DEMO
        ? `../static/data/pbp_${type}_stats.json`
        : `/api/stats/${type}?season=${season}&stype=2`;
    try {
        const payload = await safeFetchJson(endpoint, { cache: 'no-store' });
        return payload || {};
    } catch (e) {
        console.warn(`Stats ${type} API unavailable, using fallback`, e);
        const data =
            type === 'teams' ? FALLBACK_TEAM_STATS :
            type === 'skaters' ? FALLBACK_SKATERS :
            type === 'goalies' ? FALLBACK_GOALIES : [];
        return { data, meta: { source: 'fallback', updated_at: null } };
    }
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
    container.innerHTML = '<div class="loading"><div class="spinner"></div><span>Loading advanced stats...</span></div>';

    const { data, meta } = await loadStatsPayload(type);
    const source = meta?.source || 'unknown';
    const updatedAt = meta?.updated_at
        ? new Date(meta.updated_at).toLocaleString()
        : null;

    if (!data || !data.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">📊</div><h3 class="empty-title">No Data</h3><p class="empty-desc">Advanced stats are not available yet. Run <code>python update_pbp_stats.py</code> to populate them.</p></div>';
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
        html = _buildStatsTable(cols, sorted);
        notice = '📊 Team advanced metrics from PBP data (CF%, xGF%, HDCF%, PDO).';
    } else if (type === 'skaters') {
        const sorted = [...data].sort(
            (a, b) => (parseInt(b.points) || 0) - (parseInt(a.points) || 0)
        );
        const cols = [
            { key: 'name', label: 'Player', fmt: v => `<strong>${v}</strong>` },
            { key: 'gp', label: 'GP' },
            { key: 'goals', label: 'G' },
            { key: 'assists', label: 'A' },
            { key: 'points', label: 'Pts', fmt: v => `<strong>${v}</strong>` },
            { key: 'shots', label: 'SOG' },
            { key: 'gpg', label: 'G/GP', fmt: v => fmtNum(v, 2) },
            { key: 'apg', label: 'A/GP', fmt: v => fmtNum(v, 2) },
            { key: 'xgf_pg', label: 'xGF/GP', fmt: v => fmtNum(v, 2) },
        ];
        html = _buildStatsTable(cols, sorted);
        notice = '📊 Skater advanced metrics sorted by points.';
    } else if (type === 'goalies') {
        const sorted = [...data].sort(
            (a, b) => (parseFloat(b.gsax) || 0) - (parseFloat(a.gsax) || 0)
        );
        const cols = [
            { key: 'name', label: 'Player', fmt: v => `<strong>${v}</strong>` },
            { key: 'gp', label: 'GP' },
            { key: 'ga', label: 'GA' },
            { key: 'gaa', label: 'GAA', fmt: v => fmtNum(v, 2) },
            { key: 'sa', label: 'SA' },
            { key: 'sv_pct', label: 'SV%', fmt: v => fmtPct(v) },
            { key: 'gsax', label: 'GSAx', fmt: v => fmtNum(v, 1) },
            { key: 'gsax_per_60', label: 'GSAx/60', fmt: v => fmtNum(v, 2) },
        ];
        html = _buildStatsTable(cols, sorted);
        notice = '📊 Goalie advanced stats sorted by Goals Saved Above Expected (GSAx).';
    }

    const sourceBadge = source === 'cache'
        ? '<span class="stats-source cached">Cached</span>'
        : `<span class="stats-source computed">${source}</span>`;
    const updatedText = updatedAt ? `Last updated: ${updatedAt}` : 'Live computation';
    html += `<div class="cors-notice" style="margin-top:12px">${notice} ${sourceBadge} • ${updatedText}</div>`;
    container.innerHTML = html;
}

function _buildStatsTable(cols, rows) {
    let html = '<div class="table-wrap"><table class="data-table"><thead><tr>';
    cols.forEach(c => html += `<th>${c.label}</th>`);
    html += '</tr></thead><tbody>';
    rows.forEach(row => {
        html += '<tr>';
        cols.forEach(c => {
            const raw = row[c.key];
            html += `<td>${c.fmt ? c.fmt(raw) : (raw ?? '-')}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table></div>';
    return html;
}

async function runElo() {
    const container = document.getElementById('eloResults');
    container.innerHTML = '<div class="loading"><div class="spinner"></div><span>Loading Elo leaderboard...</span></div>';

    try {
        const data = await safeFetchJson('/api/elo-leaderboard');
        const teams = data.teams || [];
        if (!teams.length) {
            container.innerHTML = '<div class="empty-state"><div class="empty-icon">🏆</div><h3 class="empty-title">No Elo Data</h3><p class="empty-desc">Team Elo ratings are not available yet. Run <code>python update_elo_ratings.py --current-season --reset</code> to populate them.</p></div>';
            return;
        }

        const maxRating = Math.max(...teams.map(t => t.rating || 0));
        const minRating = Math.min(...teams.map(t => t.rating || 0));
        const range = Math.max(1, maxRating - minRating);

        let html = '<div class="table-wrap"><table class="data-table"><thead><tr>';
        html += '<th>#</th><th>Team</th><th>Rating</th><th>Games</th><th style="width:40%">Strength</th>';
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
        html += `<div class="cors-notice" style="margin-top:12px">🏆 Elo ratings for season ${data.season || 'current'}. League average is 1500; top teams are typically 1600+.</div>`;
        container.innerHTML = html;
    } catch (e) {
        console.error('Elo leaderboard failed:', e);
        container.innerHTML = `<div class="error-box">Could not load Elo leaderboard: ${e.message}</div>`;
    }
}

async function runProps() {
    const container = document.getElementById('propsResults');
    container.innerHTML = '<div class="loading"><div class="spinner"></div><span>Loading props...</span></div>';
    if (USE_DEMO) { await new Promise(r => setTimeout(r, 500)); }

    const props = [
        { player:"A. Matthews", market:"Goals", line:"O/U 0.5", over:"-145", under:"+115" },
        { player:"C. McDavid", market:"Points", line:"O/U 1.5", over:"-125", under:"+105" },
        { player:"I. Shesterkin", market:"Saves", line:"O/U 28.5", over:"-110", under:"-110" },
        { player:"M. Tkachuk", market:"Shots", line:"O/U 3.5", over:"-115", under:"-105" },
    ];

    let html = '<div class="prop-grid">';
    props.forEach(p => {
        html += `<div class="prop-card">
            <div class="prop-player">${p.player}</div>
            <div class="prop-details">
                <span class="prop-market">${p.market}</span>
                <span class="prop-line">${p.line}</span>
                <div class="prop-odds"><span class="prop-over">O ${p.over}</span><span class="prop-under">U ${p.under}</span></div>
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
        await safeFetchJson('/api/teams');
    } catch (e) { console.error('Teams load failed:', e); }
}

async function loadSeasons() {
    try {
        const data = await safeFetchJson('/api/seasons');
        const sel = document.getElementById('statsSeason');
        sel.innerHTML = '';
        (data.seasons || []).forEach(s => sel.add(new Option(s.label, s.key)));
    } catch (e) { console.error('Seasons load failed:', e); }
}

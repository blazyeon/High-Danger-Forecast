/**
 * High Danger Forecast — Midnight Ice Dashboard
 */

// ── Sample Data ──────────────────────────────────────────
const games = [
    {
        id: 1,
        away: { name: 'Toronto Maple Leafs', abbr: 'TOR', record: '46-24-12', logo: 'https://assets.nhle.com/logos/nhl/svg/TOR_light.svg' },
        home: { name: 'Montreal Canadiens', abbr: 'MTL', record: '30-36-16', logo: 'https://assets.nhle.com/logos/nhl/svg/MTL_light.svg' },
        time: '7:00 PM ET',
        status: 'scheduled',
        awayProb: 62,
        homeProb: 38,
        pick: 'TOR Moneyline',
        danger: false,
    },
    {
        id: 2,
        away: { name: 'Carolina Hurricanes', abbr: 'CAR', record: '52-22-8', logo: 'https://assets.nhle.com/logos/nhl/svg/CAR_light.svg' },
        home: { name: 'Vegas Golden Knights', abbr: 'VGK', record: '45-29-8', logo: 'https://assets.nhle.com/logos/nhl/svg/VGK_light.svg' },
        time: '10:00 PM ET',
        status: 'live',
        awayProb: 55,
        homeProb: 45,
        pick: 'CAR Puck Line -1.5',
        danger: true,
    },
];

const rosterData = {
    away: {
        name: 'Carolina Hurricanes',
        abbr: 'CAR',
        score: 2,
        logo: 'https://assets.nhle.com/logos/nhl/svg/CAR_light.svg',
        players: [
            { pos: 'C', name: 'S. Aho', g: 1, a: 1, sog: 4, toi: '19:24' },
            { pos: 'RW', name: 'M. Necas', g: 0, a: 1, sog: 3, toi: '18:12' },
            { pos: 'LW', name: 'A. Svechnikov', g: 1, a: 0, sog: 2, toi: '16:45' },
            { pos: 'D', name: 'J. Slavin', g: 0, a: 0, sog: 1, toi: '22:08' },
            { pos: 'D', name: 'B. Burns', g: 0, a: 0, sog: 0, toi: '17:33' },
            { pos: 'C', name: 'J. Staal', g: 0, a: 0, sog: 0, toi: '14:21' },
        ],
    },
    home: {
        name: 'Vegas Golden Knights',
                abbr: 'VGK',
        score: 1,
        logo: 'https://assets.nhle.com/logos/nhl/svg/VGK_light.svg',
        players: [
            { pos: 'C', name: 'J. Eichel', g: 0, a: 1, sog: 4, toi: '20:44' },
            { pos: 'RW', name: 'M. Stone', g: 1, a: 0, sog: 5, toi: '21:34' },
            { pos: 'C', name: 'W. Karlsson', g: 0, a: 0, sog: 2, toi: '18:15' },
            { pos: 'D', name: 'A. Pietrangelo', g: 0, a: 0, sog: 1, toi: '23:55' },
            { pos: 'D', name: 'B. McNabb', g: 0, a: 0, sog: 0, toi: '19:10' },
            { pos: 'C', name: 'B. Howden', g: 0, a: 0, sog: 0, toi: '12:05' },
        ],
    },
};

const props = [
    { player: 'A. Matthews', matchup: 'TOR @ MTL', category: 'Shots on Goal', line: 'O/U 3.5', projection: 4.2, edge: 0.7, high: true },
    { player: 'C. McDavid', matchup: 'EDM @ LAK', category: 'Points', line: 'O/U 1.5', projection: 1.8, edge: 0.3, high: false },
    { player: 'I. Shesterkin', matchup: 'NYR @ NJD', category: 'Saves', line: 'O/U 28.5', projection: 30.1, edge: 1.6, high: true },
    { player: 'S. Bobrovsky', matchup: 'FLA @ TBL', category: 'Goals Against', line: 'U 2.5', projection: 2.1, edge: 0.4, high: false },
];

// ── Helpers ──────────────────────────────────────────────
function formatEdge(value) {
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toFixed(1)}`;
}

function renderStatClass(value, isToi = false) {
    if (isToi) {
        return value === '00:00' ? 'zero' : value >= '20:00' ? 'highlight' : 'active';
    }
    if (value === 0) return 'zero';
    return value > 0 ? 'highlight' : 'active';
}

// ── Render Game Cards ───────────────────────────────────
function renderGames() {
    const container = document.getElementById('gamesSection');
    if (!container) return;

    container.innerHTML = games.map(game => {
        const awayWidth = game.awayProb.toFixed(1);
        const homeWidth = game.homeProb.toFixed(1);
        const statusClass = game.status === 'live' ? 'live' : '';
        const statusText = game.status === 'live' ? 'Live' : 'Scheduled';
        const pickClass = game.danger ? 'danger' : '';
        const fireIcon = game.danger ? '<i class="fa-solid fa-fire"></i>' : '<i class="fa-solid fa-arrow-trend-up"></i>';

        return `
            <article class="game-card">
                <div class="game-card-header">
                    <div class="game-meta">
                        <span class="game-status ${statusClass}">${statusText}</span>
                        <span>${game.time}</span>
                    </div>
                </div>

                <div class="game-card-teams">
                    <div class="game-team away">
                        <img class="game-team-logo" src="${game.away.logo}" alt="${game.away.abbr}">
                        <div class="game-team-info">
                            <div class="game-team-name">${game.away.name}</div>
                            <div class="game-team-record">${game.away.record}</div>
                        </div>
                    </div>
                    <div class="game-card-center">
                        <div class="game-card-time">${game.time}</div>
                        <div class="game-card-vs">VS</div>
                    </div>
                    <div class="game-team home">
                        <img class="game-team-logo" src="${game.home.logo}" alt="${game.home.abbr}">
                        <div class="game-team-info">
                            <div class="game-team-name">${game.home.name}</div>
                            <div class="game-team-record">${game.home.record}</div>
                        </div>
                    </div>
                </div>

                <div class="prob-section">
                    <div class="prob-values">
                        <div>
                            <div class="prob-value away">${game.awayProb}%</div>
                        </div>
                        <div class="prob-label">Win Probability</div>
                        <div>
                            <div class="prob-value home">${game.homeProb}%</div>
                        </div>
                    </div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-away" style="width: ${awayWidth}%"></div>
                        <div class="prob-bar-home" style="width: ${homeWidth}%"></div>
                    </div>
                    <div class="prob-footer">
                        <span class="prob-pick ${pickClass}">${fireIcon} ${game.pick}</span>
                        <span>Model Confidence: High</span>
                    </div>
                </div>
            </article>
        `;
    }).join('');
}

// ── Render Rosters ──────────────────────────────────────
function renderRosterTable(players) {
    const rows = players.map(p => {
        const gClass = renderStatClass(p.g);
        const aClass = renderStatClass(p.a);
        const sogClass = renderStatClass(p.sog);
        const toiClass = renderStatClass(p.toi, true);

        return `
            <tr>
                <td><span class="roster-pos">${p.pos}</span></td>
                <td><span class="roster-player">${p.name}</span></td>
                <td class="roster-stat ${gClass}">${p.g}</td>
                <td class="roster-stat ${aClass}">${p.a}</td>
                <td class="roster-stat ${sogClass}">${p.sog}</td>
                <td class="roster-stat ${toiClass}">${p.toi}</td>
            </tr>
        `;
    }).join('');

    return `
        <table class="roster-table">
            <thead>
                <tr>
                    <th>POS</th>
                    <th>PLAYER</th>
                    <th class="numeric">G</th>
                    <th class="numeric">A</th>
                    <th class="numeric">SOG</th>
                    <th class="numeric">TOI</th>
                </tr>
            </thead>
            <tbody>
                ${rows}
            </tbody>
        </table>
    `;
}

function renderRosters() {
    const section = document.getElementById('rostersSection');
    if (!section) return;

    section.innerHTML = `
        <div class="rosters-header">
            <h2>Rosters & Live Stats</h2>
            <span class="props-meta"><i class="fa-solid fa-circle-info"></i> Ice blue = notable stat, muted = zero action</span>
        </div>
        <div class="rosters-grid">
            <div class="roster-card">
                <div class="roster-card-header away">
                    <div class="roster-team">
                        <img src="${rosterData.away.logo}" alt="${rosterData.away.abbr}">
                        <div>
                            <div class="roster-team-name">${rosterData.away.name}</div>
                            <div class="roster-team-abbr">Away</div>
                        </div>
                    </div>
                    <div class="roster-score">${rosterData.away.score}</div>
                </div>
                ${renderRosterTable(rosterData.away.players)}
            </div>
            <div class="roster-card">
                <div class="roster-card-header home">
                    <div class="roster-team">
                        <img src="${rosterData.home.logo}" alt="${rosterData.home.abbr}">
                        <div>
                            <div class="roster-team-name">${rosterData.home.name}</div>
                            <div class="roster-team-abbr">Home</div>
                        </div>
                    </div>
                    <div class="roster-score">${rosterData.home.score}</div>
                </div>
                ${renderRosterTable(rosterData.home.players)}
            </div>
        </div>
    `;
}

// ── Render Props ────────────────────────────────────────
function renderProps() {
    const tbody = document.querySelector('#propsTable tbody');
    if (!tbody) return;

    tbody.innerHTML = props.map(prop => {
        const edgeClass = prop.high ? 'high' : '';
        const rowClass = prop.high ? 'danger' : '';
        const fireIcon = prop.high ? '<i class="fa-solid fa-fire prop-fire"></i>' : '';

        return `
            <tr class="${rowClass}">
                <td>
                    <span class="prop-player">${fireIcon}${prop.player}</span>
                </td>
                <td><span class="prop-matchup">${prop.matchup}</span></td>
                <td><span class="prop-category">${prop.category}</span></td>
                <td class="prop-odds">${prop.line}</td>
                <td class="prop-proj">${prop.projection}</td>
                <td class="prop-edge ${edgeClass}">${formatEdge(prop.edge)}</td>
            </tr>
        `;
    }).join('');
}

// ── Date Ribbon Interaction ──────────────────────────────
function initDateRibbon() {
    const tabs = document.querySelectorAll('.date-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
        });
    });
}

// ── Init ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    renderGames();
    renderRosters();
    renderProps();
    initDateRibbon();
});

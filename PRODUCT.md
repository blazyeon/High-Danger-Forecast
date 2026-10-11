# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: hockey fans and recreational bettors finding the site via a public link
(owner shared). Confirmed by owner 2026-10-11: public-facing, used on the phone
on the go — including during games and while traveling — as well as on desktop.
Success: a first-time visitor can understand what a tab offers without
instruction, and on the phone can reach the day's board and bet-slip/parlay
interactions one-handed.

## Product Purpose

Predicts NHL games and prices player props using an ensemble of Elo ratings,
Monte-Carlo simulation, and a trained ML model, then surfaces edges against
DraftKings/FanDuel odds. Tabs: Matchup Predictor, Game Lookup, Advanced Stats,
Player Props, Betting Edge, Elo Leaderboard, Today's Picks, Parlay finder,
plus a bet slip. Success means the visitor trusts the numbers and can act on
them without re-running anything.

## Positioning

A three-model ensemble (Elo + Monte-Carlo sims + trained ML) rebuilt nightly
from play-by-play data, with player props priced two-sided (not Over-only
ladders) and edges shown as model-vs-book probability deltas from cached daily
boards — not a thin "score prediction" wrapper.

## Operating Context

- Data refreshes run as a scheduled GitHub Actions job (overnight baseline +
  late-afternoon odds sweep); boards are served from on-disk caches.
- Game-day reality: most of the slate starts in the evening ET; by late
  evening most games are in progress and pregame prices vanish from the feed
  (by design). The user checks the board on the phone throughout the day.
- Betting-related content: edges and recommendations are model outputs, not
  guarantees; no wagering is processed by the site.

## Capabilities and Constraints

- Flask backend serves precomputed caches; interactive endpoints (parlay
  finder, bet slip) work on the fly but must never trigger long inline
  computation (43s sims / 112s props exceed UI timeouts).
- SharpAPI free tier meters odds per key (two keys configured); props priced
  off DraftKings, AGS off FanDuel; goals props are Over-only ladders.
- Historical team mapping ARI→UTA applies to data sources.
- Undecided: monetization, accounts, or community features — none exist.

## Brand Commitments

Keep the product name and logo/logo mark as-is (confirmed 2026-10-11);
palette, typography, and layout are explicitly replaceable by the redesign.

## Evidence on Hand

- Live caches under `static/data/` (today's picks, betting edges, odds, PBP
  stats), Elo database, model files — real data, not mockups.
- No testimonials, press, or case studies exist; future work must not
  fabricate any.

## Product Principles

1. Numbers first, chrome last — data density where it serves decision-making,
   without clutter on small screens.
2. One-glance clarity — a first-time visitor must know what each tab does and
   what action to take next without instruction.
3. Phone is a first-class surface — on-the-go use with big touch targets and
   one-handed interaction, not a shrunken desktop.
4. Honest uncertainty — show what the model does and does not know (No Data,
   stale boards) rather than hiding it behind confident chrome.

## Accessibility & Inclusion

Public-facing → WCAG 2.2 AA: semantic HTML, heading order, landmarks, visible
focus, keyboard-operable tabs/modals/dropdowns, ≥4.5:1 text contrast, not
color-only signals, form labels on every input.
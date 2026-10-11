---
name: High Danger Forecast
description: NHL predictions and prop edges in the language of a broadcast scorebug
colors:
  glass-base: "#0b0d12"
  glass-surface: "#12161f"
  glass-elevated: "#1a2030"
  glass-hover: "#232b3a"
  input-field: "#0d1017"
  line-hairline: "rgba(255,255,255,0.07)"
  line-default: "rgba(255,255,255,0.14)"
  line-strong: "rgba(255,255,255,0.45)"
  ink-primary: "#eef1f6"
  ink-secondary: "#a9b1bd"
  ink-muted: "#97a0ad"
  onair-amber: "#ffb224"
  tally-red: "#ff5470"
  edge-green: "#34d39b"
  prob-blue: "#5a9dff"
  paper-base: "#f2f3f7"
  paper-surface: "#ffffff"
  paper-ink: "#16191f"
typography:
  display:
    fontFamily: "Barlow Condensed, 'Arial Narrow', sans-serif"
    fontSize: "2rem"
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "0.01em"
  scorebug-caps:
    fontFamily: "Barlow Condensed, 'Arial Narrow', sans-serif"
    fontSize: "0.6875rem"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "0.14em"
  body:
    fontFamily: "Barlow, system-ui, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 400
    lineHeight: 1.5
  numerals:
    fontFamily: "Spline Sans Mono, ui-monospace, monospace"
    fontSize: "0.875rem"
    fontWeight: 500
rounded:
  sm: "4px"
  md: "6px"
  lg: "10px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "14px"
  lg: "22px"
  xl: "36px"
---

## Overview

Broadcast Scorebug. The app speaks the graphic language of live hockey
coverage: the scorebug that sits in the corner of every televised game. Team
identification happens through team logos in compact blocks, information is
labeled by dense micro-caps, and the model's call is stamped like an on-air
graphic. This is a fan-run analysis project — personality over corporate
polish, dense over spacious, honest over hype. No casino neon, no marketing
sections. A visitor should recognize tonight's rundown the way they recognize
a game broadcast.

## Colors

The stage is an evening under glass: cool deep charcoal-blue (not pure black,
not warm gray) reads as the rink's darkened arena. All information colors are
broadcast hardware colors:

- **glass-base / surfaces:** cool dark glass panes, hairline white rules at
  7–14% separating them, like LED panels on a dark arena wall.
- **onair-amber:** the single accent — the "on air" tally color. Reserved for
  primary actions, the model's stamped call, live status, and the featured
  pick. Never used decoratively.
- **tally-red / edge-green:** semantic pair for model-vs-book direction (red
  = negative/live, green = edges). Not team colors, not decoration.
- **prob-blue:** home/away probability bars and analytic accents.
- Team color paints the live data marks exactly as a broadcast scorebug does —
  probability bars, score numerals, and the model's call read in each team's
  color, with clash resolution when two teams wear the same hue. Interface
  chrome (rules, controls, banners) stays out of team hues; amber remains the
  only interface accent.

## Typography

- **Barlow Condensed** is the scorebug face — every label, table header, stat
  name, and nav item. Caps with 0.14em tracking at 0.6875rem is the caption
  register; 600–700 weight up to 2rem for display calls. Broadcast signage
  keeps its letterform ancestry: condensed grotesk, never serif, never
  geometric-round.
- **Barlow** carries prose: card subtitles, copy, hints.
- **Spline Sans Mono** carries every numeral that is a measurement — odds,
  Elo, lines, probabilities — tabular so columns align.

## Layout

- Content max-width 1240px, 16px side gutter minimum; main grid collapses to
  a single column below 900px.
- Navigation: horizontal tab rail under the header on desktop (scrollable, no
  wrapping); below 768px it becomes a fixed bottom thumb bar — icons large,
  labels micro-caps, safe-area inset respected.
- Same-size uniform-card scaffolds are forbidden; density is varied like a
  broadcast rundown — the day's top edge is featured as a lifted, amber-ruled
  lower third ahead of the rundown rows, then dense tile rows, then quiet
  tables.

## Elevation & Depth

Shadows carry offset and soft blur (ambient + close shadow). No zero-offset
colored glows. Emphasis comes from hairline rules, surface tone steps, and
the amber accent — never glow halos. Modals and popups lift with a real
drop shadow. `backdrop-filter` is not decor; only the fixed bottom bar may
use a surface scrim over scrolling content.

## Shapes

Small crisp radii — 4px controls, 6px blocks, 10px cards — because the
scorebug is a rectangle, not a pill. Chips and tag blocks are rectangles with
1px rules. Probability bars are 4px tracks. Every interactive control gets a
visible amber focus ring (2px offset) on keyboard focus.

## Components

- **Tab (nav):** condensed caps tile; active tab filled amber-on-ink with a
  2px bottom rule; icon above micro label on bottom bar.
- **Card:** glass surface, hairline border, 22px padding rhythm; header row
  with title + subtitle and actions right-aligned.
- **Pick row (rundown):** logo chips left, model call center, expand control
  right; detail opens below the row flush with the content edge.
- **Data table:** condensed caps headers, mono numerals, hairline row rules;
  sortable headers show an amber caret asc/desc; horizontal scroll with
  sticky first column on phone.
- **Edge badge:** green/red/amber block with mono delta, on the row's right.
- **Banner (lower third):** amber or red ribbon strip pinned full-width with
  icons; reads like an on-air lower third.
- **Buttons:** primary action filled amber (ink text), secondary ghost with
  hairline border, disabled 40% opacity with no shadow.
- **Date-picker popup:** anchored dropdown card with micro-caps weekday row;
  selected day filled amber, today outlined.

## Do's and Don'ts

- Do label with micro-caps once per region, like a scorebug; don't decorate
  every section with eyebrows.
- Do keep amber meaning "on air / attention"; don't spray it on headings,
  links, or hover states.
- Don't add glow, confetti, or casino energy; the live-dot pulse and the
  lower-third slide are the only authored motion.
- Don't replace team logos with approximated team-color blocks.
- Don't nest cards; density comes from rules and type, not boxes in boxes.
- Do keep numerals in the mono face; prose never uses it.
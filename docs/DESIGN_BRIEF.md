# Design Brief — FireTurret

Scope: the three visual surfaces — the **video overlay** (`ui/overlay.py`), the
**browser console** (`ui/webserver.py`), and the **desktop workbench**
(`studio/`). This brief is retrospective: it names the intent the built surfaces
actually express, and it is explicit about where they fall short of the floor
below rather than describing an aspiration.

[PRD.md](PRD.md) · [APP_FLOW.md](APP_FLOW.md)

## What the three surfaces must convey

**Instrument, not product.** Reading a state and a number must be faster than
recognising an icon. This is a machine that aims water; the operator's questions
are always "what is it doing", "where will the water go", and "what do I do about
it" — in that order, at a glance, possibly from across a room.

**What it must never feel like:** a consumer smart-home app. No reassuring
gradients, no friendly copy, no green tick that means "probably fine". Anything
that makes an unproven demonstrator feel *finished* is a safety problem, not a
taste problem — the honesty of the documentation has to survive into the
interface.

## The supervisor, one metre from an E-stop

- **The operator, standing next to a running machine**, often not looking at the
  screen at all. This is why every advisory also drives a physical LED/buzzer
  output: the interface's most important channel is not visual.
- **The engineer tuning a model**, sitting down, iterating: edit a parameter,
  Run (F5), read five linked plots, compare against the previous run. Density is
  a feature here; the workbench is closer to an oscilloscope than to a dashboard.
- **The reviewer watching a GIF in a README**, who has ten seconds and no
  context. The overlay has to be legible at small size and out of context, which
  is why the state name is spelled out rather than encoded in colour.

## Borrowed from

- **OpenRocket** — the model is the document. Every parameter is editable in one
  place, a run is cheap and repeatable, and results accumulate for comparison
  instead of replacing each other. The workbench's Session + Runs table is that
  idea directly.
- **ParaView / pyqtgraph dashboards** — X-linked plots with a shared crosshair
  readout, so a single cursor position answers "what was every channel doing at
  that instant" without a legend hunt.
- **Aircraft primary flight displays** — one dominant mode annunciator, plain
  words, no ambiguity about which state the machine is in. The overlay's state
  band is that: `SUPPRESS` written out, not a coloured dot.

What is being borrowed in each case is the *information structure*, not the look.

## Refusals

- **Colour as the only carrier of severity.** The advisory banner is colour-coded
  *and* always carries the severity word and the action text. A colour-blind
  operator, or a washed-out phone screen in daylight, must lose nothing.
- **A status that cannot go stale.** The browser console currently violates this:
  a stalled poll leaves the last good status on screen with no "stale" marker. It
  is listed as a defect in [APP_FLOW.md](APP_FLOW.md), not defended here.
- **Animation on a safety surface.** Nothing fades, slides or pulses. A moving
  element on a screen showing where water is about to go is noise.
- **A rearmable E-stop.** The button latches and offers no browser path back.
  Making that convenient would make it not an E-stop.
- **A "success" aesthetic.** `CONFIRM` is green because the fire is out at this
  instant, not because the mission succeeded; the design must not imply a verdict
  the system cannot support.

## Type

- **Video overlay:** `cv2.FONT_HERSHEY_SIMPLEX` at 0.38–0.55 scale — the only
  font OpenCV offers without shipping a font file. Three sizes in use: 0.55
  (state band, advisory label), 0.44–0.5 (advisory message and action), 0.38–0.42
  (blob labels, telemetry lines, inset captions). Weight 2 for the state band and
  advisory label, 1 for everything else.
- **Browser console:** `system-ui, sans-serif` — one family, whatever the device
  already renders best. The title is set in `letter-spacing: .3em; font-weight: 800`,
  which is a deliberate instrument-panel affectation and the one purely stylistic
  choice on the page.
- **Workbench:** the Qt system font at system size. Not overridden, so OS-level
  font scaling continues to work.

## Colour

Roles first; every value below is in the code.

| Role | Workbench / web | Overlay (BGR) | Used for |
|---|---|---|---|
| surface | `#10151c` | `(20,22,25)` band fill | page and window background |
| panel | `#141b24` | — | dock and control backgrounds |
| ink | `#e8edf4` | `INK (235,238,240)` | primary text |
| muted | `#8fa0b5` | `MUTED (150,160,165)` | secondary status text, inset captions |
| grid | `#253141` | — | plot gridlines, hairlines |
| accent | `#4cc3e8` | — | selection, highlight |
| water | — | `WATER (250,210,120)` | splash marker, spray timer, SUPPRESS |
| fire / target | — | `FIRE (60,110,255)`, `TARGET (40,60,255)` | detected blobs, locked target |
| ok | — | `OK (120,210,130)` | CONFIRM |
| warn | `#c47f17` banner | `WARN (60,170,250)` | ACQUIRE/RANGE/ALIGN, WARN advisories |
| danger | `#c0392b` banner | `(80,80,230)` SAFE, `(120,120,210)` HOLD | CRITICAL advisories, E-stop button |

**Contrast, measured against the surfaces they sit on:**

| Pair | Ratio | Verdict |
|---|---|---|
| `#e8edf4` ink on `#10151c` surface | **15.58:1** | passes AAA |
| `#e8edf4` ink on `#141b24` panel | **14.73:1** | passes AAA |
| `#8fa0b5` muted on `#10151c` surface | **6.86:1** | passes AA; just under AAA's 7:1 |
| white on `#c0392b` CRITICAL banner | **5.44:1** | passes AA body |
| white on `#c47f17` WARN banner | **3.28:1** | ⚠ **fails 4.5:1 body text** — passes only as large text |
| `#4cc3e8` accent on `#10151c` | **8.97:1** | passes, safe as a UI boundary |

The WARN banner is a real, measured failure, recorded rather than rounded away.
Two one-line fixes, both measured: darken the amber to `#8a5a10` (5.91:1 with
white), or keep the amber and set the banner text to black (6.40:1). Either is a
single change in `_INDEX_HTML` and in the overlay's banner fill.

## Spacing

- **Overlay:** a fixed 26 px top band, a left-aligned telemetry column at 10 px
  inset with ~18 px line spacing, an advisory banner below the band, and a
  side-view arc inset in the lower right. Everything is drawn in pixels against a
  known frame size (default 960×540) — this surface does not reflow.
- **Browser:** single centred column, `max-width: 100%` on the stream image,
  `viewport` meta set for phones. It is the only responsive surface, and it is
  responsive by being simple rather than by having breakpoints.
- **Workbench:** `QMainWindow` dock layout — Model and Calibration left,
  Ballistics / Top-down / Camera right, Runs and Advisories bottom, plots
  central. Docks are user-rearrangeable and the layout persists between sessions;
  **View ▸ Reset layout** restores the default, which is the escape hatch that
  makes rearrangeability safe.

## The components that carry it

Reused, not reinvented: Qt's `QDockWidget`, `QUndoStack` (the model editor's
undo/redo is Qt's, not a bespoke stack), `QProgressBar`/`QStatusBar`,
`pyqtgraph.PlotWidget` for all five telemetry plots and the ballistics explorer.
The only bespoke widgets are the ones with no equivalent: the top-down scene
view, the crosshair readout binding, and the sweep heatmap. The browser console
uses no framework and no components at all — one HTML string, ~30 lines of
vanilla JS, standard library HTTP server.

## States

| Element | Hover | Focus | Active | Disabled | Loading | Error |
|---|---|---|---|---|---|---|
| E-STOP button | `cursor: pointer` | **browser default ring** (not overridden) | native | n/a | n/a | fetch failure is swallowed and retried |
| Run / Stop (F5 / Shift+F5) | Qt default | Qt focus ring | toolbar pressed state | Stop disabled while idle, Run disabled while running | status bar progress + enabled Cancel | non-modal error dialog + log |
| Model editor field | Qt default | Qt focus ring | — | — | — | invalid value **rejected**, field reverts, model stays valid |
| Runs table row | row highlight | Qt focus ring | selection | — | — | stale rows marked after a model edit |

`outline: none` appears nowhere in this repository. The browser console keeps the
UA focus ring, and Qt supplies its own — a deliberate non-decision, since a
custom focus style would have to be maintained across two toolkits.

## Accessibility floor, measured rather than asserted

| Requirement | Status |
|---|---|
| 4.5:1 body text | **partial** — passes everywhere except white-on-amber WARN (3.28:1) |
| 3:1 large text and UI boundaries | passes |
| Full keyboard operation | passes on CLI and workbench; browser has one control, Tab-reachable |
| Visible focus | passes — no `outline: none` anywhere |
| Touch targets ≥ 44 px | passes — E-STOP is `0.9rem 2rem` padding at `1.1rem` type, ≈ 50 px tall |
| Colour never the only signal | passes — state name, severity word and action text always written |
| `prefers-reduced-motion` | **not honoured** — no motion to suppress in the DOM, but the MJPEG stream is unconditional |
| `prefers-color-scheme` | **not honoured** — all three surfaces are dark-only |
| 200% zoom | browser passes (single column, relative units); workbench relies on Qt/OS scaling; **the overlay does not scale at all** — its text is drawn in frame pixels |
| Screen reader | **weak** — no `aria-live` on the advisory banner, so an advisory appearing is announced to nobody; the video overlay is an image with no textual equivalent. `--headless` is the accessible path through a run |

## Responsive

- **Browser console** — the only genuinely responsive surface: one column,
  `max-width: 100%` image, viewport meta. Nothing changes at a breakpoint because
  there are no breakpoints; it works at 320 px because there is nothing to reflow.
- **Workbench** — desktop only, and honestly so. Docks can be dragged and the
  layout persists, but there is no small-window design and none is intended.
- **Overlay** — fixed pixel geometry against the configured frame size. Changing
  `camera.width`/`height` moves the elements proportionally only by accident.

## Done means

- [x] Matches the intent — instrument, not product
- [x] Every state designed, including empty (SEARCH with no fire) and error (SAFE)
- [x] Contrast checked with real values — and the one failure named above
- [x] Keyboard path walked on all three surfaces
- [ ] **Open:** WARN banner contrast (3.28:1) raised past 4.5:1
- [ ] **Open:** `aria-live` on the browser advisory banner
- [ ] **Open:** a stale-status indicator on the browser console

# Verifying Training Steps — slide deck

Pacif.ai talk deck: how an independent verifier can confirm, step by step, that a model was trained as agreed.

## View it

Open `docs/presentation/deck.html` (from this folder: `deck.html`) in a browser.

| Key | Action |
|---|---|
| → / Space / click right side | Next slide |
| ← / click left side | Previous slide |
| N | Show / hide speaker notes |
| F | Fullscreen |

Each slide has its own link, e.g. `deck.html#attacks`.

## Edit it

The source of truth is:

```
deck.json          # title, slide order, sections, fonts
slides/<id>.html   # one <section id="<id>"> per slide; speaker notes in the last <aside>
build.py           # combines the above into deck.html
```

1. Edit a file in `slides/` (or `order` in `deck.json` to reorder, add or remove slides).
2. Run `python3 build.py` to regenerate `deck.html` (or `python3 build.py --watch` while editing, and refresh the browser).
3. Commit both your source changes **and** the rebuilt `deck.html`.

### Slide format

Each slide is a fixed 1920×1080 canvas. Keep to these rules so the deck can also be loaded back into Claude Slides:

- Exactly one `<section id="…">` per file. The file name matches the id.
- All styles are inline (`style="…"`). Don't use classes, `<style>` blocks, `margin`, `z-index`, `em` or `var()`.
- Use px sizes and hex colors. Keep text at 24px or larger.
- `position:absolute` with `left`/`top`/`width` pins an element to slide coordinates.
- Arrows: `<x-connector x1 y1 x2 y2 style="color:#…;border-width:3px">` in slide pixels. Add `head="none"` for a plain line.
- Icons: `<x-icon name="ShieldCheck">` uses [Lucide](https://lucide.dev/icons) icon names in PascalCase.
- Speaker notes: plain text in one `<aside>`, as the last child of the section.

### Palette

| Use | Hex |
|---|---|
| Ink / dark | `#14213D` |
| Light backgrounds | `#F6F5F1`, `#ECEAE3` |
| Prover | `#2B63C6` |
| Verifier | `#B85C12` |
| Hash steps | `#8A3FB0` |
| Calculation check | `#2E7D3A` |
| Data step | `#0B7476` |
| Attacks | `#C62828` |

Font: DM Sans.

## Editing with Claude

The deck started as a Claude Slides artifact. To make a change with Claude, give it the relevant `slides/*.html` file (or this repo) and describe the change. Commit the updated files here, then rebuild. Treat this repo as the master copy, so edits made in different places don't drift apart.

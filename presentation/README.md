# Audio + AI Meetup — Slides

A clickable reveal.js deck driven by a single Markdown file
([slides.md](slides.md)), styled with a Voxel51 orange/grey theme
([theme/voxel51.css](theme/voxel51.css)). Built with
[reveal-md](https://github.com/webpro/reveal-md).

## Run it (live, rebuilds on save)

```bash
npm install       # first time only
npm run dev
```

Opens a local server (default `http://localhost:1948`) and watches
`slides.md` — edit the Markdown, save, and the browser auto-reloads. Keep
this tab open during the talk.

## Editing

- `---` on its own line (blank lines before/after) starts a new **horizontal**
  slide (a new Part/section).
- `----` (four dashes) on its own line starts a new **vertical** slide
  (a sub-topic nested under the current Part — press Down to reach it).
- A line starting with `Note:` turns everything below it (until the next
  separator) into speaker notes. Press `s` in the browser to open the
  speaker view (notes + next-slide preview) — that's the memory-jogger
  during the actual talk.
- `> blockquote` lines are styled as "demo cue" callouts — use them for
  "switch to the FiftyOne app and do X" reminders.
- `<!-- .slide: data-background-color="#1c1e22" --> ` above a slide's content
  gives it the lighter panel-grey background used for section dividers.

## Keyboard (during the talk)

- Arrow keys / Space — navigate
- `s` — speaker notes view
- `f` — fullscreen
- Esc — slide overview

## Static export (optional, e.g. to send around after)

```bash
npm run static    # writes build/ (gitignored)
```

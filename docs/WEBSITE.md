# K-Bench project website

The static site has an English homepage (`index.html`) and a Chinese page
(`zh/index.html`). Both share the styles, JavaScript, experiment values, figures,
and paper PDF. Existing protocol and benchmark documentation stays in this directory.

## Preview locally

From the repository root:

```bash
python3 -m http.server 8123 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8123/` or `http://127.0.0.1:8123/zh/`.
No package installation or build step is required.

## Enable GitHub Pages

After merging, a repository owner or an authorized maintainer can configure:

1. **Settings → Pages → Build and deployment**.
2. Select **Deploy from a branch**.
3. Select **main** and **/docs**, then save.

The expected URLs are:

- English: `https://onireimu.github.io/kbench/`
- Chinese: `https://onireimu.github.io/kbench/zh/`

`.nojekyll` serves the committed static files without a Jekyll build. The site
uses relative paths for navigation and assets, so it also works under the
repository URL prefix. No deployment workflow or repository setting is changed
by adding these files.

## Maintain the site

- Edit `index.html` and `zh/index.html` for the two page bodies.
- Edit `script.js` for interactive behavior and language-specific dynamic copy.
- Edit `styles.css` for shared styling and responsive rules.
- `data.js` contains the observation-budget and eligibility-gated comparison values.
  These are fixed paper results, not a live submission leaderboard.
- `assets/` contains the framework and observation-budget figures; `paper.pdf`
  is the full paper linked from both pages.
- The two `paper-visualization.config.json` files retain page metadata for
  authoring. They do not generate the custom page bodies at runtime. Keep their
  metadata consistent with the HTML head when editing titles or descriptions.

If the hosting URL changes, update the canonical, Open Graph, and language
alternate URLs in both HTML heads and metadata configs, plus `sitemap.xml`.
When adding site files, include them in `PUBLISH_MANIFEST.txt` so the existing
public-release synchronization process preserves them.

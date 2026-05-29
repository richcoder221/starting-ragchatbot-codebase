# Frontend Changes

## Light/Dark Mode Toggle Button

### What was added
A theme toggle button fixed in the top-right corner of the UI that switches between dark mode (the original design) and a new light mode.

### Files changed
- `frontend/index.html` — Added `#themeToggle` button with inline sun and moon SVG icons; bumped CSS/JS cache-busting version to `v=10`
- `frontend/style.css` — Added light theme CSS variable overrides on `body.light-theme`, toggle button styles (fixed position, circular, icon swap), and a `transition` on `body` for smooth color changes
- `frontend/script.js` — Added `initTheme()` (reads `localStorage`), `toggleTheme()` (toggles `light-theme` class + persists to `localStorage`), wired the button click handler, and added `themeToggle` to DOM element references

### Design decisions
- **Icon convention**: Sun icon is shown in dark mode (click to go lighter); moon icon is shown in light mode (click to go darker)
- **Positioning**: `position: fixed; top: 1rem; right: 1rem; z-index: 100` keeps it visible regardless of scroll
- **Smooth transition**: `transition: background-color 0.3s ease, color 0.3s ease` on `body` ensures all CSS-variable-driven colours animate together
- **Accessibility**: `aria-label` is updated dynamically to reflect the current action ("Switch to light/dark mode"); button is keyboard-focusable with a visible focus ring using `--focus-ring`
- **Persistence**: Theme preference is stored in `localStorage` under the key `theme` and restored on page load

// KaTeX auto-render bootstrap for arithmatex generic mode.
// Mirrors the snippet recommended at
// https://squidfunk.github.io/mkdocs-material/reference/math/#katex
//
// Runs on every page load (including Material's instant-loading
// navigation, hence the `document$.subscribe(...)`). Targets the
// `\\(...\\)` / `\\[...\\]` delimiters that arithmatex emits when
// `generic: true` is set.

document$.subscribe(({ body }) => {
    renderMathInElement(body, {
        delimiters: [
            { left: "\\[", right: "\\]", display: true },
            { left: "\\(", right: "\\)", display: false },
            { left: "$$", right: "$$", display: true },
            { left: "$",  right: "$",  display: false },
        ],
    });
});

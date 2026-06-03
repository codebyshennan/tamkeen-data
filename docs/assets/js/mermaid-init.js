/**
 * Mermaid diagrams: fetch .mmd source files, render with theme awareness.
 * Supports two modes:
 *   1. [data-mermaid-src] elements (preferred): fetches the .mmd file and renders.
 *   2. pre>code.language-mermaid blocks (legacy): renders inline source directly.
 */
(function () {
  'use strict';

  // Pinned to 11.4.0 to match the mmdc version used by scripts/validate-mermaid.sh.
  // 11.4.1 + htmlLabels:false mis-parses labels starting with "N." or "+" as markdown
  // lists ("Unsupported markdown: list"); keeping live and validator in lockstep avoids
  // that whole class of bug going undetected. See node_modules/mermaid (^11.4.1 resolves to 11.4.0).
  var CDN = 'https://cdn.jsdelivr.net/npm/mermaid@11.4.0/dist/mermaid.esm.min.mjs';
  var mermaid = null;
  var idCounter = 0;

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'forest';
  }

  async function renderAll() {
    var theme = currentTheme();
    mermaid.initialize({
      startOnLoad: false,
      theme: theme,
      flowchart: { useMaxWidth: true, htmlLabels: false },
      securityLevel: 'loose',
    });

    var tasks = [];

    // Mode 1: [data-mermaid-src] containers
    document.querySelectorAll('[data-mermaid-src]').forEach(function (c) {
      tasks.push((async function () {
        var text = c._mermaidSource;
        if (!text) return;
        try {
          var id = 'mmd-' + (idCounter++);
          var result = await mermaid.render(id, text);
          // Remove Mermaid's temporary rendering element BEFORE inserting the SVG
          // into the container — otherwise getElementById finds the copy we just
          // inserted (earlier in document order) and deletes the visible diagram.
          var stray = document.getElementById(id);
          if (stray) stray.remove();
          c.innerHTML = result.svg;
        } catch (e) {
          console.warn('Mermaid render failed:', c.dataset.mermaidSrc, e);
        }
      })());
    });

    // Mode 2: legacy pre>code.language-mermaid blocks
    document.querySelectorAll('pre code.language-mermaid').forEach(function (code) {
      tasks.push((async function () {
        var text = code.textContent;
        if (!text) return;
        var pre = code.parentElement;
        if (!pre) return;
        try {
          var id = 'mmd-' + (idCounter++);
          var result = await mermaid.render(id, text);
          var stray = document.getElementById(id);
          if (stray) stray.remove();
          var wrapper = document.createElement('div');
          wrapper.className = 'mermaid-diagram';
          wrapper.innerHTML = result.svg;
          pre.replaceWith(wrapper);
        } catch (e) {
          console.warn('Mermaid render failed (legacy block):', e);
        }
      })());
    });

    await Promise.all(tasks);
    attachLightboxHandlers();
  }

  function attachLightboxHandlers() {
    if (!window.tamkeenLightbox) return;
    // Mode 1: [data-mermaid-src] containers
    document.querySelectorAll('[data-mermaid-src]').forEach(function (c) {
      if (c._lightboxAttached || !c.querySelector('svg')) return;
      c._lightboxAttached = true;
      c.addEventListener('click', function () {
        window.tamkeenLightbox.openSvg(c);
      });
    });
    // Mode 2: legacy .mermaid-diagram wrappers
    document.querySelectorAll('.mermaid-diagram').forEach(function (c) {
      if (c._lightboxAttached || !c.querySelector('svg')) return;
      c._lightboxAttached = true;
      c.addEventListener('click', function () {
        window.tamkeenLightbox.openSvg(c);
      });
    });
  }

  async function run() {
    var hasSrc = document.querySelector('[data-mermaid-src]');
    var hasLegacy = document.querySelector('pre code.language-mermaid');
    if (!hasSrc && !hasLegacy) return;

    if (!mermaid) {
      var mod = await import(CDN);
      mermaid = mod.default;
    }

    // Fetch .mmd source files (once per container, cached on element)
    if (hasSrc) {
      await Promise.all(
        Array.from(document.querySelectorAll('[data-mermaid-src]')).map(async function (c) {
          if (!c._mermaidSource) {
            try {
              var r = await fetch(c.dataset.mermaidSrc);
              if (!r.ok) throw new Error('HTTP ' + r.status);
              c._mermaidSource = await r.text();
            } catch (e) {
              console.warn('Could not fetch diagram:', c.dataset.mermaidSrc, e);
            }
          }
        })
      );
    }

    await renderAll();
  }

  // Re-render when theme attribute changes on <html>
  new MutationObserver(function () {
    if (mermaid) renderAll();
  }).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
  } else {
    run();
  }
})();

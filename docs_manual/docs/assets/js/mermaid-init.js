// Mermaid (v10+) does not auto-render if initialized after the page load event.
// Render explicitly so diagrams consistently appear across all pages.
(function () {
  function markError(el, err) {
    el.style.whiteSpace = "pre-wrap";
    el.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
    el.textContent = (el.textContent || "").trim() + "\n\n[Mermaid render error]\n" + String(err);
  }

  function runMermaid() {
    if (!window.mermaid) return false;
    try {
      window.mermaid.initialize({ startOnLoad: false });
      var nodes = document.querySelectorAll(".mermaid");
      // Render each diagram independently so a single syntax error doesn't hide everything.
      nodes.forEach(function (el) {
        try {
          var p = window.mermaid.run({ nodes: [el] });
          // Mermaid.run may return a Promise; handle async errors too.
          if (p && typeof p.then === "function") {
            p.catch(function (e) {
              markError(el, e);
            });
          }
        } catch (e) {
          markError(el, e);
        }
      });
    } catch (e) {
      // Keep the page usable even if Mermaid fails for a single diagram.
      console.error("Mermaid render failed:", e);
    }
    return true;
  }

  if (runMermaid()) return;

  // If Mermaid is loaded via CDN, it should be available quickly. Poll a bit to be safe.
  var tries = 0;
  var timer = window.setInterval(function () {
    tries += 1;
    if (runMermaid() || tries > 50) {
      window.clearInterval(timer);
      if (!window.mermaid) {
        // Give a clear hint instead of leaving raw source text.
        document.querySelectorAll(".mermaid").forEach(function (el) {
          markError(el, "Mermaid library was not loaded. If you are viewing the raw .md file, use `mkdocs serve`. If you are viewing the built site, check that assets/js/mermaid.min.js is accessible.");
        });
      }
    }
  }, 100);
})();

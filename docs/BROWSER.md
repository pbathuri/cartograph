# Using Cartograph with web GenAI (ChatGPT, Gemini, claude.ai)

Web chat UIs don't expose a universal "run before every prompt" hook, so the integration has two parts:
a shared local brain (`carto serve`) and a tiny per-site **userscript** that prepends the steering brief.

## 1. Run the local API
```bash
carto serve            # http://127.0.0.1:8787/api/personalize?prompt=...
```
Returns the same steering brief as the `personalize` MCP tool / CLI — persona, field, output guidance,
and your most relevant snippets. It's local-only and CORS-enabled for browser fetches.

## 2. Install the userscript (Tampermonkey / Violentmonkey)
This minimal example, on Ctrl+Enter, fetches your brief and prepends it to the prompt box. Adjust the
`@match` and the input selector per site (ChatGPT/Gemini change their DOM often).

```javascript
// ==UserScript==
// @name         Cartograph steering
// @match        https://chatgpt.com/*
// @match        https://gemini.google.com/*
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// ==/UserScript==
(function () {
  function box() { return document.querySelector('textarea, [contenteditable="true"]'); }
  async function steer() {
    const el = box(); if (!el) return;
    const prompt = el.value || el.innerText || "";
    if (!prompt.trim()) return;
    const r = await fetch("http://127.0.0.1:8787/api/personalize?prompt=" + encodeURIComponent(prompt));
    const b = await r.json();
    const brief = "[Context for this user — from Cartograph]\n" +
      "Persona: " + (b.persona_summary || "") + "\n" +
      "Guidance: " + (b.output_guidance || []).join(" ") + "\n" +
      (b.relevant_context || []).slice(0,4).map(c => "• " + c.project + " :: " + (c.snippet||"").slice(0,200)).join("\n") +
      "\n\n---\n";
    const text = brief + prompt;
    if ("value" in el) el.value = text; else el.innerText = text;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }
  window.addEventListener("keydown", e => { if (e.ctrlKey && e.key === "Enter") steer(); }, true);
})();
```

## Notes
- This sends your prompt to your **local** Cartograph only; nothing else leaves your machine.
- Selectors/`@match` are the only per-site maintenance; the persona/graph logic is shared.
- For Claude Code / Cursor, you don't need this — use the MCP `personalize` tool directly (see README).
- Treat brief injection as *your* context: review it; it's plain text you can edit before sending.

// Assignment chatbot widget. Reads window.TAMKEEN_CHATBOT (set by the
// Jekyll include) for the lesson_key and endpoint URL, then renders a
// floating launcher + chat panel. Streams responses via Server-Sent
// Events from /api/chat, which proxies OpenRouter with per-lesson context.

(function () {
  const cfg = window.TAMKEEN_CHATBOT;
  if (!cfg || !cfg.lessonKey || !cfg.endpoint) return;

  const STORAGE_KEY = `tamkeen-chatbot-${cfg.lessonKey}`;
  const messages = loadHistory();

  const launcher = document.createElement('button');
  launcher.className = 'cb-launcher';
  launcher.setAttribute('aria-label', 'Open assignment tutor');
  launcher.setAttribute('aria-expanded', 'false');
  launcher.innerHTML = '?';

  const panel = document.createElement('section');
  panel.className = 'cb-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', 'Assignment tutor');
  panel.innerHTML = `
    <header class="cb-header">
      <div>
        <p class="cb-title">Assignment tutor</p>
        <p class="cb-subtitle">${escapeHtml(cfg.lessonTitle || cfg.lessonKey)} • guides, never gives answers</p>
      </div>
      <button class="cb-close" aria-label="Close">×</button>
    </header>
    <div class="cb-messages" role="log" aria-live="polite"></div>
    <div class="cb-quick"></div>
    <form class="cb-input-row">
      <textarea class="cb-input" rows="1" placeholder="Tell me what you've considered so far…" aria-label="Message"></textarea>
      <button type="submit" class="cb-send">Send</button>
    </form>
  `;

  document.body.appendChild(launcher);
  document.body.appendChild(panel);

  const messagesEl = panel.querySelector('.cb-messages');
  const quickEl = panel.querySelector('.cb-quick');
  const inputEl = panel.querySelector('.cb-input');
  const sendBtn = panel.querySelector('.cb-send');
  const form = panel.querySelector('form');
  const closeBtn = panel.querySelector('.cb-close');

  const QUICK_PROMPTS = [
    "I don't understand what the question is asking.",
    "I've narrowed it down to two — help me decide.",
    "I think the answer is ___ because ___. Am I on the right track?",
  ];
  QUICK_PROMPTS.forEach(p => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = p;
    b.addEventListener('click', () => { inputEl.value = p; inputEl.focus(); });
    quickEl.appendChild(b);
  });

  function open() {
    panel.dataset.open = 'true';
    launcher.setAttribute('aria-expanded', 'true');
    if (messages.length === 0) {
      addMessage('assistant', `Hi! I'm scoped to **${cfg.lessonTitle || cfg.lessonKey}**. Which question or task are you stuck on, and what have you tried so far?`);
    }
    setTimeout(() => inputEl.focus(), 50);
  }
  function close() {
    panel.dataset.open = 'false';
    launcher.setAttribute('aria-expanded', 'false');
  }
  launcher.addEventListener('click', () => panel.dataset.open === 'true' ? close() : open());
  closeBtn.addEventListener('click', close);

  renderHistory();

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = inputEl.value.trim();
    if (!text || sendBtn.disabled) return;
    inputEl.value = '';
    quickEl.style.display = 'none';
    addMessage('user', text);
    saveHistory();

    sendBtn.disabled = true;
    const assistantEl = addMessage('assistant', '');
    let accumulated = '';
    try {
      const resp = await fetch(cfg.endpoint, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ lesson_key: cfg.lessonKey, messages }),
      });
      if (!resp.ok || !resp.body) {
        const detail = await resp.text().catch(() => '');
        throw new Error(`HTTP ${resp.status}: ${detail.slice(0, 200)}`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6).trim();
          if (payload === '[DONE]') continue;
          try {
            const json = JSON.parse(payload);
            const delta = json.choices?.[0]?.delta?.content;
            if (delta) {
              accumulated += delta;
              assistantEl.innerHTML = renderMarkdown(accumulated);
              messagesEl.scrollTop = messagesEl.scrollHeight;
            }
          } catch { /* ignore parse errors mid-stream */ }
        }
      }
      // Persist the completed assistant turn.
      if (accumulated) {
        messages.push({ role: 'assistant', content: accumulated });
        saveHistory();
      }
    } catch (err) {
      assistantEl.classList.remove('cb-msg-assistant');
      assistantEl.classList.add('cb-msg-error');
      assistantEl.textContent = `Sorry — ${err.message}`;
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  });

  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });

  // --- helpers ---

  function addMessage(role, content) {
    if (role === 'user' || role === 'assistant') {
      // Only persist user; assistant gets persisted at stream end.
      if (role === 'user') messages.push({ role, content });
    }
    const el = document.createElement('div');
    el.className = `cb-msg cb-msg-${role}`;
    el.innerHTML = renderMarkdown(content);
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return el;
  }

  function renderHistory() {
    if (messages.length === 0) return;
    quickEl.style.display = 'none';
    for (const m of messages) {
      const el = document.createElement('div');
      el.className = `cb-msg cb-msg-${m.role}`;
      el.innerHTML = renderMarkdown(m.content);
      messagesEl.appendChild(el);
    }
  }

  function loadHistory() {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  }
  function saveHistory() {
    try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-20))); }
    catch { /* quota */ }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
  }

  // Tiny markdown subset: code fences, inline code, bold, italic, paragraphs.
  function renderMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);
    // fenced code blocks
    html = html.replace(/```([\s\S]*?)```/g, (_, body) => `<pre>${body.trim()}</pre>`);
    // inline code
    html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');
    // bold + italic
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>');
    // soft line breaks
    html = html.replace(/\n/g, '<br>');
    return html;
  }
})();

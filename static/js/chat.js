// ============================================================
//  SPCOET Smart AI Chatbot — Frontend Chat Engine
//  Features: Chat, Voice I/O, Multilingual, Typing Animation,
//            Confidence Display, Suggestions, Session Memory
// ============================================================

// ── State ─────────────────────────────────────────────────────
const state = {
  sessionId   : generateId(),
  lang        : 'en',
  isTyping    : false,
  isListening : false,
  messageCount: 0,
  recognition : null,
  synth       : window.speechSynthesis || null,
};

// ── Language Strings ──────────────────────────────────────────
const I18N = {
  en: {
    placeholder : 'Ask anything about SPCOET — courses, fees, admission, placements…',
    listening   : 'Listening… speak now',
    welcome_q   : 'What can I ask?',
    thinking    : 'Thinking…',
    you         : 'You',
    bot         : 'SPCOET AI',
    nomic       : 'Voice input requires Chrome or Edge browser.',
    error       : '⚠️ Server error. Please try again.',
    offline     : '⚠️ Cannot reach server. Make sure it is running (python app.py).'
  },
  hi: {
    placeholder : 'SPCOET के बारे में कुछ भी पूछें — कोर्स, फीस, एडमिशन…',
    listening   : 'सुन रहा हूँ… बोलें',
    welcome_q   : 'मैं क्या पूछ सकता हूँ?',
    thinking    : 'सोच रहा हूँ…',
    you         : 'आप',
    bot         : 'SPCOET AI',
    nomic       : 'वॉयस इनपुट के लिए Chrome या Edge ब्राउज़र चाहिए।',
    error       : '⚠️ सर्वर त्रुटि। कृपया पुनः प्रयास करें।',
    offline     : '⚠️ सर्वर से कनेक्ट नहीं हो सका।'
  },
  mr: {
    placeholder : 'SPCOET बद्दल काहीही विचारा — अभ्यासक्रम, फी, प्रवेश…',
    listening   : 'ऐकत आहे… बोला',
    welcome_q   : 'मी काय विचारू शकतो?',
    thinking    : 'विचार करत आहे…',
    you         : 'तुम्ही',
    bot         : 'SPCOET AI',
    nomic       : 'आवाज इनपुटसाठी Chrome किंवा Edge ब्राउझर वापरा।',
    error       : '⚠️ सर्व्हर त्रुटी. पुन्हा प्रयत्न करा.',
    offline     : '⚠️ सर्व्हरशी कनेक्ट होऊ शकलो नाही.'
  }
};

function t(key) { return (I18N[state.lang] || I18N.en)[key] || key; }

// ── DOM Ready ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderWelcome();
  checkServerHealth();
  document.getElementById('user-input').addEventListener('input', updateCharCount);
  window.speechSynthesis?.addEventListener?.('voiceschanged', () => {});
});

// ── Welcome Screen ────────────────────────────────────────────
function renderWelcome() {
  const container = document.getElementById('messages');
  container.innerHTML = '';

  const topics = ['📚 Courses & Departments','💰 Fee Structure','🎓 Admission Process',
    '💼 Placements & Companies','🏠 Hostel Facilities','📞 Contact Info',
    '🏆 Scholarships','🎉 Events & Activities'];

  const card = document.createElement('div');
  card.innerHTML = `
    <div class="welcome-card">
      <h2>👋 Hello! I'm SPCOET AI</h2>
      <p>I can answer <strong>any question</strong> about Sharadchandra Pawar College of Engineering and Technology using semantic AI search through the college knowledge base.</p>
      <div class="welcome-topics">
        ${topics.map(tp => `<span class="welcome-topic" onclick="quickAsk('Tell me about ${tp.replace(/[^\w\s]/g,'')}')">${tp}</span>`).join('')}
      </div>
    </div>`;
  card.style.alignSelf = 'flex-start';
  container.appendChild(card);
}

// ── Send Message ─────────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById('user-input');
  const text  = input.value.trim();
  if (!text || state.isTyping) return;

  input.value = '';
  input.style.height = 'auto';
  updateCharCount();
  hideSuggestions();

  addUserMessage(text);
  showTypingIndicator();
  state.isTyping = true;
  document.getElementById('send-btn').disabled = true;

  try {
    const res = await fetch('/chat', {
      method  : 'POST',
      headers : { 'Content-Type': 'application/json' },
      body    : JSON.stringify({ message: text, session_id: state.sessionId, lang: state.lang })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    removeTypingIndicator();

    // Add bot reply
    addBotMessage(data.reply, data);

    // Show confidence bar
    showConfidenceBar(data.intent, data.sub_intent, data.confidence);

    // Show suggestions
    if (data.suggestions?.length) {
      showSuggestions(data.suggestions);
    }

    // Auto-speak reply (if voice was used)
    if (state.voiceMode && state.synth) {
      speakText(data.reply);
    }

  } catch (err) {
    removeTypingIndicator();
    addBotMessage(err.message.includes('fetch') ? t('offline') : t('error'));
  } finally {
    state.isTyping = false;
    document.getElementById('send-btn').disabled = false;
    document.getElementById('user-input').focus();
  }
}

// ── Message Rendering ─────────────────────────────────────────
function addUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'msg-row user';
  row.innerHTML = `
    <div class="msg-avatar user">${t('you').charAt(0)}</div>
    <div class="msg-content">
      <div class="msg-bubble">${escapeHtml(text)}</div>
      <div class="msg-meta">${getTime()}</div>
    </div>`;
  appendMessage(row);
  state.messageCount++;
}

function addBotMessage(text, meta = {}) {
  const row = document.createElement('div');
  row.className = 'msg-row bot';

  const formatted = formatMarkdown(text);
  const sourceBadges = (meta.sources || [])
    .map(s => `<span class="source-badge">📎 ${s.section}</span>`)
    .join(' ');

  row.innerHTML = `
    <div class="msg-avatar">SP</div>
    <div class="msg-content">
      <div class="msg-bubble">${formatted}</div>
      <div class="msg-meta">
        ${getTime()}
        ${sourceBadges}
        ${meta.response_time ? `<span style="color:var(--txt3)">· ${meta.response_time}ms</span>` : ''}
      </div>
    </div>`;

  // Animate in
  row.style.opacity = '0';
  row.style.transform = 'translateY(8px)';
  appendMessage(row);
  requestAnimationFrame(() => {
    row.style.transition = 'opacity .3s, transform .3s';
    row.style.opacity = '1';
    row.style.transform = 'translateY(0)';
  });

  state.messageCount++;
}

// ── Typing Indicator ──────────────────────────────────────────
function showTypingIndicator() {
  const row = document.createElement('div');
  row.className = 'msg-row bot'; row.id = 'typing-row';
  row.innerHTML = `
    <div class="msg-avatar">SP</div>
    <div class="msg-bubble typing-indicator" style="padding:12px 16px">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>`;
  appendMessage(row);
}
function removeTypingIndicator() {
  document.getElementById('typing-row')?.remove();
}

// ── Confidence Bar ────────────────────────────────────────────
function showConfidenceBar(intent, subIntent, confidence) {
  const bar   = document.getElementById('confidence-bar');
  const fill  = document.getElementById('conf-fill');
  const score = document.getElementById('conf-score');
  const intentEl = document.getElementById('conf-intent');

  intentEl.textContent  = `Intent: ${intent}${subIntent ? ' → ' + subIntent : ''}`;
  score.textContent     = `Confidence: ${(confidence * 100).toFixed(0)}%`;
  fill.style.width      = (confidence * 100) + '%';
  fill.style.background = confidence > 0.7 ? 'var(--green)' : confidence > 0.4 ? 'var(--acc)' : 'var(--pri)';
  bar.style.display     = 'block';
}

// ── Suggestions ───────────────────────────────────────────────
function showSuggestions(suggestions) {
  const row   = document.getElementById('suggestions-row');
  const chips = document.getElementById('suggestions-chips');
  chips.innerHTML = suggestions.map(s =>
    `<button class="suggestion-chip" onclick="quickAsk('${s.replace(/'/g,"\\'")}')">💬 ${s}</button>`
  ).join('');
  row.style.display = 'block';
}
function hideSuggestions() {
  document.getElementById('suggestions-row').style.display = 'none';
}

// ── Quick Ask ─────────────────────────────────────────────────
function quickAsk(text) {
  document.getElementById('user-input').value = text;
  hideSuggestions();
  sendMessage();
}

// ── Voice Input ───────────────────────────────────────────────
function toggleVoice() {
  if (state.isListening) { stopListening(); return; }
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    addBotMessage(t('nomic')); return;
  }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  state.recognition = new SR();
  state.recognition.lang        = state.lang === 'hi' ? 'hi-IN' : state.lang === 'mr' ? 'mr-IN' : 'en-IN';
  state.recognition.interimResults = true;
  state.recognition.continuous  = false;

  state.recognition.onstart = () => {
    state.isListening = true;
    state.voiceMode   = true;
    document.getElementById('mic-btn').classList.add('active');
    document.getElementById('voice-indicator').style.display = 'flex';
    document.getElementById('voice-status-text').textContent = t('listening');
  };
  state.recognition.onresult = (e) => {
    const transcript = Array.from(e.results).map(r => r[0].transcript).join('');
    document.getElementById('user-input').value = transcript;
    document.getElementById('voice-status-text').textContent = transcript;
    autoResize(document.getElementById('user-input'));
    updateCharCount();
  };
  state.recognition.onend = () => {
    stopListening();
    const txt = document.getElementById('user-input').value.trim();
    if (txt) sendMessage();
  };
  state.recognition.onerror = () => stopListening();
  state.recognition.start();
}
function stopListening() {
  state.isListening = false;
  state.recognition?.stop();
  document.getElementById('mic-btn').classList.remove('active');
  document.getElementById('voice-indicator').style.display = 'none';
}

// ── Text to Speech ────────────────────────────────────────────
function speakText(text) {
  if (!state.synth) return;
  state.synth.cancel();
  const clean = text.replace(/\*\*/g,'').replace(/[*_#]/g,'').replace(/<[^>]+>/g,'').trim();
  const utt   = new SpeechSynthesisUtterance(clean);
  utt.lang    = state.lang === 'hi' ? 'hi-IN' : state.lang === 'mr' ? 'mr-IN' : 'en-IN';
  utt.rate    = 0.92; utt.pitch = 1;
  const voices = state.synth.getVoices();
  const voice  = voices.find(v => v.lang === utt.lang) || voices.find(v => v.lang.startsWith('en'));
  if (voice) utt.voice = voice;
  state.synth.speak(utt);
}

// ── Language Switching ────────────────────────────────────────
function setLang(lang, btn) {
  state.lang = lang;
  document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('user-input').placeholder = t('placeholder');
  // If synth is speaking, stop it
  state.synth?.cancel();
}

// ── New Chat ──────────────────────────────────────────────────
function newChat() {
  state.sessionId    = generateId();
  state.messageCount = 0;
  state.voiceMode    = false;
  document.getElementById('confidence-bar').style.display = 'none';
  hideSuggestions();
  renderWelcome();
  document.getElementById('user-input').focus();
}

// ── UI Helpers ────────────────────────────────────────────────
function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
function updateCharCount() {
  const len = document.getElementById('user-input').value.length;
  document.getElementById('char-count').textContent = len;
  document.getElementById('char-count').style.color = len > 450 ? '#DC2626' : '';
}
function appendMessage(el) {
  const c = document.getElementById('messages');
  c.appendChild(el);
  c.scrollTop = c.scrollHeight;
}
function getTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function generateId() {
  return 'sess_' + Math.random().toString(36).slice(2) + Date.now().toString(36);
}
function escapeHtml(str) {
  const d = document.createElement('div');
  d.appendChild(document.createTextNode(str));
  return d.innerHTML;
}

// ── Markdown Formatter ────────────────────────────────────────
function formatMarkdown(text) {
  let out = '', inList = false;
  const lines = text.split('\n');
  for (const line of lines) {
    let s = line
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g,     '<em>$1</em>')
      .replace(/`(.*?)`/g,       '<code>$1</code>')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    if (/^[•\-]\s/.test(line)) {
      if (!inList) { out += '<ul>'; inList = true; }
      out += '<li>' + s.replace(/^[•\-]\s/, '') + '</li>';
    } else if (/^\d+\.\s/.test(line)) {
      if (!inList) { out += '<ol>'; inList = true; }
      out += '<li>' + s.replace(/^\d+\.\s/, '') + '</li>';
    } else {
      if (inList) { out += inList === 'ol' ? '</ol>' : '</ul>'; inList = false; }
      if (s.trim()) out += '<p style="margin:0 0 4px">' + s + '</p>';
    }
  }
  if (inList) out += '</ul>';
  return out || escapeHtml(text);
}

// ── Sidebar Toggle (Mobile) ───────────────────────────────────
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('overlay').classList.toggle('show');
}
function toggleInfo() {
  const b = document.getElementById('info-banner');
  b.style.display = b.style.display === 'none' ? 'block' : 'none';
}

// ── Server Health Check ───────────────────────────────────────
async function checkServerHealth() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    const el = document.getElementById('kb-status');
    el.textContent = `✅ ${d.kb_entries} KB chunks · AI: ${d.ai_ready ? 'ready' : '⚠ no key'}`;
    el.style.color = d.ai_ready ? 'var(--green)' : 'var(--acc)';
  } catch(e) {
    document.getElementById('kb-status').textContent = '❌ Server offline';
  }
}

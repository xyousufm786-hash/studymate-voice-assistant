"""
StudyMate – AI Student Assistant (Flask web app with voice)
Backend: Flask + Ollama (llama3.2:1b)
Voice:   browser Web Speech API (speech-to-text + text-to-speech), best in Chrome / Edge
Run:     python app.py   ->  open http://127.0.0.1:5000
"""
import json
import re
import time

import ollama
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context

MODEL = "llama3.2:1b"
MAX_MESSAGE_CHARS = 1500
MAX_HISTORY = 10            # last N messages sent to the model (small model = small context)

BASE_RULES = (
    "You are StudyMate, a friendly and encouraging AI assistant for university students. "
    "Keep answers clear, accurate and well organised, using short paragraphs and '-' bullet points. "
    "Use **bold** only for key terms. Keep most answers under 250 words unless a plan or quiz needs more. "
    "If you are not sure about something, say so instead of guessing. "
    "For questions about a specific university's fees, deadlines or rules, give general guidance and "
    "advise the student to confirm with their university's official website or student services office. "
    "Never write a full graded assignment for the student; instead, explain, outline and guide them."
)

MODES = {
    "ask": {
        "label": "Ask Anything",
        "prompt": BASE_RULES + " Answer the student's academic or student-life question directly.",
    },
    "plan": {
        "label": "Study Planner",
        "prompt": BASE_RULES + (
            " You are in STUDY PLANNER mode. Build a realistic study plan. If the student did not give "
            "subjects, exam dates or hours available, make sensible assumptions and state them. "
            "Structure it as: Goal, Weekly/Daily schedule (with time blocks), Techniques "
            "(e.g. active recall, spaced repetition, Pomodoro), and Tips to stay on track."
        ),
    },
    "explain": {
        "label": "Explain a Topic",
        "prompt": BASE_RULES + (
            " You are in EXPLAIN mode. Explain the concept step by step for a beginner: "
            "1) a one-line simple definition, 2) how it works, 3) a real-world example or analogy, "
            "4) a common mistake students make, 5) one quick check question at the end."
        ),
    },
    "quiz": {
        "label": "Quiz Me",
        "prompt": BASE_RULES + (
            " You are in QUIZ mode. Create 5 multiple-choice questions (A-D) on the topic the student gives. "
            "Number them. Put all the answers with a one-line explanation in an 'Answer Key' section at "
            "the end. If the student replies with their answers, mark them and explain any mistakes."
        ),
    },
    "career": {
        "label": "Career & CV",
        "prompt": BASE_RULES + (
            " You are in CAREER mode. Help with internships, CVs, cover letters and interview preparation. "
            "Give specific, actionable suggestions and short example phrasing the student can adapt."
        ),
    },
}

app = Flask(__name__)


# ------------------------------------------------------------------ helpers
def check_ollama():
    """Return (ok, message)."""
    try:
        resp = ollama.list()
    except Exception:
        return False, "Ollama is not running. Open the Ollama app or run 'ollama serve'."
    models = resp["models"] if isinstance(resp, dict) else getattr(resp, "models", [])
    names = []
    for m in models:
        name = (m.get("model") or m.get("name")) if isinstance(m, dict) else getattr(m, "model", None)
        if name:
            names.append(name)
    if not any(n == MODEL or n.startswith(MODEL) for n in names):
        return False, f"Model '{MODEL}' not found. Run: ollama pull {MODEL}"
    return True, f"Connected to {MODEL}"


def clean_history(raw):
    """Validate the chat history sent from the browser. Returns (messages, error)."""
    if not isinstance(raw, list) or not raw:
        return None, "No message received."
    msgs = []
    for m in raw[-MAX_HISTORY:]:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            msgs.append({"role": role, "content": content[:4000]})
    if not msgs or msgs[-1]["role"] != "user":
        return None, "Please type or say a message."
    last = msgs[-1]["content"]
    if len(last) > MAX_MESSAGE_CHARS:
        return None, f"Your message is too long (max {MAX_MESSAGE_CHARS} characters)."
    if not re.search(r"[A-Za-z0-9À-￿]", last):
        return None, "Please type a real question."
    return msgs, None


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return render_template_string(PAGE, modes=MODES, model=MODEL)


@app.route("/api/health")
def health():
    ok, msg = check_ollama()
    return jsonify(ok=ok, message=msg, model=MODEL)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "ask")
    if mode not in MODES:
        return jsonify(error="Unknown mode."), 400
    msgs, err = clean_history(data.get("messages"))
    if err:
        return jsonify(error=err), 400

    messages = [{"role": "system", "content": MODES[mode]["prompt"]}] + msgs
    print(f"[{time.strftime('%H:%M:%S')}] ({mode}) {msgs[-1]['content'][:70]}")

    def generate():
        try:
            stream = ollama.chat(model=MODEL, messages=messages, stream=True,
                                 options={"temperature": 0.6, "num_ctx": 4096, "num_predict": 1200})
            got_text = False
            for chunk in stream:
                text = chunk["message"]["content"]
                if text:
                    got_text = True
                    yield text
            if not got_text:
                yield "\n[[ERROR]]The model returned an empty answer. Please try again."
        except ollama.ResponseError as e:
            if getattr(e, "status_code", None) == 404:
                yield f"\n[[ERROR]]Model '{MODEL}' is not installed. Run: ollama pull {MODEL}"
            else:
                yield f"\n[[ERROR]]Ollama error: {getattr(e, 'error', e)}"
        except ConnectionError:
            yield "\n[[ERROR]]Can't reach Ollama. Open the Ollama app or run 'ollama serve', then try again."
        except Exception as e:
            yield f"\n[[ERROR]]Unexpected error: {type(e).__name__}: {e}"

    return Response(stream_with_context(generate()), mimetype="text/plain; charset=utf-8",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.errorhandler(404)
def not_found(e):
    return jsonify(error="Not found."), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify(error="Server error. Check the terminal for details."), 500


# ------------------------------------------------------------------ page
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StudyMate – AI Student Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#f4f1ea; --panel:#fffdf8; --ink:#1f2421; --muted:#6b706b; --line:#e3ddd0;
    --accent:#2f5d50; --accent-ink:#ffffff; --accent-soft:#e3ece7; --user:#2f5d50;
    --bot:#fffdf8; --warn:#b4432c; --live:#c2410c;
  }
  @media (prefers-color-scheme: dark){
    :root{ --bg:#151816; --panel:#1d211f; --ink:#e9ece8; --muted:#9aa29c; --line:#2d332f;
      --accent:#7fb5a0; --accent-ink:#0f1412; --accent-soft:#24302b; --user:#3c6e5f; --bot:#1d211f;
      --warn:#f08b73; --live:#fb923c; }
  }
  *{box-sizing:border-box}
  html,body{height:100%;margin:0}
  body{background:var(--bg);color:var(--ink);font:15px/1.6 Inter,system-ui,sans-serif;display:flex;justify-content:center}
  .app{width:100%;max-width:880px;height:100%;display:flex;flex-direction:column;padding:0 16px}
  header{display:flex;align-items:center;gap:12px;padding:18px 0 10px;border-bottom:1px solid var(--line)}
  .logo{width:42px;height:42px;border-radius:12px;background:var(--accent);color:var(--accent-ink);display:grid;place-items:center;font-size:22px}
  h1{font:600 22px/1.1 Fraunces,Georgia,serif;margin:0}
  .sub{color:var(--muted);font-size:13px}
  .status{margin-left:auto;font-size:12px;color:var(--muted);display:flex;align-items:center;gap:6px;white-space:nowrap}
  .dot{width:8px;height:8px;border-radius:50%;background:#aaa}
  .dot.ok{background:#22a06b}.dot.bad{background:var(--warn)}

  .modes{display:flex;gap:8px;overflow-x:auto;padding:12px 0 4px;scrollbar-width:none}
  .mode{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:7px 14px;border-radius:999px;cursor:pointer;font:500 13px Inter,sans-serif;white-space:nowrap}
  .mode.on{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}

  #chat{flex:1;overflow-y:auto;padding:14px 2px 8px;display:flex;flex-direction:column;gap:14px}
  .msg{max-width:85%;padding:11px 15px;border-radius:16px;word-wrap:break-word}
  .msg.user{align-self:flex-end;background:var(--user);color:#fff;border-bottom-right-radius:4px}
  .msg.bot{align-self:flex-start;background:var(--bot);border:1px solid var(--line);border-bottom-left-radius:4px}
  .msg.err{align-self:center;background:transparent;color:var(--warn);border:1px dashed var(--warn);font-size:13px}
  .msg .tag{font-size:11px;color:var(--muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em}
  .msg.bot ul{margin:6px 0;padding-left:20px}
  .msg.bot p{margin:6px 0}
  .msg .tools{margin-top:6px;display:flex;gap:6px}
  .mini{border:none;background:var(--accent-soft);color:var(--ink);border-radius:8px;padding:3px 9px;font-size:12px;cursor:pointer}
  .typing{display:inline-flex;gap:4px}.typing span{width:6px;height:6px;border-radius:50%;background:var(--muted);animation:b 1s infinite}
  .typing span:nth-child(2){animation-delay:.15s}.typing span:nth-child(3){animation-delay:.3s}
  @keyframes b{0%,80%,100%{opacity:.25}40%{opacity:1}}

  .welcome{margin:auto 0;text-align:center;color:var(--muted)}
  .welcome h2{font:600 26px Fraunces,Georgia,serif;color:var(--ink);margin:0 0 6px}
  .ideas{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:18px;text-align:left}
  .idea{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 12px;cursor:pointer;color:var(--ink);font-size:14px}
  .idea:hover{border-color:var(--accent)}
  .idea b{display:block;font-size:12px;color:var(--accent);font-weight:600}

  .composer{padding:10px 0 16px}
  .bar{display:flex;align-items:flex-end;gap:8px;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:8px}
  textarea{flex:1;resize:none;border:none;outline:none;background:transparent;color:var(--ink);font:15px/1.5 Inter,sans-serif;max-height:140px;padding:6px 4px}
  .btn{border:none;border-radius:12px;width:42px;height:42px;cursor:pointer;font-size:18px;display:grid;place-items:center;flex:none}
  .btn.mic{background:var(--accent-soft);color:var(--ink)}
  .btn.mic.live{background:var(--live);color:#fff;animation:pulse 1.2s infinite}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(194,65,12,.5)}100%{box-shadow:0 0 0 12px rgba(194,65,12,0)}}
  .btn.send{background:var(--accent);color:var(--accent-ink)}
  .btn:disabled{opacity:.45;cursor:not-allowed}
  .opts{display:flex;flex-wrap:wrap;gap:14px;align-items:center;font-size:12px;color:var(--muted);margin-top:8px}
  .opts label{display:flex;align-items:center;gap:5px;cursor:pointer}
  .opts select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:2px 4px;font-size:12px}
  .link{background:none;border:none;color:var(--muted);text-decoration:underline;cursor:pointer;font-size:12px;padding:0}
  .hint{min-height:18px;font-size:12px;color:var(--live);margin-top:4px}
  @media (max-width:560px){ .sub{display:none} .msg{max-width:94%} }
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="logo">🎓</div>
    <div><h1>StudyMate</h1><div class="sub">Your AI student assistant · type or talk</div></div>
    <div class="status"><span class="dot" id="dot"></span><span id="statusText">Checking…</span></div>
  </header>

  <div class="modes" id="modes">
    {% for key, m in modes.items() %}
      <button class="mode{% if key=='ask' %} on{% endif %}" data-mode="{{key}}">{{ m.label }}</button>
    {% endfor %}
  </div>

  <div id="chat">
    <div class="welcome" id="welcome">
      <h2>Hi! What are we studying today?</h2>
      <div>Pick a mode above, type below, or tap 🎤 and just ask.</div>
      <div class="ideas">
        <div class="idea" data-mode="plan" data-q="Make me a 2-week study plan for my Data Mining and Statistics finals. I can study 3 hours a day."><b>📅 Study Planner</b>2-week plan for my finals, 3 hrs a day</div>
        <div class="idea" data-mode="explain" data-q="Explain linear regression"><b>🧠 Explain a Topic</b>Explain linear regression simply</div>
        <div class="idea" data-mode="quiz" data-q="Quiz me on the basics of SQL joins"><b>📝 Quiz Me</b>5 questions on SQL joins</div>
        <div class="idea" data-mode="career" data-q="How do I write a strong CV for a data analyst internship?"><b>💼 Career & CV</b>CV tips for a data analyst internship</div>
        <div class="idea" data-mode="ask" data-q="How can I stop procrastinating before exams?"><b>💬 Ask Anything</b>How do I stop procrastinating?</div>
        <div class="idea" data-mode="ask" data-q="What is the difference between a GPA and a CGPA?"><b>💬 Ask Anything</b>GPA vs CGPA?</div>
      </div>
    </div>
  </div>

  <div class="composer">
    <div class="bar">
      <button class="btn mic" id="micBtn" title="Speak (voice input)">🎤</button>
      <textarea id="input" rows="1" placeholder="Ask a question, or tap 🎤 to speak…" maxlength="1500"></textarea>
      <button class="btn send" id="sendBtn" title="Send">➤</button>
    </div>
    <div class="hint" id="hint"></div>
    <div class="opts">
      <label><input type="checkbox" id="speakToggle" checked> 🔊 Read answers aloud</label>
      <label>Voice
        <select id="lang">
          <option value="en-US">English (US)</option>
          <option value="en-GB">English (UK)</option>
          <option value="en-IN">English (India)</option>
          <option value="ms-MY">Bahasa Melayu</option>
        </select>
      </label>
      <button class="link" id="stopVoice">🔇 Stop voice</button>
      <button class="link" id="newChat">↺ New chat</button>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const chat = $("#chat"), input = $("#input"), sendBtn = $("#sendBtn"), micBtn = $("#micBtn"), hint = $("#hint");
let mode = "ask", history = [], busy = false, voiceStarted = false;

/* ---------------- status ---------------- */
async function checkHealth(){
  try{
    const r = await fetch("/api/health"); const d = await r.json();
    $("#dot").className = "dot " + (d.ok ? "ok" : "bad");
    $("#statusText").textContent = d.ok ? d.model : "Offline";
    if(!d.ok) showError(d.message);
  }catch(e){ $("#dot").className = "dot bad"; $("#statusText").textContent = "Server offline"; }
}
checkHealth();

/* ---------------- modes ---------------- */
function setMode(m){
  mode = m;
  document.querySelectorAll(".mode").forEach(b => b.classList.toggle("on", b.dataset.mode === m));
  const ph = {ask:"Ask a question", plan:"Tell me your subjects, exam dates and hours per day",
              explain:"Which topic should I explain", quiz:"Which topic should I quiz you on",
              career:"Ask about CVs, internships or interviews"}[m];
  input.placeholder = ph + ", or tap 🎤 to speak…";
}
document.querySelectorAll(".mode").forEach(b => b.onclick = () => setMode(b.dataset.mode));
document.querySelectorAll(".idea").forEach(c => c.onclick = () => { setMode(c.dataset.mode); input.value = c.dataset.q; send(); });

/* ---------------- rendering ---------------- */
function esc(t){ return t.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function format(t){
  let h = esc(t).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/^#{1,6}\s*(.+)$/gm, "<b>$1</b>");
  const lines = h.split("\n"); let out = "", inList = false;
  for(const l of lines){
    const m = l.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)/);
    if(m){ if(!inList){ out += "<ul>"; inList = true; } out += "<li>" + m[1] + "</li>"; }
    else { if(inList){ out += "</ul>"; inList = false; } out += l.trim() ? "<p>" + l + "</p>" : ""; }
  }
  return out + (inList ? "</ul>" : "");
}
function addMsg(role, text){
  $("#welcome")?.remove();
  const d = document.createElement("div");
  d.className = "msg " + role;
  if(role === "user") d.textContent = text;
  else if(role === "err") d.textContent = "⚠️ " + text;
  else d.innerHTML = text;
  chat.appendChild(d); chat.scrollTop = chat.scrollHeight;
  return d;
}
function showError(t){ addMsg("err", t); }

/* ---------------- chat ---------------- */
async function send(){
  const text = input.value.trim();
  if(busy) return;
  if(!text){ hint.textContent = "Type or say something first."; return; }
  if(text.length > 1500){ hint.textContent = "Message too long (max 1500 characters)."; return; }
  hint.textContent = ""; stopSpeaking();
  input.value = ""; autoGrow();
  addMsg("user", text);
  history.push({role:"user", content:text});
  setBusy(true);

  const bot = addMsg("bot", '<div class="tag">' + esc(document.querySelector(".mode.on").textContent) + '</div><div class="typing"><span></span><span></span><span></span></div>');
  let full = "";
  try{
    const r = await fetch("/api/chat", {method:"POST", headers:{"Content-Type":"application/json"},
                                        body: JSON.stringify({mode, messages: history})});
    if(!r.ok){
      const d = await r.json().catch(() => ({error:"Request failed (" + r.status + ")."}));
      throw new Error(d.error);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    const tag = bot.querySelector(".tag").outerHTML;
    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      full += dec.decode(value, {stream:true});
      const shown = full.split("[[ERROR]]")[0];
      bot.innerHTML = tag + format(shown);
      chat.scrollTop = chat.scrollHeight;
    }
    if(full.includes("[[ERROR]]")){
      const [answer, err] = full.split("[[ERROR]]");
      if(!answer.trim()) bot.remove();
      throw new Error(err.trim());
    }
    history.push({role:"assistant", content: full});
    if(history.length > 20) history = history.slice(-20);
    addTools(bot, full);
    if($("#speakToggle").checked) speak(full);
  }catch(e){
    history.pop();                               // let the user retry
    if(bot.isConnected && !full.trim()) bot.remove();
    showError(e.message || "Something went wrong. Is the Flask server still running?");
  }finally{
    setBusy(false); input.focus();
  }
}
function addTools(el, text){
  const t = document.createElement("div"); t.className = "tools";
  const sp = document.createElement("button"); sp.className = "mini"; sp.textContent = "🔊 Listen"; sp.onclick = () => speak(text);
  const cp = document.createElement("button"); cp.className = "mini"; cp.textContent = "📋 Copy";
  cp.onclick = () => navigator.clipboard.writeText(text).then(() => cp.textContent = "✅ Copied");
  t.append(sp, cp); el.appendChild(t);
}
function setBusy(b){ busy = b; sendBtn.disabled = b; micBtn.disabled = b; }

sendBtn.onclick = send;
input.addEventListener("keydown", e => { if(e.key === "Enter" && !e.shiftKey){ e.preventDefault(); send(); } });
function autoGrow(){ input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 140) + "px"; }
input.addEventListener("input", autoGrow);

$("#newChat").onclick = () => { if(busy) return; stopSpeaking(); history = []; location.reload(); };

/* ---------------- voice OUTPUT (text-to-speech) ---------------- */
const synth = window.speechSynthesis;
function speakable(t){
  return t.replace(/\[\[ERROR\]\].*/s, "").replace(/[*#_`>|]/g, "")
          .replace(/^\s*[-•]\s+/gm, "").replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, "");
}
function pickVoice(lang){
  const vs = synth.getVoices();
  return vs.find(v => v.lang === lang) || vs.find(v => v.lang.startsWith(lang.split("-")[0])) || null;
}
function speak(text){
  if(!synth){ hint.textContent = "Your browser doesn't support voice output."; return; }
  stopSpeaking();
  const lang = $("#lang").value;
  // Chrome stops long utterances, so speak sentence by sentence
  const parts = speakable(text).match(/[^.!?\n]+[.!?]?/g) || [];
  parts.map(p => p.trim()).filter(Boolean).forEach(p => {
    const u = new SpeechSynthesisUtterance(p);
    u.lang = lang; u.rate = 1.02;
    const v = pickVoice(lang); if(v) u.voice = v;
    synth.speak(u);
  });
}
function stopSpeaking(){ if(synth) synth.cancel(); }
$("#stopVoice").onclick = stopSpeaking;
if(synth) synth.onvoiceschanged = () => synth.getVoices();

/* ---------------- voice INPUT (speech-to-text) ---------------- */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, listening = false;
if(!SR){
  micBtn.title = "Voice input needs Chrome or Edge";
  micBtn.onclick = () => hint.textContent = "🎤 Voice input isn't supported in this browser. Please use Chrome or Edge.";
}else{
  rec = new SR();
  rec.interimResults = true; rec.continuous = false; rec.maxAlternatives = 1;
  let finalText = "";
  rec.onstart = () => { listening = true; finalText = ""; micBtn.classList.add("live"); hint.textContent = "🎤 Listening… speak now"; };
  rec.onresult = e => {
    let interim = "";
    for(let i = e.resultIndex; i < e.results.length; i++){
      const r = e.results[i];
      if(r.isFinal) finalText += r[0].transcript; else interim += r[0].transcript;
    }
    input.value = (finalText + interim).trim(); autoGrow();
  };
  rec.onerror = e => {
    const msg = {
      "not-allowed":"Microphone blocked. Click the 🔒 icon in the address bar and allow the microphone.",
      "service-not-allowed":"Microphone blocked. Allow it in your browser settings.",
      "no-speech":"I didn't hear anything. Tap 🎤 and try again.",
      "audio-capture":"No microphone found. Plug one in and try again.",
      "network":"Voice input needs an internet connection."
    }[e.error] || ("Voice error: " + e.error);
    hint.textContent = "🎤 " + msg;
  };
  rec.onend = () => {
    listening = false; micBtn.classList.remove("live");
    if(input.value.trim() && !busy){ hint.textContent = ""; send(); }   // auto-send what was said
    else if(hint.textContent.includes("Listening")) hint.textContent = "";
  };
  micBtn.onclick = () => {
    if(busy) return;
    if(listening){ rec.stop(); return; }
    stopSpeaking();
    input.value = "";
    rec.lang = $("#lang").value;
    try{ rec.start(); }catch(e){ hint.textContent = "🎤 Couldn't start the microphone. Try again."; }
  };
}
input.focus();
</script>
</body>
</html>"""


if __name__ == "__main__":
    ok, msg = check_ollama()
    print(("✅ " if ok else "⚠️ ") + msg)
    print("🌐 Open http://127.0.0.1:5000 in Chrome or Edge (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

/* common.js — ฟังก์ชันที่ใช้ร่วมกันทุกหน้า */

const API = "";

function saveSession(s){ localStorage.setItem("lha_session", JSON.stringify(s)); }
function getSession(){
  try { return JSON.parse(localStorage.getItem("lha_session")); }
  catch(e){ return null; }
}
function clearSession(){ localStorage.removeItem("lha_session"); }

function requireLogin(){
  const s = getSession();
  if(!s || !s.token){ location.href = "index.html"; return null; }
  return s;
}

async function api(path, options = {}){
  const s = getSession();
  const headers = Object.assign(
    {"Content-Type": "application/json"},
    s && s.token ? {"X-Token": s.token} : {},
    options.headers || {}
  );
  const res = await fetch(API + path, Object.assign({}, options, {headers}));
  if(res.status === 401){ clearSession(); location.href = "index.html"; throw new Error("session หมดอายุ"); }
  if(!res.ok){
    let detail = "เกิดข้อผิดพลาด";
    try { detail = (await res.json()).detail || detail; } catch(e){}
    throw new Error(detail);
  }
  return res.json();
}

async function apiForm(path, formData){
  const s = getSession();
  const res = await fetch(API + path, {
    method: "POST",
    headers: s && s.token ? {"X-Token": s.token} : {},
    body: formData
  });
  if(res.status === 401){ clearSession(); location.href = "index.html"; throw new Error("session หมดอายุ"); }
  if(!res.ok){
    let detail = "เกิดข้อผิดพลาด";
    try { detail = (await res.json()).detail || detail; } catch(e){}
    throw new Error(detail);
  }
  return res.json();
}

function renderTopbar(activePage){
  const s = getSession();
  if(!s) return;
  const nav = [
    {href:"select.html", label:"เลือกโหมด", key:"select"},
    {href:"medicine.html", label:"ปรึกษาอาการ", key:"medicine"},
    {href:"wound.html", label:"ประเมินบาดแผล", key:"wound"},
    {href:"stock.html", label:"บัญชียา", key:"stock"},
  ].map(n => `<a href="${n.href}" style="${n.key===activePage?'text-decoration:underline;font-weight:700;':''}">${n.label}</a>`).join("");

  document.body.insertAdjacentHTML("afterbegin", `
    <div class="topbar">
      <div>
        <div class="brand">Local Health AI</div>
        <div class="meta">${s.unit_name} — ${s.display}</div>
      </div>
      <div class="nav">${nav}<a href="#" onclick="doLogout(event)">ออกจากระบบ</a></div>
    </div>`);
}

async function doLogout(e){
  if(e) e.preventDefault();
  try { await api("/api/auth/logout", {method:"POST"}); } catch(err){}
  clearSession();
  location.href = "index.html";
}

function escapeHtml(str){
  return String(str == null ? "" : str)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;").replace(/'/g,"&#039;");
}

function confidenceBadge(level){
  const map = {high:["high","ความมั่นใจสูง"], medium:["","ความมั่นใจปานกลาง"], low:["low","ความมั่นใจต่ำ"]};
  const [cls, label] = map[level] || map.medium;
  return `<span class="badge ${cls}">${label}</span>`;
}

/* ── ความยินยอมตาม PDPA (ข้อ 25) ─────────────────────────────
   ถามทุกครั้งก่อนบันทึก ข้อความกำหนดตายตัวจาก backend ไม่ให้ AI แต่งเอง */
async function askConsentAndSave(mode, transcript, outcome, referred, redFlagIds){
  let prompt;
  try { prompt = await api("/api/consent/prompt"); }
  catch(e){ return; }

  const sid = (await api("/api/consent/session", {method:"POST"})).session_id;

  return new Promise(resolve => {
    const el = document.createElement("div");
    el.className = "modal-backdrop show";
    el.innerHTML = `
      <div class="modal consent">
        <h3>ขอความยินยอมก่อนบันทึกข้อมูล</h3>
        <div class="body">${escapeHtml(prompt.text)}</div>
        <div class="btn-row">
          <button id="cYes">${escapeHtml(prompt.options[0])}</button>
          <button class="secondary" id="cNo">${escapeHtml(prompt.options[1])}</button>
        </div>
      </div>`;
    document.body.appendChild(el);

    const finish = async (given) => {
      el.remove();
      try {
        const r = await api("/api/consent/save", {
          method:"POST",
          body: JSON.stringify({
            session_id: sid, mode, consent_given: given,
            transcript: transcript || [], outcome: outcome || "",
            referred: !!referred, red_flag_ids: redFlagIds || []
          })
        });
        alert(r.message);
        resolve(r);
      } catch(err){ alert("บันทึกไม่สำเร็จ: " + err.message); resolve(null); }
    };
    el.querySelector("#cYes").onclick = () => finish(true);
    el.querySelector("#cNo").onclick  = () => finish(false);
  });
}

/* ── แสดงรายการอ้างอิงจาก PubMed (ข้อ 20, 21) ───────────── */
function renderEvidence(evidence){
  if(!evidence || !evidence.length){
    return `<p class="muted">ไม่มีผลการค้นจาก PubMed สำหรับคำถามนี้ คำแนะนำข้างต้นจึงอ้างอิงจากบัญชียาของหน่วยบริการและความรู้ทั่วไปของโมเดลเท่านั้น</p>`;
  }
  const items = evidence.map(e => `
    <li>
      <a href="${escapeHtml(e.url)}" target="_blank" rel="noopener">${escapeHtml(e.title)}</a>
      <span class="muted"> — PMID ${escapeHtml(e.pmid)}${e.year ? ", " + escapeHtml(e.year) : ""}${e.from_cache ? " (จาก cache)" : ""}</span>
    </li>`).join("");
  return `<ul class="tight evidence">${items}</ul>`;
}

function renderSources(sources){
  if(!sources || !sources.length) return "";
  return `<div class="result-block"><h3>แหล่งที่มาของคำแนะนำ</h3>
    <ul class="tight">${sources.map(s => `<li>${escapeHtml(s)}</li>`).join("")}</ul></div>`;
}

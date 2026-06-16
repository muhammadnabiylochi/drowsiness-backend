const $ = id => document.getElementById(id);

const STATE_LABELS = {
  awake: 'HUSHYOR',
  drowsy: 'UYQULI',
  drowsy_yawning: 'UYQULI + ESNAMOQDA',
  yawning: 'ESNAMOQDA',
  no_face: 'YUZ TOPILMADI',
  falling_forward: 'BOSH OLDINGA TUSHMOQDA',
  falling_back: 'BOSH ORQAGA TUSHMOQDA',
  falling_left: 'BOSH CHAPGA TUSHMOQDA',
  falling_right: "BOSH O'NGGA TUSHMOQDA",
};

let ws = null;
let prevState = '';
let frameInterval = null;
let facingMode = 'user';
let stream = null;
let frameCount = 0;
let fpsTimer = Date.now();

// ═══ Camera ═══

window.startCamera = async function () {
  // Check if browser supports camera
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showStatus('Bu brauzer kamerani qollab-quvvatlamaydi', 'danger');
    addLog('getUserMedia qollab-quvvatlanmaydi', 'alert');
    return;
  }

  showStatus('Kamera ruxsati so\'ralmoqda...', 'info');

  try {
    if (stream) stream.getTracks().forEach(t => t.stop());

    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 320 }, height: { ideal: 240 } },
      audio: false,
    });

    const video = $('localVideo');
    video.srcObject = stream;
    await video.play().catch(() => {});

    $('startBtn').style.display = 'none';
    $('flipBtn').style.display = 'inline-flex';
    $('liveBadge').style.display = '';
    showStatus('Kamera yoqildi. Server bilan ulanmoqda...', 'success');
    addLog('Kamera yoqildi (' + facingMode + ')', 'info');
    connectWS();
  } catch (e) {
    const msg = e.name === 'NotAllowedError'
      ? 'Kamera ruxsati rad etildi. Brauzer sozlamalaridan ruxsat bering.'
      : e.name === 'NotFoundError'
      ? 'Kamera topilmadi.'
      : 'Kamera xatosi: ' + e.message;
    showStatus(msg, 'danger');
    addLog(msg, 'alert');
  }
};

window.flipCamera = function () {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  window.startCamera();
};

function showStatus(msg, type) {
  let el = $('statusMsg');
  if (!el) {
    el = document.createElement('div');
    el.id = 'statusMsg';
    el.className = 'mt-2 small';
    $('startBtn').parentNode.appendChild(el);
  }
  const colors = { info: 'text-cyan', success: 'text-green', danger: 'text-danger', warn: 'text-warning' };
  el.className = 'mt-2 small ' + (colors[type] || 'text-secondary');
  el.textContent = msg;
}

// ═══ WebSocket ═══

function connectWS() {
  if (ws) { try { ws.close(); } catch(e) {} }
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/ws/stream`;
  addLog('WS ulanmoqda: ' + url, 'info');

  try {
    ws = new WebSocket(url);
  } catch(e) {
    addLog('WS yaratishda xato: ' + e.message, 'alert');
    return;
  }

  ws.onopen = () => {
    $('wsDot').className = 'live-dot on';
    $('wsLabel').textContent = 'Ulandi';
    showStatus('Ulandi! Kamera tahlil qilinmoqda...', 'success');
    addLog('Server bilan ulandi', 'info');
    startFrameLoop();
  };

  ws.onmessage = e => {
    let d;
    try { d = JSON.parse(e.data); } catch { return; }

    applyState(d.state);
    if ($('vC')) $('vC').textContent = d.confidence != null ? d.confidence.toFixed(2) : '--';
    if ($('vEar')) $('vEar').textContent = d.ear_avg != null ? d.ear_avg.toFixed(3) : '--';
    if ($('vMar')) $('vMar').textContent = d.mar != null ? d.mar.toFixed(3) : '--';
    if ($('vBlink')) $('vBlink').textContent = d.total_blinks ?? '--';
    if ($('vPerclos')) $('vPerclos').textContent = d.perclos != null ? (d.perclos * 100).toFixed(1) + '%' : '--';

    frameCount++;
    const now = Date.now();
    if (now - fpsTimer >= 1000) {
      $('fpsLabel').textContent = frameCount + ' FPS';
      frameCount = 0;
      fpsTimer = now;
    }

    if (d.state !== prevState) {
      if (d.state === 'drowsy' || d.state === 'drowsy_yawning')
        addLog('⚠ UYQU aniqlandi!', 'alert');
      else if (d.state === 'yawning')
        addLog('Esnash aniqlandi', 'warn');
      else if (d.state === 'awake' && prevState)
        addLog('Haydovchi hushyor', 'info');
      else if (d.state?.startsWith('falling_'))
        addLog('Bosh tushmoqda: ' + (STATE_LABELS[d.state] || d.state), 'alert');
      else if (d.state === 'no_face')
        addLog('Yuz topilmadi', 'warn');
      else if (d.state === 'error')
        addLog('Server xatosi: ' + (d.message || ''), 'alert');
      prevState = d.state;
    }
  };

  ws.onclose = (ev) => {
    $('wsDot').className = 'live-dot off';
    $('wsLabel').textContent = 'Uzildi';
    stopFrameLoop();
    addLog('WS uzildi (kod: ' + ev.code + '). Qayta ulanmoqda...', 'warn');
    if (stream) setTimeout(connectWS, 3000);
  };

  ws.onerror = () => {
    addLog('WS xatosi yuz berdi', 'alert');
  };
}

// ═══ Frame capture ═══

function startFrameLoop() {
  stopFrameLoop();
  frameInterval = setInterval(sendFrame, 100);
}

function stopFrameLoop() {
  if (frameInterval) { clearInterval(frameInterval); frameInterval = null; }
}

function sendFrame() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const video = $('localVideo');
  if (!video || !video.videoWidth || video.paused || video.ended) return;

  const canvas = $('captureCanvas');
  const w = 320;
  const h = Math.round(w * video.videoHeight / video.videoWidth) || 240;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
  const b64 = canvas.toDataURL('image/jpeg', 0.7);
  try {
    ws.send(JSON.stringify({ type: 'frame', data: b64 }));
  } catch(e) {}
}

// ═══ State UI ═══

function applyState(state) {
  const text = $('stateText');
  const card = $('stateCard');
  const icon = $('stateIcon');
  if (!text) return;
  text.className = 'state-badge ';
  card.style.borderTop = '';
  const label = STATE_LABELS[state] || (state || "NOMA'LUM").toUpperCase().replace(/_/g, ' ');
  if (state === 'drowsy' || state === 'drowsy_yawning') {
    text.classList.add('state-drowsy');
    card.style.borderTop = '3px solid #d63939';
    icon.className = 'ti ti-alert-octagon';
  } else if (state === 'yawning') {
    text.classList.add('state-yawning');
    card.style.borderTop = '3px solid #f76707';
    icon.className = 'ti ti-mood-tongue-wink';
  } else if (state === 'awake') {
    text.classList.add('state-awake');
    card.style.borderTop = '3px solid #2fb344';
    icon.className = 'ti ti-steering-wheel';
  } else {
    text.classList.add('state-default');
    icon.className = 'ti ti-steering-wheel';
  }
  text.textContent = label;
}

// ═══ Log ═══

function addLog(msg, type) {
  const box = $('log');
  if (!box) return;
  const d = document.createElement('div');
  const t = new Date().toLocaleTimeString('uz-UZ', { hour12: false });
  const cls = type === 'alert' ? 'text-danger' : type === 'warn' ? 'text-warning' : 'text-cyan';
  d.innerHTML = `<span class="${cls}">[${t}]</span> <span class="text-secondary">${msg}</span>`;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
  while (box.children.length > 100) box.removeChild(box.firstChild);
}

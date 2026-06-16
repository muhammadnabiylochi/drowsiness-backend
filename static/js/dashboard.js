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

async function startCamera() {
  try {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
    }
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 320 }, height: { ideal: 240 } },
      audio: false,
    });
    $('localVideo').srcObject = stream;
    $('startBtn').style.display = 'none';
    $('flipBtn').style.display = '';
    $('liveBadge').style.display = '';
    addLog('Kamera yoqildi', 'info');
    connectWS();
  } catch (e) {
    addLog('Kamera xatosi: ' + e.message, 'alert');
  }
}

function flipCamera() {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  startCamera();
}

// ═══ WebSocket ═══

function connectWS() {
  if (ws) ws.close();
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws/stream`);

  ws.onopen = () => {
    $('wsDot').className = 'live-dot on';
    $('wsLabel').textContent = 'Ulandi';
    addLog('Server bilan ulandi', 'info');
    startFrameLoop();
  };

  ws.onmessage = e => {
    const d = JSON.parse(e.data);
    applyState(d.state);
    if ($('vC')) $('vC').textContent = d.confidence?.toFixed(2) ?? '--';
    if ($('vEar')) $('vEar').textContent = d.ear_avg?.toFixed(3) ?? '--';
    if ($('vMar')) $('vMar').textContent = d.mar?.toFixed(3) ?? '--';
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
        addLog('UYQU aniqlandi!', 'alert');
      else if (d.state === 'yawning')
        addLog('Esnash aniqlandi', 'warn');
      else if (d.state === 'awake' && prevState)
        addLog('Haydovchi hushyor', 'info');
      else if (d.state?.startsWith('falling_'))
        addLog('Bosh tushmoqda: ' + (STATE_LABELS[d.state] || d.state), 'alert');
      prevState = d.state;
    }
  };

  ws.onclose = () => {
    $('wsDot').className = 'live-dot off';
    $('wsLabel').textContent = 'Uzildi';
    stopFrameLoop();
    if (stream) setTimeout(connectWS, 2000);
  };

  ws.onerror = () => ws.close();
}

// ═══ Frame capture loop ═══

function startFrameLoop() {
  stopFrameLoop();
  frameInterval = setInterval(sendFrame, 100); // 10 fps
}

function stopFrameLoop() {
  if (frameInterval) { clearInterval(frameInterval); frameInterval = null; }
}

function sendFrame() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const video = $('localVideo');
  if (!video.videoWidth) return;
  const canvas = $('captureCanvas');
  canvas.width = 320;
  canvas.height = Math.round(320 * video.videoHeight / video.videoWidth);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const b64 = canvas.toDataURL('image/jpeg', 0.7);
  ws.send(JSON.stringify({ type: 'frame', data: b64 }));
}

// ═══ State UI ═══

function applyState(state) {
  const text = $('stateText');
  const card = $('stateCard');
  const icon = $('stateIcon');
  text.className = 'state-badge ';
  card.style.borderTop = '';
  const label = STATE_LABELS[state] || (state || 'NOMA\'LUM').toUpperCase().replace(/_/g, ' ');
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
  const d = document.createElement('div');
  const t = new Date().toLocaleTimeString('uz-UZ', { hour12: false });
  const cls = type === 'alert' ? 'text-danger' : type === 'warn' ? 'text-warning' : 'text-cyan';
  d.innerHTML = `<span class="${cls}">[${t}]</span> <span class="text-secondary">${msg}</span>`;
  $('log').appendChild(d);
  $('log').scrollTop = $('log').scrollHeight;
  while ($('log').children.length > 100) $('log').removeChild($('log').firstChild);
}

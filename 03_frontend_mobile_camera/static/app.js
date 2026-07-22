const state = {
  running: false,
  refreshTimer: null,
  lineTimer: null,
};

const elements = {
  stream: document.querySelector("#stream"),
  emptyVideo: document.querySelector("#emptyVideo"),
  runningStatus: document.querySelector("#runningStatus"),
  totalCount: document.querySelector("#totalCount"),
  fpsValue: document.querySelector("#fpsValue"),
  trackCount: document.querySelector("#trackCount"),
  sessionId: document.querySelector("#sessionId"),
  resolution: document.querySelector("#resolution"),
  errorMessage: document.querySelector("#errorMessage"),
  cameraSource: document.querySelector("#cameraSource"),
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  lineButton: document.querySelector("#lineButton"),
  historyList: document.querySelector("#historyList"),
  lineInputs: {
    x1: document.querySelector("#x1"),
    y1: document.querySelector("#y1"),
    x2: document.querySelector("#x2"),
    y2: document.querySelector("#y2"),
  },
};

elements.startButton.addEventListener("click", startSession);
elements.stopButton.addEventListener("click", stopSession);
elements.lineButton.addEventListener("click", updateLine);

Object.values(elements.lineInputs).forEach((input) => {
  input.addEventListener("input", () => {
    window.clearTimeout(state.lineTimer);
    state.lineTimer = window.setTimeout(updateLine, 250);
  });
});

refreshStatus();
refreshHistory();
state.refreshTimer = window.setInterval(() => {
  refreshStatus();
  refreshHistory();
}, 900);

async function startSession() {
  const source = elements.cameraSource.value.trim() || "0";
  const response = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source }),
  });
  await handleResponse(response);
  elements.stream.src = `/api/stream?cache=${Date.now()}`;
  await refreshStatus();
}

async function stopSession() {
  const response = await fetch("/api/stop", { method: "POST" });
  await handleResponse(response);
  elements.stream.removeAttribute("src");
  await refreshStatus();
  await refreshHistory();
}

async function updateLine() {
  const response = await fetch("/api/line", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(readLine()),
  });
  await handleResponse(response);
  await refreshStatus();
}

async function refreshStatus() {
  const response = await fetch("/api/status");
  const data = await response.json();
  state.running = Boolean(data.running);

  elements.runningStatus.textContent = state.running ? "rodando" : "parado";
  elements.runningStatus.classList.toggle("running", state.running);
  elements.emptyVideo.classList.toggle("hidden", state.running);
  elements.totalCount.textContent = data.total ?? 0;
  elements.fpsValue.textContent = Number(data.fps ?? 0).toFixed(1);
  elements.trackCount.textContent = data.tracks ?? 0;
  elements.sessionId.textContent = data.session_id ?? "-";
  elements.resolution.textContent =
    data.frame_width && data.frame_height ? `${data.frame_width} x ${data.frame_height}` : "-";
  elements.errorMessage.textContent = data.error || "-";

  if (data.line) {
    for (const [key, input] of Object.entries(elements.lineInputs)) {
      if (document.activeElement !== input) {
        input.value = data.line[key];
      }
    }
  }
}

async function refreshHistory() {
  const response = await fetch("/api/sessions?limit=5");
  const sessions = await response.json();

  if (!sessions.length) {
    elements.historyList.innerHTML = '<div class="history-item"><span>Nenhuma sessao registrada.</span></div>';
    return;
  }

  elements.historyList.innerHTML = sessions
    .map((session) => {
      const status = session.status === "running" ? "em andamento" : "finalizada";
      return `
        <article class="history-item">
          <strong>#${session.id} | total ${session.total} | ${status}</strong>
          <span>${session.started_at} | fonte ${session.source}</span>
        </article>
      `;
    })
    .join("");
}

function readLine() {
  return Object.fromEntries(
    Object.entries(elements.lineInputs).map(([key, input]) => [key, Number(input.value)])
  );
}

async function handleResponse(response) {
  if (response.ok) {
    return response.json();
  }

  const payload = await response.json().catch(() => ({}));
  throw new Error(payload.detail || "Erro inesperado na API.");
}

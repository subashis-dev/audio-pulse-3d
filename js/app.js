/**
 * SONIX3D - Main Application Coordinator & UI Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize 3D Visualizer
  const visualizer = new Visualizer3D('viewport-3d');

  // DOM Elements
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('audio-upload');
  const playBtn = document.getElementById('btn-play');
  const playIcon = document.getElementById('play-icon');
  const trackNameEl = document.getElementById('track-title');
  const trackArtistEl = document.getElementById('track-artist');
  const currentTimeEl = document.getElementById('time-current');
  const totalTimeEl = document.getElementById('time-total');
  const volumeSlider = document.getElementById('volume-slider');
  const waveformCanvas = document.getElementById('waveform-canvas');
  const waveformProgress = document.getElementById('waveform-progress');
  const waveformContainer = document.getElementById('waveform-container');

  // HUD Readouts
  const bpmValueEl = document.getElementById('hud-bpm');
  const keyValueEl = document.getElementById('hud-key');
  const camelotEl = document.getElementById('hud-camelot');
  const dynamicRangeEl = document.getElementById('hud-dynamic-range');
  const peakDbEl = document.getElementById('hud-peak');
  const sampleRateEl = document.getElementById('hud-sample-rate');

  // Neural AI & Mood
  const topGenreBadge = document.getElementById('top-genre-badge');
  const genreListContainer = document.getElementById('genre-breakdown-list');
  const moodEnergyBar = document.getElementById('mood-energy');
  const moodValenceBar = document.getElementById('mood-valence');
  const moodDanceBar = document.getElementById('mood-dance');
  const moodAcousticBar = document.getElementById('mood-acoustic');

  // 7-Band Bars
  const bandBars = {
    subBass: document.getElementById('band-sub-bass'),
    bass: document.getElementById('band-bass'),
    lowMid: document.getElementById('band-low-mid'),
    mid: document.getElementById('band-mid'),
    highMid: document.getElementById('band-high-mid'),
    presence: document.getElementById('band-presence'),
    brilliance: document.getElementById('band-brilliance')
  };

  let currentAnalysis = null;
  let isDraggingScrub = false;

  // Formatting helpers
  const formatTime = (secs) => {
    if (isNaN(secs) || secs < 0) return "0:00";
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  /* ----------------- Visualizer Mode & Theme Controls ----------------- */
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      visualizer.setMode(btn.dataset.mode);
    });
  });

  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      visualizer.setTheme(btn.dataset.theme);
    });
  });

  /* ----------------- Audio Loading & Pipeline ----------------- */
  async function processAudioBuffer(fileOrName, customBuffer = null) {
    // Show loading state
    trackNameEl.textContent = "Analyzing & Deconstructing Audio...";
    trackArtistEl.textContent = "Running DSP Algorithms & Neural Spectrogram AI...";

    try {
      let analysis;
      if (customBuffer) {
        window.SonixDsp.audioBuffer = customBuffer;
        window.SonixDsp.duration = customBuffer.duration;
        analysis = await window.SonixDsp.analyzeFullTrack(customBuffer);
      } else {
        analysis = await window.SonixDsp.loadAudioFile(fileOrName);
      }

      currentAnalysis = analysis;
      const fileName = fileOrName.name || fileOrName;
      trackNameEl.textContent = fileName.replace(/\.[^/.]+$/, "");
      trackArtistEl.textContent = `${analysis.duration.toFixed(1)}s • ${analysis.sampleRate} Hz • ${analysis.numberOfChannels === 2 ? 'Stereo' : 'Mono'}`;
      totalTimeEl.textContent = formatTime(analysis.duration);

      // Render DSP Metrics
      bpmValueEl.textContent = analysis.bpm;
      keyValueEl.textContent = analysis.key;
      camelotEl.textContent = analysis.camelotCode;
      dynamicRangeEl.textContent = `${analysis.dynamicRange} dB`;
      peakDbEl.textContent = `${analysis.peakDb} dBFS`;
      sampleRateEl.textContent = `${(analysis.sampleRate / 1000).toFixed(1)} kHz`;

      // Draw Waveform
      drawWaveform(analysis.waveformPeaks);

      // Run Neural Spectrogram AI Classification
      const aiResults = await window.SonixClassifier.classify(window.SonixDsp.audioBuffer, analysis);
      renderAiPredictions(aiResults);

      // Start Playback
      window.SonixDsp.play(0);
      updatePlayButton(true);

    } catch (err) {
      console.error("Audio processing error:", err);
      trackNameEl.textContent = "Error decoding audio file";
      trackArtistEl.textContent = "Please try a standard MP3 or WAV file";
    }
  }

  function renderAiPredictions(ai) {
    topGenreBadge.textContent = `${ai.topGenre.genre} (${ai.topGenre.probability}%)`;
    
    // Populate Genre Breakdown Bars
    genreListContainer.innerHTML = '';
    ai.genres.forEach(g => {
      const item = document.createElement('div');
      item.className = 'genre-bar-item';
      item.innerHTML = `
        <div class="genre-label">
          <span>${g.genre}</span>
          <span>${g.probability}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width: ${g.probability}%;"></div>
        </div>
      `;
      genreListContainer.appendChild(item);
    });

    // Populate Mood Metrics
    if (moodEnergyBar) moodEnergyBar.style.width = `${ai.mood.energy}%`;
    if (moodValenceBar) moodValenceBar.style.width = `${ai.mood.valence}%`;
    if (moodDanceBar) moodDanceBar.style.width = `${ai.mood.danceability}%`;
    if (moodAcousticBar) moodAcousticBar.style.width = `${ai.mood.acousticness}%`;

    document.getElementById('val-energy').textContent = `${ai.mood.energy}%`;
    document.getElementById('val-valence').textContent = `${ai.mood.valence}%`;
    document.getElementById('val-dance').textContent = `${ai.mood.danceability}%`;
    document.getElementById('val-acoustic').textContent = `${ai.mood.acousticness}%`;
  }

  /* ----------------- Waveform Canvas ----------------- */
  function drawWaveform(peaks) {
    if (!peaks || !peaks.length) return;
    const canvas = waveformCanvas;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = 54;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const barWidth = canvas.width / peaks.length;
    const midY = canvas.height / 2;

    const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
    grad.addColorStop(0, '#00f3ff');
    grad.addColorStop(0.5, '#00b4d8');
    grad.addColorStop(1, '#ff007f');

    ctx.fillStyle = grad;
    for (let i = 0; i < peaks.length; i++) {
      const h = Math.max(3, peaks[i] * canvas.height * 0.95);
      const x = i * barWidth;
      const y = midY - h / 2;
      ctx.fillRect(x, y, Math.max(1.5, barWidth - 1), h);
    }
  }

  /* ----------------- Playback & UI Update Loop ----------------- */
  function updatePlayButton(isPlaying) {
    if (isPlaying) {
      playIcon.innerHTML = `<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>`;
      playBtn.setAttribute('title', 'Pause');
    } else {
      playIcon.innerHTML = `<path d="M8 5v14l11-7z"/>`;
      playBtn.setAttribute('title', 'Play');
    }
  }

  playBtn.addEventListener('click', () => {
    if (!window.SonixDsp.audioBuffer) return;
    if (window.SonixDsp.isPlaying) {
      window.SonixDsp.pause();
      updatePlayButton(false);
    } else {
      window.SonixDsp.play();
      updatePlayButton(true);
    }
  });

  volumeSlider.addEventListener('input', (e) => {
    window.SonixDsp.setVolume(parseFloat(e.target.value));
  });

  // Waveform Scrubbing
  const seekAtPosition = (e) => {
    if (!window.SonixDsp.duration) return;
    const rect = waveformContainer.getBoundingClientRect();
    const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const targetSec = pos * window.SonixDsp.duration;
    window.SonixDsp.seek(targetSec);
    waveformProgress.style.width = `${pos * 100}%`;
  };

  waveformContainer.addEventListener('mousedown', (e) => {
    isDraggingScrub = true;
    seekAtPosition(e);
  });

  window.addEventListener('mousemove', (e) => {
    if (isDraggingScrub) seekAtPosition(e);
  });

  window.addEventListener('mouseup', () => {
    isDraggingScrub = false;
  });

  /* ----------------- Live HUD UI Update Loop (60 FPS) ----------------- */
  function updateUiLoop() {
    requestAnimationFrame(updateUiLoop);

    if (!window.SonixDsp) return;
    const curTime = window.SonixDsp.getCurrentTime();
    const duration = window.SonixDsp.duration;

    if (duration > 0 && !isDraggingScrub) {
      currentTimeEl.textContent = formatTime(curTime);
      const progressPercent = (curTime / duration) * 100;
      waveformProgress.style.width = `${progressPercent}%`;
    }

    if (window.SonixDsp.isPlaying) {
      const data = window.SonixDsp.getRealtimeData();
      if (data && data.bands) {
        if (bandBars.subBass) bandBars.subBass.style.height = `${Math.min(100, data.bands.subBass * 140)}%`;
        if (bandBars.bass) bandBars.bass.style.height = `${Math.min(100, data.bands.bass * 130)}%`;
        if (bandBars.lowMid) bandBars.lowMid.style.height = `${Math.min(100, data.bands.lowMid * 120)}%`;
        if (bandBars.mid) bandBars.mid.style.height = `${Math.min(100, data.bands.mid * 120)}%`;
        if (bandBars.highMid) bandBars.highMid.style.height = `${Math.min(100, data.bands.highMid * 110)}%`;
        if (bandBars.presence) bandBars.presence.style.height = `${Math.min(100, data.bands.presence * 100)}%`;
        if (bandBars.brilliance) bandBars.brilliance.style.height = `${Math.min(100, data.bands.brilliance * 90)}%`;
      }
    }
  }
  updateUiLoop();

  /* ----------------- Drag & Drop / File Input ----------------- */
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      processAudioBuffer(e.target.files[0]);
    }
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
      processAudioBuffer(e.dataTransfer.files[0]);
    }
  });

  /* ----------------- Built-in Demo Synthesizer ----------------- */
  // Allows immediate testing with 0 downloads! Generates rich Synthwave & Cyberpunk beats
  async function generateDemoTrack(type = 'synthwave') {
    window.SonixDsp.initAudioContext();
    const ctx = window.SonixDsp.audioCtx;
    const sampleRate = ctx.sampleRate;
    const duration = 24.0; // 24 seconds
    const buffer = ctx.createBuffer(2, sampleRate * duration, sampleRate);
    const left = buffer.getChannelData(0);
    const right = buffer.getChannelData(1);

    const bpm = (type === 'synthwave') ? 128 : 95;
    const beatInterval = 60 / bpm;
    const totalBeats = Math.floor(duration / beatInterval);

    // Root chord progression in A Minor (Am - F - C - G)
    const chords = [
      [220, 261.63, 329.63], // Am
      [174.61, 220, 261.63], // F
      [130.81, 164.81, 196.0], // C
      [196.0, 246.94, 293.66]  // G
    ];

    for (let b = 0; b < totalBeats; b++) {
      const beatTime = b * beatInterval;
      const startSample = Math.floor(beatTime * sampleRate);
      const chord = chords[Math.floor(b / 8) % chords.length];

      // 1. Kick on every 4/4 beat
      if (b % 1 === 0) {
        const kickLen = Math.floor(sampleRate * 0.28);
        for (let i = 0; i < kickLen; i++) {
          const t = i / sampleRate;
          const freq = 140 * Math.exp(-t * 22) + 40;
          const amp = Math.exp(-t * 12);
          const val = Math.sin(2 * Math.PI * freq * t) * amp * 0.85;
          if (startSample + i < left.length) {
            left[startSample + i] += val;
            right[startSample + i] += val;
          }
        }
      }

      // 2. Snare / Clap on beats 2 and 4
      if (b % 2 === 1) {
        const snareLen = Math.floor(sampleRate * 0.2);
        for (let i = 0; i < snareLen; i++) {
          const t = i / sampleRate;
          const noise = (Math.random() * 2 - 1) * Math.exp(-t * 16) * 0.45;
          const body = Math.sin(2 * Math.PI * 180 * t) * Math.exp(-t * 20) * 0.4;
          if (startSample + i < left.length) {
            left[startSample + i] += noise + body;
            right[startSample + i] += noise + body;
          }
        }
      }

      // 3. Arpeggiated 16th note synth bassline
      for (let sub = 0; sub < 4; sub++) {
        const subTime = beatTime + (sub * beatInterval / 4);
        const subSample = Math.floor(subTime * sampleRate);
        const noteFreq = chord[sub % chord.length] * (sub % 2 === 0 ? 0.5 : 1.0);
        const noteLen = Math.floor(sampleRate * (beatInterval / 4) * 0.85);

        for (let i = 0; i < noteLen; i++) {
          const t = i / sampleRate;
          const saw = (2 * ((t * noteFreq) % 1) - 1) * Math.exp(-t * 8) * 0.28;
          if (subSample + i < left.length) {
            left[subSample + i] += saw;
            right[subSample + i] += saw * 0.95;
          }
        }
      }
    }

    // Master normalize
    let maxAmp = 0;
    for (let i = 0; i < left.length; i++) {
      if (Math.abs(left[i]) > maxAmp) maxAmp = Math.abs(left[i]);
    }
    if (maxAmp > 0) {
      for (let i = 0; i < left.length; i++) {
        left[i] = (left[i] / maxAmp) * 0.92;
        right[i] = (right[i] / maxAmp) * 0.92;
      }
    }

    const demoName = type === 'synthwave' ? "Cyber_Neon_Overdrive_128BPM.wav" : "Sub_Bass_Groove_95BPM.wav";
    processAudioBuffer(demoName, buffer);
  }

  document.querySelectorAll('.demo-track-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      generateDemoTrack(btn.dataset.track);
    });
  });

  /* ----------------- Export Report ----------------- */
  const exportBtn = document.getElementById('btn-export');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      if (!currentAnalysis) {
        alert("Please load or play a track first to generate the intelligence report!");
        return;
      }
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentAnalysis, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `Sonix3D_Intelligence_${trackNameEl.textContent.trim()}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    });
  }

  // Auto-load demo on first visit
  setTimeout(() => {
    generateDemoTrack('synthwave');
  }, 400);
});

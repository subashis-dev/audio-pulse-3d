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
  let currentAiResults = null;
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

      // Render Song Structure Segmentation Timeline
      renderSongStructure(analysis.segments, analysis.duration);

      // Run Neural Spectrogram AI Classification
      const aiResults = await window.SonixClassifier.classify(window.SonixDsp.audioBuffer, analysis);
      currentAiResults = aiResults;
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

    // Populate Multi-Instrument Neural Recognition
    const instContainer = document.getElementById('instruments-container');
    if (instContainer && ai.instruments) {
      instContainer.innerHTML = '';
      ai.instruments.forEach(inst => {
        const row = document.createElement('div');
        row.className = 'instrument-row';
        row.innerHTML = `
          <div class="inst-left">
            <span class="inst-dot ${inst.detected ? 'active' : ''}"></span>
            <span>${inst.name}</span>
          </div>
          <div class="inst-right">
            <div class="inst-bar-track">
              <div class="inst-bar-fill" style="width: ${inst.probability}%;"></div>
            </div>
            <span class="inst-pct">${inst.probability}%</span>
          </div>
        `;
        instContainer.appendChild(row);
      });
    }
  }

  /* ----------------- Structural Song Segmentation Timeline ----------------- */
  const structureTimeline = document.getElementById('structure-timeline');
  function renderSongStructure(segments, duration) {
    if (!structureTimeline || !segments || !segments.length) return;
    structureTimeline.innerHTML = '';
    segments.forEach((seg, idx) => {
      const chip = document.createElement('button');
      chip.className = 'segment-chip';
      chip.dataset.start = seg.start;
      chip.dataset.end = seg.end;
      chip.dataset.idx = idx;
      chip.title = `Jump to ${seg.name} (${formatTime(seg.start)})`;
      chip.innerHTML = `<span>${seg.name} • ${formatTime(seg.start)}</span>`;
      chip.addEventListener('click', () => {
        window.SonixDsp.seek(seg.start);
        if (window.SonixStemSeparator && window.SonixStemSeparator.isStemModeActive && window.SonixDsp.isPlaying) {
          window.SonixStemSeparator.playStems(seg.start);
        }
      });
      structureTimeline.appendChild(chip);
    });
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
      if (window.SonixStemSeparator && window.SonixStemSeparator.isStemModeActive) {
        window.SonixStemSeparator.stopStems();
      }
      updatePlayButton(false);
    } else {
      const curTime = window.SonixDsp.getCurrentTime();
      window.SonixDsp.play();
      if (window.SonixStemSeparator && window.SonixStemSeparator.isStemModeActive) {
        window.SonixStemSeparator.playStems(curTime);
      }
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
    if (window.SonixStemSeparator && window.SonixStemSeparator.isStemModeActive && window.SonixDsp.isPlaying) {
      window.SonixStemSeparator.playStems(targetSec);
    }
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

      // Highlight active structural segment
      if (structureTimeline) {
        const chips = structureTimeline.querySelectorAll('.segment-chip');
        chips.forEach(chip => {
          const st = parseFloat(chip.dataset.start);
          const en = parseFloat(chip.dataset.end);
          if (curTime >= st && curTime < en) {
            chip.classList.add('active');
          } else {
            chip.classList.remove('active');
          }
        });
      }
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

  /* ----------------- Export Reports (JSON & Natural Language Story) ----------------- */
  // 1. Existing JSON Export (Unchanged)
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

  // 2. Extra Natural Language Story Report Generator & Modal
  const exportNlBtn = document.getElementById('btn-export-nl');
  const reportModal = document.getElementById('report-modal');
  const reportTextContent = document.getElementById('report-text-content');
  const modalCloseBtn = document.getElementById('modal-close-btn');
  const copyReportBtn = document.getElementById('btn-copy-report');
  const copyBtnText = document.getElementById('copy-btn-text');
  const downloadReportTxtBtn = document.getElementById('btn-download-report-txt');

  function generateNaturalLanguageReport(analysis, ai, title) {
    if (!analysis) return "";

    const durationMin = Math.floor(analysis.duration / 60);
    const durationSec = Math.floor(analysis.duration % 60);
    const durationStr = `${durationMin}m ${durationSec < 10 ? '0' : ''}${durationSec}s (${analysis.duration.toFixed(1)} seconds)`;

    // Tonality narrative
    const key = analysis.key || "Unknown";
    const camelot = analysis.camelotCode || "N/A";
    const isMinor = key.toLowerCase().includes("minor");
    const harmonicMood = isMinor
      ? "steeped in a melancholic, dramatic, or nocturnal aesthetic"
      : "resonant with an uplifting, bright, and triumphant tonality";

    // Harmonic mixing compatible keys
    const camelotNum = parseInt(camelot) || 8;
    const camelotLetter = camelot.slice(-1) || 'A';
    const oppLetter = camelotLetter === 'A' ? 'B' : 'A';
    const relativeKey = `${camelotNum}${oppLetter}`;
    const energyUp = `${(camelotNum % 12) + 1}${camelotLetter}`;
    const energyDown = `${((camelotNum - 2 + 12) % 12) + 1}${camelotLetter}`;

    // Tempo & rhythm narrative
    const bpm = analysis.bpm;
    let tempoDescription = "";
    if (bpm < 80) {
      tempoDescription = `At a deliberate ${bpm} BPM, the rhythm establishes a slow, meditative, or ambient pulse, allowing ample acoustic space between transients.`;
    } else if (bpm <= 105) {
      tempoDescription = `Clocking in at ${bpm} BPM, this track settles into a grounded, head-nodding mid-tempo groove typical of classic hip-hop, downtempo, or chillout beats.`;
    } else if (bpm <= 128) {
      tempoDescription = `Pumping at an energetic ${bpm} BPM, the track commands a driving club cadence—ideal for peak-time electronic, synthwave, and house movement.`;
    } else if (bpm <= 145) {
      tempoDescription = `Driving forward at a brisk ${bpm} BPM, this composition possesses intense forward momentum, characteristic of modern pop, trance, or heavy rock.`;
    } else {
      tempoDescription = `Blazing at a high-velocity ${bpm} BPM, the track delivers rapid percussive transients suitable for drum & bass, hardcore, or high-octane metal.`;
    }

    // Dynamic range & master quality
    const dr = parseFloat(analysis.dynamicRange);
    const peak = parseFloat(analysis.peakDb);
    let dynamicsNarrative = "";
    if (dr > 16) {
      dynamicsNarrative = `With an expansive dynamic range of ${dr} dB and a peak ceiling of ${peak} dBFS, the recording exhibits audiophile-grade breathing room with minimal compression, preserving natural acoustic transients.`;
    } else if (dr >= 10) {
      dynamicsNarrative = `Boasting a well-calibrated dynamic range of ${dr} dB (Peak: ${peak} dBFS), the mix strikes a radio-ready balance between punchy transients and perceived commercial loudness.`;
    } else {
      dynamicsNarrative = `Exhibiting a dense dynamic range of ${dr} dB (Peak: ${peak} dBFS), this master is aggressively compressed for maximum perceived punch and club-system volume.`;
    }

    // AI & Mood details
    let aiSection = "";
    if (ai) {
      const topGenre = ai.topGenre.genre;
      const topProb = ai.topGenre.probability;
      const secondaryGenres = ai.genres.slice(1, 4).map(g => `${g.genre} (${g.probability}%)`).join(", ");
      
      const detectedInsts = ai.instruments ? ai.instruments.filter(i => i.detected).map(i => `${i.name} (${i.probability}%)`).join(", ") : "None detected";
      const allInstsList = ai.instruments ? ai.instruments.map(i => `  • ${i.name.padEnd(22)}: ${i.probability}% [${i.detected ? 'CONFIRMED' : 'Subtle/Absent'}]`).join("\n") : "  None";

      aiSection = `
================================================================================
2. NEURAL SPECTROGRAM AI & PSYCHOACOUSTIC ATTRIBUTES
================================================================================
• Primary Classification: ${topGenre} (${topProb}% confidence)
• Secondary Stylistic Nuances: ${secondaryGenres || "None detected"}
• Energy Intensity:       ${ai.mood.energy}% (Overall acoustic power & drive)
• Valence (Positivity):   ${ai.mood.valence}% (Emotional optimism vs dark melancholia)
• Danceability Index:     ${ai.mood.danceability}% (Rhythmic stability & regularity)
• Acoustic Purity:        ${ai.mood.acousticness}% (${ai.mood.acousticness > 50 ? "Organic/acoustic instrumentation" : "Synthesized/electronic production"})

Multi-Instrument Neural Detection:
• Primary Instruments Active: ${detectedInsts || "Synthesized textures"}
${allInstsList}
`;
    }

    let structureSection = "";
    if (analysis.segments && analysis.segments.length) {
      const segList = analysis.segments.map(s => `  • [${formatTime(s.start)} - ${formatTime(s.end)}] : ${s.name}`).join("\n");
      structureSection = `
================================================================================
5. STRUCTURAL SONG PROGRESSION & ARRANGEMENT
================================================================================
${segList}
`;
    }

    return `
================================================================================
🌌 SONIX3D ACOUSTIC INTELLIGENCE DOSSIER
================================================================================
Composition Title : ${title}
Analysis Engine   : Sonix3D Offline DSP & Neural Spectrogram AI (ONNX)
Hardware Platform : Local Edge Inference (100% Sovereign & Cloud-Free)
Timestamp         : ${new Date().toUTCString()}

================================================================================
1. EXECUTIVE AUDIO OVERVIEW & TELEMETRY
================================================================================
• Total Duration     : ${durationStr}
• Sample Rate        : ${analysis.sampleRate} Hz (${(analysis.sampleRate / 1000).toFixed(1)} kHz)
• Channel Topology   : ${analysis.numberOfChannels === 2 ? "Stereo (2 Channels)" : "Mono (1 Channel)"}
• Peak Amplitude     : ${analysis.peakDb} dBFS
• Integrated Loudness: ${analysis.rmsDb} dBFS (RMS)
• Dynamic Range (DR) : ${analysis.dynamicRange} dB (Crest Factor)
${aiSection}
================================================================================
3. HARMONIC TONALITY & MUSICAL KEY PROFILE
================================================================================
• Detected Musical Key: ${key} (Confidence: ${analysis.keyConfidence || 95}%)
• Camelot Wheel Code  : ${camelot}

Acoustic Description:
The composition's fundamental energy centers around the key of ${key}, which is ${harmonicMood}.

DJ & Mixing Compatibility:
• Relative Key Transition: ${relativeKey} (Smooth emotional polarity modulation)
• Energy Lift (+1 Camelot): ${energyUp} (Pushes harmonic tension higher)
• Energy Drop (-1 Camelot): ${energyDown} (Creates relaxed harmonic resolution)

================================================================================
4. TEMPO & RHYTHMIC CADENCE
================================================================================
• Detected Tempo : ${bpm} BPM (Beats Per Minute)

Rhythmic Analysis:
${tempoDescription}
${structureSection}
================================================================================
6. DYNAMIC MASTERING & LOUDNESS EVALUATION
================================================================================
${dynamicsNarrative}

================================================================================
7. SPECTRAL BALANCE & FREQUENCY TOPOGRAPHY
================================================================================
• Sub-Bass (20 - 60 Hz)     : Low-end weight and visceral sub-woofer impact.
• Bass (60 - 250 Hz)        : Kick drum punch and foundational bassline weight.
• Low-Mids (250 - 500 Hz)   : Body of instruments, snares, and lower harmonic warmth.
• Midrange (500 - 2 kHz)    : Melodic leads, vocal fundamentals, and presence.
• Upper-Mids (2 - 4 kHz)    : Articulation, attack transients, and acoustic bite.
• Presence (4 - 6 kHz)      : Definition, vocal clarity, and synth openness.
• Brilliance (6 - 20 kHz)   : Airy sheen, cymbal sizzle, and high-frequency sparkle.

================================================================================
8. PRODUCER & CURATOR VERDICT
================================================================================
"${title}" is an evocative piece possessing a well-defined ${bpm} BPM cadence anchored in ${key} (${camelot}). Its balance of dynamic headroom (${analysis.dynamicRange} dB) and harmonic resonance delivers an engaging acoustic profile suitable for high-fidelity playback, DJ performance sets, and immersive 3D spatial visualization.

================================================================================
Generated 100% offline by Sonix3D Autonomous Audio Intelligence Engine.
================================================================================
`.trim();
  }

  function downloadStoryFile(text, filename) {
    const dataStr = "data:text/plain;charset=utf-8," + encodeURIComponent(text);
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", filename);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  }

  if (exportNlBtn) {
    exportNlBtn.addEventListener('click', () => {
      if (!currentAnalysis) {
        alert("Please load or play a track first to generate the natural language report!");
        return;
      }
      const title = trackNameEl.textContent.trim();
      const storyReport = generateNaturalLanguageReport(currentAnalysis, currentAiResults, title);
      
      reportTextContent.textContent = storyReport;
      reportModal.classList.add('active');
    });
  }

  if (modalCloseBtn) {
    modalCloseBtn.addEventListener('click', () => {
      reportModal.classList.remove('active');
    });
  }

  if (reportModal) {
    reportModal.addEventListener('click', (e) => {
      if (e.target === reportModal) {
        reportModal.classList.remove('active');
      }
    });
  }

  if (copyReportBtn) {
    copyReportBtn.addEventListener('click', () => {
      const text = reportTextContent.textContent;
      navigator.clipboard.writeText(text).then(() => {
        copyBtnText.textContent = "Copied to Clipboard!";
        setTimeout(() => {
          copyBtnText.textContent = "Copy to Clipboard";
        }, 2000);
      }).catch(err => {
        console.error("Clipboard copy error:", err);
      });
    });
  }

  if (downloadReportTxtBtn) {
    downloadReportTxtBtn.addEventListener('click', () => {
      const text = reportTextContent.textContent;
      const title = trackNameEl.textContent.trim();
      downloadStoryFile(text, `Sonix3D_Story_Report_${title}.txt`);
    });
  }

  /* ----------------- 4-Stem Neural Demixer UI Handlers ----------------- */
  const runDemixBtn = document.getElementById('btn-run-demix');
  const demixText = document.getElementById('btn-demix-text');
  const demixerBadge = document.getElementById('demixer-status-badge');

  if (runDemixBtn) {
    runDemixBtn.addEventListener('click', async () => {
      if (!window.SonixDsp.audioBuffer) {
        alert("Please load or play a track first to extract stems!");
        return;
      }

      runDemixBtn.disabled = true;
      demixerBadge.textContent = "EXTRACTING...";
      demixerBadge.style.color = "var(--neon-pink)";

      try {
        await window.SonixStemSeparator.separateStems(window.SonixDsp.audioBuffer, (pct, msg) => {
          demixText.textContent = `${msg} (${pct}%)`;
        });

        demixText.textContent = "⚡ Stems Active (Demixed)";
        demixerBadge.textContent = "4 STEMS ACTIVE";
        demixerBadge.style.color = "var(--neon-green)";
        runDemixBtn.style.background = "rgba(0, 255, 102, 0.15)";
        runDemixBtn.style.borderColor = "var(--neon-green)";
        runDemixBtn.style.color = "var(--neon-green)";
        runDemixBtn.disabled = false;

        // Switch playback to multi-track stems
        const curTime = window.SonixDsp.getCurrentTime();
        window.SonixStemSeparator.playStems(curTime);

      } catch (e) {
        console.error("Stem separation error:", e);
        demixText.textContent = "Demix Error";
        runDemixBtn.disabled = false;
      }
    });
  }

  // Mute / Solo / Volume Handlers
  document.querySelectorAll('.stem-ctrl-btn.btn-mute').forEach(btn => {
    btn.addEventListener('click', () => {
      const stem = btn.dataset.stem;
      const isMuted = window.SonixStemSeparator.toggleMute(stem);
      btn.classList.toggle('active', isMuted);
      const row = document.querySelector(`.stem-channel-row[data-stem="${stem}"]`);
      if (row) row.classList.toggle('muted', isMuted);
    });
  });

  document.querySelectorAll('.stem-ctrl-btn.btn-solo').forEach(btn => {
    btn.addEventListener('click', () => {
      const stem = btn.dataset.stem;
      const isSolo = window.SonixStemSeparator.toggleSolo(stem);
      btn.classList.toggle('active', isSolo);
      const row = document.querySelector(`.stem-channel-row[data-stem="${stem}"]`);
      if (row) row.classList.toggle('solo', isSolo);
    });
  });

  document.querySelectorAll('.stem-vol-slider').forEach(slider => {
    slider.addEventListener('input', (e) => {
      const stem = e.target.dataset.stem;
      window.SonixStemSeparator.setStemVolume(stem, parseFloat(e.target.value));
    });
  });

  // Download Individual Stem WAV
  document.querySelectorAll('.stem-ctrl-btn.btn-dl').forEach(btn => {
    btn.addEventListener('click', () => {
      const stem = btn.dataset.stem;
      const wavBlob = window.SonixStemSeparator.exportStemWav(stem);
      if (!wavBlob) {
        alert(`Please click 'Extract 4 Stems' first to generate ${stem} audio!`);
        return;
      }
      const title = trackNameEl.textContent.trim();
      const url = URL.createObjectURL(wavBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${title}_${stem}.wav`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });
  });

  // Auto-load demo on first visit
  setTimeout(() => {
    generateDemoTrack('synthwave');
  }, 400);
});

/**
 * session.js — one Heimdall connection shared by the Hardware, Exhibition and
 * Study screens. It owns the /realtime_hw WebSocket, the command strings sent
 * to the firmware, per-sector freshness, the event log, the level history and
 * the stop-then-download flow for recordings.
 *
 * GUI→firmware commands are sent verbatim and must stay byte-identical:
 *   F  C  G  X  S,<n>  M  I (via request_init)  R,1  R,0  D
 */

import { API_BASE, WS_BASE, apiJson } from '../shared/api.js';
import { download } from '../shared/dom.js';
import { fileStamp } from '../shared/format.js';
import { activeGrid, directionGridCommand, directionGridFailure, directionGridReady, gridMatches } from './dirgrid.js';
import { median } from './field.js';
import { outputReplyCode } from './replies.js';
import { BAND_HIGH_HZ, BAND_LOW_HZ, listenRecord, listenSuffix, listenWav } from './listenfilter.js';
import { LABEL_KEY, labelPayload, loadLabel } from './reclabel.js';
import { dsp } from '../study/dsp.js';

const LISTEN_KEY = 'heimdall.listenCopy';
/** The listening copy of a saved recording: the band filter of the MCU window, the bank's own filters, or others. */
export const LISTEN_PRESETS = {
  mcu: 'MCU window band (as the board measures now)',
  bank: 'Live detectors (the bank filters)',
  drone: 'Drone band 0.1-4 kHz',
  custom: 'Custom band',
};

function loadListenCopy() {
  const fallback = {enabled: false, preset: 'mcu', low: BAND_LOW_HZ, high: BAND_HIGH_HZ};
  try {
    const saved = JSON.parse(localStorage.getItem(LISTEN_KEY));
    return saved && LISTEN_PRESETS[saved.preset] ? {...fallback, ...saved} : fallback;
  } catch (_) {
    return fallback;
  }
}

export const STALE_MS = 5000;
export const HISTORY_WINDOW_S = 30;
const HISTORY_MAX = 600;
const EVENT_LOG_MAX = 50;
const MOVE_LOG_MS = 5000;

const MODE_NAMES = {F: 'Single sweep', C: 'Continuous scan', G: 'Tracking', B: 'Live', IDLE: 'Idle'};

function side(value, positive, negative) {
  const rounded = Math.round(Math.abs(value));
  return rounded === 0 ? '0°' : `${rounded}° ${value > 0 ? positive : negative}`;
}

/** "at 24° right, 13° up" from the bank in Live, the record or its sector; else the sector number. */
function where(frame, record) {
  const bank = frame.firmware_mode === 'B' ? frame.bank?.target : null;
  const azimuth = bank?.azimuth_deg ?? record?.azimuth_deg ?? frame.azimuth_deg?.[record?.column];
  const elevation = bank?.elevation_deg ?? record?.elevation_deg ?? frame.elevation_deg?.[record?.row];
  if (!Number.isFinite(azimuth) || !Number.isFinite(elevation)) return `in sector ${record?.sector}`;
  return `at ${side(azimuth, 'right', 'left')}, ${side(elevation, 'up', 'down')}`;
}

export class HardwareSession extends EventTarget {
  constructor() {
    super();
    this.ws = null;
    this.init = null;
    this.frame = null;
    this.knownConfiguration = null;
    this.homeGrid = null;             // grid in place before the first direction set
    this.link = 'disconnected';       // disconnected | connecting | online | fault
    this.linkDetail = 'Not connected';
    this.cellTimes = [];
    this.events = [];
    this.history = [];
    this.monitoring = false;
    this.monitorTimer = null;
    this.monitorRate = 20;
    this.frozen = false;
    this.downloadRequested = false;
    this.downloadCommandSent = false;
    this.downloadStopSent = false;
    this.listenCopy = loadListenCopy(); // {enabled, preset, low, high}: the filtered copy that Save WAV adds
    this.recordLabel = loadLabel(globalThis.localStorage); // text as typed; Save WAV writes it into the record
    this.direction = 'fit';           // Live direction for the history: 'firmware' or 'fit' (set by the view)
    this.lastLoggedSequence = -1;
    this.lastTargetSector = null;
    this.lastMoveLogged = 0;
    this.lockedSince = null;          // Date.now() when the current lock began
    this.pending = null;              // {kind, port, baud} of the last connect request
    this.discovered = false;
  }

  // ── State accessors ────────────────────────────────────────────────────

  get open() {
    return Boolean(this.ws && this.ws.readyState === WebSocket.OPEN);
  }

  get latest() {
    return this.frame || this.init;
  }

  get configuration() {
    return this.latest?.configuration || null;
  }

  get ready() {
    return this.open && this.link === 'online' && Boolean(this.configuration);
  }

  get transport() {
    return String(this.latest?.transport || '');
  }

  get firmwareMode() {
    return this.frame?.firmware_mode || this.init?.firmware_mode || 'IDLE';
  }

  get recordingState() {
    return this.frame?.recording_state || this.init?.recording_state || 'idle';
  }

  get recordingCapable() {
    return this.ready && this.transport.startsWith('serial:');
  }

  get arrayLayout() {
    return this.init?.array_layout || null;
  }

  /** {entry, title, state} while a Study test runs: the backend then refuses commands from this page. */
  get studyRun() {
    return this.latest?.study_run || null;
  }

  /** Human description of what we are connected to. */
  get transportLabel() {
    const name = this.transport;
    if (name.startsWith('serial:')) {
      const port = name.slice('serial:'.length);
      return this.pending?.kind === 'serial' && this.pending.port === port && this.pending.baud
        ? `${port} at ${this.pending.baud} baud` : port;
    }
    if (name === 'acoustic-emulator') return 'acoustic room emulator';
    if (name === 'emulator') return 'emulator';
    return name || 'device';
  }

  emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, {detail}));
  }

  setLink(state, detail) {
    this.link = state;
    if (detail) this.linkDetail = detail;
    this.emit('link');
  }

  // ── Connection ─────────────────────────────────────────────────────────

  /** Attach to a transport the backend already has, if any. */
  async discover() {
    if (this.open || this.link === 'connecting') return;
    this.discovered = true;
    this.setLink('connecting', 'Checking the backend…');
    try {
      const status = await apiJson('/hw_status');
      if (status.available) this.openSocket();
      else this.setLink('disconnected', 'Backend ready. Choose a connection.');
    } catch (error) {
      this.setLink('fault', 'No backend on 127.0.0.1:8766. Start sim_server.py.');
      this.addEvent('ERROR', `Backend unavailable: ${error.message}`, 'error');
    }
  }

  async connectSerial(port, baud) {
    this.pending = {kind: 'serial', port, baud};
    this.setLink('connecting', `Opening ${port} at ${baud} baud…`);
    try {
      await apiJson('/hw_connect', {method: 'POST', body: {transport: 'serial', port, baud}});
      this.openSocket();
    } catch (error) {
      this.setLink('fault', `Could not open ${port}: ${error.message}`);
      this.addEvent('ERROR', `Serial connection failed: ${error.message}`, 'error');
      throw error;
    }
  }

  async connectEmulator(config) {
    this.pending = {kind: 'emulator'};
    this.setLink('connecting', 'Starting the emulator…');
    try {
      await apiJson('/hw_connect', {method: 'POST', body: {transport: 'emulator', ...config}});
      this.openSocket();
    } catch (error) {
      this.setLink('fault', `Emulator failed: ${error.message}`);
      this.addEvent('ERROR', `Emulator failed: ${error.message}`, 'error');
      throw error;
    }
  }

  async connectAcoustic(cacheId) {
    this.pending = {kind: 'acoustic'};
    this.setLink('connecting', 'Starting the acoustic room emulator…');
    try {
      await apiJson('/hw_connect', {method: 'POST', body: {transport: 'acoustic', acoustic_cache_id: cacheId}});
      this.openSocket();
    } catch (error) {
      this.setLink('fault', `Acoustic emulator failed: ${error.message}`);
      this.addEvent('ERROR', `Acoustic emulator failed: ${error.message}`, 'error');
      throw error;
    }
  }

  async disconnect() {
    this.addEvent('LINK', 'Disconnected by operator');
    try {
      await fetch(API_BASE + '/hw_disconnect', {method: 'POST'});
    } finally {
      this.closeSocket();
      this.setLink('disconnected', 'Not connected');
    }
  }

  openSocket() {
    this.closeSocket();
    this.setLink('connecting', 'Connecting to the Heimdall service…');
    const socket = new WebSocket(WS_BASE + '/realtime_hw');
    this.ws = socket;
    socket.onopen = () => {
      if (this.ws !== socket) return;
      this.setLink('online', 'Link open, requesting firmware information');
      this.addEvent('LINK', 'Connected');
      this.send({type: 'resume'});
      this.send({type: 'set_true_dir', az_deg: null, el_deg: null});
      this.send({type: 'request_init'});
    };
    socket.onmessage = (event) => {
      if (this.ws !== socket) return;
      let message;
      try { message = JSON.parse(event.data); } catch (_) { return; }
      this.onMessage(message);
    };
    socket.onclose = () => {
      if (this.ws !== socket) return;
      this.ws = null;
      this.stopMonitor(false);
      this.clearPresentation();
      this.setLink('disconnected', 'Not connected');
      this.addEvent('LINK', 'Link closed', 'warning');
      this.emit('frame');
    };
    socket.onerror = () => {
      if (this.ws !== socket) return;
      this.setLink('fault', 'The Heimdall WebSocket is unavailable');
      this.addEvent('ERROR', 'WebSocket unavailable', 'error');
    };
  }

  closeSocket() {
    this.stopMonitor(false);
    if (this.ws) {
      const socket = this.ws;
      this.ws = null;
      socket.close();
    }
    this.clearPresentation();
    this.emit('frame');
  }

  clearPresentation() {
    this.init = null;
    this.frame = null;
    this.knownConfiguration = null;
    this.homeGrid = null;
    this.frozen = false;
    this.lastLoggedSequence = -1;
    this.lastTargetSector = null;
    this.lockedSince = null;
    this.cellTimes = [];
    this.history = [];
    this.downloadRequested = false;
    this.downloadCommandSent = false;
    this.downloadStopSent = false;
  }

  send(message) {
    if (!this.open) {
      this.linkDetail = 'No active link';
      this.emit('link');
      return false;
    }
    this.ws.send(JSON.stringify(message));
    return true;
  }

  /** {enabled, preset, low, high}; saved in this browser. */
  setListenCopy(patch) {
    this.listenCopy = {...this.listenCopy, ...patch};
    try { localStorage.setItem(LISTEN_KEY, JSON.stringify(this.listenCopy)); } catch (_) { /* storage unavailable */ }
    this.emit('listen');
  }

  /** The labels of the next saved recordings; saved in this browser. */
  setRecordLabel(patch) {
    this.recordLabel = {...this.recordLabel, ...patch};
    try { localStorage.setItem(LABEL_KEY, JSON.stringify(this.recordLabel)); } catch (_) { /* storage unavailable */ }
    this.emit('label');
  }

  /** The filter of the listening copy: {kind: 'band', low, high} or {kind: 'bank', antiAlias, bandPass}. */
  async listenFilter() {
    const {preset, low, high} = this.listenCopy;
    if (preset === 'bank') {
      await dsp.load();
      const taps = dsp.bankListenTaps();
      if (!taps) throw new Error('this DSP program has no bank');
      return {kind: 'bank', ...taps};
    }
    if (preset === 'drone') return {kind: 'band', low: 100, high: 4000};
    if (preset === 'custom') return {kind: 'band', low, high};
    const settings = this.latest?.settings || {};
    return {kind: 'band', low: settings.band_low_hz ?? BAND_LOW_HZ, high: settings.band_high_hz ?? BAND_HIGH_HZ};
  }

  // ── Commands ───────────────────────────────────────────────────────────

  command(command) {
    return this.send({type: 'command', command});
  }

  sweepOnce() { this.modeCommand('F', 'Single sweep requested'); }
  continuous() { this.modeCommand('C', 'Continuous scan requested'); }
  track() { this.modeCommand('G', 'Tracking requested'); }
  stare() { this.modeCommand('B', 'Live requested'); }
  stop() { this.modeCommand('X', 'Stop requested', 'warning'); }

  selectOutput(source, index) {
    const names = ['the steered beam', `microphone ${index}`, `bank beam ${index}`];
    if (this.command(`O,${source},${index}`)) this.addEvent('AUDIO', `Ch1 set to ${names[source]}`);
  }

  get hasSelectors() {
    return Number(this.latest?.outputs?.microphones || 0) > 0;
  }

  /** Route Ch1 and wait for the device to confirm it. */
  async selectOutputChecked(source, index) {
    const beforeSequence = Number(this.latest?.sequence || 0);
    if (!this.command(`O,${source},${index}`)) throw new Error('No active hardware link.');
    const deadline = performance.now() + 2000;
    while (performance.now() < deadline) {
      const code = outputReplyCode(this.latest, source, index, beforeSequence);
      if (code === 0) return;
      if (code !== null) throw new Error(`The device refused Ch1 = ${source},${index} (code ${code}).`);
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    throw new Error('No reply to the Ch1 selection. The DSP program may have no output selector.');
  }

  async setSetting(key, value, label = key, log = true) {
    try {
      await apiJson('/dsp/set', {method: 'POST', body: {key, value}});
      if (log) this.addEvent('SETTING', `${label}: ${value}`);
      return true;
    } catch (error) {
      this.addEvent('ERROR', `Could not set ${label}: ${error.message}`, 'error');
      throw error;
    }
  }

  async setMeasurementSettings(settings) {
    for (const [key, value] of Object.entries(settings)) {
      await this.setSetting(key, value, key.replace('_us', ''));
    }
  }

  async setDirectionGrid(spec) {
    if (this.firmwareMode !== 'IDLE') {
      this.command('X');
      const deadline = performance.now() + 3000;
      while (this.firmwareMode !== 'IDLE' && performance.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
    }
    this.homeGrid ||= activeGrid(this.latest);
    const beforeSequence = Number(this.latest?.sequence || 0);
    if (!this.command(directionGridCommand(spec))) throw new Error('No active hardware link.');
    const deadline = performance.now() + 5000;
    while (performance.now() < deadline) {
      if (directionGridReady(this.latest, spec, beforeSequence)) return this.latest.direction_set;
      const failure = directionGridFailure(this.latest, beforeSequence);
      if (failure === 'UNSUPPORTED') throw new Error('Direction sets need hardware or the fast emulator; the acoustic emulator only has its prepared grid.');
      if (failure) throw new Error(`The device refused the direction set (${failure}).`);
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    throw new Error('Timed out waiting for the direction set.');
  }

  /** Put back the grid from before the first direction set; I restores the header grid. */
  async restoreGrid() {
    const home = this.homeGrid;
    if (!home) return;
    this.send({type: 'request_init'});
    const deadline = performance.now() + 5000;
    while (performance.now() < deadline) {
      if (gridMatches(this.latest, home)) {
        this.homeGrid = null;
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    throw new Error('Timed out restoring the deployment grid.');
  }

  modeCommand(command, text, tone = '') {
    this.stopMonitor();
    if (this.command(command)) this.addEvent('MODE', text, tone);
  }

  validSector(sector) {
    const sectors = this.configuration?.sectors;
    return Number.isInteger(sector) && sector >= 0 && (!sectors || sector < sectors);
  }

  steer(sector, source = '') {
    this.stopMonitor();
    if (!this.validSector(sector)) {
      this.linkDetail = 'Invalid sector';
      this.emit('link');
      return false;
    }
    const sent = this.command(`S,${sector}`);
    if (sent) this.addEvent('STEER', `Beam held on sector ${sector}${source}`);
    return sent;
  }

  startMonitor(sector) {
    if (!this.validSector(sector)) return false;
    if (!this.open) return false;
    this.command(`S,${sector}`);
    this.monitoring = true;
    this.addEvent('MODE', `Monitoring sector ${sector}`);
    this.restartMonitorTimer();
    this.emit('frame');
    return true;
  }

  stopMonitor(log = true) {
    const was = this.monitoring;
    this.monitoring = false;
    if (this.monitorTimer) clearInterval(this.monitorTimer);
    this.monitorTimer = null;
    if (was && log) this.addEvent('MODE', 'Monitoring stopped');
    if (was) this.emit('frame');
  }

  setMonitorRate(rate) {
    this.monitorRate = Math.max(1, Number(rate) || 20);
    if (this.monitoring) this.restartMonitorTimer();
  }

  restartMonitorTimer() {
    if (this.monitorTimer) clearInterval(this.monitorTimer);
    this.monitorTimer = setInterval(() => this.command('M'), 1000 / this.monitorRate);
  }

  toggleFreeze() {
    this.frozen = !this.frozen;
    this.send({type: this.frozen ? 'pause' : 'resume'});
    this.addEvent('DISPLAY', this.frozen
      ? 'Display frozen. The device keeps running.'
      : 'Display live again', 'warning');
    this.emit('frame');
  }

  toggleRecording() {
    const command = this.recordingState === 'recording' ? 'R,0' : 'R,1';
    if (this.command(command)) {
      this.addEvent('AUDIO', command === 'R,1' ? 'Recording started' : 'Recording stopped');
    }
  }

  requestDownload() {
    if (this.frame?.audio_download_ready) {
      this.saveWav();
      return;
    }
    if (!this.open) return;
    this.downloadRequested = true;
    this.downloadCommandSent = false;
    this.downloadStopSent = false;
    this.advanceDownload(this.latest);
    this.emit('frame');
  }

  /** Stop any scan first, then send D once the firmware reports IDLE. */
  advanceDownload(frame) {
    if (!this.downloadRequested || this.downloadCommandSent || frame?.audio_download_ready) return;
    if (!['ready', 'downloaded'].includes(frame?.recording_state)) return;
    if (frame?.firmware_mode !== 'IDLE') {
      if (!this.downloadStopSent && this.command('X')) {
        this.downloadStopSent = true;
        this.addEvent('AUDIO', 'Stopping the scan before downloading');
      }
      return;
    }
    if (this.command('D')) {
      this.downloadCommandSent = true;
      this.addEvent('AUDIO', 'Download started');
    }
  }

  async saveWav() {
    this.downloadRequested = false;
    this.downloadCommandSent = false;
    this.downloadStopSent = false;
    try {
      const filter = this.listenCopy.enabled ? await this.listenFilter() : null;
      const {label, problems} = labelPayload(this.recordLabel);
      problems.forEach((problem) => this.addEvent('AUDIO', `Label left out: ${problem}`, 'warning'));
      let record;
      try {
        record = await apiJson('/hw_recording/save', {method: 'POST', body: {
          label, listen_copy: filter ? listenRecord(filter, this.listenCopy.preset) : null}});
      } catch (error) {
        this.addEvent('ERROR', `Not saved in the dataset folder: ${error.message}`, 'error');
        record = await apiJson('/hw_recording_metadata.json');
      }
      const audioResponse = await fetch(API_BASE + '/hw_recording.wav');
      if (!audioResponse.ok) throw new Error(await audioResponse.text());
      const audio = await audioResponse.blob();
      const stem = record.files ? record.files.wav.split('/').pop().replace(/\.wav$/, '') : `heimdall-${fileStamp()}`;
      download(audio, `${stem}.wav`);
      if (filter) {
        download(new Blob([listenWav(await audio.arrayBuffer(), filter)], {type: 'audio/wav'}),
          record.listen_copy?.file || `${stem}-${listenSuffix(filter)}.wav`);
      }
      const {dataset_folder: folder, ...sidecar} = record;
      download(new Blob([JSON.stringify(sidecar, null, 2) + '\n'], {type: 'application/json'}), `${stem}.json`);
      const what = filter ? 'WAV, filtered copy and record' : 'WAV and record';
      this.addEvent('AUDIO', folder
        ? `${what} downloaded (checksum-verified); WAV and record also saved in ${folder}`
        : `${what} downloaded (checksum-verified)`);
    } catch (error) {
      this.addEvent('ERROR', `Saving the recording failed: ${error.message}`, 'error');
    }
  }

  // ── Incoming messages ──────────────────────────────────────────────────

  onMessage(message) {
    if (message.type === 'init') {
      this.init = message;
      const cfg = message.configuration;
      this.noteConfiguration(cfg);
      this.setLink('online', cfg ? 'Connected' : 'Connected, waiting for firmware information');
      this.emit('init');
      this.emit('frame');
      return;
    }
    if (message.type === 'frame') {
      const previous = this.frame;
      const advanced = message.sequence !== previous?.sequence;
      // Serial firmware answers I after the init message, so the grid often arrives in a frame.
      const configChanged = this.noteConfiguration(message.configuration);
      if (advanced) this.updateFreshness(message, previous);
      this.frame = message;
      if (configChanged) this.emit('init');
      if (!message.connected) this.setLink('fault', `Link fault on ${message.transport}`);
      else if (this.link !== 'online') this.setLink('online', 'Connected');
      if (advanced) {
        this.logRecord(message);
        this.logTransitions(message, previous);
        this.noteLock(message);
        if (message.firmware_mode !== 'B' && message.levels_db && message.configuration &&
            message.last_measurement?.count !== previous?.last_measurement?.count) {
          const level = this.fixedBeamLevel(message) ?? message.argmax_db;
          this.pushHistory(message.timestamp_s, message.est_az_deg, message.est_el_deg, level, message.levels_db.flat());
        }
        const bank = message.bank;
        if (message.firmware_mode === 'B' && bank?.levels_db && bank.maps !== previous?.bank?.maps) {
          const levels = bank.levels_db.flat();
          const target = bank.target;
          const level = target ? levels[target.beam] : Math.max(...levels.filter(Number.isFinite));
          const where = target && this.direction === 'fit' && bank.fit ? bank.fit : target;
          this.pushHistory(message.timestamp_s, where?.azimuth_deg ?? null, where?.elevation_deg ?? null, level, levels);
        }
      }
      this.advanceDownload(message);
      if (this.downloadRequested && message.audio_download_ready) this.saveWav();
      this.emit('frame');
      return;
    }
    if (message.type === 'command_error') {
      const detail = String(message.detail || 'unknown command error');
      this.linkDetail = `Command error: ${detail}`;
      this.addEvent('ERROR', detail, 'error');
      this.emit('link');
    }
  }

  fixedBeamLevel(frame) {
    const steer = frame?.last_steer;
    if (!steer || !frame.levels_db?.[steer.row]) return null;
    const level = frame.levels_db[steer.row][steer.column];
    return Number.isFinite(level) ? level : null;
  }

  /** One point of the last 30 s; "margin" is the level above the median of the map it came from. */
  pushHistory(t, azimuth, elevation, level, levels) {
    const background = median(levels);
    const finite = Number.isFinite(level);
    this.history.push({
      t,
      azimuth,
      elevation,
      level: finite ? level : null,
      margin: finite && background !== null ? level - background : null,
    });
    if (this.history.length > HISTORY_MAX) this.history.shift();
  }

  noteLock(frame) {
    const locked = Boolean(frame.target || (frame.firmware_mode === 'B' && frame.bank?.target));
    if (!locked) this.lockedSince = null;
    else if (this.lockedSince === null) this.lockedSince = Date.now();
  }

  get lockedForMs() {
    return this.lockedSince === null ? null : Date.now() - this.lockedSince;
  }

  // ── Freshness ──────────────────────────────────────────────────────────

  noteConfiguration(cfg) {
    const known = this.knownConfiguration;
    if (!cfg || (known && known.rows === cfg.rows && known.columns === cfg.columns &&
        known.microphones === cfg.microphones)) return false;
    this.knownConfiguration = cfg;
    this.resetFreshness(cfg);
    this.addEvent('CONFIG', `${cfg.rows} × ${cfg.columns} grid, ${cfg.sectors} sectors, ${cfg.microphones} microphones`);
    return true;
  }

  resetFreshness(cfg = null) {
    this.cellTimes = cfg
      ? Array.from({length: cfg.rows}, () => Array(cfg.columns).fill(null))
      : [];
  }

  updateFreshness(frame, previous) {
    const cfg = frame.configuration;
    if (!cfg) return;
    if (this.cellTimes.length !== cfg.rows || this.cellTimes.some((row) => row.length !== cfg.columns)) {
      this.resetFreshness(cfg);
    }
    const at = Number.isFinite(frame.timestamp_s) ? frame.timestamp_s * 1000 : Date.now();
    const record = frame.last_record;
    if (record?.type === 'measurement' && this.cellTimes[record.row]?.[record.column] !== undefined) {
      this.cellTimes[record.row][record.column] = at;
    }
    for (let row = 0; row < cfg.rows; row++) {
      for (let column = 0; column < cfg.columns; column++) {
        const value = frame.levels_raw?.[row]?.[column];
        if (value != null && value !== previous?.levels_raw?.[row]?.[column]) this.cellTimes[row][column] = at;
      }
    }
  }

  cellAge(row, column) {
    const at = this.cellTimes[row]?.[column];
    return Number.isFinite(at) ? Math.max(0, Date.now() - at) : null;
  }

  // ── Derived answer ─────────────────────────────────────────────────────

  strongest(frame = this.frame) {
    let best = null;
    const cfg = frame?.configuration;
    if (!cfg) return null;
    for (let row = 0; row < cfg.rows; row++) {
      for (let column = 0; column < cfg.columns; column++) {
        const level = frame.levels_db?.[row]?.[column];
        if (Number.isFinite(level) && (!best || level > best.level)) {
          best = {row, column, sector: row * cfg.columns + column, level};
        }
      }
    }
    return best;
  }

  /**
   * What the operator should look at: the tracked target, the held beam, or
   * the strongest measured sector — with its angles, level and age.
   */
  solution(frame = this.frame) {
    if (!frame?.configuration) return null;
    let marker;
    let kind;
    if (frame.target) {
      marker = frame.target;
      kind = 'target';
    } else if (frame.last_steer) {
      marker = frame.last_steer;
      kind = 'beam';
    } else {
      marker = this.strongest(frame);
      kind = marker ? 'strongest' : null;
    }
    if (!marker) return null;
    // In stare mode the bank interpolates between its beams, finer than the sector grid.
    const bankTarget = kind === 'target' ? frame.bank?.target : null;
    const azimuth = Number.isFinite(bankTarget?.azimuth_deg) ? bankTarget.azimuth_deg
      : Number.isFinite(marker.azimuth_deg) ? marker.azimuth_deg : frame.azimuth_deg?.[marker.column];
    const elevation = Number.isFinite(bankTarget?.elevation_deg) ? bankTarget.elevation_deg
      : Number.isFinite(marker.elevation_deg) ? marker.elevation_deg : frame.elevation_deg?.[marker.row];
    const level = frame.levels_db?.[marker.row]?.[marker.column];
    return {
      kind,
      row: marker.row,
      column: marker.column,
      sector: marker.sector ?? marker.row * frame.configuration.columns + marker.column,
      azimuth,
      elevation,
      level: Number.isFinite(level) ? level : null,
      ageMs: this.cellAge(marker.row, marker.column),
    };
  }

  // ── Event log ──────────────────────────────────────────────────────────

  addEvent(kind, text, tone = '') {
    const last = this.events.at(-1);
    if (last && last.kind === kind && last.text === text) return;
    this.events.push({time: new Date(), kind, text, tone});
    if (this.events.length > EVENT_LOG_MAX) this.events.shift();
    this.emit('events');
  }

  clearEvents() {
    this.events = [];
    this.emit('events');
  }

  logMove(frame, record) {
    const now = Date.now();
    if (now - this.lastMoveLogged < MOVE_LOG_MS) return;
    this.lastMoveLogged = now;
    this.addEvent('TARGET', `Moved to ${where(frame, record).replace(/^(at|in) /, '')}`, 'target');
  }

  logRecord(frame) {
    if (!Number.isFinite(frame.sequence) || frame.sequence === this.lastLoggedSequence) return;
    this.lastLoggedSequence = frame.sequence;
    const record = frame.last_record;
    if (!record) return;
    switch (record.type) {
    case 'scan_started':
      this.addEvent('MODE', `${MODE_NAMES[record.mode] || record.mode} started`);
      break;
    case 'scan_done':
      if (frame.firmware_mode === 'IDLE') this.addEvent('MODE', 'Sweep complete');
      break;
    case 'scan_stopped':
      this.addEvent('MODE', 'Stopped');
      break;
    case 'target_acquired':
      this.lastTargetSector = record.sector;
      this.lastMoveLogged = Date.now();
      this.addEvent('TARGET', `Acquired ${where(frame, record)}`, 'target');
      break;
    case 'target_updated':
      if (record.sector !== this.lastTargetSector) {
        this.lastTargetSector = record.sector;
        this.logMove(frame, record);
      }
      break;
    case 'target_lost':
      this.lastTargetSector = null;
      this.addEvent('TARGET', `Lost ${where(frame, record)}`, 'warning');
      break;
    case 'steer_error':
      this.addEvent('ERROR', `Steering failed at microphone ${record.microphone}, code ${record.code}`, 'error');
      break;
    case 'error':
      this.addEvent('ERROR', [record.reason, ...(record.details || [])].join(' · '), 'error');
      break;
    default:
      break;
    }
  }

  logTransitions(frame, previous) {
    if (!previous) return;
    if (frame.scan_count > previous.scan_count && frame.firmware_mode === 'IDLE') {
      this.addEvent('MODE', `Sweep complete, pass ${frame.scan_count}`);
    }
    const before = previous.target;
    const now = frame.target;
    if (!before && now) {
      this.lastTargetSector = now.sector;
      this.lastMoveLogged = Date.now();
      this.addEvent('TARGET', `Acquired ${where(frame, now)}`, 'target');
    } else if (before && !now) {
      this.lastTargetSector = null;
      this.addEvent('TARGET', `Cleared ${where(previous, before)}`, 'warning');
    } else if (now && now.sector !== before?.sector && now.sector !== this.lastTargetSector) {
      this.lastTargetSector = now.sector;
      this.logMove(frame, now);
    }
    if (frame.last_error && JSON.stringify(frame.last_error) !== JSON.stringify(previous.last_error)) {
      this.addEvent('ERROR', frame.last_error.reason || 'Hardware command failed', 'error');
    }
  }
}

export const session = new HardwareSession();

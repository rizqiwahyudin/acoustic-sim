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

export const STALE_MS = 5000;
export const HISTORY_WINDOW_S = 30;
const HISTORY_MAX = 600;
const EVENT_LOG_MAX = 50;

const MODE_NAMES = {F: 'Single sweep', C: 'Continuous scan', G: 'Tracking', IDLE: 'Idle'};

export class HardwareSession extends EventTarget {
  constructor() {
    super();
    this.ws = null;
    this.init = null;
    this.frame = null;
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
    this.lastLoggedSequence = -1;
    this.lastTargetSector = null;
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
    return this.open && this.link === 'online' && Boolean(this.init?.configuration);
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
    this.addEvent('LINK', 'Disconnected by operator', 'warning');
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
    this.frozen = false;
    this.lastLoggedSequence = -1;
    this.lastTargetSector = null;
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

  // ── Commands ───────────────────────────────────────────────────────────

  command(command) {
    return this.send({type: 'command', command});
  }

  sweepOnce() { this.modeCommand('F', 'Single sweep requested'); }
  continuous() { this.modeCommand('C', 'Continuous scan requested'); }
  track() { this.modeCommand('G', 'Tracking requested'); }
  stop() { this.modeCommand('X', 'Stop requested', 'warning'); }

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
        this.addEvent('AUDIO', 'Stopping the scan before downloading', 'warning');
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
      const response = await fetch(API_BASE + '/hw_recording.wav');
      if (!response.ok) throw new Error(await response.text());
      download(await response.blob(), `heimdall-${fileStamp()}.wav`);
      this.addEvent('AUDIO', 'WAV saved and checksum-verified');
    } catch (error) {
      this.addEvent('ERROR', `Saving the recording failed: ${error.message}`, 'error');
    }
  }

  // ── Incoming messages ──────────────────────────────────────────────────

  onMessage(message) {
    if (message.type === 'init') {
      const previous = this.init?.configuration;
      this.init = message;
      const cfg = message.configuration;
      const changed = cfg && (!previous || previous.rows !== cfg.rows ||
        previous.columns !== cfg.columns || previous.microphones !== cfg.microphones);
      if (changed) {
        this.resetFreshness(cfg);
        this.addEvent('CONFIG', `${cfg.rows} × ${cfg.columns} grid, ${cfg.sectors} sectors, ${cfg.microphones} microphones`);
      }
      this.setLink('online', cfg ? 'Connected' : 'Connected, waiting for firmware information');
      this.emit('init');
      this.emit('frame');
      return;
    }
    if (message.type === 'frame') {
      const previous = this.frame;
      const advanced = message.sequence !== previous?.sequence;
      if (advanced) this.updateFreshness(message, previous);
      this.frame = message;
      if (!message.connected) this.setLink('fault', `Link fault on ${message.transport}`);
      else if (this.link !== 'online') this.setLink('online', 'Connected');
      if (advanced) {
        this.logRecord(message);
        this.logTransitions(message, previous);
        if (message.levels_db && message.configuration && message.last_record?.type === 'measurement') {
          this.history.push({
            t: message.timestamp_s,
            azimuth: message.est_az_deg,
            elevation: message.est_el_deg,
            level: this.fixedBeamLevel(message) ?? message.argmax_db,
          });
          if (this.history.length > HISTORY_MAX) this.history.shift();
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

  // ── Freshness ──────────────────────────────────────────────────────────

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

  newestSampleAge() {
    let newest = null;
    for (const row of this.cellTimes) {
      for (const at of row) if (Number.isFinite(at) && (newest === null || at > newest)) newest = at;
    }
    return newest === null ? null : Math.max(0, Date.now() - newest);
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
    const azimuth = Number.isFinite(marker.azimuth_deg) ? marker.azimuth_deg : frame.azimuth_deg?.[marker.column];
    const elevation = Number.isFinite(marker.elevation_deg) ? marker.elevation_deg : frame.elevation_deg?.[marker.row];
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
      this.addEvent('MODE', 'Stopped', 'warning');
      break;
    case 'target_acquired':
      this.lastTargetSector = record.sector;
      this.addEvent('TARGET', `Target acquired in sector ${record.sector}`, 'target');
      break;
    case 'target_updated':
      if (record.sector !== this.lastTargetSector) {
        this.lastTargetSector = record.sector;
        this.addEvent('TARGET', `Target moved to sector ${record.sector}`, 'target');
      }
      break;
    case 'target_lost':
      this.lastTargetSector = null;
      this.addEvent('TARGET', `Target lost from sector ${record.sector}`, 'warning');
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
      this.addEvent('TARGET', `Target acquired in sector ${now.sector}`, 'target');
    } else if (before && !now) {
      this.lastTargetSector = null;
      this.addEvent('TARGET', `Target cleared from sector ${before.sector}`, 'warning');
    } else if (now && now.sector !== before?.sector && now.sector !== this.lastTargetSector) {
      this.lastTargetSector = now.sector;
      this.addEvent('TARGET', `Target moved to sector ${now.sector}`, 'target');
    }
    if (frame.last_error && JSON.stringify(frame.last_error) !== JSON.stringify(previous.last_error)) {
      this.addEvent('ERROR', frame.last_error.reason || 'Hardware command failed', 'error');
    }
  }
}

export const session = new HardwareSession();

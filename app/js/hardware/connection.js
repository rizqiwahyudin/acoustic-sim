/**
 * connection.js — how the operator connects: the MAX78002 over a serial port
 * (picked from the detected ports, or typed by hand), the fast emulator, or
 * the acoustic room emulator (prepared scene + audio audition).
 */

import { apiJson, base64ToBlobUrl } from '../shared/api.js';
import { h, options, segmented, setText } from '../shared/dom.js';

const BAUDS = [['921600', '921600'], ['115200', '115200']];
const FLIGHT_PROFILES = [
  ['crossing', 'Crossing pass'], ['approach', 'Approach and retreat'], ['orbit', 'Orbit'],
  ['patrol', 'Patrol'], ['evasive', 'Evasive stress'], ['legacy', 'Legacy angular sweep'],
];
const SPEEDS = [['0.5', '0.5×'], ['1', '1.0×'], ['1.5', '1.5×'], ['2', '2.0×']];
const SCENARIOS = [
  ['conference_evasive', 'Conference room, evasive drone 2×'],
  ['handheld_2m_drone', 'Handheld drone at 2 m, drone only'],
  ['handheld_2m_mixed', 'Handheld drone at 2 m, crowd and speech'],
  ['stationary_2m', 'Stationary speaker at 2 m, broadside'],
];
const AUDITION_MODES = [
  ['selected', 'Selected or tracked beam'], ['truth', 'Beam steered at the truth'],
  ['unsteered', 'Unsteered sum'], ['single_mic', 'Single microphone'], ['generated', 'Generated mix'],
];
const MANUAL = '__manual__';

function numberField(id, label, value, {min, max, step = 1, width = 88} = {}) {
  const input = h('input', {id, class: 'input input--mono', type: 'number', min, max, step, value, style: {width: `${width}px`}});
  return {input, el: h('div', {class: 'field'}, h('label', {for: id}, label), input)};
}

function gridFields(prefix, defaults) {
  const rows = numberField(`${prefix}-rows`, 'Rows', defaults.rows, {min: 1, max: 20, width: 72});
  const columns = numberField(`${prefix}-cols`, 'Columns', defaults.columns, {min: 1, max: 20, width: 72});
  const azMin = numberField(`${prefix}-azmin`, 'Azimuth from (°)', defaults.azimuth_min_deg, {min: -90, max: 89});
  const azMax = numberField(`${prefix}-azmax`, 'to (°)', defaults.azimuth_max_deg, {min: -89, max: 90});
  const elMin = numberField(`${prefix}-elmin`, 'Elevation from (°)', defaults.elevation_min_deg, {min: -90, max: 89});
  const elMax = numberField(`${prefix}-elmax`, 'to (°)', defaults.elevation_max_deg, {min: -89, max: 90});
  return {
    el: h('div', {class: 'form-row'}, rows.el, columns.el, azMin.el, azMax.el, elMin.el, elMax.el),
    inputs: [rows, columns, azMin, azMax, elMin, elMax].map((f) => f.input),
    value: () => ({
      rows: parseInt(rows.input.value, 10),
      columns: parseInt(columns.input.value, 10),
      azimuth_min_deg: parseFloat(azMin.input.value),
      azimuth_max_deg: parseFloat(azMax.input.value),
      elevation_min_deg: parseFloat(elMin.input.value),
      elevation_max_deg: parseFloat(elMax.input.value),
    }),
  };
}

const DEPLOYED_GRID = {rows: 6, columns: 6, azimuth_min_deg: -40, azimuth_max_deg: 40, elevation_min_deg: -40, elevation_max_deg: 40};

/** The prepared acoustic scene, shared by the connect form and the audition controls. */
export class AcousticScene extends EventTarget {
  constructor() {
    super();
    this.cacheId = null;
    this.status = {state: 'idle'};
    this.pollTimer = null;
    this.auditionUrls = {};
  }

  async prepare(scenario, grid) {
    this.cacheId = null;
    this.status = await apiJson('/hw_acoustic_prepare', {method: 'POST', body: {scenario, ...grid}});
    this.changed();
    this.poll();
  }

  async cancel() {
    try { await apiJson('/hw_acoustic_cancel', {method: 'POST'}); } finally { this.poll(); }
  }

  async poll() {
    clearTimeout(this.pollTimer);
    try {
      this.status = await apiJson('/hw_acoustic_status');
      this.cacheId = this.status.state === 'ready' ? this.status.cache_id : null;
      if (this.status.state === 'preparing') this.pollTimer = setTimeout(() => this.poll(), 500);
    } catch (error) {
      this.status = {state: 'offline', error: error.message};
    }
    this.changed();
  }

  invalidate() {
    this.cacheId = null;
    this.status = {state: 'idle'};
    this.clearAudition();
    this.changed();
  }

  describe() {
    const s = this.status || {};
    const percent = Number(s.overall_percent ?? s.percent ?? 0);
    switch (s.state) {
    case 'ready': {
      const rt60 = Number.isFinite(s.measured_rt60_s) ? `, RT60 ${s.measured_rt60_s.toFixed(2)} s` : '';
      const parity = s.grid_mode === 'deployment_contract' ? ', exact firmware delays' : ', simulated delays';
      return `Scene ready (${Number(s.preparation_seconds || 0).toFixed(1)} s to prepare${rt60}${parity}). Levels are not calibrated.`;
    }
    case 'preparing': {
      const eta = Number.isFinite(s.eta_s) ? `, about ${Math.ceil(s.eta_s)} s left` : '';
      return `${s.stage || 'Preparing'} · ${percent.toFixed(0)}%${eta}`;
    }
    case 'failed': return `Preparation failed: ${s.error || 'unknown error'}`;
    case 'canceled': return 'Preparation canceled.';
    case 'offline': return 'The backend is not reachable.';
    default: return 'No scene prepared. Preparing renders the room once and caches it.';
    }
  }

  clearAudition() {
    for (const url of Object.values(this.auditionUrls)) URL.revokeObjectURL(url);
    this.auditionUrls = {};
  }

  async audition(startS, sector) {
    const result = await apiJson('/hw_acoustic_audition', {
      method: 'POST',
      body: {acoustic_cache_id: this.cacheId, start_s: startS, duration_s: 3.0, selected_sector: sector},
    });
    this.clearAudition();
    for (const [name, encoded] of Object.entries(result.clips_b64)) this.auditionUrls[name] = base64ToBlobUrl(encoded);
    return result;
  }

  changed() {
    this.dispatchEvent(new Event('change'));
  }
}

export class ConnectionPanel {
  constructor(session, scene) {
    this.session = session;
    this.scene = scene;
    this.kind = 'serial';

    this.tabs = segmented([['serial', 'Serial port'], ['emulator', 'Emulator'], ['acoustic', 'Acoustic room']], {
      label: 'Connection type', size: 'lg', onPick: (kind) => this.setKind(kind),
    });

    // Serial
    this.portSelect = h('select', {id: 'hw-port', class: 'select input--lg', style: {minWidth: '280px'},
      onChange: () => this.syncManual()});
    this.portManual = h('input', {id: 'hw-port-manual', class: 'input input--mono input--lg', placeholder: 'COM7',
      'aria-label': 'Serial port name', style: {width: '140px'}});
    this.baud = h('select', {id: 'hw-baud', class: 'select input--lg'}, options(BAUDS, '921600'));
    this.portHint = h('p', {class: 'field__hint'});
    this.serialForm = h('form', {class: 'connect-form', onSubmit: (event) => { event.preventDefault(); this.connectSerial(); }},
      h('div', {class: 'form-row'},
        h('div', {class: 'field'}, h('label', {for: 'hw-port'}, 'Port'),
          h('div', {class: 'inline-field'}, this.portSelect, this.portManual,
            h('button', {type: 'button', class: 'btn btn--lg btn--ghost', onClick: () => this.loadPorts()}, 'Rescan'))),
        h('div', {class: 'field'}, h('label', {for: 'hw-baud'}, 'Baud rate'), this.baud),
      ),
      this.portHint,
      h('div', {class: 'toolbar'}, h('button', {type: 'submit', class: 'btn btn--primary btn--lg'}, 'Connect')),
    );

    // Fast emulator
    this.emuGrid = gridFields('emu', DEPLOYED_GRID);
    this.flight = h('select', {id: 'emu-flight', class: 'select'}, options(FLIGHT_PROFILES, 'crossing'));
    this.speed = h('select', {id: 'emu-speed', class: 'select'}, options(SPEEDS, '1'));
    this.emulatorForm = h('form', {class: 'connect-form', hidden: true, onSubmit: (event) => { event.preventDefault(); this.connectEmulator(); }},
      h('p', {class: 'lede'}, 'A protocol-faithful stand-in for the firmware. It answers the same commands with a simulated drone flight. The deployed table is 6 × 6 over ±40°.'),
      this.emuGrid.el,
      h('div', {class: 'form-row'},
        h('div', {class: 'field'}, h('label', {for: 'emu-flight'}, 'Flight profile'), this.flight),
        h('div', {class: 'field'}, h('label', {for: 'emu-speed'}, 'Flight speed'), this.speed),
      ),
      h('div', {class: 'toolbar'}, h('button', {type: 'submit', class: 'btn btn--primary btn--lg'}, 'Start emulator')),
    );

    // Acoustic room
    this.acGrid = gridFields('ac', DEPLOYED_GRID);
    for (const input of this.acGrid.inputs) input.addEventListener('change', () => scene.invalidate());
    this.scenario = h('select', {id: 'ac-scenario', class: 'select', style: {minWidth: '320px'},
      onChange: () => scene.invalidate()}, options(SCENARIOS, 'conference_evasive'));
    this.prepareButton = h('button', {type: 'button', class: 'btn btn--lg', onClick: () => this.prepare()}, 'Prepare scene');
    this.cancelButton = h('button', {type: 'button', class: 'btn btn--lg btn--ghost', onClick: () => scene.cancel()}, 'Cancel');
    this.startAcoustic = h('button', {type: 'submit', class: 'btn btn--primary btn--lg'}, 'Start acoustic emulator');
    this.acProgress = h('progress', {max: 100, value: 0});
    this.acStatus = h('p', {class: 'field__hint', role: 'status'});
    this.acousticForm = h('form', {class: 'connect-form', hidden: true, onSubmit: (event) => { event.preventDefault(); this.connectAcoustic(); }},
      h('p', {class: 'lede'}, 'Simulates the real array in a room with PyRoomAcoustics, then replays the levels through the emulator. The 6 × 6, ±40° grid reuses the exact firmware delays.'),
      h('div', {class: 'form-row'}, h('div', {class: 'field'}, h('label', {for: 'ac-scenario'}, 'Scenario'), this.scenario)),
      this.acGrid.el,
      h('div', {class: 'toolbar'}, this.prepareButton, this.cancelButton),
      this.acProgress,
      this.acStatus,
      h('div', {class: 'toolbar'}, this.startAcoustic),
    );

    this.status = h('p', {class: 'connect-status', role: 'status'});
    this.retry = h('button', {type: 'button', class: 'btn', hidden: true, onClick: () => session.discover()}, 'Try again');

    this.el = h('section', {class: 'connect', 'aria-labelledby': 'connect-title'},
      h('div', {class: 'connect__head'},
        h('h1', {class: 'page-title', id: 'connect-title'}, 'Connect to Heimdall'),
        h('p', {class: 'lede'}, 'Pick the MAX78002 serial port, or start an emulator to work without the hardware.'),
      ),
      this.tabs.el,
      this.serialForm,
      this.emulatorForm,
      this.acousticForm,
      h('div', {class: 'connect__foot'}, this.status, this.retry),
    );

    this.tabs.set('serial');
    scene.addEventListener('change', () => this.renderScene());
    session.addEventListener('link', () => this.renderStatus());
    this.renderScene();
    this.renderStatus();
    this.loadPorts();
  }

  setKind(kind) {
    this.kind = kind;
    this.tabs.set(kind);
    this.serialForm.hidden = kind !== 'serial';
    this.emulatorForm.hidden = kind !== 'emulator';
    this.acousticForm.hidden = kind !== 'acoustic';
    if (kind === 'serial') this.loadPorts();
    if (kind === 'acoustic') this.scene.poll();
  }

  async loadPorts() {
    const previous = this.portSelect.value;
    let ports = [];
    try {
      ports = (await apiJson('/hw_ports')).ports || [];
    } catch (_) {
      ports = [];
    }
    const items = ports.map((port) => [port.device, port.description && port.description !== port.device
      ? `${port.device} · ${port.description}` : port.device]);
    items.push([MANUAL, 'Type a port name…']);
    this.portSelect.replaceChildren(...options(items, previous || items[0][0]));
    if (previous && items.some(([value]) => value === previous)) this.portSelect.value = previous;
    setText(this.portHint, ports.length
      ? `${ports.length} port${ports.length === 1 ? '' : 's'} found. The MAX78002 console usually shows up as a USB serial device.`
      : 'No serial ports found. Check the USB cable, or type the port name, for example COM7.');
    this.syncManual();
  }

  syncManual() {
    const manual = this.portSelect.value === MANUAL;
    this.portManual.hidden = !manual;
    if (manual) this.portManual.focus();
  }

  portName() {
    return (this.portSelect.value === MANUAL ? this.portManual.value : this.portSelect.value).trim();
  }

  async connectSerial() {
    const port = this.portName();
    if (!port) {
      setText(this.status, 'Choose or type a serial port first.');
      return;
    }
    try { await this.session.connectSerial(port, parseInt(this.baud.value, 10)); } catch (_) { /* shown via link status */ }
  }

  async connectEmulator() {
    try {
      await this.session.connectEmulator({
        ...this.emuGrid.value(),
        flight_profile: this.flight.value,
        flight_speed: parseFloat(this.speed.value),
      });
    } catch (_) { /* shown via link status */ }
  }

  async prepare() {
    try {
      await this.scene.prepare(this.scenario.value, this.acGrid.value());
    } catch (error) {
      this.scene.status = {state: 'failed', error: error.message};
      this.renderScene();
    }
  }

  async connectAcoustic() {
    if (!this.scene.cacheId) {
      setText(this.status, 'Prepare the scene first.');
      return;
    }
    try { await this.session.connectAcoustic(this.scene.cacheId); } catch (_) { /* shown via link status */ }
  }

  renderScene() {
    const state = this.scene.status?.state;
    this.acProgress.value = Number(this.scene.status?.overall_percent ?? this.scene.status?.percent ?? 0);
    this.acProgress.hidden = state !== 'preparing';
    this.prepareButton.disabled = state === 'preparing';
    this.cancelButton.hidden = state !== 'preparing';
    this.startAcoustic.disabled = !this.scene.cacheId;
    setText(this.acStatus, this.scene.describe());
  }

  renderStatus() {
    const {link, linkDetail} = this.session;
    const text = link === 'disconnected' && linkDetail === 'Not connected' ? '' : linkDetail;
    setText(this.status, text);
    this.status.dataset.state = link;
    this.retry.hidden = link !== 'fault';
  }
}

/** Audition controls shown while the acoustic room emulator is connected. */
export class AuditionPanel {
  constructor(session, scene) {
    this.session = session;
    this.scene = scene;
    this.mode = h('select', {id: 'aud-mode', class: 'select select--sm', onChange: () => this.select()}, options(AUDITION_MODES, 'selected'));
    this.button = h('button', {type: 'button', class: 'btn btn--sm', onClick: () => this.render()}, 'Render 3 s');
    this.player = h('audio', {controls: true, preload: 'none', class: 'audition__player'});
    this.link = h('a', {class: 'btn btn--sm btn--ghost', hidden: true, download: 'heimdall-audition.wav'}, 'Download');
    this.status = h('p', {class: 'note'}, 'Render a 3 s clip of the current moment to hear what the beam hears.');
    this.el = h('section', {class: 'section', hidden: true, 'aria-labelledby': 'aud-title'},
      h('h2', {class: 'section__title', id: 'aud-title'}, 'Listen'),
      h('div', {class: 'toolbar toolbar--tight'}, this.mode, this.button),
      this.player,
      h('div', {class: 'section__row'}, this.status, this.link),
    );
  }

  render() {
    const frame = this.session.frame;
    const answer = this.session.solution(frame);
    this.button.disabled = true;
    setText(this.status, 'Rendering…');
    this.scene.audition(frame?.emulator_truth?.elapsed_s ?? 0, answer?.sector ?? null)
      .then((result) => {
        this.select();
        setText(this.status, `3.0 s at sector ${result.selected_sector}, joint gain across clips.`);
      })
      .catch((error) => setText(this.status, `Render failed: ${error.message}`))
      .finally(() => { this.button.disabled = false; });
  }

  select() {
    const url = this.scene.auditionUrls[this.mode.value];
    if (!url) return;
    this.player.src = url;
    this.link.href = url;
    this.link.download = `heimdall-${this.mode.value}.wav`;
    this.link.hidden = false;
  }

  update() {
    this.el.hidden = this.session.transport !== 'acoustic-emulator';
  }
}

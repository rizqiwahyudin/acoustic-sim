/**
 * status.js — the console's panels around the Field: Source (the answer),
 * Audio (Ch1 and recording), the log rail, and the drawer pages (manual
 * beam, measurement, tonal detector, diagnostics).
 */

import { h, options, segmented, setText } from '../shared/dom.js';
import { clockTime, deg, hz, num, signed } from '../shared/format.js';
import { look } from '../shared/look.js';
import { ledBoard } from '../shared/led.js';
import { ledText, sideWord } from './lookparts.js';
import { LABEL_CLASSES, LABEL_FIELDS, labelSummary, parseNumber } from './reclabel.js';
import { LISTEN_PRESETS } from './session.js';

export function operatingMode(session) {
  const frame = session.frame;
  if (session.monitoring) return 'fixed';
  const mode = frame?.firmware_mode;
  if (mode === 'G') return 'track';
  if (mode === 'B') return 'stare';
  if (mode === 'C') return 'continuous';
  if (mode === 'F') return 'once';
  if (frame?.last_steer) return 'fixed';
  return 'idle';
}

const RATES = [['10', '10 Hz'], ['20', '20 Hz'], ['50', '50 Hz']];

function arrow(value, positive, negative) {
  return !Number.isFinite(value) || Math.abs(value) < 0.5 ? '' : value > 0 ? positive : negative;
}

function stat(name, unit) {
  const value = h('span', {}, '—');
  const el = h('div', {class: 'stat'},
    h('div', {class: 'stat__v mono'}, value, h('small', {}, unit)),
    h('div', {class: 'stat__k'}, name));
  return {el, value};
}

/** What the Source panel shows: the lock in Live, else the scan's answer. */
export function sourceAnswer(session, field, mode) {
  const background = field?.background ?? null;
  const above = (level) => (Number.isFinite(level) && background !== null ? level - background : null);
  if (field?.kind === 'bank') {
    const marker = field.marker;
    if (!marker) return {tag: mode === 'stare' ? 'SEARCHING' : 'IDLE', tone: ''};
    const level = marker.cell?.level ?? null;
    return {tag: 'LOCKED', tone: 'accent', azimuth: marker.azimuth, elevation: marker.elevation,
      level, margin: above(level), lockedMs: session.lockedForMs};
  }
  const answer = session.solution(session.frame);
  if (!answer) return {tag: mode === 'track' ? 'SEARCHING' : mode === 'idle' ? 'IDLE' : 'SCANNING', tone: ''};
  const tag = {target: 'LOCKED', beam: 'HOLDING'}[answer.kind] || 'STRONGEST';
  return {tag, tone: answer.kind === 'strongest' ? '' : 'accent', azimuth: answer.azimuth, elevation: answer.elevation,
    level: answer.level, margin: above(answer.level), lockedMs: answer.kind === 'target' ? session.lockedForMs : null};
}

export class SourcePanel {
  constructor() {
    const readout = (name, title) => {
      const value = h('span', {class: 'ro__num'}, '—');
      const dir = h('span', {class: 'ro__dir'});
      const el = h('div', {class: 'ro'}, h('span', {class: 'label'}, look ? title : name), value, dir);
      if (!look) return {value, dir, el};
      // The hidden looks show the angle on a dot-matrix board, under its direction in words.
      const word = h('span', {class: 'ro__word'});
      const board = ledBoard();
      el.append(word, board.el);
      return {value, dir, el, word, board};
    };
    this.tag = h('span', {class: 'tag'}, 'IDLE');
    this.az = readout('AZ', 'Azimuth');
    this.elev = readout('EL', 'Elevation');
    this.level = stat('Level', 'dBFS');
    this.margin = stat('Above background', 'dB');
    this.locked = stat('Locked for', 's');
    this.el = h('section', {class: 'panel source', 'aria-labelledby': 'hw-source-title'},
      h('header', {class: 'panel__head'}, h('h2', {class: 'label', id: 'hw-source-title'}, 'Source'), this.tag),
      h('div', {class: 'src-answer', 'aria-live': 'polite'}, this.az.el, this.elev.el),
      h('div', {class: 'stats'}, this.level.el, this.margin.el, this.locked.el));
  }

  render(answer) {
    setText(this.tag, answer.tag);
    this.tag.className = `tag${answer.tone ? ' tag--' + answer.tone : ''}`;
    const angle = (value) => (Number.isFinite(value) ? `${Math.round(Math.abs(value))}°` : '—');
    setText(this.az.value, angle(answer.azimuth));
    setText(this.az.dir, arrow(answer.azimuth, '▶', '◀'));
    setText(this.elev.value, angle(answer.elevation));
    setText(this.elev.dir, arrow(answer.elevation, '▲', '▼'));
    if (look) {
      // During the 3D build the azimuth board counts the microphones in (answer.counting), as the mockups.
      const counting = answer.counting;
      const azWord = counting ? 'mic' : sideWord(answer.azimuth, 'right', 'left', 'ahead');
      const elWord = sideWord(answer.elevation, 'up', 'down', 'level');
      setText(this.az.word, azWord);
      setText(this.elev.word, elWord);
      this.az.board.set(counting ? String(counting).padStart(2, '0') : ledText(answer.azimuth),
        counting ? `Microphone ${counting}` : `Azimuth ${angle(answer.azimuth)} ${azWord}`);
      this.elev.board.set(ledText(answer.elevation), `Elevation ${angle(answer.elevation)} ${elWord}`);
    }
    setText(this.level.value, num(answer.level, 1));
    setText(this.margin.value, signed(answer.margin, 1));
    setText(this.locked.value, Number.isFinite(answer.lockedMs) ? (answer.lockedMs / 1000).toFixed(1) : '—');
  }
}

export class AudioPanel {
  constructor(session) {
    this.session = session;
    this.key = '';
    this.output = h('select', {class: 'select select--sm', 'aria-label': 'Ch1 listens to',
      onChange: () => {
        const [source, index] = this.output.value.split(':').map(Number);
        session.selectOutput(source, index);
      }});
    this.listen = h('div', {class: 'line', hidden: true}, h('span', {class: 'line__k'}, 'Listen'), this.output);
    this.recIcon = h('span', {class: 'btn__rec', 'aria-hidden': 'true'});
    this.recLabel = h('span', {}, 'Record');
    this.recButton = h('button', {type: 'button', class: 'btn btn--sm', onClick: () => session.toggleRecording()}, this.recIcon, this.recLabel);
    this.saveButton = h('button', {type: 'button', class: 'btn btn--sm', onClick: () => session.requestDownload()}, 'Save WAV');
    const bandBox = h('input', {type: 'checkbox', checked: session.listenCopy.enabled,
      onChange: () => session.setListenCopy({enabled: bandBox.checked})});
    const preset = h('select', {class: 'select select--sm', 'aria-label': 'Filter of the copy',
      onChange: () => session.setListenCopy({preset: preset.value})},
    Object.entries(LISTEN_PRESETS).map(([key, label]) => h('option', {value: key}, label)));
    this.bandBox = bandBox;
    this.preset = preset;
    this.bandCopy = h('div', {class: 'line', title: 'The beam outputs are full band (they feed the tonal detector and ' +
      'classification). The copy is filtered as you choose; set custom corners in Study, Parameters, Filters.'},
    h('label', {class: 'check'}, bandBox, 'Also save a copy through'), preset);
    session.addEventListener('listen', () => this.renderListen());
    this.renderListen();
    this.labelInputs = LABEL_FIELDS.map((field) => {
      const id = `rec-label-${field.key}`;
      const input = field.kind === 'class'
        ? h('select', {id, class: 'select select--sm'}, options(LABEL_CLASSES, session.recordLabel.class))
        : h(field.kind === 'text' ? 'textarea' : 'input', {id, class: 'input input--sm', rows: field.kind === 'text' ? 2 : undefined,
          type: field.kind === 'text' ? undefined : 'text', inputmode: field.kind === 'number' ? 'decimal' : undefined,
          placeholder: field.placeholder, value: session.recordLabel[field.key]});
      input.addEventListener('change', () => session.setRecordLabel({[field.key]: input.value}));
      return {field, input, el: h('div', {class: `field${field.wide ? ' field--wide' : ''}`}, h('label', {for: id}, field.label), input)};
    });
    this.labelSummary = h('span', {class: 'labels__summary'});
    this.labels = h('details', {class: 'disclosure labels'},
      h('summary', {}, 'Labels: ', this.labelSummary),
      h('div', {class: 'labels__grid'}, this.labelInputs.map(({el}) => el)),
      h('p', {class: 'note'}, 'Save WAV writes these labels into the record and saves the WAV and the record in the ' +
        'dataset folder. The labels stay until you change them.'));
    session.addEventListener('label', () => this.renderLabels());
    this.renderLabels();
    this.recText = h('p', {class: 'note'});
    this.recProgress = h('progress', {max: 1, value: 0, hidden: true});
    this.el = h('section', {class: 'panel audio', 'aria-labelledby': 'hw-audio-title'},
      h('header', {class: 'panel__head'}, h('h2', {class: 'label', id: 'hw-audio-title'}, 'Audio')),
      h('div', {class: 'audio__body'},
        this.listen,
        h('div', {class: 'line'}, h('span', {class: 'line__k'}, 'Record'), this.recButton, this.saveButton),
        this.bandCopy,
        this.labels,
        this.recProgress,
        this.recText));
  }

  renderListen() {
    const {enabled, preset, low, high} = this.session.listenCopy;
    this.bandBox.checked = enabled;
    this.preset.value = preset;
    const custom = this.preset.querySelector('option[value="custom"]');
    if (custom) custom.textContent = `Custom band ${low / 1000}-${high / 1000} kHz`;
  }

  renderLabels() {
    const label = this.session.recordLabel;
    setText(this.labelSummary, labelSummary(label));
    for (const {field, input} of this.labelInputs) {
      if (document.activeElement !== input) input.value = label[field.key] ?? '';
      const bad = field.kind === 'number' && Number.isNaN(parseNumber(label[field.key]));
      input.setAttribute('aria-invalid', String(bad));
      input.title = bad ? 'Not a number: Save WAV leaves this label out' : '';
    }
  }

  build(frame) {
    const outputs = frame.outputs;
    const bank = frame.bank;
    const key = `${outputs.microphones}|${outputs.bank_beams}|${bank?.azimuth_deg?.join(',')}`;
    if (key === this.key) return;
    this.key = key;
    const choices = [['0:0', 'Steered beam, follows the source']];
    for (let beam = 0; beam < (outputs.bank_beams || 0); beam++) {
      const az = bank?.azimuth_deg?.[beam];
      const el = bank?.elevation_deg?.[beam];
      choices.push([`2:${beam}`, `Bank beam ${beam}${Number.isFinite(az) ? ` (${deg(az, 0)}, ${deg(el, 0)})` : ''}`]);
    }
    for (let mic = 0; mic < (outputs.microphones || 0); mic++) choices.push([`1:${mic}`, `Microphone ${mic}`]);
    this.output.replaceChildren(...choices.map(([value, text]) => h('option', {value}, text)));
  }

  render(frame) {
    this.listen.hidden = !frame?.outputs;
    if (frame?.outputs) {
      this.build(frame);
      const ch1 = frame.ch1;
      if (ch1 && ch1.code === 0 && document.activeElement !== this.output) {
        this.output.value = `${ch1.source}:${ch1.source === 0 ? 0 : ch1.index}`;
      }
      this.output.disabled = !this.session.ready;
    }
    this.renderRecording(frame);
  }

  renderRecording(frame) {
    const session = this.session;
    const state = session.recordingState;
    const capable = session.recordingCapable;
    const recording = frame?.last_recording;
    const progress = Number(frame?.audio_download_progress || 0);
    let text = capable ? '' : 'Recording needs the serial link to the MAX78002.';
    if (state === 'recording') text = 'Recording…';
    else if (state === 'ready' && recording) {
      const channels = Number(frame?.audio_format?.channels || 1);
      text = `Ready to save · ${(Number(recording.samples || 0) / 48000).toFixed(1)} s, 48 kHz ${channels === 2 ? 'stereo' : 'mono'}, ${recording.overruns || 0} overruns`;
    } else if (state === 'downloading') text = `Downloading · ${Math.round(progress * 100)}%`;
    else if (state === 'downloaded') text = 'Saved and checksum-verified.';
    else if (state === 'error') text = frame?.audio_download_error || 'Recording error';
    if (session.downloadRequested && !session.downloadCommandSent && session.downloadStopSent) text = 'Stopping the scan before downloading…';
    setText(this.recText, text);
    this.recText.hidden = !text;
    this.recProgress.hidden = state !== 'downloading';
    this.recProgress.value = progress;
    const isRecording = state === 'recording';
    setText(this.recLabel, isRecording ? 'Stop' : 'Record');
    this.recIcon.classList.toggle('btn__rec--square', isRecording);
    this.recButton.disabled = !capable || state === 'downloading';
    this.saveButton.disabled = !capable || !['ready', 'downloaded'].includes(state);
  }
}

export class EventLog {
  constructor(session, {onFold}) {
    this.session = session;
    this.count = h('span', {class: 'rail__count'}, '0');
    this.stripCount = h('span', {class: 'rail__count'}, '0');
    this.list = h('ol', {class: 'events'});
    this.el = h('aside', {class: 'rail', 'aria-label': 'Log'},
      h('header', {class: 'panel__head'},
        h('h2', {class: 'label'}, 'Log'), this.count,
        h('button', {type: 'button', class: 'btn btn--sm btn--quiet rail__clear', onClick: () => session.clearEvents()}, 'Clear'),
        h('button', {type: 'button', class: 'btn btn--sm btn--ghost', title: 'Hide the log', 'aria-label': 'Hide the log', onClick: () => onFold(true)}, '«')),
      this.list,
      h('button', {type: 'button', class: 'rail__strip', title: 'Show the log', 'aria-label': 'Show the log', onClick: () => onFold(false)},
        this.stripCount, h('span', {class: 'rail__vlabel'}, 'LOG')));
    session.addEventListener('events', () => this.render());
    this.render();
  }

  render() {
    const events = this.session.events;
    setText(this.count, String(events.length));
    setText(this.stripCount, String(events.length));
    if (!events.length) {
      this.list.replaceChildren(h('li', {class: 'events__empty'}, 'No events yet.'));
      return;
    }
    this.list.replaceChildren(...[...events].reverse().map((event) =>
      h('li', {class: `events__item${event.tone ? ' is-' + event.tone : ''}`},
        h('div', {class: 'events__meta'}, h('b', {}, event.kind), h('time', {}, clockTime(event.time))),
        h('p', {}, event.text))));
  }
}

export class ManualBeamPanel {
  constructor(session) {
    this.session = session;
    this.touched = false;
    this.sector = h('input', {class: 'input input--mono input--sm', type: 'number', min: 0, value: 24,
      style: {width: '84px'}, onInput: () => { this.touched = true; }});
    this.hold = h('button', {type: 'button', class: 'btn btn--sm',
      onClick: () => session.startMonitor(parseInt(this.sector.value, 10))}, 'Hold beam');
    this.release = h('button', {type: 'button', class: 'btn btn--sm', onClick: () => session.stopMonitor()}, 'Release');
    this.rate = h('select', {class: 'select select--sm', 'aria-label': 'Re-read rate',
      onChange: () => session.setMonitorRate(parseInt(this.rate.value, 10))}, options(RATES, String(session.monitorRate)));
    this.el = h('div', {class: 'drawer-page'},
      h('div', {class: 'form'},
        h('label', {}, 'Sector', this.sector), this.hold, this.release,
        h('label', {}, 'Re-read', this.rate)),
      h('p', {class: 'note'}, 'The beam stays on the sector and its level is re-read at this rate. During a diagnostic scan, clicking a tile points the beam there once.'));
  }

  setSector(sector) {
    this.sector.value = sector;
    this.touched = true;
  }

  render() {
    const session = this.session;
    const cfg = session.configuration;
    this.sector.disabled = !session.ready;
    this.hold.disabled = !session.ready;
    this.release.disabled = !session.monitoring;
    this.hold.setAttribute('aria-pressed', String(session.monitoring));
    if (cfg) {
      this.sector.max = cfg.sectors - 1;
      if (!this.touched) this.sector.value = Math.floor(cfg.rows / 2) * cfg.columns + Math.floor(cfg.columns / 2);
    }
  }
}

export class MeasurementPanel {
  constructor(session) {
    this.session = session;
    this.mode = segmented([['0', 'DSP detector'], ['1', 'MCU window']], {
      label: 'Measurement source',
      onPick: async (key) => {
        this.mode.setDisabled(true);
        try { await session.setSetting('measure', Number(key), 'measurement source'); }
        finally { this.render(); }
      },
    });
    this.aim = segmented([['1', 'Pattern fit'], ['0', 'Peak beam']], {
      label: 'Live aim',
      onPick: async (key) => {
        this.aim.setDisabled(true);
        try { await session.setSetting('fit', Number(key), 'Live aim'); }
        finally { this.render(); }
      },
    });
    const input = (key, minimum, maximum, step) => {
      this.inputs[key] = h('input', {class: 'input input--mono input--sm', type: 'number', min: minimum, max: maximum, step, style: {width: '96px'}});
      return this.inputs[key];
    };
    this.inputs = {};
    this.apply = h('button', {type: 'button', class: 'btn btn--sm', onClick: async () => {
      this.apply.disabled = true;
      try {
        await session.setMeasurementSettings({
          settle_us: Number(this.inputs.settle_us.value),
          guard_us: Number(this.inputs.guard_us.value),
          window_us: Number(this.inputs.window_us.value),
        });
      } finally { this.render(); }
    }}, 'Apply');
    this.el = h('div', {class: 'drawer-page'},
      h('div', {class: 'form'},
        this.mode.el,
        h('label', {}, 'Settle', input('settle_us', 50, 20000, 50), 'µs'),
        h('label', {}, 'Guard', input('guard_us', 0, 20000, 50), 'µs'),
        h('label', {}, 'Window', input('window_us', 1000, 100000, 500), 'µs'),
        this.apply),
      h('div', {class: 'form'}, h('span', {class: 'pick', title: 'Where Live steers the beam (Ch0): the pattern fit of the bank map, '
        + 'or the peak bank beam'}, 'Live aim', this.aim.el)),
      h('p', {class: 'note'}, 'Settings change only while the device is stopped.'));
  }

  render() {
    const settings = this.session.frame?.settings || {};
    const idle = this.session.ready && this.session.firmwareMode === 'IDLE';
    this.mode.set(String(settings.measure ?? 0));
    this.mode.setDisabled(!idle);
    this.aim.set(String(settings.fit ?? 0));
    this.aim.setDisabled(!idle || settings.fit === undefined);
    for (const [key, input] of Object.entries(this.inputs)) {
      if (document.activeElement !== input && Number.isFinite(settings[key])) input.value = settings[key];
      input.disabled = !idle;
    }
    this.apply.disabled = !idle;
  }
}

export class TonePanel {
  constructor(session) {
    this.session = session;
    this.list = h('div', {class: 'kv'});
    this.period = h('select', {class: 'select select--sm', 'aria-label': 'Tone report period',
      onChange: async () => {
        this.period.disabled = true;
        try { await session.setSetting('tone', Number(this.period.value), 'tone reports'); }
        catch (_) { /* logged by the session */ }
        finally { this.render(); }
      }},
    options([['0', 'Off'], ['100', 'Every 100 ms'], ['250', 'Every 250 ms'], ['500', 'Every 500 ms'], ['1000', 'Every 1 s']], '0'));
    this.el = h('div', {class: 'drawer-page'},
      h('div', {class: 'form'}, h('label', {}, 'Reports', this.period)),
      this.list);
  }

  render() {
    const frame = this.session.frame;
    const period = Number(frame?.settings?.tone || 0);
    const idle = this.session.ready && this.session.firmwareMode === 'IDLE';
    if (document.activeElement !== this.period) this.period.value = String(period);
    this.period.disabled = !idle;
    this.period.title = idle ? '' : 'Stop the device to change settings.';
    const tones = Object.values(frame?.tones || {}).sort((a, b) => a.channel - b.channel);
    if (!tones.length) {
      this.list.replaceChildren(h('p', {class: 'note'}, period > 0 ? 'No tonal estimate yet.' : 'Tone reports are off.'));
      return;
    }
    this.list.replaceChildren(...tones.flatMap((tone) => [
      h('div', {}, `Ch${tone.channel} fundamental`, h('b', {}, `${num(tone.f0_hz, 1)} Hz`)),
      h('div', {}, 'Score', h('b', {}, `${num(tone.score_db, 1)} dB`)),
      h('div', {}, 'Harmonics', h('b', {}, String(tone.harmonics))),
    ]));
  }
}

const RATE_NOTES = {
  stare: 'The bank measures all its directions together every 50 ms; the steered beam follows the strongest.',
  track: 'Waiting for a lock. Tracking starts with a full search.',
  continuous: 'Every pass re-measures all sectors.',
  once: 'The map holds the result of one pass until you start another.',
  fixed: 'Only the held sector updates while the beam is fixed.',
  idle: 'Nothing is being measured.',
};

export class DiagnosticsPanel {
  constructor(session, {extra} = {}) {
    this.session = session;
    this.kv = h('div', {class: 'kv'});
    this.note = h('p', {class: 'note'});
    this.freeze = h('button', {type: 'button', class: 'btn btn--sm', onClick: () => session.toggleFreeze()}, 'Freeze display');
    this.el = h('div', {class: 'drawer-page'},
      this.kv,
      this.note,
      h('div', {class: 'form'}, this.freeze, h('span', {class: 'note'}, 'Freezing pauses this display only; the device keeps running.')),
      extra || null);
  }

  render(mode) {
    const session = this.session;
    const frame = session.frame;
    const program = frame?.dsp;
    const tables = frame?.tables;
    const taper = frame?.taper;
    const truth = frame?.emulator_truth;
    const track = frame?.target_update_rate_hz;
    const trackCurrent = Number.isFinite(track) && Number.isFinite(frame?.target_update_age_s) && frame.target_update_age_s < 2;
    const rows = [
      ['Program', program ? `${program.program} · ${program.style === 'lut' ? 'LUT' : 'per mic'}` : '—'],
      ['Signature', program ? `${program.matched} / ${program.read}` : '—'],
      ['DSP load', program?.load ? `${(program.load.elapsed_us / 1e6).toFixed(2)} s · code ${program.load.code}` : '—'],
      ['Tables', tables ? `${tables.verified} / ${tables.words}` : '—'],
      ['Taper', taper ? `${taper.verified} / ${taper.written} · σ ${num(taper.sigma, 2)}` : '—'],
      ['Direction updates', trackCurrent ? hz(track, 0) : '—'],
      ['Full scans', hz(frame?.scan_rate_hz, 1)],
      ['Passes', String(frame?.scan_count ?? 0)],
      ['Ignored lines', String(frame?.protocol_errors ?? 0)],
    ];
    if (program?.core) {
      rows.push(['DSP core', `status ${program.core.status} · panic 0x${program.core.panic_code.toString(16).padStart(4, '0')}`]);
    }
    if (truth) {
      rows.push(['Emulator source', `${deg(truth.azimuth_deg)}, ${deg(truth.elevation_deg)}${truth.source_active === false ? ' · dropout' : ''}`]);
      if (Number.isFinite(frame.est_az_deg) && Number.isFinite(frame.est_el_deg)) {
        rows.push(['Answer error', deg(Math.hypot(frame.est_az_deg - truth.azimuth_deg, frame.est_el_deg - truth.elevation_deg))]);
      }
    }
    this.kv.replaceChildren(...rows.map(([key, value]) => h('div', {}, key, h('b', {}, value))));
    setText(this.note, mode === 'track' && trackCurrent && frame?.scan_rate_hz > 0
      ? `Tracking re-measures the sectors around the target, so the direction refreshes ${(track / frame.scan_rate_hz).toFixed(1)}× faster than a full scan.`
      : RATE_NOTES[mode] || '');
    setText(this.freeze, session.frozen ? 'Resume display' : 'Freeze display');
    this.freeze.setAttribute('aria-pressed', String(session.frozen));
    this.freeze.disabled = !session.open;
  }
}

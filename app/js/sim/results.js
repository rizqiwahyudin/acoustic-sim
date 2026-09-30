/**
 * results.js — the Simulator's answer column: estimate and error, the SRP map
 * with top candidates or the beam-scan polar plot, the DSP budget, the room
 * impulse response, audio A/B/C with downloads, the ML spectrogram and the
 * run details.
 */

import { base64ToBlobUrl } from '../shared/api.js';
import { h, s, segmented, setText } from '../shared/dom.js';
import { rampColor, rampGradient, rgbCss } from '../shared/colormap.js';
import { MINUS, deg, num } from '../shared/format.js';

const METHOD_NAMES = {srp_phat: 'SRP-PHAT', steered_das: 'Steered DAS', beam_bank_das: 'Beam bank'};
const AUDIO = [['raw', 'One microphone'], ['unsteered', 'Unsteered sum'], ['beam', 'Beamformed'], ['ml', 'ML input']];

function angleError(a, b) {
  return Math.abs(((a - b) + 540) % 360 - 180);
}

export class ResultsColumn {
  constructor() {
    this.urls = {};
    this.audioMode = 'beam';

    this.chip = h('span', {class: 'chip'}, 'Not run yet');
    this.az = h('span', {class: 'answer__value answer__value--sm'}, '—');
    this.elValue = h('span', {class: 'answer__value answer__value--sm'}, '—');
    this.azErr = h('span', {class: 'answer__sub'}, 'Azimuth');
    this.elErr = h('span', {class: 'answer__sub'}, 'Elevation');
    this.truth = h('p', {class: 'answer__detail'}, 'Set up the scenario on the left, then run the simulation.');
    this.notes = h('ul', {class: 'clip-notes', hidden: true});

    this.mapSvg = s('svg', {class: 'srp-map', viewBox: '0 0 400 196', role: 'img'});
    this.candidates = h('ol', {class: 'candidates'});
    this.mapSection = h('section', {class: 'section', hidden: true, 'aria-labelledby': 'sim-map-h'},
      h('h2', {class: 'section__title', id: 'sim-map-h'}, 'SRP-PHAT power over all directions'),
      this.mapSvg,
      h('div', {class: 'srp-legend'},
        h('span', {class: 'inline-field'}, 'Low', h('span', {class: 'srp-legend__bar', style: {background: rampGradient('to right')}}), 'High'),
        h('span', {class: 'inline-field'},
          h('span', {class: 'srp-legend__truth', 'aria-hidden': 'true'}), 'True',
          h('span', {class: 'srp-legend__est', 'aria-hidden': 'true'}, '×'), 'Estimate'),
      ),
      this.candidates,
    );

    this.polarSvg = s('svg', {class: 'scan-polar', viewBox: '0 0 240 240', role: 'img'});
    this.polarRange = h('span', {class: 'note'});
    this.polarSection = h('section', {class: 'section', hidden: true, 'aria-labelledby': 'sim-polar-h'},
      h('div', {class: 'section__row'}, h('h2', {class: 'section__title', id: 'sim-polar-h'}, 'Beam scan power'), this.polarRange),
      this.polarSvg,
    );

    this.mips = h('dl', {class: 'figures'});
    this.mipsSection = h('section', {class: 'section', hidden: true, 'aria-labelledby': 'sim-mips-h'},
      h('h2', {class: 'section__title', id: 'sim-mips-h'}, 'DSP budget'), this.mips);

    this.rirSvg = s('svg', {class: 'rir', viewBox: '0 0 400 118', role: 'img', 'aria-label': 'Room impulse response'});
    this.rirSection = h('section', {class: 'section', hidden: true, 'aria-labelledby': 'sim-rir-h'},
      h('h2', {class: 'section__title', id: 'sim-rir-h'}, 'Room impulse response, microphone 1'), this.rirSvg);

    this.audioSeg = segmented(AUDIO, {label: 'Audio source', onPick: (key) => this.setAudio(key)});
    this.player = h('audio', {controls: true, preload: 'auto', class: 'sim-player'});
    this.downloads = h('div', {class: 'downloads'});
    this.spectrogram = h('img', {class: 'spectrogram', alt: 'Log-mel spectrogram of the ML input', hidden: true});
    this.audioSection = h('section', {class: 'section', hidden: true, 'aria-labelledby': 'sim-audio-h'},
      h('h2', {class: 'section__title', id: 'sim-audio-h'}, 'Listen'),
      this.audioSeg.el, this.player, this.downloads, this.spectrogram);

    this.details = h('dl', {class: 'figures figures--details'});
    this.detailsSection = h('details', {class: 'disclosure section', hidden: true},
      h('summary', {}, 'Run details'), this.details);

    this.el = h('aside', {class: 'sim-results', 'aria-label': 'Result'},
      h('section', {class: 'section', 'aria-labelledby': 'sim-res-h'},
        h('div', {class: 'section__row'}, h('h2', {class: 'section__title', id: 'sim-res-h'}, 'Estimated direction'), this.chip),
        h('div', {class: 'answer'},
          h('div', {class: 'answer__figure'}, this.az, this.azErr),
          h('div', {class: 'answer__figure'}, this.elValue, this.elErr),
        ),
        this.truth,
        this.notes,
      ),
      this.mapSection, this.polarSection, this.mipsSection, this.rirSection, this.audioSection, this.detailsSection,
    );
  }

  setBusy(busy) {
    this.el.classList.toggle('is-busy', busy);
    if (busy) {
      setText(this.chip, 'Running…');
      this.chip.className = 'chip chip--accent';
    }
  }

  showError(message) {
    setText(this.chip, 'Failed');
    this.chip.className = 'chip chip--danger';
    setText(this.truth, `${message}. Make sure sim_server.py is running on 127.0.0.1:8766.`);
  }

  clearAudio() {
    for (const url of Object.values(this.urls)) if (url) URL.revokeObjectURL(url);
    this.urls = {};
  }

  setAudio(mode) {
    if (!this.urls[mode]) return;
    this.audioMode = mode;
    this.audioSeg.set(mode);
    const playing = !this.player.paused;
    this.player.src = this.urls[mode];
    if (playing) this.player.play().catch(() => {});
  }

  render(data, params) {
    const method = data.beam_method || params.beam_method;
    setText(this.chip, METHOD_NAMES[method] || method);
    this.chip.className = 'chip chip--plain';
    setText(this.az, deg(data.est_az_deg));
    setText(this.elValue, deg(data.est_el_deg));
    const azErr = angleError(data.est_az_deg, data.true_az_deg);
    const elErr = Math.abs(data.est_el_deg - data.true_el_deg);
    setText(this.azErr, `Azimuth, off by ${num(azErr, 1)}°`);
    setText(this.elErr, `Elevation, off by ${num(elErr, 1)}°`);
    const moving = params.moving_source ? ' The drone moves during the run.' : '';
    setText(this.truth, `True direction ${deg(data.true_az_deg)}, ${deg(data.true_el_deg)}.${moving} Computed in ${data.elapsed_s} s.`);

    const notes = [...(data.mic_clip_notes || []), ...(data.beam_scan?.clamp_notes || [])];
    this.notes.hidden = notes.length === 0;
    this.notes.replaceChildren(...notes.map((note) => {
      const mic = Number.isInteger(note.idx) ? note.idx + 1 : '?';
      return h('li', {}, note.axis
        ? `Microphone ${mic} moved on ${note.axis} to fit the room: ${num(note.before, 3)} → ${num(note.after, 3)} m`
        : `Fractional delay clamped${note.angle_deg !== undefined ? ` at ${num(note.angle_deg, 0)}°` : ''}: microphone ${mic} needed ${num(note.before_samples, 2)} samples, maximum ${num(note.max_samples, 0)}`);
    }));

    const srp = method === 'srp_phat' && Array.isArray(data.power) && data.power.length;
    this.mapSection.hidden = !srp;
    if (srp) this.renderMap(data);
    const scan = !srp && data.beam_scan && Array.isArray(data.beam_scan.angles_deg) && data.beam_scan.angles_deg.length;
    this.polarSection.hidden = !scan;
    if (scan) this.renderPolar(data.beam_scan);

    const onDsp = data.mips && data.mips.method !== 'srp_phat';
    this.mipsSection.hidden = !onDsp;
    if (onDsp) {
      const m = data.mips;
      const rows = [
        ['Method', m.method],
        ['Operations per sample', Number(m.ops_per_sample || 0).toFixed(0)],
        ['Share of the 6144-op budget', `${Number(m.budget_used_pct || 0).toFixed(1)} %`],
      ];
      if (m.scan_latency_s !== null && m.scan_latency_s !== undefined) rows.push(['Scan latency', `${Number(m.scan_latency_s).toFixed(2)} s`]);
      this.mips.replaceChildren(...rows.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {class: 'mono'}, v)]));
    }

    this.rirSection.hidden = !data.rir?.length;
    if (data.rir?.length) this.renderRir(data.rir);

    this.renderAudio(data, params);
    this.renderDetails(data, params);
  }

  renderMap(data) {
    const grid = data.power;
    const nC = grid.length;
    const nA = grid[0].length;
    let lo = Infinity;
    let hi = -Infinity;
    for (const row of grid) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const span = Math.max(hi - lo, 1e-12);
    const x0 = 36; const x1 = 392; const y0 = 8; const y1 = 172;
    const w = (x1 - x0) / nA;
    const hh = (y1 - y0) / nC;
    const nodes = [];
    for (let c = 0; c < nC; c++) {
      for (let a = 0; a < nA; a++) {
        nodes.push(s('rect', {x: (x0 + a * w).toFixed(2), y: (y0 + c * hh).toFixed(2), width: (w + 0.3).toFixed(2), height: (hh + 0.3).toFixed(2),
          fill: rgbCss(rampColor((grid[c][a] - lo) / span))}));
      }
    }
    const px = (az) => x0 + (((az % 360) + 360) % 360) / 360 * (x1 - x0);
    const py = (el) => y0 + (90 - el) / 180 * (y1 - y0);
    const tx = px(data.true_az_deg); const ty = py(data.true_el_deg);
    const ex = px(data.est_az_deg); const ey = py(data.est_el_deg);
    nodes.push(s('circle', {cx: tx, cy: ty, r: 7, fill: 'none', stroke: '#ffffff', 'stroke-width': 2}));
    nodes.push(s('path', {d: `M${ex - 5} ${ey - 5} L${ex + 5} ${ey + 5} M${ex + 5} ${ey - 5} L${ex - 5} ${ey + 5}`, stroke: 'var(--accent)', 'stroke-width': 2.5}));
    for (const [label, y] of [['90°', 16], ['0°', 94], [`${MINUS}90°`, 172]]) nodes.push(s('text', {class: 'tick', x: 30, y, 'text-anchor': 'end'}, label));
    for (const az of [0, 90, 180, 270, 360]) nodes.push(s('text', {class: 'tick', x: px(az === 360 ? 359.999 : az), y: 188, 'text-anchor': 'middle'}, `${az}°`));
    this.mapSvg.replaceChildren(...nodes);
    this.mapSvg.setAttribute('aria-label', `Steered response power by azimuth and elevation. Estimate ${deg(data.est_az_deg)}, ${deg(data.est_el_deg)}; truth ${deg(data.true_az_deg)}, ${deg(data.true_el_deg)}.`);
    const peaks = data.top_peaks || [];
    this.candidates.replaceChildren(...peaks.map((p, i) => h('li', {class: 'candidates__item'},
      h('span', {class: 'candidates__rank'}, String(i + 1)),
      h('span', {class: 'mono'}, `${deg(p.az_deg)}, ${deg(p.el_deg)}`),
      h('span', {class: 'mono candidates__db'}, i === 0 ? 'peak' : `${num(p.rel_db, 1)} dB`),
    )));
  }

  renderPolar(scan) {
    const angles = scan.angles_deg;
    const powers = scan.powers_db || [];
    const arg = Number(scan.argmax_idx || 0);
    const lo = Math.min(...powers);
    const hi = Math.max(...powers);
    const span = Math.max(hi - lo, 1e-9);
    setText(this.polarRange, `${num(lo, 1)} to ${num(hi, 1)} dB`);
    const cx = 120; const cy = 120; const rMax = 96;
    const wedge = (2 * Math.PI) / angles.length;
    const nodes = [1, 2, 3, 4].map((i) => s('circle', {cx, cy, r: rMax * i / 4, fill: 'none', stroke: 'var(--line)'}));
    angles.forEach((angle, i) => {
      const a = angle * Math.PI / 180;
      const r = rMax * (0.15 + 0.85 * (powers[i] - lo) / span);
      const a0 = a - wedge * 0.45; const a1 = a + wedge * 0.45;
      const d = `M${cx} ${cy} L${cx + r * Math.cos(a0)} ${cy - r * Math.sin(a0)} A${r} ${r} 0 0 0 ${cx + r * Math.cos(a1)} ${cy - r * Math.sin(a1)} Z`;
      nodes.push(s('path', {d, fill: i === arg ? 'var(--accent)' : 'var(--raised)'}));
    });
    for (const [label, x, y, anchor] of [['0°', 232, 124, 'end'], ['90°', 120, 14, 'middle'], ['180°', 8, 124, 'start'], ['270°', 120, 236, 'middle']]) {
      nodes.push(s('text', {class: 'tick', x, y, 'text-anchor': anchor}, label));
    }
    this.polarSvg.replaceChildren(...nodes);
    this.polarSvg.setAttribute('aria-label', `Beam scan power by azimuth; strongest beam at ${deg(angles[arg], 0)}.`);
  }

  renderRir(rir) {
    const peak = Math.max(...rir.map((v) => Math.abs(v))) || 1;
    const x0 = 40; const x1 = 380; const mid = 55; const amp = 42;
    const stepX = (x1 - x0) / Math.max(1, rir.length - 1);
    const d = rir.map((v, i) => `${i ? 'L' : 'M'}${(x0 + i * stepX).toFixed(1)} ${(mid - v / peak * amp).toFixed(1)}`).join(' ');
    const ms = (rir.length / 16000 * 1000).toFixed(0);
    this.rirSvg.replaceChildren(
      s('line', {class: 'grid', x1: x0, x2: x1, y1: mid, y2: mid}),
      s('path', {d, fill: 'none', stroke: 'var(--text)', 'stroke-width': 1}),
      s('text', {class: 'tick', x: x0, y: 114, 'text-anchor': 'middle'}, '0'),
      s('text', {class: 'tick', x: x1, y: 114, 'text-anchor': 'middle'}, `${ms} ms`),
    );
  }

  renderAudio(data, params) {
    this.clearAudio();
    if (!data.audio_b64) {
      this.audioSection.hidden = true;
      return;
    }
    this.audioSection.hidden = false;
    this.urls.beam = base64ToBlobUrl(data.audio_b64);
    if (data.raw_audio_b64) this.urls.raw = base64ToBlobUrl(data.raw_audio_b64);
    if (data.unsteered_audio_b64) this.urls.unsteered = base64ToBlobUrl(data.unsteered_audio_b64);
    if (data.ml_audio_b64) this.urls.ml = base64ToBlobUrl(data.ml_audio_b64);
    for (const [key, button] of this.audioSeg.buttons) button.hidden = !this.urls[key];
    const tag = `${params.geometry}_${data.seed_used ?? params.seed}`;
    const names = {raw: `raw_${tag}.wav`, unsteered: `unsteered_${tag}.wav`, beam: `beamformed_${tag}.wav`, ml: `ml_preview_${tag}.wav`};
    this.downloads.replaceChildren(h('span', {class: 'note'}, 'Download'),
      ...AUDIO.filter(([key]) => this.urls[key]).map(([key, label]) => h('a', {class: 'downloads__link', href: this.urls[key], download: names[key]}, label)));
    this.setAudio(this.urls[this.audioMode] ? this.audioMode : 'beam');
    if (data.ml_spectrogram_png_b64) {
      this.spectrogram.src = `data:image/png;base64,${data.ml_spectrogram_png_b64}`;
      this.spectrogram.hidden = false;
    } else {
      this.spectrogram.hidden = true;
      this.spectrogram.removeAttribute('src');
    }
  }

  renderDetails(data, p) {
    const rows = [
      ['Microphones', String(data.mic_positions?.length ?? '—')],
      ['Room', `${p.room_length} × ${p.room_width} × ${p.room_height} m`],
      ['Absorption', p.absorption_mode === 'materials'
        ? `Materials${data.rt60_actual != null ? `, measured RT60 ${data.rt60_actual} s` : ''}` : `RT60 ${p.rt60} s`],
      ['Seed', `${p.seed < 0 ? 'random ' : ''}#${data.seed_used ?? p.seed}`],
      ['Drone, crowd, PA, floor', `${p.drone_spl_db} / ${p.crowd_spl_db} / ${p.pa_spl_db} / ${p.mic_noise_floor_db} dB`],
      ['Interference', p.diffuse ? `${p.crowd_count} talkers, ${p.pa_count} PA${p.crowd_model === 'plane_wave' ? `, plane waves (${p.n_plane_waves})` : ''}` : 'none'],
      ['Direction band', `${p.fmin_hz}–${p.fmax_hz} Hz${p.harmonic_comb ? `, comb at ${p.drone_fundamental_hz} Hz` : ''}${data.n_freq_bins !== undefined ? ` (${data.n_freq_bins} bins)` : ''}`],
      ['Atmosphere', `${p.temperature_c} °C, ${p.humidity_pct} % RH, ${p.temp_gradient_c_per_m} °C/m`],
      ['Impairments', [p.mic_mismatch && 'mismatch', p.crosstalk && `crosstalk ${p.crosstalk_db} dB`, p.quantization && `${p.bit_depth}-bit`].filter(Boolean).join(', ') || 'none'],
    ];
    if (Number.isFinite(data.atmospheric_bias_deg) && Math.abs(data.atmospheric_bias_deg) >= 0.05) rows.push(['Atmospheric elevation bias', `${num(data.atmospheric_bias_deg, 2)}°`]);
    if (data.ml_path_snr_db != null) rows.push(['ML audio SNR', `${data.ml_path_snr_db} dB`]);
    if (data.feature_snr_db != null) rows.push(['ML feature SNR', `${data.feature_snr_db} dB`]);
    if ((data.image_sources || []).length) rows.push(['Reflections drawn', String(data.image_sources.length)]);
    this.details.replaceChildren(...rows.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {class: 'mono'}, v)]));
    this.detailsSection.hidden = false;
  }
}

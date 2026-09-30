/**
 * status.js — the right-hand column: the answer (direction), how fast it
 * refreshes, scenario truth for emulators, recording, and the event log.
 */

import { h, setText } from '../shared/dom.js';
import { clockTime, deg, formatAge, hz, num } from '../shared/format.js';

export function operatingMode(session) {
  const frame = session.frame;
  if (session.monitoring) return 'fixed';
  const mode = frame?.firmware_mode;
  if (mode === 'G') return 'track';
  if (mode === 'C') return 'continuous';
  if (mode === 'F') return 'once';
  if (frame?.last_steer) return 'fixed';
  return 'idle';
}

function side(value, positive, negative) {
  if (!Number.isFinite(value) || Math.abs(value) < 0.05) return 'centre';
  return value > 0 ? positive : negative;
}

export class StatusColumn {
  constructor(session, {extra} = {}) {
    this.session = session;

    this.title = h('h2', {class: 'section__title', id: 'hw-answer-title'}, 'Source direction');
    this.chip = h('span', {class: 'chip'}, 'Idle');
    this.az = h('span', {class: 'answer__value'}, '—');
    this.elValue = h('span', {class: 'answer__value'}, '—');
    this.azSub = h('span', {class: 'answer__sub'}, 'Azimuth');
    this.elSub = h('span', {class: 'answer__sub'}, 'Elevation');
    this.detail = h('p', {class: 'answer__detail', 'aria-live': 'polite'});
    const answer = h('section', {class: 'section', 'aria-labelledby': 'hw-answer-title'},
      h('div', {class: 'section__row'}, this.title, this.chip),
      h('div', {class: 'answer'},
        h('div', {class: 'answer__figure'}, this.az, this.azSub),
        h('div', {class: 'answer__figure'}, this.elValue, this.elSub),
      ),
      this.detail,
      h('p', {class: 'note'}, 'Left and right as seen from behind the array.'),
    );

    this.rateValues = [h('span', {class: 'figure-value'}, '—'), h('span', {class: 'figure-value'}, '—')];
    this.rateLabels = [h('span', {class: 'figure-label'}, 'Direction updates'), h('span', {class: 'figure-label'}, 'Full scans')];
    this.rateNote = h('p', {class: 'note note--body'});
    const refresh = h('section', {class: 'section', 'aria-labelledby': 'hw-rate-title'},
      h('h2', {class: 'section__title', id: 'hw-rate-title'}, 'Refresh'),
      h('div', {class: 'rates'},
        h('div', {class: 'rate'}, this.rateValues[0], this.rateLabels[0]),
        h('div', {class: 'rate'}, this.rateValues[1], this.rateLabels[1]),
      ),
      this.rateNote,
    );

    this.truthChip = h('span', {class: 'chip chip--plain'}, 'Source active');
    this.truthValue = h('span', {class: 'truth__value'});
    this.truthMeta = h('p', {class: 'note note--body'});
    this.truth = h('section', {class: 'section', 'aria-labelledby': 'hw-truth-title', hidden: true},
      h('div', {class: 'section__row'}, h('h2', {class: 'section__title', id: 'hw-truth-title'}, 'Scenario truth'), this.truthChip),
      this.truthValue,
      this.truthMeta,
    );

    this.recText = h('p', {class: 'recording__text'}, 'No recording yet.');
    this.recIcon = h('span', {class: 'btn__rec', 'aria-hidden': 'true'});
    this.recLabel = h('span', {}, 'Record');
    this.recButton = h('button', {type: 'button', class: 'btn btn--lg', onClick: () => session.toggleRecording()}, this.recIcon, this.recLabel);
    this.saveButton = h('button', {type: 'button', class: 'btn btn--lg', onClick: () => session.requestDownload()}, 'Save WAV');
    this.recProgress = h('progress', {max: 1, value: 0, hidden: true});
    const recording = h('section', {class: 'section', 'aria-labelledby': 'hw-rec-title'},
      h('h2', {class: 'section__title', id: 'hw-rec-title'}, 'Recording'),
      this.recText,
      this.recProgress,
      h('div', {class: 'toolbar toolbar--tight'}, this.recButton, this.saveButton),
      h('p', {class: 'note'}, 'For clean audio, record on a fixed beam. Tracking switches beams, and the switches are audible.'),
    );

    this.eventList = h('ol', {class: 'events'});
    const events = h('section', {class: 'section', 'aria-labelledby': 'hw-events-title'},
      h('div', {class: 'section__row'},
        h('h2', {class: 'section__title', id: 'hw-events-title'}, 'Events'),
        h('button', {type: 'button', class: 'btn btn--sm btn--quiet', onClick: () => session.clearEvents()}, 'Clear'),
      ),
      this.eventList,
    );

    this.freezeButton = h('button', {type: 'button', class: 'btn', onClick: () => session.toggleFreeze()}, 'Freeze display');
    const advanced = h('section', {class: 'section'},
      h('details', {class: 'disclosure'},
        h('summary', {}, 'Advanced'),
        h('div', {class: 'advanced'},
          this.freezeButton,
          h('p', {class: 'note'}, 'Freezing pauses this display only. The device keeps scanning, and commands still work.'),
        ),
      ),
    );

    this.el = h('aside', {class: 'hw-aside', 'aria-label': 'Status'},
      answer, refresh, this.truth, extra || null, recording, events, advanced);
    session.addEventListener('events', () => this.renderEvents());
    this.renderEvents();
  }

  render() {
    const session = this.session;
    const frame = session.frame;
    const mode = operatingMode(session);
    const answer = session.solution(frame);

    let title = 'Source direction';
    let chip = 'Idle';
    let tone = '';
    let detail = 'Nothing is being measured. Start a sweep or tracking to locate a source.';
    if (answer) {
      const where = `Sector ${answer.sector}`;
      const level = `${num(answer.level, 1)} dBFS`;
      const margin = Number.isFinite(frame.margin_db) ? `${num(frame.margin_db, 1)} dB` : null;
      if (answer.kind === 'target') {
        chip = 'Tracking'; tone = 'accent';
        detail = `${where} at ${level}${margin ? `, ${margin} above the next-strongest sector` : ''}. Updated ${formatAge(answer.ageMs)} ago.`;
      } else if (answer.kind === 'beam') {
        title = 'Beam direction';
        chip = session.monitoring ? 'Monitoring' : 'Fixed beam'; tone = 'accent';
        detail = `Beam held on sector ${answer.sector}, reading ${level}. Other sectors keep their last measurement.`;
      } else {
        title = 'Strongest sector';
        tone = 'plain';
        if (mode === 'continuous') {
          chip = 'Scanning';
          const period = Number.isFinite(frame.scan_rate_hz) && frame.scan_rate_hz > 0 ? ` The whole map refreshes every ${(1 / frame.scan_rate_hz).toFixed(2)} s.` : '';
          detail = `${where} at ${level}${margin ? `, ${margin} above the next` : ''}.${period}`;
        } else if (mode === 'track') {
          chip = 'Searching';
          detail = `Searching the field for a source. ${where} is strongest so far at ${level}.`;
        } else {
          chip = mode === 'once' ? 'Single sweep' : 'Last sweep';
          detail = `${where} at ${level}, from the last full pass ${formatAge(answer.ageMs)} ago.`;
        }
      }
    } else if (mode === 'track') {
      chip = 'Searching'; tone = 'plain';
      detail = 'Searching the field for a source.';
    }
    setText(this.title, title);
    setText(this.chip, chip);
    this.chip.className = `chip ${tone ? 'chip--' + tone : ''}`.trim();
    setText(this.az, answer ? deg(answer.azimuth) : '—');
    setText(this.elValue, answer ? deg(answer.elevation) : '—');
    setText(this.azSub, answer ? `Azimuth, ${side(answer.azimuth, 'right', 'left')}` : 'Azimuth');
    setText(this.elSub, answer ? `Elevation, ${side(answer.elevation, 'up', 'down')}` : 'Elevation');
    setText(this.detail, detail);

    this.renderRates(mode, frame);
    this.renderTruth(frame);
    this.renderRecording(frame);
    setText(this.freezeButton, session.frozen ? 'Resume display' : 'Freeze display');
    this.freezeButton.setAttribute('aria-pressed', String(session.frozen));
    this.freezeButton.disabled = !session.open;
  }

  renderRates(mode, frame) {
    const session = this.session;
    const cfg = frame?.configuration;
    const scan = frame?.scan_rate_hz;
    const track = frame?.target_update_rate_hz;
    const trackCurrent = Number.isFinite(track) && Number.isFinite(frame?.target_update_age_s) && frame.target_update_age_s < 2;
    let values = ['—', '—'];
    let labels = ['Direction updates', 'Full scans'];
    let note = 'Nothing is being measured.';
    if (mode === 'track') {
      values = [trackCurrent ? hz(track, 0) : '—', hz(scan, 1)];
      note = trackCurrent && Number.isFinite(scan) && scan > 0
        ? `Tracking re-measures the sectors around the target, so the direction refreshes ${(track / scan).toFixed(1)}× faster than a full scan.`
        : 'Waiting for a lock. Tracking starts with a full search.';
    } else if (mode === 'continuous') {
      values = [hz(scan, 1), hz(scan, 1)];
      note = cfg ? `Every pass re-measures all ${cfg.sectors} sectors.` : '';
    } else if (mode === 'once') {
      values = ['—', '1 pass'];
      note = 'The map holds the result of one pass until you start another.';
    } else if (mode === 'fixed') {
      labels = ['Level reads', 'Full scans'];
      values = [session.monitoring ? `${session.monitorRate} Hz` : 'On demand', '—'];
      note = 'Only the held sector updates while the beam is fixed.';
    }
    values.forEach((value, i) => setText(this.rateValues[i], value));
    labels.forEach((label, i) => setText(this.rateLabels[i], label));
    setText(this.rateNote, note);
  }

  renderTruth(frame) {
    const truth = frame?.emulator_truth;
    this.truth.hidden = !truth;
    if (!truth) return;
    const active = truth.source_active !== false;
    setText(this.truthChip, active ? 'Source active' : 'Acoustic dropout');
    this.truthChip.className = `chip ${active ? 'chip--plain' : 'chip--warn'}`;
    setText(this.truthValue, `${deg(truth.azimuth_deg)}, ${deg(truth.elevation_deg)}`);
    const error = Number.isFinite(frame.est_az_deg) && Number.isFinite(frame.est_el_deg)
      ? ` · answer off by ${num(Math.hypot(frame.est_az_deg - truth.azimuth_deg, frame.est_el_deg - truth.elevation_deg), 1)}°`
      : '';
    setText(this.truthMeta, `Range ${num(truth.range_m, 1)} m${error}. Shown on the map as a green cross.`);
  }

  renderRecording(frame) {
    const session = this.session;
    const state = session.recordingState;
    const capable = session.recordingCapable;
    const recording = frame?.last_recording;
    const progress = Number(frame?.audio_download_progress || 0);
    let text = capable ? 'No recording yet.' : 'Recording needs the serial connection to the MAX78002.';
    if (state === 'recording') text = 'Recording the beam output…';
    else if (state === 'ready' && recording) {
      text = `Ready to save · ${(Number(recording.samples || 0) / 48000).toFixed(1)} s, 48 kHz mono, ${recording.overruns || 0} overruns`;
    } else if (state === 'downloading') text = `Downloading · ${Math.round(progress * 100)}%`;
    else if (state === 'downloaded') text = 'Downloaded and checksum-verified.';
    else if (state === 'error') text = frame?.audio_download_error || 'Recording error';
    if (session.downloadRequested && !session.downloadCommandSent && session.downloadStopSent) text = 'Stopping the scan before downloading…';
    setText(this.recText, text);
    this.recProgress.hidden = state !== 'downloading';
    this.recProgress.value = progress;
    const isRecording = state === 'recording';
    setText(this.recLabel, isRecording ? 'Stop recording' : 'Record');
    this.recIcon.classList.toggle('btn__rec--square', isRecording);
    this.recButton.disabled = !capable || state === 'downloading';
    this.saveButton.disabled = !capable || !['ready', 'downloaded'].includes(state);
  }

  renderEvents() {
    const items = [...this.session.events].reverse().slice(0, 8);
    if (items.length === 0) {
      this.eventList.replaceChildren(h('li', {class: 'events__empty'}, 'No events yet.'));
      return;
    }
    this.eventList.replaceChildren(...items.map((event) =>
      h('li', {class: `events__item ${event.tone ? 'events__item--' + event.tone : ''}`.trim()},
        h('span', {class: 'events__time'}, clockTime(event.time)),
        h('span', {}, event.text),
      )));
  }
}

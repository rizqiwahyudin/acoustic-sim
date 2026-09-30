/**
 * controls.js — every Simulator setting, grouped, with rarely used ones in
 * "More settings". Element ids match the old app so saved presets and
 * #preset= links keep working.
 */

import { h, options, setText } from '../shared/dom.js';
import { MINUS } from '../shared/format.js';
import { customArray, customArrayEditor } from '../shared/custom-array.js';
import { heimdallToolPositions } from '../shared/heimdall-array.js';

const MATERIAL_FALLBACK = [
  'hard_surface', 'rough_concrete', 'unpainted_concrete', 'brickwork', 'marble_floor',
  'concrete_floor', 'linoleum_on_concrete', 'wood_1.6cm', 'carpet_thin', 'carpet_hairy',
  'carpet_tufted_9.5mm', 'plasterboard', 'gypsum_board', 'wooden_lining',
  'ceiling_plasterboard', 'ceiling_fissured_tile', 'ceiling_metal_panel',
  'ceiling_perforated_gypsum_board', 'mineral_wool_50mm_40kgm3',
  'glass_window', 'double_glazing_30mm', 'curtains_0.2', 'curtains_cotton_0.33',
  'curtains_cotton_0.5', 'curtains_velvet', 'audience_1_m2', 'chairs_medium_upholstered',
];
export const EXHIBITION_HALL = {
  floor: 'carpet_hairy', ceiling: 'ceiling_fissured_tile', east: 'plasterboard',
  west: 'plasterboard', south: 'curtains_cotton_0.5', north: 'plasterboard',
};
const WALLS = [['floor', 'matFloor', 'Floor'], ['ceiling', 'matCeiling', 'Ceiling'], ['east', 'matEast', 'East'],
  ['west', 'matWest', 'West'], ['south', 'matSouth', 'South'], ['north', 'matNorth', 'North']];

export const PRESET_SLIDER_IDS = [
  'simMicCount', 'simRadius', 'simSep', 'simRoomL', 'simRoomW', 'simRoomH', 'simRT60',
  'simSrcAz', 'simSrcEl', 'simSrcDist', 'simDroneSPL', 'simCrowdSPL', 'simPASPL', 'simMicFloor',
  'simSeed', 'simCrowd', 'simPA', 'simInt', 'simCrosstalkDb', 'simTemp', 'simHumidity', 'simTempGrad',
  'simSpeed', 'simHeading', 'simChunks', 'simNPlaneWaves', 'simCrosstalkCorner', 'simMlNMels',
  'simFmin', 'simFmax', 'simFundamental', 'simBeamK', 'simFracMax', 'simFracTaps', 'simTargetFs',
];
export const PRESET_SELECT_IDS = [
  'simGeo', 'simBitDepth', 'simTrajType', 'simBeamMethod',
  'matFloor', 'matCeiling', 'matEast', 'matWest', 'matSouth', 'matNorth',
  'simCrowdModel', 'simCrosstalkModel', 'simMlBitDepth', 'simMlFeatBitDepth',
];
export const PRESET_CHECK_IDS = [
  'simRandSeed', 'simDiffuse', 'simMismatch', 'simCrosstalk', 'simQuant',
  'simMoving', 'simMlPreview', 'simHarmonicComb', 'simNormalizeAudio',
];

const fixed = (digits, unit = '') => (v) => `${(v < 0 ? MINUS : '') + Math.abs(v).toFixed(digits)}${unit}`;

export function buildControls({onChange}) {
  const byId = new Map();
  const changed = () => onChange?.();

  const range = (id, label, min, max, step, value, format = fixed(0)) => {
    const output = h('span', {class: 'slider__value'});
    const input = h('input', {id, type: 'range', class: 'slider__input', min, max, step, value});
    const update = () => setText(output, format(Number(input.value)));
    input.addEventListener('input', () => { update(); changed(); });
    update();
    const el = h('div', {class: 'slider'}, h('div', {class: 'slider__head'}, h('label', {for: id}, label), output), input);
    byId.set(id, {input, update, el});
    return el;
  };
  const check = (id, label, checked = false) => {
    const input = h('input', {id, type: 'checkbox', checked});
    input.addEventListener('change', () => { sync(); changed(); });
    const el = h('label', {class: 'check'}, input, label);
    byId.set(id, {input, el});
    return el;
  };
  const select = (id, label, pairs, value) => {
    const input = h('select', {id, class: 'select'}, options(pairs, value));
    input.addEventListener('change', () => { sync(); changed(); });
    const el = h('div', {class: 'field'}, h('label', {for: id}, label), input);
    byId.set(id, {input, el});
    return el;
  };
  const get = (id) => byId.get(id)?.input;
  const section = (title, ...children) => h('section', {class: 'section'}, h('h2', {class: 'section__title'}, title), ...children);
  const group = (...children) => h('div', {class: 'sub-group'}, ...children);

  const editor = customArrayEditor({getSeed: () => ({
    count: Number(get('simMicCount').value), radius: Number(get('simRadius').value), separation: Number(get('simSep').value),
  })});
  customArray.addEventListener('change', () => changed());

  const scanAngles = h('input', {id: 'simScanAngles', class: 'input input--mono', value: '0,45,90,135,180,225,270,315'});
  scanAngles.addEventListener('change', changed);
  const scanAnglesField = h('div', {class: 'field'}, h('label', {for: 'simScanAngles'}, 'Scan angles (degrees, comma separated)'), scanAngles);

  const absRt60 = h('input', {type: 'radio', name: 'absMode', id: 'absRT60', value: 'rt60', checked: true});
  const absMaterials = h('input', {type: 'radio', name: 'absMode', id: 'absMaterials', value: 'materials'});
  for (const radio of [absRt60, absMaterials]) radio.addEventListener('change', () => { sync(); changed(); });
  const materialGrid = h('div', {class: 'material-grid'},
    ...WALLS.map(([, id, label]) => {
      const input = h('select', {id, class: 'select select--sm'});
      input.addEventListener('change', changed);
      byId.set(id, {input});
      return h('div', {class: 'field'}, h('label', {for: id}, label), input);
    }));
  const hallButton = h('button', {type: 'button', class: 'btn btn--sm', onClick: () => applyExhibitionHall()}, 'Exhibition hall');


  const el = h('div', {class: 'sim-controls'},
    section('Array and method',
      select('simGeo', 'Geometry', [['UCA', 'Circular ring'], ['CROSS', 'Standing cross'], ['ULA', 'Linear'],
        ['CYLINDER', 'Stacked rings'], ['HEIMDALL', 'Heimdall, 44 microphones'], ['CUSTOM', 'Custom coordinates']], 'UCA'),
      editor,
      select('simBeamMethod', 'Direction estimate', [['srp_phat', 'SRP-PHAT, reference'],
        ['steered_das', 'Steered delay-and-sum (scan and lock)'], ['beam_bank_das', 'Bank of K fixed beams']], 'srp_phat'),
      scanAnglesField,
      range('simBeamK', 'Fixed beams (K)', 4, 16, 1, 8),
      range('simMicCount', 'Microphones', 4, 32, 1, 12),
      range('simRadius', 'Radius', 0.03, 1.0, 0.01, 0.15, fixed(2, ' m')),
      range('simSep', 'Ring separation', 0.02, 0.5, 0.01, 0.12, fixed(2, ' m')),
    ),
    section('Room',
      range('simRoomL', 'Length', 10, 100, 1, 50, fixed(0, ' m')),
      range('simRoomW', 'Width', 10, 80, 1, 40, fixed(0, ' m')),
      range('simRoomH', 'Height', 4, 15, 1, 12, fixed(0, ' m')),
      h('fieldset', {class: 'radio-row'},
        h('legend', {}, 'Absorption'),
        h('label', {class: 'check'}, absRt60, 'From RT60'),
        h('label', {class: 'check'}, absMaterials, 'Per-wall materials'),
      ),
      range('simRT60', 'RT60', 0, 2.5, 0.1, 1.5, fixed(1, ' s')),
      materialGrid,
      hallButton,
    ),
    section('Drone',
      range('simSrcAz', 'Azimuth', 0, 355, 5, 60, fixed(0, '°')),
      range('simSrcEl', 'Elevation', -90, 90, 5, 30, fixed(0, '°')),
      range('simSrcDist', 'Distance', 2, 20, 0.5, 6, fixed(1, ' m')),
      range('simDroneSPL', 'Level at 1 m', 60, 95, 1, 78, fixed(0, ' dB SPL')),
      check('simMoving', 'Moving source'),
      group(
        select('simTrajType', 'Trajectory', [['straight', 'Straight line'], ['arc', 'Circular arc']], 'straight'),
        range('simSpeed', 'Speed', 0, 30, 0.5, 5, fixed(1, ' m/s')),
        range('simHeading', 'Heading', 0, 355, 5, 0, fixed(0, '°')),
        range('simChunks', 'Trajectory chunks', 4, 16, 1, 8),
      ),
    ),
    section('Interference',
      check('simDiffuse', 'Crowd and PA speakers', true),
      group(
        range('simCrowd', 'Crowd talkers', 0, 100, 1, 30),
        range('simCrowdSPL', 'Crowd level at 1 m', 55, 80, 1, 67, fixed(0, ' dB SPL')),
        range('simPA', 'PA speakers', 0, 12, 1, 8),
        range('simPASPL', 'PA level at 1 m', 70, 100, 1, 80, fixed(0, ' dB SPL')),
        select('simCrowdModel', 'Crowd model', [['point_source', 'Point sources'], ['plane_wave', 'Plane-wave diffuse field']], 'point_source'),
        range('simNPlaneWaves', 'Plane waves', 8, 128, 8, 64),
      ),
      range('simMicFloor', 'Microphone noise floor', 20, 50, 1, 30, fixed(0, ' dB SPL')),
    ),
    h('details', {class: 'disclosure sim-more'},
      h('summary', {}, 'More settings'),
      h('div', {class: 'sim-more__body'},
        section('Direction band',
          range('simFmin', 'Lowest frequency', 50, 4000, 50, 200, fixed(0, ' Hz')),
          range('simFmax', 'Highest frequency', 200, 7000, 50, 2000, fixed(0, ' Hz')),
          check('simHarmonicComb', 'Weight the drone’s harmonics'),
          group(range('simFundamental', 'Drone fundamental', 80, 400, 5, 200, fixed(0, ' Hz'))),
        ),
        section('Hardware impairments',
          check('simMismatch', 'Microphone gain and phase mismatch'),
          check('simCrosstalk', 'Channel crosstalk'),
          group(
            range('simCrosstalkDb', 'Coupling', -60, -20, 1, -40, fixed(0, ' dB')),
            select('simCrosstalkModel', 'Crosstalk model', [['simple', 'Flat'], ['fir_capacitive', 'Capacitive (FIR)']], 'simple'),
            range('simCrosstalkCorner', 'Corner frequency', 20, 4000, 20, 500, fixed(0, ' Hz')),
          ),
          check('simQuant', 'Codec quantisation'),
          group(select('simBitDepth', 'Bit depth', [['24', '24 bit'], ['16', '16 bit'], ['12', '12 bit'], ['8', '8 bit']], '16')),
          h('button', {type: 'button', class: 'btn btn--sm', onClick: () => applyHardwarePreset()}, 'Apply the hardware-realistic set'),
        ),
        section('DSP',
          range('simFracMax', 'Fractional delay maximum', 4, 64, 1, 8, fixed(0, ' samples')),
          range('simFracTaps', 'Fractional delay taps', 8, 32, 1, 16),
          range('simTargetFs', 'DSP sample rate', 16000, 96000, 8000, 48000, fixed(0, ' Hz')),
          range('simInt', 'Integration time', 50, 1000, 50, 1000, fixed(0, ' ms')),
        ),
        section('Atmosphere',
          range('simTemp', 'Temperature', 0, 40, 1, 20, fixed(0, ' °C')),
          range('simHumidity', 'Humidity', 10, 90, 5, 50, fixed(0, ' %')),
          range('simTempGrad', 'Temperature gradient', -5, 5, 0.1, 0, fixed(1, ' °C/m')),
        ),
        section('ML input preview (MAX78000)',
          check('simMlPreview', 'Render the ML input'),
          group(
            select('simMlBitDepth', 'Audio bit depth', [['16', '16 bit'], ['8', '8 bit']], '8'),
            select('simMlFeatBitDepth', 'Feature bit depth', [['16', '16 bit'], ['8', '8 bit']], '8'),
            range('simMlNMels', 'Mel bands', 32, 96, 16, 64),
          ),
        ),
        section('Run',
          check('simRandSeed', 'New random seed every run', true),
          range('simSeed', 'Seed', 0, 999, 1, 0),
          check('simNormalizeAudio', 'Normalise downloaded audio', true),
          h('p', {class: 'note'}, 'Turn normalisation off to keep absolute levels across runs, for example to compare distances.'),
        ),
      ),
    ),
  );

  const show = (id, visible) => {
    const entry = byId.get(id);
    const target = entry?.el?.closest?.('.field') || entry?.el;
    if (target) target.hidden = !visible;
  };
  const groupOf = (id) => byId.get(id)?.el?.closest('.sub-group') || byId.get(id)?.el?.closest('.field')?.closest('.sub-group');

  function sync() {
    const geo = get('simGeo').value;
    const builtIn = !['CUSTOM', 'HEIMDALL'].includes(geo);
    editor.hidden = geo !== 'CUSTOM';
    show('simMicCount', builtIn);
    show('simRadius', builtIn);
    show('simSep', geo === 'CYLINDER');
    byId.get('simRadius').el.querySelector('label').textContent = geo === 'ULA' || geo === 'CROSS' ? 'Half-length' : 'Radius';
    const method = get('simBeamMethod').value;
    scanAnglesField.hidden = method !== 'steered_das';
    show('simBeamK', method === 'beam_bank_das');
    const materials = absMaterials.checked;
    byId.get('simRT60').el.classList.toggle('is-disabled', materials);
    get('simRT60').disabled = materials;
    materialGrid.hidden = !materials;
    hallButton.hidden = !materials;
    groupOf('simTrajType').hidden = !get('simMoving').checked;
    show('simHeading', get('simTrajType').value !== 'arc');
    groupOf('simCrowd').hidden = !get('simDiffuse').checked;
    show('simNPlaneWaves', get('simCrowdModel').value === 'plane_wave');
    groupOf('simFundamental').hidden = !get('simHarmonicComb').checked;
    groupOf('simCrosstalkDb').hidden = !get('simCrosstalk').checked;
    show('simCrosstalkCorner', get('simCrosstalkModel').value === 'fir_capacitive');
    groupOf('simBitDepth').hidden = !get('simQuant').checked;
    groupOf('simMlBitDepth').hidden = !get('simMlPreview').checked;
    show('simSeed', !get('simRandSeed').checked);
  }

  function setValue(id, value, fire = false) {
    const entry = byId.get(id);
    if (!entry) return false;
    if (entry.input.type === 'checkbox') entry.input.checked = Boolean(value);
    else entry.input.value = value;
    entry.update?.();
    if (fire) changed();
    return true;
  }

  function populateMaterials(choices = MATERIAL_FALLBACK, defaults = EXHIBITION_HALL) {
    for (const [wall, id] of WALLS) {
      const input = get(id);
      const previous = input.value;
      input.replaceChildren(...options(choices.map((name) => [name, name.replace(/_/g, ' ')]), previous || defaults[wall]));
      input.value = choices.includes(previous) ? previous : defaults[wall] || choices[0];
    }
  }

  function applyExhibitionHall() {
    for (const [wall, id] of WALLS) {
      const input = get(id);
      if ([...input.options].some((o) => o.value === EXHIBITION_HALL[wall])) input.value = EXHIBITION_HALL[wall];
    }
    // The known baseline: warm hall air, default models, the original 200–2000 Hz band.
    for (const [id, value] of [['simTemp', 22], ['simHumidity', 55], ['simTempGrad', 1.5], ['simFmin', 200], ['simFmax', 2000], ['simFundamental', 200]]) setValue(id, value);
    get('simCrowdModel').value = 'point_source';
    get('simCrosstalkModel').value = 'simple';
    get('simMlPreview').checked = false;
    get('simHarmonicComb').checked = false;
    get('simNormalizeAudio').checked = true;
    sync();
    changed();
  }

  function applyHardwarePreset() {
    get('simMismatch').checked = true;
    get('simCrosstalk').checked = true;
    setValue('simCrosstalkDb', -40);
    get('simQuant').checked = true;
    get('simBitDepth').value = '16';
    sync();
    changed();
  }

  function scanAnglesDeg() {
    const values = String(scanAngles.value || '').split(',').map((v) => v.trim()).filter(Boolean)
      .map(Number).filter(Number.isFinite).map((v) => ((v % 360) + 360) % 360);
    const unique = [...new Set(values.map((v) => Number(v.toFixed(6))))];
    return unique.length ? unique : [0, 45, 90, 135, 180, 225, 270, 315];
  }

  /** Request body for POST /simulate (same fields as before). */
  function params() {
    const n = (id) => Number(get(id).value);
    const geo = get('simGeo').value;
    const method = get('simBeamMethod').value;
    const k = n('simBeamK');
    let fmin = n('simFmin');
    let fmax = n('simFmax');
    if (fmax <= fmin) { fmax = fmin + 50; setValue('simFmax', fmax); }
    return {
      geometry: geo === 'HEIMDALL' ? 'CUSTOM' : geo,
      custom_mic_positions: geo === 'CUSTOM' ? customArray.positions.map((p) => [p[0], p[1], p[2]])
        : geo === 'HEIMDALL' ? heimdallToolPositions() : null,
      beam_method: method,
      beam_bank_angles_deg: method === 'beam_bank_das' ? Array.from({length: k}, (_, i) => i * (360 / k)) : scanAnglesDeg(),
      frac_delay_max_samples: n('simFracMax'),
      frac_delay_taps: n('simFracTaps'),
      target_fs_hz: n('simTargetFs'),
      mic_count: n('simMicCount'),
      radius: n('simRadius'),
      ring_separation: n('simSep'),
      room_length: n('simRoomL'),
      room_width: n('simRoomW'),
      room_height: n('simRoomH'),
      rt60: n('simRT60'),
      source_az_deg: n('simSrcAz'),
      source_el_deg: n('simSrcEl'),
      source_distance: n('simSrcDist'),
      drone_spl_db: n('simDroneSPL'),
      crowd_spl_db: n('simCrowdSPL'),
      pa_spl_db: n('simPASPL'),
      mic_noise_floor_db: n('simMicFloor'),
      seed: get('simRandSeed').checked ? -1 : n('simSeed'),
      diffuse: get('simDiffuse').checked,
      crowd_count: n('simCrowd'),
      pa_count: n('simPA'),
      mic_mismatch: get('simMismatch').checked,
      crosstalk: get('simCrosstalk').checked,
      crosstalk_db: n('simCrosstalkDb'),
      quantization: get('simQuant').checked,
      bit_depth: Number(get('simBitDepth').value),
      absorption_mode: absMaterials.checked ? 'materials' : 'rt60',
      floor_material: get('matFloor').value,
      ceiling_material: get('matCeiling').value,
      east_material: get('matEast').value,
      west_material: get('matWest').value,
      south_material: get('matSouth').value,
      north_material: get('matNorth').value,
      integration_ms: n('simInt'),
      temperature_c: n('simTemp'),
      humidity_pct: n('simHumidity'),
      temp_gradient_c_per_m: n('simTempGrad'),
      moving_source: get('simMoving').checked,
      trajectory_type: get('simTrajType').value,
      speed_mps: n('simSpeed'),
      heading_deg: n('simHeading'),
      n_trajectory_chunks: n('simChunks'),
      crowd_model: get('simCrowdModel').value,
      n_plane_waves: n('simNPlaneWaves'),
      crosstalk_model: get('simCrosstalkModel').value,
      crosstalk_corner_hz: n('simCrosstalkCorner'),
      ml_preview: get('simMlPreview').checked,
      ml_bit_depth: Number(get('simMlBitDepth').value),
      ml_feature_bit_depth: Number(get('simMlFeatBitDepth').value),
      ml_n_mels: n('simMlNMels'),
      fmin_hz: fmin,
      fmax_hz: fmax,
      harmonic_comb: get('simHarmonicComb').checked,
      drone_fundamental_hz: n('simFundamental'),
      normalize_audio: get('simNormalizeAudio').checked,
    };
  }

  function captureState() {
    const state = {v: 1, sliders: {}, selects: {}, checks: {}, absMode: '', customMicPositions: []};
    for (const id of PRESET_SLIDER_IDS) state.sliders[id] = get(id).value;
    for (const id of PRESET_SELECT_IDS) state.selects[id] = get(id).value;
    for (const id of PRESET_CHECK_IDS) state.checks[id] = get(id).checked;
    state.absMode = absMaterials.checked ? 'materials' : 'rt60';
    state.customMicPositions = customArray.positions.map((p) => [p[0], p[1], p[2]]);
    state.simScanAngles = scanAngles.value;
    return state;
  }

  /** Accepts presets from this and the old app (old files used "live*" ids). */
  function applyState(state) {
    if (!state) return;
    const legacy = (bag) => {
      const out = {...(bag || {})};
      for (const [id, value] of Object.entries(bag || {})) {
        if (id.startsWith('live') && out['sim' + id.slice(4)] === undefined) out['sim' + id.slice(4)] = value;
      }
      return out;
    };
    for (const [id, value] of Object.entries(legacy(state.sliders))) setValue(id, value);
    for (const [id, value] of Object.entries(legacy(state.selects))) {
      const input = get(id);
      if (input) input.value = id === 'simGeo' && value === 'MICCANVAS' ? 'CUSTOM' : value;
    }
    for (const [id, value] of Object.entries(state.checks || {})) setValue(id, value);
    if (typeof state.simScanAngles === 'string') scanAngles.value = state.simScanAngles;
    else if (typeof state.liveScanAngles === 'string') scanAngles.value = state.liveScanAngles;
    if (state.absMode === 'materials') absMaterials.checked = true; else absRt60.checked = true;
    customArray.set(Array.isArray(state.customMicPositions) ? state.customMicPositions : []);
    sync();
    changed();
  }

  populateMaterials();
  sync();
  return {el, params, captureState, applyState, populateMaterials, get, sync};
}

/**
 * dsp.js — Study state: the register map (parsed by the backend from the
 * SigmaStudio export), the device's current words, staged changes, the
 * session history and saved snapshots.
 *
 * Nothing reaches the device until apply(). Staged words equal to the device
 * value are dropped, so the pending list always shows real changes.
 */

import { apiJson } from '../shared/api.js';

const SNAP_KEY = 'heimdall.study.snapshots';
export const ONE = 1 << 24;

export function toFloat(word, type) {
  return type === 'int32' ? word : word / ONE;
}

export function toWord(value, type) {
  const v = type === 'int32' ? Math.round(value) : Math.round(value * ONE);
  return Math.max(-(2 ** 31), Math.min(2 ** 31 - 1, v));
}

export function hexWord(word) {
  return '0x' + ((word >>> 0).toString(16).toUpperCase().padStart(8, '0'));
}

class DspStore extends EventTarget {
  constructor() {
    super();
    this.registry = null;
    this.error = null;
    this.index = new Map();
    this.byGroup = new Map();
    this.status = null;
    this.memory = new Map();
    this.pending = new Map();        // address → {word, label}
    this.setting = null;             // {key, value, label} staged firmware setting
    this.history = [];
    this.snapshots = this.loadSnapshots();
    this.plan = null;
    this.planTimer = null;
    this.loading = null;
  }

  changed(what = 'change') { this.dispatchEvent(new Event(what)); }

  async load() {
    if (this.registry) return this.registry;
    if (!this.loading) {
      this.loading = apiJson('/dsp/registry').then((registry) => {
        this.registry = registry;
        this.error = null;
        for (const param of registry.parameters) {
          for (let offset = 0; offset < param.words; offset++) this.index.set(param.address + offset, {param, offset});
          if (!this.byGroup.has(param.group)) this.byGroup.set(param.group, []);
          this.byGroup.get(param.group).push(param);
        }
        this.changed();
        return registry;
      }).catch((error) => {
        this.error = error.message;
        this.loading = null;
        this.changed();
        throw error;
      });
    }
    return this.loading;
  }

  async refresh() {
    try {
      this.status = await apiJson('/dsp/status');
    } catch (error) {
      this.status = {supported: false, reason: 'The backend is not reachable.', offline: true};
    }
    if (this.status.supported) {
      try {
        const {words} = await apiJson('/dsp/memory');
        this.memory = new Map(Object.entries(words).map(([address, word]) => [Number(address), word]));
      } catch (_) { /* keep the previous values */ }
    } else {
      this.memory = new Map();
    }
    this.dropUnchanged();
    this.changed();
    return this.status;
  }

  /** Device value of a word: the device memory if known, else the export default. */
  deviceWord(address) {
    if (this.memory.has(address)) return this.memory.get(address);
    const entry = this.index.get(address);
    return entry ? entry.param.default_words[entry.offset] : 0;
  }

  defaultWord(address) {
    const entry = this.index.get(address);
    return entry ? entry.param.default_words[entry.offset] : 0;
  }

  proposedWord(address) {
    return this.pending.has(address) ? this.pending.get(address).word : this.deviceWord(address);
  }

  param(group, mic, name) {
    return (this.byGroup.get(group) || []).find((p) => (mic === undefined || p.mic === mic) && (!name || p.name === name));
  }

  /** writes: [{address, word}]. Replaces staged words for these addresses. */
  stage(writes, label) {
    for (const {address, word} of writes) {
      if (word === this.deviceWord(address)) this.pending.delete(address);
      else this.pending.set(address, {word, label});
    }
    this.pendingChanged();
  }

  unstage(predicate) {
    for (const [address, entry] of [...this.pending]) if (predicate(address, entry)) this.pending.delete(address);
    this.pendingChanged();
  }

  stageSetting(key, value, label) {
    const current = this.status?.settings?.[key];
    this.setting = Number(value) === Number(current) ? null : {key, value: Number(value), label};
    this.pendingChanged();
  }

  discard() {
    this.pending.clear();
    this.setting = null;
    this.pendingChanged();
  }

  dropUnchanged() {
    for (const [address, entry] of [...this.pending]) if (entry.word === this.deviceWord(address)) this.pending.delete(address);
  }

  pendingChanged() {
    this.changed();
    clearTimeout(this.planTimer);
    this.planTimer = setTimeout(() => this.updatePlan(), 150);
  }

  async updatePlan() {
    if (!this.pending.size) {
      this.plan = {command_count: 0, word_count: 0, estimate_ms: 0};
      this.changed('plan');
      return;
    }
    try {
      this.plan = await apiJson('/dsp/plan', {method: 'POST', body: {writes: this.pendingWrites()}});
    } catch (error) {
      this.plan = {error: error.message};
    }
    this.changed('plan');
  }

  pendingWrites() {
    return [...this.pending].map(([address, entry]) => ({address, word: entry.word}));
  }

  /** Groups of pending words by label, for the pending list. */
  pendingGroups() {
    const groups = new Map();
    for (const [address, entry] of this.pending) {
      if (!groups.has(entry.label)) groups.set(entry.label, []);
      groups.get(entry.label).push(address);
    }
    return [...groups].map(([label, addresses]) => ({label, addresses: addresses.sort((a, b) => a - b)}));
  }

  modifiedAddresses() {
    const out = [];
    for (const [address, word] of this.memory) if (word !== this.defaultWord(address)) out.push(address);
    return out;
  }

  async apply(label) {
    const writes = this.pendingWrites();
    const previous = writes.map(({address}) => ({address, word: this.deviceWord(address)}));
    const setting = this.setting;
    const previousSetting = setting ? {key: setting.key, value: this.status?.settings?.[setting.key]} : null;
    let result = null;
    if (writes.length) result = await apiJson('/dsp/write', {method: 'POST', body: {writes}});
    if (setting) await apiJson('/dsp/set', {method: 'POST', body: {key: setting.key, value: setting.value}});
    this.history.unshift({
      time: new Date(), label, words: writes.length, commands: result?.command_count ?? 0,
      previous, previousSetting, reverted: false,
    });
    this.pending.clear();
    this.setting = null;
    await this.refresh();
    this.pendingChanged();
    return result;
  }

  async revert(entry) {
    if (entry.previous.length) await apiJson('/dsp/write', {method: 'POST', body: {writes: entry.previous}});
    if (entry.previousSetting?.value !== undefined) {
      await apiJson('/dsp/set', {method: 'POST', body: entry.previousSetting});
    }
    entry.reverted = true;
    await this.refresh();
  }

  /** Stage every writable word back to the export default. */
  stageFlashed() {
    const writes = [];
    for (const [address, {param}] of this.index) {
      if (param.writable && this.deviceWord(address) !== this.defaultWord(address)) writes.push({address, word: this.defaultWord(address)});
    }
    this.stage(writes, 'Return to the flashed values');
  }

  loadSnapshots() {
    try { return JSON.parse(localStorage.getItem(SNAP_KEY)) || []; } catch (_) { return []; }
  }

  saveSnapshots() {
    try { localStorage.setItem(SNAP_KEY, JSON.stringify(this.snapshots)); } catch (_) { /* storage unavailable */ }
  }

  saveSnapshot(name) {
    const words = {};
    for (const [address, {param}] of this.index) {
      if (param.writable && this.deviceWord(address) !== this.defaultWord(address)) words[address] = this.deviceWord(address);
    }
    this.snapshots.push({name, time: new Date().toISOString(), export: this.registry?.export?.sha256, words,
      settings: {...(this.status?.settings || {})}});
    this.saveSnapshots();
    this.changed();
  }

  /** Stage the difference between the device and a snapshot (flashed values elsewhere). */
  loadSnapshot(snapshot) {
    const writes = [];
    for (const [address, {param}] of this.index) {
      if (!param.writable) continue;
      const target = snapshot.words[address] ?? this.defaultWord(address);
      if (target !== this.deviceWord(address)) writes.push({address, word: target});
    }
    this.stage(writes, `Snapshot: ${snapshot.name}`);
    if (snapshot.settings?.settle_us) this.stageSetting('settle_us', snapshot.settings.settle_us, `Snapshot: ${snapshot.name}`);
  }

  deleteSnapshot(snapshot) {
    this.snapshots = this.snapshots.filter((s) => s !== snapshot);
    this.saveSnapshots();
    this.changed();
  }
}

export const dsp = new DspStore();

# Study mode: proposed firmware commands (protocol v3)

Status: **proposal**. The GUI and the emulator implement this. The MAX78002
firmware does not yet; until it does, the Study screen treats serial hardware
as read-only and says so.

The goal is to change DSP parameters at runtime (gain tapers, FIR taps, detector
settings, mutes) to characterise the array without reflashing. This document is
the contract the GUI expects. The firmware and DSP owners decide whether and
how to implement it.

## Principles

1. **Additive.** Existing commands (`F C G X S,<n> M I R,1 R,0 D`) and replies do
   not change. A firmware without v3 simply answers `ERR,UNKNOWN` (or nothing)
   to the new commands.
2. **Only while idle.** Parameter writes are accepted only when the scan engine
   is `IDLE` — not scanning, tracking or holding a beam for monitoring. The GUI
   checks this, but the firmware must enforce it too (`ERR,BUSY`), because the
   GUI's check can race a command.
3. **No address assumptions.** SigmaStudio places parameters at compile time.
   Blocks of one type are not guaranteed to be contiguous or to increase by one.
   A command only ever covers consecutive addresses, and the host decides the
   batching from the export (see "Write planning").
4. **The export is the map.** The host parses the SigmaStudio export JSON
   (`ModuleList` / `ModuleParameter` / `AddressMap`) into a register map
   (`heimdall-dsp-registry-v1`, see `dsp_registry.py`). The firmware should
   carry an allow-list generated from the same export.

## Commands

Lines are ASCII, end with `\r\n`, and must fit in 128 bytes (the current
command buffer is 16 bytes and needs to grow). `*CS` is the XOR of all bytes
between the start of the line and `*`, as two upper-case hex digits (NMEA style).

| Command | Meaning | Replies |
| --- | --- | --- |
| `PW,<addr>,<w1>[,<w2>…<w5>]*CS` | Safeload write of 1–5 words to consecutive addresses starting at `<addr>` (decimal). Words are 8 hex digits, two's complement (8.24 or int32 as the parameter defines). | `PW_OK,<addr>,<n>` or `ERR,PW,<code>` |
| `PR,<addr>,<n>` | Read `n` (1–16) words starting at `<addr>`. | `PV,<addr>,<w1>,…,<wn>` or `ERR,PR,<code>` |
| `SET,<key>,<value>` | Change a firmware setting at runtime. First key: `settle_us` (50–20000), the wait after steering before the level is read. | `SET_OK,<key>,<value>` or `ERR,SET,<code>` |
| `I` (extended) | Existing info reply, plus one line `MAP,<sha256-16>` with the first 16 hex digits of the register map the firmware was built for, and `SETTINGS,settle_us=<v>`. | as today, plus the two lines |

Error codes: `BUSY` (not idle), `ADDR` (address not in the allow-list or
read-only), `LEN` (bad word count), `CS` (checksum), `FMT` (malformed).

### Safeload

`PW` maps directly onto the ADAU1467 safeload mechanism the firmware already
uses for the delays: data words to `0x6000…0x6004`, target address to `0x6005`,
word count to `0x6006`, then wait at least one frame (≥ 20.83 µs at 48 kHz)
before the next safeload. Because safeload writes up to five consecutive
words, one `PW` is one safeload.

### Allow-list

The firmware should reject writes outside the parameter words that the export
marks as tunable, and in particular:

- delay words (`Delay_DSPxx.DelayPercentage`) while the scan engine owns them —
  a host-steered fine sweep needs them writable **only while idle**;
- `MaxDelay` words (fixed by the program);
- safeload registers, detector state words and program memory.

The current GUI allow-list is in `dsp_registry.py` (`_policy`).

## Write planning (host side)

The host merges the requested words, sorts by address, and cuts them into runs
of consecutive addresses of at most five words. Examples from the Golden Image:

| Change | Words | Commands |
| --- | --- | --- |
| Gain taper on all 44 microphones (gains happen to be contiguous, 1134–1177) | 44 | 9 |
| One FIR stage on all 44 microphones (11 contiguous taps per filter, filters scattered) | 484 | 132 |
| All 44 fractional delays (addresses step by −24) | 44 | 44 |
| Detector time constant | 1 | 1 |

If a later compile interleaves blocks differently, the same code simply produces
more single-word commands; nothing in the GUI hard-codes address ranges.

## Consistency

- After every `PW` the host reads the words back with `PR` and shows the device
  values, not the requested ones.
- On connect, the host compares the `MAP` hash from `I` with the export it
  parsed and refuses writes if they differ.
- Values live in DSP RAM only. A DSP restart or program reload restores the
  flashed values, so the host re-reads after reconnecting instead of trusting its
  own history.
- A boot-time sanity check could read back each `MaxDelay` word and expect 68.

## What the GUI does today

- Parses the export, shows every block, and plans writes as above.
- Writes and reads through the emulator's parameter memory
  (`/dsp/write`, `/dsp/read`, `/dsp/memory`, `/dsp/set`), which answers as the
  firmware would. Levels in the emulator do not react to the parameters yet.
- For serial transports, `/dsp/status` reports "not supported" until the
  firmware implements these commands.

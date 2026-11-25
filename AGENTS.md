# Instructions for AI agents (acoustic-sim)

acoustic-sim is the host software of Heimdall, an acoustic drone detector: the FastAPI backend
(`sim_server.py`), the browser GUI (`app/`), the firmware protocol parser and emulators, the bench
scripts (`scripts/`) and the classifier tools (`heimdall_ml/`). It is a public repository and a
submodule of the project repository. When `../AGENTS.md` exists, read it first: it has the project
rules, the owners and the firmware commands.

## Commands (Windows, PowerShell 5.1)

```powershell
.\.venv\Scripts\python.exe -m pytest -q                      # all Python tests
cd app; node --test --test-reporter=tap "tests/*.test.js"    # GUI tests
```

Setup and starting the backend and the GUI: [README.md](README.md), sections "Setup" and "Run".

## Rules

- **Public repository.** Never add recordings, exported windows or arrays, trained models, internal
  documents or bench data. `ml_out/` and `ml_cnn/` are ignored for this reason.
- **The firmware is the reference for the protocol.** `heimdall_protocol.py` is strict: an unknown or
  malformed record raises `ProtocolError`. A new or changed record needs the firmware change, the
  parser, the emulator (`heimdall_transport.py`) and tests.
- **Settings.** `FIRMWARE_SETTINGS` (`heimdall_protocol.py`) and `DspParameterService.SETTINGS`
  (`dsp_registry.py`) must equal the firmware table. A test compares them with the firmware source
  when the project repository is present.
- **Firmware copies.** `heimdall_ml/tonal.py` (decimator, tonal detector) and
  `heimdall_ml/model_input.py` (the classifier input) are bit-exact copies of firmware code. Their
  tests use values from the firmware host tests. Do not change them without the firmware side.
- **Git.** Terse lowercase commit messages, no AI signature. Push only when a person asks. Never
  touch `main`.
- **PowerShell.** Chain commands with `;`. Native programs lose embedded double quotes: put
  `python -c` code in a file. The npm test output is garbled: use the `node --test` command above.

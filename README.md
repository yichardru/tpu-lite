# TPU-Lite

A multi-core int8 matrix-multiply accelerator, inspired by Google's TPU, designed in Verilog and taken from RTL to GDS on the open-source IHP SG13G2 130nm PDK.

The chip has four TensorCore-style compute cores, each built around a 16×16 weight-stationary systolic array, connected to banked shared SRAM through a crossbar.

> **Status:** RTL. See [Roadmap](#roadmap).

---

## Contents

- [Specifications](#specifications)
- [Architecture](#architecture)
- [Instruction Set](#instruction-set)
- [Toolchain](#toolchain)
- [Repository Layout](#repository-layout)
- [Design Flow](#design-flow)
- [Verification](#verification)
- [Getting Started](#getting-started)
- [Roadmap](#roadmap)
- [License](#license)

---

## Specifications

| Parameter | Value |
|---|---|
| Cores | 4 (2×2) |
| Array per core | 16×16 weight-stationary systolic |
| Total MACs | 1,024 |
| Input / weight type | int8 |
| Accumulation | int32 |
| Output type | int8 (after ReLU + requantization) |
| Process | IHP SG13G2, 130nm |
| Target clock | 100–200 MHz (to be confirmed after first synthesis) |
| Peak throughput | ~0.2–0.4 TOPS (1,024 MACs × 2 ops × f_clk) |

Array size and core count are Verilog parameters, so the design can be scaled down (e.g., 8×8 tiles) if area or runtime requires it.

---

## Architecture

```
                 ┌──────────────────────────── Host Interface ───────────────────────────┐
                 │   Control/status registers         Memory load/readback port          │
                 └───────────────┬───────────────────────────────┬────────────────────────┘
                                 │                               │
   ┌─────────┐ ┌─────────┐ ┌─────┴───┐ ┌─────────┐               │
   │ Core 0  │ │ Core 1  │ │ Core 2  │ │ Core 3  │               │
   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘               │
        └───────────┴─────┬─────┴───────────┴────────────────────┘
                          │
                ┌─────────┴──────────┐
                │      Crossbar      │   5 requesters × 8 banks, round-robin
                └─────────┬──────────┘
          ┌──────┬──────┬─┴────┬──────┬──────┬──────┬──────┐
          │ B0   │ B1   │ B2   │ B3   │ B4   │ B5   │ B6   │ B7     Global SRAM banks
          └──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

### Compute core

Each core contains:

- **Systolic array** — 16×16 weight-stationary processing elements (PEs). Each PE holds a weight, multiplies it by the activation passing through, and adds to the partial sum flowing down the column.
- **Weight buffer** — SRAM for weight tiles, plus a double-buffered weight register set so the next tile loads while the current one computes.
- **Activation buffer** — SRAM feeding one 16-byte activation row per cycle.
- **Accumulators** — int32 storage below the array for summing partial results across K-tiles. Implemented as ping-pong banks (or dual-port SRAM, if available) since they read and write every cycle.
- **Activation unit** — ReLU, then requantization (scale, shift, round, saturate to int8).
- **Controller** — instruction queue and sequencer.
- **Local DMA** — moves data between global SRAM and the core's buffers.

Buffer sizes are set to match the SRAM macros available in the IHP PDK (TBD in Phase 0).

### Shared memory and interconnect

- **Global SRAM** — 8 banks, 128 bits wide, address-interleaved across banks.
- **Crossbar** — connects the 4 cores plus the host port to the 8 banks with round-robin arbitration, 128 bits per port.
- **Synchronization** — a global barrier register and per-core done flags.

### Parallelism model

The initial version splits work by **output tile**: each core computes its own block of the output matrix and only communicates through shared memory. Direct core-to-core links (ring or 2D mesh) are a possible extension.

### Host interface

- Control/status registers: per-core instruction base address, start, status, interrupt.
- A wide port for loading inputs into and reading results from global SRAM.
- In simulation, the testbench acts as the host.

---

## Instruction Set

64-bit fixed-format instructions, executed per core.

| Instruction | Description |
|---|---|
| `LOAD_W` | DMA a weight tile from global SRAM into the weight buffer |
| `LOAD_A` | DMA activations from global SRAM into the activation buffer |
| `MATMUL` | Stream rows through the array; flag selects overwrite or accumulate |
| `ACT` | Apply ReLU and requantize accumulators to int8 (scale, shift) |
| `STORE` | DMA results from the core back to global SRAM |
| `SYNC` | Barrier across all cores |
| `HALT` | Stop and raise the done flag |

Exact bit encodings are defined in [`docs/spec.md`](docs/spec.md).

---

## Toolchain

| Step | Tool |
|---|---|
| Performance model, golden model | Python, NumPy |
| RTL simulation | Cadence Xcelium |
| Testbenches | cocotb |
| Lint | Xcelium lint / Verilator `--lint-only` |
| Synthesis | Cadence Genus |
| Equivalence checking | Cadence Conformal |
| Place and route | Cadence Innovus |
| Static timing analysis | Cadence Tempus |
| Power / IR drop | Cadence Voltus |
| DRC / LVS | KLayout (IHP rule decks) |
| Netlist conversion for LVS | Calibre `v2lvs` |
| Layout viewing | KLayout |

### PDK

[IHP SG13G2 Open PDK](https://github.com/IHP-GmbH/IHP-Open-PDK) (Apache 2.0), which provides:

- Standard-cell library (Liberty, LEF, GDS, CDL, Verilog)
- SRAM macros
- Tech LEF
- KLayout DRC and LVS rule decks

The PDK is not vendored in this repository. Point `PDK_ROOT` at your local clone (see [Getting Started](#getting-started)).

---

## Repository Layout

```
tpu-lite/
├── README.md
├── docs/
│   ├── spec.md              # Interfaces, ISA encodings, memory map, crossbar protocol
│   └── diagrams/
├── model/
│   ├── perf_model.py        # Utilization and bandwidth estimates
│   └── golden.py            # Bit-exact int8 matmul + requantization reference
├── rtl/
│   ├── pe/                  # Processing element
│   ├── array/               # 16×16 systolic array
│   ├── core/                # Buffers, accumulators, activation unit, controller, DMA
│   ├── xbar/                # Crossbar and arbitration
│   ├── mem/                 # Behavioral SRAM models matching IHP macro ports
│   └── top/                 # Top level and host interface
├── tb/
│   ├── pe/
│   ├── array/
│   ├── core/
│   └── top/                 # cocotb testbenches, one directory per level
├── syn/
│   ├── constraints/         # SDC files
│   └── scripts/             # Genus scripts
├── pnr/
│   ├── core/                # Innovus scripts for the core macro
│   └── top/                 # Innovus scripts for the top level
├── signoff/
│   ├── sta/                 # Tempus scripts
│   ├── power/               # Voltus scripts
│   └── pv/                  # KLayout DRC/LVS, layer map, fill
└── results/                 # Reports, screenshots (generated outputs are git-ignored)
```

---

## Design Flow

Each phase ends with a check that must pass before moving on.

### Phase 0 — Architecture and spec
Performance model, spec document, golden model, and confirmation of available IHP SRAM sizes and cells.
**Check:** spec frozen enough for RTL and tests to proceed in parallel.

### Phase 1 — RTL
Built bottom-up: PE → array → core → crossbar and global SRAM → top level. Behavioral SRAM models use the same ports and timing as the IHP macros.
**Check:** lint clean.

### Phase 2 — Functional verification
Unit tests at every level, then random tests against the golden model. Corner cases include int8 extremes, accumulator overflow, back-to-back instructions, and bank contention.
**Check:** regression passes, coverage goals met. Rerun after every RTL change.

### Phase 3 — Synthesis (Genus)
Synthesize the core alone, then the top level with the core as a black box. Run equivalence checking and gate-level simulation.
**Check:** timing met with ~10% margin, equivalence passes, gate-level sim passes.

### Phase 4 — Core physical design (Innovus)
Import → floorplan (SRAM macro placement, halos, pins) → power plan → place → CTS → route → fillers → verify. Export the core as a hard macro (LEF, .lib, GDS, netlist, SPEF).
**Check:** no setup/hold violations, clean geometry and connectivity.

### Phase 5 — Top-level physical design (Innovus)
Place 4 core macros, global SRAM, crossbar, and host interface. Top-level power grid, CTS (watching cross-core skew), route.
**Check:** same as Phase 4.

### Phase 6 — Signoff
STA at all corners (Tempus), power and IR drop (Voltus), GDS merge, metal fill, DRC and LVS in KLayout.
**Check:** DRC/LVS clean, timing met at all corners.

### Phase 7 — Results
Measured utilization and TOPS vs. the Phase 0 model, area breakdown, power, max frequency, TOPS/W and TOPS/mm², layout screenshots.

### Iteration
- Timing failures → add pipeline stages in RTL (back to Phase 1).
- Routing congestion → adjust floorplan, macro placement, or crossbar width.
- Area too large → reduce tile size or core count.

> **Note:** Innovus RC extraction uses LEF-based RC values, since the IHP open PDK does not provide a QRC tech file. Post-layout timing should be read with that in mind.

---

## Verification

- **Golden model:** `model/golden.py` is the single source of truth for expected results, matching the hardware's rounding and saturation bit-for-bit.
- **Testbenches:** cocotb, organized by hierarchy level under `tb/`.
- **Levels:** PE → array → single core → full 4-core system.
- **Regression:** all levels run on every RTL change.
- **Gate-level:** a subset of tests rerun on synthesized and post-layout netlists.

---

## Getting Started

```bash
# 1. Clone this repo and the PDK
git clone <this-repo-url> tpu-lite
git clone https://github.com/IHP-GmbH/IHP-Open-PDK.git
export PDK_ROOT=$PWD/IHP-Open-PDK

# 2. Python environment for models and testbenches
cd tpu-lite
python3 -m venv .venv && source .venv/bin/activate
pip install numpy cocotb

# 3. Run a testbench (example: processing element)
make -C tb/pe SIM=xcelium
```

Synthesis, place-and-route, and signoff commands will be added as those phases come online.

---

## Roadmap

- [ ] **Phase 0** — Performance model, spec, golden model, IHP SRAM/cell survey
- [ ] **Flow smoke test** — Small design through Genus → Innovus → KLayout DRC/LVS on IHP
- [ ] **Phase 1–2** — PE, then 16×16 array, RTL + tests
- [ ] **Milestone** — Single 16×16 array through the full physical flow
- [ ] **Phase 1–2** — Full core (buffers, controller, DMA, activation unit)
- [ ] **Phase 3–4** — Core synthesized and hardened as a macro
- [ ] **Phase 1–2** — Crossbar, global SRAM, 4-core top level
- [ ] **Phase 5–6** — Top-level PnR and signoff
- [ ] **Phase 7** — Results and writeup
- [ ] **Stretch** — Core-to-core ring or mesh interconnect

---

## License

TBD for project source. The IHP SG13G2 Open PDK is licensed separately under Apache 2.0.

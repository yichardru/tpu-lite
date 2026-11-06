# TPU-Lite

A multi-core int8 matrix-multiply accelerator, inspired by Google's TPU, designed in Verilog and taken from RTL to GDS on the open-source IHP SG13G2 130nm PDK.

The chip is a 2×2 mesh of compute tiles. Each tile has a TensorCore-style core built around a 16×16 weight-stationary systolic array, a slice of shared SRAM, and a router on a buffered on-chip network (NoC).

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
| Cores | 4 (2×2 mesh) |
| Array per core | 16×16 weight-stationary systolic |
| Total MACs | 1,024 |
| Input | int8 |
| Accumulation | int32 |
| Output type | int8 (after ReLU + requantization) |
| Process | IHP SG13G2, 130nm |
| Target clock | TBD |
| Peak throughput | ~0.2–0.4 TOPS (1,024 MACs × 2 ops × f_clk) |

---

## Architecture

```
   Host Interface
        │
  ┌─────┴──────────────┐          ┌────────────────────┐
  │ Tile (0,0)         │          │ Tile (1,0)         │
  │  Core   SRAM slice │          │  Core   SRAM slice │
  │     └──┬──┘        │          │     └──┬──┘        │
  │      Router ◄──────┼──────────┼──► Router          │
  └────────▲───────────┘          └────────▲───────────┘
           │                               │
  ┌────────▼───────────┐          ┌────────▼───────────┐
  │      Router ◄──────┼──────────┼──► Router          │
  │     ┌──┴──┐        │          │     ┌──┴──┐        │
  │  Core   SRAM slice │          │  Core   SRAM slice │
  │ Tile (0,1)         │          │ Tile (1,1)         │
  └────────────────────┘          └────────────────────┘
```

### Compute core

Each core contains:

- **Systolic array** — 16×16 weight-stationary processing elements (PEs). Each PE holds a weight, multiplies it by the activation passing through, and adds to the partial sum flowing down the column.
- **Weight buffer** — SRAM for weight tiles, plus a double-buffered weight register set so the next tile loads while the current one computes.
- **Activation buffer** — SRAM feeding one 16-byte activation row per cycle.
- **Accumulators** — int32 storage below the array for summing partial results across K-tiles. Implemented as ping-pong banks (or dual-port SRAM, if available) since they read and write every cycle.
- **Activation unit** — ReLU, then requantization (scale, shift, round, saturate to int8).
- **Controller** — instruction queue and sequencer.
- **Local DMA** — moves data between the core's buffers and any tile's SRAM slice, issuing NoC packets for remote addresses.

Buffer sizes are set to match the SRAM macros available in the IHP PDK (TBD in Phase 0).

### Shared memory and interconnect

- **Distributed SRAM** — each tile holds one slice of a single global address space, 128 bits wide. A core reaches its own slice locally and other tiles' slices over the mesh, so access latency depends on distance (NUMA).
- **Address map** — upper address bits select the home tile; lower bits select the location within its slice.

### Network-on-chip
 
- **Topology** — 2D mesh, one router per tile. Each router has a local port (core + SRAM slice) and up to four neighbor ports (N/S/E/W); tile (0,0) also connects the host.
- **Routing** — dimension-order (XY) routing: travel along X first, then Y. Simple, and deadlock-free on a mesh.
- **Flow control** — input-buffered routers with credit-based flow control between neighbors; 128-bit flits.
- **Packets** — read requests, read responses, and writes. Requests and responses travel on separate virtual channels (or separate physical networks) to avoid protocol deadlock.
- **Arbitration** — round-robin per output port.
- **Synchronization** — barrier and per-core done flags implemented as messages to a register block on tile (0,0).

### Parallelism model
 
- **Output-tile split (first)** — each core computes its own block of the output matrix, reading inputs from any tile's SRAM over the mesh. Placing each core's data in its own slice keeps most traffic local.
- **Core-to-core (extension)** — cores pass partial sums or activations to neighbors, e.g., splitting the K dimension and reducing across tiles, or pipelining consecutive layers across tiles.
### Host interface
 
- Control/status registers: per-core instruction base address, start, status, interrupt.
- A wide port, attached to the router on tile (0,0), for loading inputs into and reading results from SRAM over the mesh.
- In simulation, the testbench acts as the host.
---
 
## Instruction Set
 
64-bit fixed-format instructions, executed per core.
 
| Instruction | Description |
|---|---|
| `LOAD_W` | DMA a weight tile from any tile's SRAM slice into the weight buffer |
| `LOAD_A` | DMA activations from any tile's SRAM slice into the activation buffer |
| `MATMUL` | Stream rows through the array; flag selects overwrite or accumulate |
| `ACT` | Apply ReLU and requantize accumulators to int8 (scale, shift) |
| `STORE` | DMA results from the core to any tile's SRAM slice |
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
│   ├── spec.md              # Interfaces, ISA encodings, memory map, NoC packet format
│   └── diagrams/
├── model/
│   ├── perf_model.py        # Utilization and bandwidth estimates
│   └── golden.py            # Bit-exact int8 matmul + requantization reference
├── rtl/
│   ├── pe/                  # Processing element
│   ├── array/               # 16×16 systolic array
│   ├── core/                # Buffers, accumulators, activation unit, controller, DMA
│   ├── noc/                 # Router, flit/packet format, XY routing, credits
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
Built bottom-up: PE → array → core → router → tile (core + SRAM slice + router) → 2×2 mesh top level. Behavioral SRAM models use the same ports and timing as the IHP macros.
**Check:** lint clean.
 
### Phase 2 — Functional verification
Unit tests at every level, then random tests against the golden model. Corner cases include int8 extremes, accumulator overflow, back-to-back instructions, all cores targeting one tile's SRAM, and NoC deadlock/backpressure (full buffers, credit exhaustion).
**Check:** regression passes, coverage goals met. Rerun after every RTL change.
 
### Phase 3 — Synthesis (Genus)
Synthesize the tile alone, then the mesh top level with the tile as a black box. Run equivalence checking and gate-level simulation.
**Check:** timing met with ~10% margin, equivalence passes, gate-level sim passes.
 
### Phase 4 — Tile physical design (Innovus)
Import → floorplan (SRAM macro placement, halos, pins) → power plan → place → CTS → route → fillers → verify. Pins for N/S/E/W links are placed on matching edges so neighboring tiles abut cleanly. Export the tile as a hard macro (LEF, .lib, GDS, netlist, SPEF).
**Check:** no setup/hold violations, clean geometry and connectivity.
 
### Phase 5 — Top-level physical design (Innovus)
Place 4 tile macros in a 2×2 grid and the host interface. Top-level power grid, CTS (watching cross-tile skew), route the short neighbor links.
**Check:** same as Phase 4.
 
### Phase 6 — Signoff
STA at all corners (Tempus), power and IR drop (Voltus), GDS merge, metal fill, DRC and LVS in KLayout.
**Check:** DRC/LVS clean, timing met at all corners.
 
### Phase 7 — Results
Measured utilization and TOPS vs. the Phase 0 model, area breakdown, power, max frequency, TOPS/W and TOPS/mm², layout screenshots.
 
### Iteration
- Timing failures → add pipeline stages in RTL (back to Phase 1).
- Routing congestion → adjust floorplan, macro placement, or NoC link width.
- Area too large → reduce array size or mesh dimensions.
> **Note:** Innovus RC extraction uses LEF-based RC values, since the IHP open PDK does not provide a QRC tech file. Post-layout timing should be read with that in mind.
 
---
 
## Verification
 
- **Golden model:** `model/golden.py` is the single source of truth for expected results, matching the hardware's rounding and saturation bit-for-bit.
- **Testbenches:** cocotb, organized by hierarchy level under `tb/`.
- **Levels:** PE → array → single core → router → single tile → full 2×2 mesh.
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
 
Synthesis, place-and-route, and signoff commands will be added as we go.
 
---
 
## Roadmap
 
- [x] **Phase 0** — Performance model, spec, golden model, IHP SRAM/cell survey
- [x] **Flow smoke test** — Small design through Genus → Innovus → KLayout DRC/LVS on IHP
- [ ] **Phase 1–2** — PE, then 16×16 array, RTL + tests
- [ ] **Milestone** — Single 16×16 array through the full physical flow
- [ ] **Phase 1–2** — Full core (buffers, controller, DMA, activation unit)
- [ ] **Phase 1–2** — Router and NoC, then single tile (core + SRAM slice + router)
- [ ] **Phase 3–4** — Tile synthesized and hardened as a macro
- [ ] **Phase 1–2** — 2×2 mesh top level
- [ ] **Phase 5–6** — Top-level PnR and signoff
- [ ] **Phase 7** — Results and writeup
- [ ] **Stretch** — Core-to-core dataflow (cross-tile reduction, layer pipelining); larger meshes
---
 
## License
 
TBD for project source. The IHP SG13G2 Open PDK is licensed separately under Apache 2.0.
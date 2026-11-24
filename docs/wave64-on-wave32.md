# Wave64 compute on hosts with 32-lane waves

Console (RDNA2) compute programs are mostly wave64. A translation that depends
on the lane layout declares `WaveSize(64)`; hosts whose D3D12 wave lane count
range stops below 64 (NVIDIA: 32..32) refuse such pipelines ("Required
WaveSize range [64, 64], but supported range is [32, 32]" under vkd3d-proton).

## When the translator declares a wave size (unchanged on 64-lane hosts)

`emit/lower.rs`, `Lower::new`: a compute program needs the console's lanes
(`console_lanes`) when it has `ds_swizzle`, `*permute*`/`v_permlane*`,
`v_mbcnt*`, `v_readlane`/`v_writelane`, DPP, a lane mask read as a number
(`reads_mask_as_data`), or a guard built on an opaque or constant lane mask.
Then, or when the group is one console wave sharing LDS (the console compiler
drops barriers there), it declares `WaveSize(64)` (wave64) or `WaveSize(32)`.
Every other program declares no wave size and runs on any host as it is.

## The split form (`Options::host_wave32`)

Opt-in by capability: `victory-gpu` asks the device once
(`D3D12_FEATURE_D3D12_OPTIONS1.WaveLaneCountMax < 64`, `Host::wave32_only`)
and calls `victory_shader::emit::set_host_wave32`; `VICTORY_HOST_WAVE32=1`
(or `0`) overrides it; `emit::translate_with` / `emit_with` take the option
explicitly. The shader cache keys on it. On 64-lane hosts nothing changes.

With it, a console wave64 compute program that needs console lanes is first
lowered *split*: each console wave runs as two host waves of 32 ("halves").

* **Lane ids.** Console thread index `f` in the group; console lane
  `f % 64`, console wave `f / 64` (`Lower::split_flat`, `console_lane`).
  - No DPP / swizzle / permlane: `f = SV_GroupIndex`, thread ids are the
    host's, **no wave size declared** (host picks any).
  - With them (they read lanes inside a row of 16 / group of 32): threads
    are relabelled so each host wave holds a 32-aligned run of console
    threads, and the program declares `WaveSize(32)`. Every thread zeroes a
    group-shared counter, group barrier (entry block, uniform), the first
    lane of each full wave adds 32 and the wave takes the old value as its
    base; a partial wave takes the last `n` threads (`total - n`). This does
    not assume the host forms waves from consecutive indices (NVIDIA does not
    for 3-D groups, per xcl).
* **Uses of the console lane**: own bit of an opaque / constant lane mask
  (`lane_bit`), `v_mbcnt_lo/hi`, `v_writelane`'s target, DPP row masks.
* **Refusal.** Lowering an operation that needs the other half sets
  `split_refused` (`Lower::whole_wave`): a ballot read as a number
  (`mask_dword`), `v_readlane`, a whole-wave reduction or prefix (`shape`).
  So do one-wave groups with LDS (decided up front). Any refusal or failure
  of the split form lowers the program again exactly as before
  (`WaveSize(64)`), so the result is never worse than without the option.

## Audit of wave intrinsics in the emitter (32 vs 64)

| intrinsic (DXIL op) | where | wave64 console on 32-lane host |
|---|---|---|
| `waveGetLaneIndex` (111) | `lane_index`: DPP and swizzle source lanes | exact: relabelled, host lane = console lane % 32, reads stay in 16/32 |
| | `lane_bit`, `v_mbcnt`, `v_writelane`, DPP row mask | split uses `console_lane` (`f & 63`) instead |
| | split relabelling | host lane within its wave (by design) |
| `waveReadLaneAt` (117) | DPP, `ds_swizzle` | exact (within 16 / 32 lanes) |
| | `v_readlane` | refused (lane may be in the other half) |
| `waveReadLaneFirst` (118) | `v_readfirstlane` | per half: exact for uniform values and waterfall loops; an election of one lane per console wave becomes one per half (risk; such code usually also ballots, which refuses) |
| | non-split relabel (`thread_id`) | unchanged (64-lane hosts; assumes wave-aligned runs) |
| `waveActiveBallot` (116) | `mask_dword` (mask read as data) | refused (needs the other half's word) |
| | split relabelling | counts the host wave's own lanes (by design) |
| `waveActiveOp` (119), `waveActiveBit` (120), `wavePrefixOp` (121) | `shape`: whole-wave sections | refused (needs the other half's partial) |
| `quadReadLaneAt` (122) | mesh quad swizzles / DPP | mesh only; unaffected |
| pixel / vertex (`MaskData::Host`) | ballots, `v_mbcnt`, readlane at computed lanes | already host-wave-size agnostic; unaffected |

## Not done yet (TODO, emulation "B")

The refused operations need an exchange between the halves through
group-shared memory (as xcl does: `../xcl/docs/wave64-on-wave32.md`):
ballot words (1 dword per half), `v_readlane` (owner's value), reductions /
prefixes (each half's partial; the upper half adds the lower's total),
`ds_bpermute` (32 dwords per half). The obstacle specific to VIctory: its
lowering keeps console control flow per lane (masks become per-lane
booleans), so an exchange point may be reached by one half only; a group
barrier there is undefined behaviour. A correct design needs either
(1) exchanges only at points whose block is reached by all lanes of the
console wave (provable from the structurizer: the mask is written in a block
dominated only by uniform branches), with a group barrier when the group is
one console wave and xcl's counter-paired spin otherwise, or (2) both halves
executing every exchange unconditionally (hoisted to uniform points with
the value masked by the lane's activity). Until then those programs keep
`WaveSize(64)` and are refused by 32-lane hosts, as before.

## Verify

* `cargo +stable test --offline --locked -p victory-shader` — tests
  `a_lane_id_program_is_split_on_a_wave32_host`,
  `a_dpp_program_is_split_with_relabelled_threads`,
  `whole_wave_operations_keep_wave64`, `one_wave_lds_programs_keep_wave64`,
  `lane_independent_programs_are_unchanged`, `the_host_wave32_switch_parses`.
* DXIL validation on Linux: `VICTORY_WAVE64_DUMP=/tmp/d cargo test ... wave64`
  then `LD_LIBRARY_PATH=dxc-linux/lib dxc-linux/bin/dxv /tmp/d/*.dxil`.
* In the game on NVIDIA: the log line `host wave lane counts ... wave32=true`,
  then fewer "Required WaveSize range [64, 64]" refusals; the remaining
  ones are programs that need the emulation above.

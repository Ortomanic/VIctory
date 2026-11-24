# Linux (Wine)

VIctory is a Windows program. On Linux it runs under Wine with vkd3d-proton
for Direct3D 12 and DXVK's `dxgi.dll`. Proton Experimental brings both; a
system Wine (tested here: Wine 11.22, vkd3d-proton 3.1, NVIDIA; upstream
reports RADV with vkd3d-proton 3.2) needs them added. Adapted from Marvisak/VIctory#994 (`docs/linux-proton.md`,
Slaktarn700), which also has a Proton recipe and `scripts/proton-run.sh`.

## What the prefix needs

- `d3d12.dll` and `d3d12core.dll` from vkd3d-proton, `dxgi.dll` from DXVK,
  in `system32`, with `WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=n"`.
- `dxcompiler.dll` and `dxil.dll` from a Windows DXC release, in `system32`
  with `dxcompiler,dxil=n`: the rectangle-list geometry stage is compiled
  with DXC, and the plane composition and the upscaler stand-in fall back
  to it where Wine's compiler refuses their source.
- No Microsoft `d3dcompiler_47`: Wine's own `D3DCompile` (vkd3d-shader)
  builds the indirect argument clamp and the rectangle expander, which use
  structured buffers (it has no `RWByteAddressBuffer` methods) and get
  their root signatures from code (it does not embed one).

## Building

Cross-build for `x86_64-pc-windows-msvc` with `lld-link`/`clang-cl` and an
xwin-downloaded CRT and SDK (`INCLUDE`/`LIB` pointing at it), or with
`cargo xwin`. The tests run under Wine with the same prefix (set
`CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER=wine`); GPU tests return early
where there is no D3D12 device.

## Where vkd3d-proton differs from Windows drivers (handled)

- An empty pipeline library must be asked for with a null blob; an empty
  slice's dangling pointer is refused. The library is still off under Wine
  by default (`VICTORY_PIPELINE_LIBRARY=1` turns it on): with entries that
  miss, vkd3d-proton made pipelines several times slower.
- DXIL container hashes are not checked (a shader test skips that check
  under Wine).
- `ws2_32` lacks `WSAConnectByList` and `GetAddrInfoExA`: their overrides
  fail the call cleanly.
- Write watching is emulated per page under Wine's virtual-memory lock:
  the watch caches allocation kinds and answers repeat askings within a
  submission from its journal.

## Switches

- `VICTORY_PIPELINE_LIBRARY`: 1 keeps host pipelines in
  `host-pipeline-library.bin` under the cache folder (the default on Windows), 0
  does not (the default under Wine). Delete the file to reset it.
- `VICTORY_PIPELINE_WAIT_MS`: how long a title frame may wait for pipelines
  being made (500 by default; 0 never waits, so pipelines appear a few
  frames later and one-off passes may be lost).

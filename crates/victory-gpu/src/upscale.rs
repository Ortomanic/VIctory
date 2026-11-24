//! A bilinear stand-in for the console's FidelityFX upscaler.
//!
//! Grand Theft Auto VI renders its scene at a lower resolution (1280x720
//! for a 1920x1080 plane on Series S) and has FidelityFX scale it up. The
//! upscaler's passes are precompiled console ISA (an `ACCC` part holding
//! an `FSPX` image, no `DXIL` part), which the shader translator cannot
//! take: their dispatches were dropped, the output was never written, and
//! everything after it (bloom, tone mapping, the plane) came out black.
//! `victory-hle` answers the FidelityFX loader's API instead and calls
//! [`upscale`] for each upscale dispatch: one compute pass, recorded in the
//! title's list, samples the rendered part of the colour input with a
//! linear filter into the output. No temporal accumulation (the loader
//! reports no jitter, so the image is stable), no sharpening.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{s, w};

use crate::object::This;

const SOURCE: &str = r#"
cbuffer Scale : register(b0) { float2 scale; uint2 size; float pattern; float3 pad; };
Texture2D<float4> color : register(t0);
RWTexture2D<float4> output : register(u0);
SamplerState linear_clamp : register(s0);
[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
    if (id.x >= size.x || id.y >= size.y) return;
    float4 c = color.SampleLevel(linear_clamp, (float2(id.xy) + 0.5) * scale, 0);
    float2 uv = (float2(id.xy) + 0.5) / float2(size);
    c.rgb += pattern * float3(uv, 1.0 - uv.x);
    output[id.xy] = c;
}
"#;

/// `VICTORY_FFX_PATTERN=<level>` (diagnostics): a colour gradient of that
/// level is added to the upscaled image, to tell whether what the stand-in
/// writes reaches the presented plane.
fn pattern() -> f32 {
    static LEVEL: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| {
        std::env::var("VICTORY_FFX_PATTERN")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0.0)
    })
}

struct Pipeline {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pipeline {}
// SAFETY: as above.
unsafe impl Sync for Pipeline {}

fn pipeline() -> Option<&'static Pipeline> {
    static PIPELINE: std::sync::OnceLock<Option<Pipeline>> = std::sync::OnceLock::new();
    PIPELINE
        .get_or_init(|| match make() {
            Ok(p) => {
                tracing::info!("upscaler stand-in ready (bilinear)");
                Some(p)
            }
            Err(e) => {
                tracing::warn!(error = %e, "no upscaler stand-in: upscale dispatches do nothing");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Pipeline> {
    let device = &crate::host::host()
        .ok_or_else(windows::core::Error::empty)?
        .device;
    let root = root_signature(device)?;
    // Wine's d3dcompiler_47 may refuse what DXC takes, and the other way.
    let code = match fxc() {
        Ok(code) => code,
        Err(e) => {
            tracing::info!(error = %e, "upscaler stand-in: the system compiler failed; trying DXC");
            crate::indirect::dxc_compile(SOURCE, w!("cs_6_0"), w!("cs"))?
        }
    };
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        // SAFETY: a borrowed pointer the description never releases.
        pRootSignature: unsafe { std::mem::transmute_copy(&root) },
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: code.as_ptr().cast(),
            BytecodeLength: code.len(),
        },
        ..Default::default()
    };
    // SAFETY: the description points at live data.
    let pso = unsafe { device.CreateComputePipelineState(&desc)? };
    Ok(Pipeline { root, pso })
}

fn fxc() -> windows::core::Result<Vec<u8>> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("upscale"),
            None,
            None,
            s!("cs"),
            s!("cs_5_0"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    match (compiled, code) {
        (Ok(()), Some(code)) => Ok(bytes(&code).to_vec()),
        (result, _) => Err(windows::core::Error::new(
            result
                .err()
                .map_or(windows::Win32::Foundation::E_FAIL, |e| e.code()),
            errors.as_ref().map_or(String::new(), |e| {
                String::from_utf8_lossy(bytes(e)).trim().to_owned()
            }),
        )),
    }
}

fn bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: the blob's own bytes, alive as long as the blob.
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    }
}

/// Four constants (`scale`, `size`), a table of the colour view and the
/// output view, a linear clamping sampler.
fn root_signature(device: &ID3D12Device) -> windows::core::Result<ID3D12RootSignature> {
    let ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 1,
        },
    ];
    let parameters = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 8,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: ranges.len() as u32,
                    pDescriptorRanges: ranges.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let sampler = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MipLODBias: 0.0,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        MinLOD: 0.0,
        MaxLOD: D3D12_FLOAT32_MAX,
        ShaderRegister: 0,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: parameters.len() as u32,
        pParameters: parameters.as_ptr(),
        NumStaticSamplers: 1,
        pStaticSamplers: &sampler,
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let mut blob: Option<ID3DBlob> = None;
    // SAFETY: the description points at live locals; out pointer ours.
    unsafe {
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)?;
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        device.CreateRootSignature(0, bytes(&blob))
    }
}

/// Records, in title list `list`, the top-left `render` texels of title
/// texture `color` scaled into the top-left `size` texels of title texture
/// `output` (which must allow unordered access). Replaces the list's
/// descriptor heaps, compute root signature and pipeline: the title's are
/// bound again at its next draw or dispatch. False if nothing was
/// recorded.
pub fn upscale(
    list: *mut std::ffi::c_void,
    color: *mut std::ffi::c_void,
    render: (u32, u32),
    output: *mut std::ffi::c_void,
    size: (u32, u32),
) -> bool {
    let (list, color, output) = (list as This, color as This, output as This);
    if list.is_null() || color.is_null() || output.is_null() {
        return false;
    }
    let mut done = false;
    crate::list::with_state_pub(list, |host_list, state| {
        // The title's bound view heap, when it has room: no heap of the
        // stand-in's own, made and bound for each upscale.
        let slots = if own_heap() { None } else { state.own_slots(host_list, 2) };
        done = record(host_list, color, render, output, size, slots);
        if done && slots.is_some() {
            state.compute_bindings_lost();
        } else if done {
            state.host_bindings_lost();
        }
    });
    done
}

/// `VICTORY_FFX_OWN_HEAP=1` (diagnostics): each upscale makes and binds a
/// descriptor heap of its own, as the stand-in first did.
fn own_heap() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FFX_OWN_HEAP").is_some())
}

fn record(
    list: &crate::host::HostList,
    color: This,
    (rw, rh): (u32, u32),
    output: This,
    (ow, oh): (u32, u32),
    slots: Option<(&crate::descriptor::host::HostHeap, u32)>,
) -> bool {
    let (Some(host), Some(p)) = (crate::host::host(), pipeline()) else {
        return false;
    };
    let (Some(src), Some(dst)) = (
        crate::resource::host_resource(color),
        crate::resource::host_resource(output),
    ) else {
        return false;
    };
    // SAFETY: live resources.
    let (s, d) = unsafe { (src.GetDesc(), dst.GetDesc()) };
    if s.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || s.SampleDesc.Count != 1
        || !d.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
        || s.Flags.contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
    {
        return false;
    }
    let (rw, rh) = (rw.clamp(1, s.Width as u32), rh.clamp(1, s.Height));
    let (ow, oh) = (ow.clamp(1, d.Width as u32), oh.clamp(1, d.Height));
    const NON_PIXEL_SHADER_RESOURCE: u32 = 0x40;
    const UNORDERED_ACCESS: u32 = 0x8;
    crate::list::ensure_state(list, color, NON_PIXEL_SHADER_RESOURCE);
    crate::list::ensure_state(list, output, UNORDERED_ACCESS);
    let device = &host.device;
    let constants: [u32; 8] = [
        (rw as f32 / s.Width as f32 / ow as f32).to_bits(),
        (rh as f32 / s.Height as f32 / oh as f32).to_bits(),
        ow,
        oh,
        pattern().to_bits(),
        0,
        0,
        0,
    ];
    let n = UPSCALES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let probing = probe::due(n);
    if probing {
        probe::take(list, color, src, n, "colour before");
    }
    // SAFETY: plain creation; an open list; our pipeline; live textures the
    // title's list holds.
    unsafe {
        let step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        // The title's bound heap (already bound), or one of our own.
        let (cpu, gpu, own) = match slots {
            Some((heap, first)) => {
                heap.hold(first, src.clone());
                heap.hold(first + 1, dst.clone());
                let (cpu, gpu) = heap.slot_handles(first);
                (cpu, gpu, None)
            }
            None => {
                let Ok(heap) = device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(
                    &D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: 2,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    },
                ) else {
                    return false;
                };
                list.keep(&heap);
                let (cpu, gpu) = (
                    heap.GetCPUDescriptorHandleForHeapStart(),
                    heap.GetGPUDescriptorHandleForHeapStart(),
                );
                (cpu, gpu, Some(heap))
            }
        };
        let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: crate::compose::sampled_format(s.Format.0 as u32),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        device.CreateShaderResourceView(src, Some(&view), cpu);
        let unordered = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: crate::compose::sampled_format(d.Format.0 as u32),
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        };
        device.CreateUnorderedAccessView(
            dst,
            None,
            Some(&unordered),
            D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu.ptr + step as usize,
            },
        );
        let l = &list.list;
        if let Some(heap) = own {
            l.SetDescriptorHeaps(&[Some(heap)]);
        }
        l.SetComputeRootSignature(&p.root);
        l.SetPipelineState(&p.pso);
        l.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(1, gpu);
        l.Dispatch(ow.div_ceil(8), oh.div_ceil(8), 1);
        l.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::transmute_copy(dst),
                }),
            },
        }]);
    }
    if probing {
        probe::take(list, output, dst, n, "output after");
    }
    pass_log(color, output, (rw, rh), (ow, oh));
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!(
            from = format_args!("{rw}x{rh} of {}x{} f{}", s.Width, s.Height, s.Format.0),
            to = format_args!("{ow}x{oh} of {}x{} f{}", d.Width, d.Height, d.Format.0),
            "first upscale by the stand-in"
        );
    });
    true
}

static UPSCALES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// With `VICTORY_PASS_LOG`, the stand-in's pass among the title's in a
/// listed frame.
fn pass_log(color: This, output: This, render: (u32, u32), size: (u32, u32)) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_PASS_LOG").is_some())
        || !crate::capture::listed(crate::queue::presented() + 1)
    {
        return;
    }
    let (color, output) = (color as usize, output as usize);
    tracing::info!(
        "pass upscale-stand-in 1 [] targets [] depth None reads [\"Srv: {color:#x} {}x{}\", \"Uav: uav Some(\\\"{output:#x} {}x{}\\\")\"]",
        render.0,
        render.1,
        size.0,
        size.1
    );
}

/// `VICTORY_FFX_PROBE=1` (diagnostics): the colour input before and the
/// output after an upscale are copied in the title's list, at the first
/// upscales, those of listed frames and every 40th, and summarized 30
/// upscales later (largest finite value, set, negative and non-finite
/// texels per channel).
mod probe {
    use super::*;

    struct Taken {
        at: u64,
        what: &'static str,
        resource: usize,
        buffer: ID3D12Resource,
        footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    }
    // SAFETY: D3D12 objects are free-threaded.
    unsafe impl Send for Taken {}

    static TAKEN: std::sync::Mutex<Vec<Taken>> = std::sync::Mutex::new(Vec::new());

    fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_FFX_PROBE").is_some())
    }

    /// Whether upscale `n` is probed; summarizes earlier probes that have
    /// had time to run.
    pub(super) fn due(n: u64) -> bool {
        if !on() {
            return false;
        }
        let ready: Vec<Taken> = {
            let mut taken = TAKEN.lock().unwrap_or_else(|e| e.into_inner());
            let (ready, wait) = std::mem::take(&mut *taken)
                .into_iter()
                .partition(|t| t.at + 30 <= n);
            *taken = wait;
            ready
        };
        for t in ready {
            summarize(&t);
        }
        n < 3 || n % 40 == 0 || crate::capture::listed(crate::queue::presented() + 1)
    }

    pub(super) fn take(
        list: &crate::host::HostList,
        resource: This,
        texture: &ID3D12Resource,
        at: u64,
        what: &'static str,
    ) {
        let Some(host) = crate::host::host() else { return };
        const COPY_SOURCE: u32 = 0x800;
        crate::list::ensure_state(list, resource, COPY_SOURCE);
        let Some((buffer, footprint)) = crate::capture::read_back(&host.device, &list.list, texture)
        else {
            return;
        };
        list.keep(&buffer);
        TAKEN.lock().unwrap_or_else(|e| e.into_inner()).push(Taken {
            at,
            what,
            resource: resource as usize,
            buffer,
            footprint,
        });
    }

    fn summarize(t: &Taken) {
        let f = t.footprint.Footprint;
        let texel = match f.Format.0 {
            10 => 8usize,
            2 => 16,
            24 | 26 | 28 | 87 => 4,
            _ => {
                tracing::info!(
                    at = t.at,
                    what = t.what,
                    format = f.Format.0,
                    "upscale probe: format not summarized"
                );
                return;
            }
        };
        let mut data = std::ptr::null_mut();
        // SAFETY: a readback buffer of the full pitched rows; its copy ran
        // many frames ago.
        unsafe {
            if t.buffer.Map(0, None, Some(&mut data)).is_err() {
                return;
            }
            let pitch = f.RowPitch as usize;
            let rows = std::slice::from_raw_parts(data.cast::<u8>(), pitch * f.Height as usize);
            let mut max = [0f32; 4];
            let mut set = [0usize; 4];
            let mut negative = [0usize; 4];
            let mut odd = [0usize; 4];
            for y in 0..f.Height as usize {
                for x in 0..f.Width as usize {
                    let p = &rows[y * pitch + x * texel..][..texel];
                    let v: [f32; 4] = match f.Format.0 {
                        10 => std::array::from_fn(|i| {
                            half(u16::from_le_bytes([p[2 * i], p[2 * i + 1]]))
                        }),
                        2 => std::array::from_fn(|i| {
                            f32::from_le_bytes(p[4 * i..4 * i + 4].try_into().unwrap_or([0; 4]))
                        }),
                        _ => std::array::from_fn(|i| f32::from(p[i])),
                    };
                    for (c, &v) in v.iter().enumerate() {
                        if !v.is_finite() {
                            odd[c] += 1;
                        } else {
                            set[c] += usize::from(v != 0.0);
                            negative[c] += usize::from(v < 0.0);
                            max[c] = max[c].max(v);
                        }
                    }
                }
            }
            t.buffer.Unmap(0, None);
            tracing::info!(
                at = t.at,
                what = t.what,
                resource = format_args!("{:#x}", t.resource),
                size = format_args!("{}x{} f{}", f.Width, f.Height, f.Format.0),
                ?max,
                ?set,
                ?negative,
                non_finite = ?odd,
                "upscale probe"
            );
        }
    }

    fn half(h: u16) -> f32 {
        let (sign, exp, man) = (h >> 15, (h >> 10) & 0x1f, f32::from(h & 0x3ff));
        let v = match exp {
            0 => man * 2f32.powi(-24),
            31 if man == 0.0 => f32::INFINITY,
            31 => f32::NAN,
            e => (1.0 + man / 1024.0) * 2f32.powi(i32::from(e) - 15),
        };
        if sign == 1 { -v } else { v }
    }
}

//! `amd_fidelityfx_loader_xbox.dll`: the FidelityFX API, with a bilinear
//! upscaler in place of the console's.
//!
//! The console upscaler (`amd_fidelityfx_upscaler_xbox.dll`, which the
//! loader loads) creates its passes from precompiled console ISA: `ACCC`
//! parts holding `FSPX` images, with no `DXIL` part for the shader
//! translator. Its dispatches were dropped, its output never written, and
//! Grand Theft Auto VI's scene plane came out black after it. Here the
//! loader answers as the upscaler does for the queries the title makes
//! (render size from the quality mode, so the title renders as on the
//! console), reports one jitter phase at offset zero (no temporal
//! accumulation, so an unjittered camera gives a stable image), and each
//! upscale dispatch becomes one bilinear pass in the title's list
//! ([`victory_gpu::upscale`]). Other effects (frame generation) report no
//! provider, as the console does when their DLL is absent.
//!
//! Descriptor types and field offsets are the public FidelityFX API's:
//! each descriptor starts with `u64 type, header *next`. The upscale
//! dispatch has the command list at +0x10, the colour resource at +0x18,
//! the output resource at +0x138, the render size at +0x178 and the
//! upscale size at +0x180 (as rebox documents, checked against the
//! console upscaler's dispatch). Query types are the SDK's: 0x10002 ratio,
//! 0x10003 render size, 0x10004 jitter phase count, 0x10005 jitter offset
//! (rebox's stub has them one higher; the title's phase count query showed
//! which is right).
//!
//! Opt-in: `VICTORY_FFX=bilinear` serves the loader; by default the
//! title's own loader (and the console upscaler, whose passes are dropped)
//! is used. With the stand-in the world's presented frames carry the scene
//! (the opening night flight: city lights, bloom); without it they are
//! black. World runs on NVIDIA often end in a GPU hang (Xid 109, context
//! switch timeout) with or without it, the host queue stopped at GPU
//! particle or hair dispatches (`VICTORY_GPU_BREADCRUMBS`).
#![allow(non_snake_case)]
// Title code passes pointers the real library would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{HleModule, hle_export};

const FFX_OK: u32 = 0;
const FFX_ERROR_UNKNOWN_DESCTYPE: u32 = 2;
const FFX_ERROR_RUNTIME: u32 = 3;
const FFX_NO_PROVIDER: u32 = 4;
const FFX_ERROR_MEMORY: u32 = 5;
const FFX_ERROR_PARAMETER: u32 = 6;

const EFFECT_MASK: u64 = 0xff_0000;
const EFFECT_UPSCALE: u64 = 0x1_0000;
const CREATE_UPSCALE: u64 = 0x1_0000;
const DISPATCH_UPSCALE: u64 = 0x1_0001;
// The FidelityFX SDK's numbering. Grand Theft Auto VI asks 0x10004 with
// (render width, display width, out pointer): the phase count, which it
// divides its frame index by.
const QUERY_UPSCALE_RATIO: u64 = 0x1_0002;
const QUERY_RENDER_RESOLUTION: u64 = 0x1_0003;
const QUERY_JITTER_PHASE_COUNT: u64 = 0x1_0004;
const QUERY_JITTER_OFFSET: u64 = 0x1_0005;
const QUERY_GET_VERSIONS: u64 = 4;

/// Reads a field of a descriptor at `offset`.
fn field<T: Copy>(desc: *const u8, offset: usize) -> T {
    // SAFETY: the title passes a descriptor of the type it names, which
    // holds the field.
    unsafe { desc.add(offset).cast::<T>().read_unaligned() }
}

/// Writes through a pointer field of a descriptor, if it is set.
fn put<T>(desc: *const u8, offset: usize, value: T) {
    let at: *mut T = field(desc, offset);
    if !at.is_null() {
        // SAFETY: the title's out pointer.
        unsafe { at.write_unaligned(value) };
    }
}

/// The descriptor of `kind` in a chain.
fn find(mut desc: *const u8, kind: u64) -> Option<*const u8> {
    while !desc.is_null() {
        if field::<u64>(desc, 0) == kind {
            return Some(desc);
        }
        desc = field(desc, 8);
    }
    None
}

/// Render size divisor of a quality mode: native AA, quality, balanced,
/// performance, ultra performance.
fn ratio(mode: u32) -> f32 {
    [1.0, 1.5, 1.7, 2.0, 3.0]
        .get(mode as usize)
        .copied()
        .unwrap_or(1.0)
}

/// An upscale context: the sizes it was made for.
struct Context {
    max_render: (u32, u32),
    max_upscale: (u32, u32),
}

pub extern "system" fn ffxCreateContext(
    context: *mut *mut c_void,
    desc: *const u8,
    _alloc: *const c_void,
) -> u32 {
    if context.is_null() || desc.is_null() {
        return FFX_ERROR_PARAMETER;
    }
    // SAFETY: the title's out pointer.
    unsafe { *context = std::ptr::null_mut() };
    let Some(up) = find(desc, CREATE_UPSCALE) else {
        tracing::info!(
            effect = format_args!("{:#x}", field::<u64>(desc, 0) & EFFECT_MASK),
            "FidelityFX: no provider for the effect"
        );
        return FFX_NO_PROVIDER;
    };
    let c = Context {
        max_render: (field(up, 0x14), field(up, 0x18)),
        max_upscale: (field(up, 0x1c), field(up, 0x20)),
    };
    tracing::info!(
        render = ?c.max_render,
        upscale = ?c.max_upscale,
        "FidelityFX: upscale context (bilinear stand-in, no jitter)"
    );
    let raw = Box::into_raw(Box::new(c));
    if raw.is_null() {
        return FFX_ERROR_MEMORY;
    }
    // SAFETY: as above.
    unsafe { *context = raw.cast() };
    FFX_OK
}

pub extern "system" fn ffxDestroyContext(context: *mut *mut c_void, _alloc: *const c_void) -> u32 {
    if context.is_null() {
        return FFX_ERROR_PARAMETER;
    }
    // SAFETY: a context ffxCreateContext made (or null), taken once.
    unsafe {
        let c = std::mem::replace(&mut *context, std::ptr::null_mut());
        if !c.is_null() {
            drop(Box::from_raw(c.cast::<Context>()));
        }
    }
    FFX_OK
}

pub extern "system" fn ffxConfigure(context: *mut *mut c_void, desc: *const u8) -> u32 {
    if !desc.is_null() {
        log_first(desc, context);
    }
    if desc.is_null() {
        FFX_ERROR_PARAMETER
    } else {
        FFX_OK
    }
}

pub extern "system" fn ffxQuery(context: *mut *mut c_void, desc: *const u8) -> u32 {
    if desc.is_null() {
        return FFX_ERROR_PARAMETER;
    }
    log_first(desc, context);
    match field::<u64>(desc, 0) {
        QUERY_UPSCALE_RATIO => {
            put(desc, 0x18, ratio(field(desc, 0x10)));
            FFX_OK
        }
        QUERY_RENDER_RESOLUTION => {
            let r = ratio(field(desc, 0x18));
            let (w, h): (u32, u32) = (field(desc, 0x10), field(desc, 0x14));
            put(desc, 0x20, (w as f32 / r) as u32);
            put(desc, 0x28, (h as f32 / r) as u32);
            FFX_OK
        }
        QUERY_JITTER_PHASE_COUNT => {
            put(desc, 0x18, 1i32);
            FFX_OK
        }
        QUERY_JITTER_OFFSET => {
            put(desc, 0x18, 0f32);
            put(desc, 0x20, 0f32);
            FFX_OK
        }
        QUERY_GET_VERSIONS => {
            // { header; create type; device; u64 *count; u64 *ids; const char **names }
            static NAME: &[u8] = b"VIctory bilinear\0";
            let ours = field::<u64>(desc, 0x10) & EFFECT_MASK == EFFECT_UPSCALE;
            let count: *mut u64 = field(desc, 0x20);
            if count.is_null() {
                return FFX_ERROR_PARAMETER;
            }
            // SAFETY: the title's count and arrays of at least that many.
            unsafe {
                if ours && *count > 0 {
                    put(desc, 0x28, 1u64);
                    put(desc, 0x30, NAME.as_ptr());
                }
                *count = u64::from(ours);
            }
            FFX_OK
        }
        kind => {
            tracing::debug!(kind = format_args!("{kind:#x}"), "FidelityFX: unknown query");
            if context.is_null() {
                FFX_NO_PROVIDER
            } else {
                FFX_ERROR_UNKNOWN_DESCTYPE
            }
        }
    }
}

/// Each query type's first descriptor, raw (what the title asks).
fn log_first(desc: *const u8, context: *mut *mut c_void) {
    static SEEN: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
    let kind = field::<u64>(desc, 0);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.contains(&kind) {
        return;
    }
    seen.push(kind);
    let words: Vec<String> = (0..12)
        .map(|i| format!("{:08x}", field::<u32>(desc, 4 * i)))
        .collect();
    tracing::info!(
        kind = format_args!("{kind:#x}"),
        context = ?context,
        raw = %words.join(" "),
        "FidelityFX: first query of the type"
    );
}

pub extern "system" fn ffxDispatch(context: *mut *mut c_void, desc: *const u8) -> u32 {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    // SAFETY: the title passes its context slot.
    if context.is_null() || unsafe { (*context).is_null() } || desc.is_null() {
        return FFX_ERROR_PARAMETER;
    }
    if field::<u64>(desc, 0) != DISPATCH_UPSCALE {
        return FFX_ERROR_UNKNOWN_DESCTYPE;
    }
    let list: *mut c_void = field(desc, 0x10);
    let color: *mut c_void = field(desc, 0x18);
    let output: *mut c_void = field(desc, 0x138);
    let render: (u32, u32) = (field(desc, 0x178), field(desc, 0x17c));
    let size: (u32, u32) = (field(desc, 0x180), field(desc, 0x184));
    let done = victory_gpu::upscale(list, color, render, output, size);
    if !LOGGED.swap(true, Ordering::Relaxed) || !done {
        tracing::info!(
            ?render,
            ?size,
            list = ?list,
            color = ?color,
            output = ?output,
            done,
            "FidelityFX: upscale dispatch"
        );
    }
    if done { FFX_OK } else { FFX_ERROR_RUNTIME }
}

pub(crate) fn module() -> HleModule {
    let title = !std::env::var("VICTORY_FFX").is_ok_and(|v| v == "bilinear");
    HleModule {
        name: "ffx",
        dlls: if title {
            &[]
        } else {
            &["amd_fidelityfx_loader_xbox.dll"]
        },
        exports: vec![
            hle_export!(ffxConfigure),
            hle_export!(ffxCreateContext),
            hle_export!(ffxDestroyContext),
            hle_export!(ffxDispatch),
            hle_export!(ffxQuery),
        ],
    }
}

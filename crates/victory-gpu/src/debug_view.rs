//! `VICTORY_DEBUG_VIEW=albedo|normals|depth|rt<N>`: the window shows the
//! title's G-buffer instead of the lit, tone-mapped scene (plane 0), with
//! its UI plane still blended over it.
//!
//! The G-buffer is found by its pass: the render-target set bound with a
//! depth target that has the most colour targets of the depth's size (GTA
//! VI's main pass binds six at 1280x720 with its depth: R8G8B8A8 typeless,
//! four R8G8B8A8, one R16G16). The chosen view is one of that set's colour
//! slots, or its depth. Present samples it as it stands after the frame's
//! submissions, so the view costs one fullscreen draw and no copies; the
//! tracking costs one compare per multi-target binding.
//!
//! When no such pass was bound for a while (menus, loading, movies), the
//! normal composition comes back.

use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, Ordering::Relaxed},
};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

use crate::object::This;
use crate::resource::HostTexture;

/// GTA VI's main G-buffer pass slots (see `reports/unlit-view.md`).
const ALBEDO_SLOT: usize = 1;
const NORMALS_SLOT: usize = 3;
/// Fewest colour targets a G-buffer pass binds.
const MIN_TARGETS: usize = 3;
/// Presented frames the last G-buffer is shown for without a new pass.
const HOLD_FRAMES: u64 = 60;

/// How the shown texture is turned into colour (the composition pixel
/// shader's `mode`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    /// As it is (unsigned normalized targets: colours, or normals already
    /// stored as `n * 0.5 + 0.5`).
    Raw = 0,
    /// Signed values in [-1, 1] mapped to [0, 1] (signed or float normals).
    Signed = 1,
    /// Reversed-Z depth, brighter nearer, on a log scale.
    Depth = 2,
}

/// What `VICTORY_DEBUG_VIEW` asks for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum View {
    /// A colour slot of the G-buffer pass; whether it holds normals.
    Slot(usize, bool),
    Depth,
}

pub(crate) fn parse(value: &str) -> Option<View> {
    match value.trim().to_ascii_lowercase().as_str() {
        "albedo" | "color" | "colour" => Some(View::Slot(ALBEDO_SLOT, false)),
        "normals" | "normal" => Some(View::Slot(NORMALS_SLOT, true)),
        "depth" => Some(View::Depth),
        other => other
            .strip_prefix("rt")
            .and_then(|n| n.parse().ok())
            .filter(|&n: &usize| n < 8)
            .map(|n| View::Slot(n, false)),
    }
}

fn view() -> Option<View> {
    static VIEW: OnceLock<Option<View>> = OnceLock::new();
    *VIEW.get_or_init(|| {
        let raw = std::env::var("VICTORY_DEBUG_VIEW").ok()?;
        let view = parse(&raw);
        match view {
            Some(view) => tracing::info!(?view, "debug view: the G-buffer is shown instead of the scene"),
            None => tracing::warn!(value = raw, "VICTORY_DEBUG_VIEW: not albedo, normals, depth or rt0-rt7"),
        }
        view
    })
}

/// Whether a debug view is on (cheap after the first call).
pub(crate) fn enabled() -> bool {
    view().is_some()
}

/// One texture of the pass, held (so it outlives the title's object) with
/// the format to sample it in.
#[derive(Clone)]
struct Held {
    copy: Arc<HostTexture>,
    format: DXGI_FORMAT,
}

/// Colour slots a render-target set has at most.
const SLOTS: usize = 8;

/// A G-buffer pass candidate. Fixed-size, so tracking allocates nothing.
struct Pass {
    /// The title's objects bound, colour then depth (0: unused), to tell a
    /// binding of the same set apart cheaply.
    key: [usize; SLOTS + 1],
    colors: [Option<Held>; SLOTS],
    depth: Held,
    area: u64,
    /// The presented frame it was last bound before.
    frame: u64,
}

impl Pass {
    fn count(&self) -> usize {
        self.colors.iter().flatten().count()
    }
}

struct State {
    /// The best pass bound since the last present.
    building: Option<Pass>,
    /// The pass shown.
    shown: Option<Pass>,
    /// Frames presented.
    presented: u64,
    /// What was last logged as shown (texture, format), to log changes.
    logged: Option<(usize, i32)>,
}

static STATE: Mutex<State> = Mutex::new(State {
    building: None,
    shown: None,
    presented: 0,
    logged: None,
});

/// The first presented frame that showed a G-buffer texture (0: none yet).
static FIRST_SHOWN: AtomicU64 = AtomicU64::new(0);

/// Diagnostics captures: every 64th presented frame from the first one the
/// debug view showed, so a run saves the view without listing frames.
#[cfg(feature = "diagnostics")]
pub(crate) fn capture_wanted(frame: u64) -> bool {
    let first = FIRST_SHOWN.load(Relaxed);
    first != 0 && frame >= first && (frame - first) % 64 == 0
}

/// The format to sample a host texture in: a depth's red plane, a typed
/// member of a colour family.
fn sampled(texture: &ID3D12Resource, title: Option<u32>) -> DXGI_FORMAT {
    // SAFETY: a live resource.
    let host = unsafe { texture.GetDesc() }.Format.0 as u32;
    let depth = match host {
        19..=22 => Some(21),
        39..=41 => Some(41),
        44..=47 => Some(46),
        53 | 55 | 56 => Some(56),
        _ => None,
    };
    depth.map_or_else(
        || crate::compose::sampled_format(title.unwrap_or(host)),
        |f| DXGI_FORMAT(f as i32),
    )
}

fn hold(this: This) -> Option<(Held, (u64, u32))> {
    let copy = crate::resource::host_copy(this)?;
    // SAFETY: the Arc keeps the texture live.
    let desc = unsafe { copy.texture.GetDesc() };
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || desc.SampleDesc.Count != 1
        || desc.Flags.contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
    {
        return None;
    }
    let format = sampled(&copy.texture, crate::resource::view_format(this));
    Some((Held { copy, format }, (desc.Width, desc.Height)))
}

/// A render-target set bound on a title list: `targets` and `depth` are
/// the title's resources (0: none).
pub(crate) fn note_targets(targets: &[usize], depth: usize) {
    if depth == 0 || targets.len() < MIN_TARGETS || targets.len() > SLOTS || !enabled() {
        return;
    }
    let mut key = [0; SLOTS + 1];
    key[..targets.len()].copy_from_slice(targets);
    key[SLOTS] = depth;
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let frame = state.presented;
    if let Some(pass) = state.building.as_mut()
        && pass.key == key
    {
        pass.frame = frame;
        return;
    }
    // A set with fewer targets than the best one cannot replace it.
    if state
        .building
        .as_ref()
        .is_some_and(|best| best.key[..SLOTS].iter().filter(|&&t| t != 0).count() > targets.len())
    {
        return;
    }
    let Some((depth_held, size)) = hold(depth as This) else {
        return;
    };
    let mut colors: [Option<Held>; SLOTS] = Default::default();
    for (slot, &t) in colors.iter_mut().zip(targets) {
        *slot = hold(t as This).filter(|(_, s)| *s == size).map(|(h, _)| h);
    }
    let pass = Pass {
        key,
        colors,
        depth: depth_held,
        area: size.0 * u64::from(size.1),
        frame,
    };
    if pass.count() < MIN_TARGETS {
        return;
    }
    let better = state
        .building
        .as_ref()
        .is_none_or(|best| (pass.count(), pass.area) > (best.count(), best.area));
    if better {
        state.building = Some(pass);
    }
}

/// What to show for presented frame `frame`, if a debug view is on and a
/// G-buffer pass was bound lately: the texture, the format to sample it
/// in, and how to colour it.
pub(crate) fn take(frame: u64) -> Option<(Arc<HostTexture>, DXGI_FORMAT, Mode)> {
    let view = view()?;
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    state.presented = frame;
    if let Some(pass) = state.building.take() {
        state.shown = Some(pass);
    }
    let pass = state.shown.as_ref()?;
    if frame.saturating_sub(pass.frame) > HOLD_FRAMES {
        return None;
    }
    let (held, mode) = match view {
        View::Depth => (pass.depth.clone(), Mode::Depth),
        View::Slot(n, normals) => {
            let held = pass.colors.get(n)?.clone()?;
            // Unsigned normalized normals are stored mapped already.
            let signed = normals
                && matches!(held.format.0, 2 | 10 | 13 | 16 | 26 | 31 | 34 | 37 | 41 | 54);
            (held, if signed { Mode::Signed } else { Mode::Raw })
        }
    };
    let key = (Arc::as_ptr(&held.copy) as usize, held.format.0);
    if state.logged != Some(key) {
        // SAFETY: a live resource.
        let desc = unsafe { held.copy.texture.GetDesc() };
        // The pass's sampled formats by slot (0: no usable texture), then
        // its depth's, to tell the slots apart.
        let mut formats = [0; SLOTS];
        for (f, c) in formats.iter_mut().zip(&pass.colors) {
            *f = c.as_ref().map_or(0, |h| h.format.0);
        }
        tracing::info!(
            frame,
            ?view,
            ?mode,
            width = desc.Width,
            height = desc.Height,
            format = held.format.0,
            targets = pass.count(),
            ?formats,
            depth_format = pass.depth.format.0,
            "debug view: showing the G-buffer pass's texture"
        );
        state.logged = Some(key);
        let _ = FIRST_SHOWN.compare_exchange(0, frame, Relaxed, Relaxed);
    }
    Some((held.copy, held.format, mode))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_parse() {
        assert_eq!(parse("albedo"), Some(View::Slot(ALBEDO_SLOT, false)));
        assert_eq!(parse(" Normals"), Some(View::Slot(NORMALS_SLOT, true)));
        assert_eq!(parse("depth"), Some(View::Depth));
        assert_eq!(parse("rt4"), Some(View::Slot(4, false)));
        assert_eq!(parse("rt8"), None);
        assert_eq!(parse("lit"), None);
    }
}

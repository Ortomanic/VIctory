//! `VICTORY_DEBUG_VIEW=albedo|normals|depth|cycle|rt<N>`: the window shows
//! the title's G-buffer instead of the lit, tone-mapped scene (plane 0),
//! with its UI plane still blended over it.
//!
//! The G-buffer is found by its pass: the render-target set bound with a
//! depth target that has the most colour targets of the depth's size, a
//! landscape one before a square one, then the largest (GTA
//! VI's main pass binds six at 1280x720 with its R32G8X24 depth: slot 0
//! R8G8B8A8 typeless, flat albedo; slot 1 R8G8B8A8, packed normals (shown
//! as stored, in bands); slots 2, 3, 5 R8G8B8A8; slot 4 R16G16 float). The
//! chosen view is one of that set's colour slots, or its depth. Present samples it as it stands after the frame's
//! submissions, so the view costs one fullscreen draw and no copies; the
//! tracking costs one compare per multi-target binding.
//!
//! When no such pass was bound for a while (menus, loading, movies), the
//! normal composition comes back.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering::Relaxed},
};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

use crate::object::This;
use crate::resource::HostTexture;

/// GTA VI's main G-buffer pass slots, told apart with `cycle` (see
/// `reports/unlit-view.md`): peds come out in skin and cloth colours in
/// slot 0, in bands of a packed encoding in slot 1.
const ALBEDO_SLOT: usize = 0;
const NORMALS_SLOT: usize = 1;
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
    /// Each colour slot, then the depth, for [`CYCLE_FRAMES`] presented
    /// frames each, round and round (to tell the slots apart).
    Cycle,
}

/// Presented frames each texture is shown for by [`View::Cycle`].
const CYCLE_FRAMES: u64 = 16;

pub(crate) fn parse(value: &str) -> Option<View> {
    match value.trim().to_ascii_lowercase().as_str() {
        "albedo" | "color" | "colour" => Some(View::Slot(ALBEDO_SLOT, false)),
        "normals" | "normal" => Some(View::Slot(NORMALS_SLOT, true)),
        "depth" => Some(View::Depth),
        "cycle" => Some(View::Cycle),
        other => other
            .strip_prefix("rt")
            .and_then(|n| n.parse().ok())
            .filter(|&n: &usize| n < 8)
            .map(|n| View::Slot(n, false)),
    }
}

fn view() -> Option<View> {
    decode(VIEW.get(|| {
        let Ok(raw) = std::env::var("VICTORY_DEBUG_VIEW") else {
            return 0;
        };
        let view = parse(&raw);
        match view {
            Some(view) => tracing::info!(?view, "debug view: the G-buffer is shown instead of the scene"),
            None => tracing::warn!(
                value = raw,
                "VICTORY_DEBUG_VIEW: not albedo, normals, depth, cycle or rt0-rt7"
            ),
        }
        encode(view)
    }))
}

/// The view, as [`encode`] gives it (read per binding: an atomic).
static VIEW: crate::live::Number = crate::live::Number::new();

/// 0 none, 1 depth, 2 cycle, 3 + 2 * slot + normals.
fn encode(view: Option<View>) -> u64 {
    match view {
        None => 0,
        Some(View::Depth) => 1,
        Some(View::Cycle) => 2,
        Some(View::Slot(n, normals)) => 3 + 2 * n as u64 + u64::from(normals),
    }
}

fn decode(v: u64) -> Option<View> {
    match v {
        0 => None,
        1 => Some(View::Depth),
        2 => Some(View::Cycle),
        v => Some(View::Slot(((v - 3) / 2) as usize, (v - 3) % 2 == 1)),
    }
}

/// Live controls: `VICTORY_DEBUG_VIEW` from now on (`None`: off); the pass
/// tracked so far is let go (it holds the title's textures).
#[cfg(feature = "diagnostics")]
pub(crate) fn set(value: Option<&str>) -> Result<String, String> {
    let view = match value {
        Some(v) => Some(parse(v).ok_or("not albedo, normals, depth, cycle or rt0-rt7")?),
        None => None,
    };
    VIEW.set(encode(view));
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    state.building = None;
    state.shown = None;
    state.logged = None;
    FIRST_SHOWN.store(0, Relaxed);
    Ok(String::new())
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
    /// Wider than tall, then the area: the main view beats square passes
    /// of the same targets (GTA VI also binds six at 1024x1024 some frames).
    rank: (bool, u64),
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
/// debug view showed (with [`View::Cycle`], the first frame of each
/// texture), so a run saves the view without listing frames.
#[cfg(feature = "diagnostics")]
pub(crate) fn capture_wanted(frame: u64) -> bool {
    let first = FIRST_SHOWN.load(Relaxed);
    let every = if view() == Some(View::Cycle) { CYCLE_FRAMES } else { 64 };
    first != 0 && frame >= first && (frame - first) % every == 0
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
        rank: (size.0 > u64::from(size.1), size.0 * u64::from(size.1)),
        frame,
    };
    if pass.count() < MIN_TARGETS {
        return;
    }
    let better = state
        .building
        .as_ref()
        .is_none_or(|best| (pass.count(), pass.rank) > (best.count(), best.rank));
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
    let first = match FIRST_SHOWN.compare_exchange(0, frame, Relaxed, Relaxed) {
        Ok(_) => frame,
        Err(first) => first,
    };
    let view = match view {
        View::Cycle => {
            let used = pass.count() as u64;
            let k = ((frame - first) / CYCLE_FRAMES % (used + 1)) as usize;
            pass.colors
                .iter()
                .enumerate()
                .filter(|(_, c)| c.is_some())
                .nth(k)
                .map_or(View::Depth, |(n, _)| View::Slot(n, false))
        }
        view => view,
    };
    let (held, mode) = match view {
        View::Depth | View::Cycle => (pass.depth.clone(), Mode::Depth),
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
        assert_eq!(parse("cycle"), Some(View::Cycle));
        assert_eq!(parse("rt4"), Some(View::Slot(4, false)));
        assert_eq!(parse("rt8"), None);
        assert_eq!(parse("lit"), None);
    }

    #[test]
    fn views_round_trip_their_encoding() {
        for v in ["albedo", "normals", "depth", "cycle", "rt0", "rt4", "rt7"] {
            let view = parse(v);
            assert_eq!(decode(encode(view)), view, "{v}");
        }
        assert_eq!(decode(encode(None)), None);
    }
}

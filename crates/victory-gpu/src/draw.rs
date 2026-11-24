//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    pub(crate) address: u64,
    pub(crate) size: u32,
    pub(crate) stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    #[cfg(feature = "diagnostics")]
    target_views: Vec<Option<D3D12_RENDER_TARGET_VIEW_DESC>>,
    depth: usize,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, what it holds) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count, owner
    /// token, and whether it holds them now; lent back when a run of the
    /// list has finished, given back when the list is reset or released.
    dynamic: Vec<(usize, u32, u32, u32, bool)>,
    /// Runs of the list since it was recorded.
    runs: u32,
    /// Whether its dynamic slots are lent back (it has run).
    lent: bool,
    /// Another list took some of its lent slots: its tables are gone, and
    /// it cannot run again until it is recorded again.
    lost: bool,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n, token, held) in &self.dynamic {
            crate::descriptor::with_host_heap(heap, |h| h.release(first, n, token, held));
        }
    }
}

/// Dynamic slots lent back by lists that ran, lent runs taken back for
/// another run, and runs of lists not run because other lists took their
/// lent slots, since the last call (diagnostics).
static LENT: AtomicU64 = AtomicU64::new(0);
static RECLAIMED: AtomicU64 = AtomicU64::new(0);
static LOST_RUNS: AtomicU64 = AtomicU64::new(0);
/// Tables made afresh found made by another list (shared), since the
/// last call.
static SHARED_HITS: AtomicU64 = AtomicU64::new(0);
/// Tables made afresh: shared (no constant buffers) and their slots,
/// with constant buffers (the list's own) and their slots.
static MADE_SHARED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
static MADE_OWN: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Title submissions under way (from before their lists' slots are taken
/// back to after they are lent back), and lends since the start, with a
/// condition variable for recording threads waiting for slots.
pub(crate) static SUBMITTING: AtomicU64 = AtomicU64::new(0);
static LENDS: (std::sync::Mutex<u64>, std::sync::Condvar) =
    (std::sync::Mutex::new(0), std::sync::Condvar::new());
/// The start of the current second and the microseconds recording threads
/// waited in it for dynamic slots to be lent back.
static SLOT_WAITS: std::sync::Mutex<Option<(std::time::Instant, u64)>> = std::sync::Mutex::new(None);
/// Of each second (about a world frame), the most recording threads
/// together wait for slots.
const SLOT_WAIT_BUDGET_US: u64 = 50_000;
/// Waits for dynamic slots to be lent back (another submission finished),
/// within the frame's budget; false if none was or the budget is spent.
static SLOT_WAITED: AtomicU64 = AtomicU64::new(0);
static SLOT_WAIT_US: AtomicU64 = AtomicU64::new(0);

/// Slots were lent back: waiting recording threads try again.
fn lent_back() {
    let (count, lent) = &LENDS;
    *count.lock().unwrap_or_else(|e| e.into_inner()) += 1;
    lent.notify_all();
}

/// No dynamic slots were free: waits (within the frame's budget) until a
/// title submission under way finishes and lends its slots back. True if
/// some were lent back meanwhile.
fn wait_for_lent_slots() -> bool {
    if SUBMITTING.load(Ordering::Acquire) == 0 {
        return false;
    }
    let left = {
        let mut waits = SLOT_WAITS.lock().unwrap_or_else(|e| e.into_inner());
        let window = waits.get_or_insert((std::time::Instant::now(), 0));
        if window.0.elapsed() > std::time::Duration::from_secs(1) {
            *window = (std::time::Instant::now(), 0);
        }
        SLOT_WAIT_BUDGET_US.saturating_sub(window.1)
    };
    if left == 0 {
        return false;
    }
    let started = std::time::Instant::now();
    let (count, lent) = &LENDS;
    let guard = count.lock().unwrap_or_else(|e| e.into_inner());
    let before = *guard;
    let (guard, _) = lent
        .wait_timeout_while(guard, std::time::Duration::from_micros(left), |n| *n == before)
        .unwrap_or_else(|e| e.into_inner());
    let got = *guard != before;
    drop(guard);
    let us = started.elapsed().as_micros() as u64;
    if let Some(window) = SLOT_WAITS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        window.1 += us;
    }
    SLOT_WAITED.fetch_add(1, Ordering::Relaxed);
    SLOT_WAIT_US.fetch_add(us, Ordering::Relaxed);
    got
}

/// `VICTORY_LEND_SLOTS=0`: a list keeps its dynamic slots until it is
/// reset (the behaviour before: the world ran out of them within its first
/// frame, and the draws needing more were dropped).
fn lend_slots() -> bool {
    static LEND: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LEND.get_or_init(|| std::env::var("VICTORY_LEND_SLOTS").map_or(true, |v| v != "0"))
}

/// `VICTORY_SHARE_TABLES=0`: tables made afresh are not shared between
/// lists (each list makes its own, as before).
fn share_tables() -> bool {
    static SHARE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SHARE.get_or_init(|| std::env::var("VICTORY_SHARE_TABLES").map_or(true, |v| v != "0"))
}

/// Since the last call: dynamic slots lent back, lent runs reclaimed,
/// lists not run (their slots were taken), and draws and dispatches
/// dropped because a table to make afresh could not be made.
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn take_slot_stats() -> [u64; 11] {
    [
        SLOT_WAITED.swap(0, Ordering::Relaxed),
        SLOT_WAIT_US.swap(0, Ordering::Relaxed),
        MADE_SHARED[0].swap(0, Ordering::Relaxed),
        MADE_SHARED[1].swap(0, Ordering::Relaxed),
        MADE_OWN[0].swap(0, Ordering::Relaxed),
        MADE_OWN[1].swap(0, Ordering::Relaxed),
        SHARED_HITS.swap(0, Ordering::Relaxed),
        LENT.swap(0, Ordering::Relaxed),
        RECLAIMED.swap(0, Ordering::Relaxed),
        LOST_RUNS.swap(0, Ordering::Relaxed),
        UNMADE_SINCE.swap(0, Ordering::Relaxed),
    ]
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory. Also the bytes there translated shaders may read
/// (their root view bounds, `victory_shader::emit::Options::root_bounds`):
/// to the copy's end for SRVs, the constant buffer's 64 KiB for CBVs (both
/// cut at the buffer's end, CBVs rounded up to whole 16-byte rows).
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB,
/// other views as 1 MiB (both cut at their buffer's end) for refreshes.
/// Largest root UAV range refreshed, kept and copied back per submission.
const ROOT_UAV_MAX: u64 = 4 << 20;

fn root_view(list: &HostList, address: u64, srv: bool, uav: bool) -> Option<(u64, u32)> {
    let shadow = crate::shadow::find(address, 1)?;
    let left = shadow.left(address);
    if uav {
        // Written by shaders: refreshed, then copied back after the
        // submission, up to a cap (the snapshot that finds what the GPU wrote
        // costs per byte and submission). The shader drops writes and zeroes
        // reads past the length it is given, so nothing outside is touched.
        let len = left.min(ROOT_UAV_MAX);
        if left > ROOT_UAV_MAX {
            static CUT: std::sync::Once = std::sync::Once::new();
            CUT.call_once(|| tracing::warn!(left, "a root UAV cut to its refresh and copy-back cap"));
        }
        shadow.refresh(list, address, len);
        shadow.gpu_writes(list, address, len);
        return Some((shadow.gpu(address), u32::try_from(len).unwrap_or(u32::MAX)));
    }
    let len = left.min(if srv { 1 << 20 } else { 1 << 16 });
    let bounds = if srv { left } else { len.next_multiple_of(16) };
    let bounds = u32::try_from(bounds).unwrap_or(u32::MAX);
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        return Some((gpu, bounds));
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some((gpu, bounds))
}

/// Times the rest of the scope into a recording counter (diagnostics).
macro_rules! rec_timed {
    ($c:ident) => {
        #[cfg(feature = "diagnostics")]
        let _rec_timed = crate::copystat::timed(crate::copystat::C::$c);
    };
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// Draws and dispatches dropped because a table to make afresh could not
/// be made (no free dynamic slots): all, and those past the host heap.
static UNMADE: AtomicU64 = AtomicU64::new(0);
static UNMADE_PAST_HEAP: AtomicU64 = AtomicU64::new(0);
static UNMADE_SINCE: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_KEEP_UNMADE_TABLES=1`: run such draws anyway, reading the
/// title's slots (the behaviour before; hangs the GPU within minutes).
fn keep_unmade() -> bool {
    static KEEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KEEP.get_or_init(|| std::env::var("VICTORY_KEEP_UNMADE_TABLES").is_ok_and(|v| v == "1"))
}

fn unmade_table(past_heap: bool) {
    UNMADE_SINCE.fetch_add(1, Ordering::Relaxed);
    let n = UNMADE.fetch_add(1, Ordering::Relaxed) + 1;
    let past = UNMADE_PAST_HEAP.fetch_add(u64::from(past_heap), Ordering::Relaxed)
        + u64::from(past_heap);
    if n.is_power_of_two() {
        tracing::warn!(
            n,
            past,
            "a draw or dispatch dropped: a table to make afresh could not be made"
        );
    }
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// Before the list runs: its lent dynamic slots taken back. False if
    /// another list took some of them (the list's tables are gone): the
    /// list must not run on the GPU.
    pub(crate) fn before_run(&mut self) -> bool {
        if self.lent {
            self.lent = false;
            for (heap, first, n, token, held) in &mut self.dynamic {
                if *held {
                    continue;
                }
                let (first, n, token) = (*first, *n, *token);
                *held = crate::descriptor::with_host_heap(*heap, |h| h.reclaim(first, n, token))
                    == Some(true);
                if !*held {
                    self.lost = true;
                }
            }
            RECLAIMED.fetch_add(1, Ordering::Relaxed);
        }
        if self.lost {
            let n = LOST_RUNS.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::warn!(n, "a list run again lost its dynamic descriptor slots: not run on the GPU");
            }
        }
        !self.lost
    }

    /// After a run of the list finished on the GPU: its dynamic slots lent
    /// back, unless it has run before since it was recorded (a list the
    /// title runs again and again keeps them).
    pub(crate) fn after_run(&mut self) {
        self.runs += 1;
        if self.runs > 1 || self.lent || self.lost || !lend_slots() {
            return;
        }
        let mut lent = 0;
        for (heap, first, n, token, held) in &mut self.dynamic {
            if std::mem::replace(held, false) {
                let (first, n, token) = (*first, *n, *token);
                lent += crate::descriptor::with_host_heap(*heap, |h| h.lend(first, n, token)).unwrap_or(0);
            }
        }
        LENT.fetch_add(u64::from(lent), Ordering::Relaxed);
        self.lent = true;
        if lent > 0 {
            lent_back();
        }
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    /// Our own commands replaced the host list's heaps, root signatures
    /// and pipeline: the title's are set again at its next draw or
    /// dispatch.
    /// `n` dynamic slots of the title's bound view heap, held until the
    /// list is reset, with that heap bound on the host list: the heap and
    /// the first slot. For passes of the host's own in the title's list,
    /// which then call [`State::compute_bindings_lost`].
    pub(crate) fn own_slots(
        &mut self,
        list: &HostList,
        n: u32,
    ) -> Option<(&'static crate::descriptor::host::HostHeap, u32)> {
        let (title, host) = self.heaps.iter().filter(|&&h| h != 0).find_map(|&h| {
            let heap: &'static crate::descriptor::Heap = crate::descriptor::heap(h as This);
            (!heap.samplers()).then(|| heap.host()).flatten().map(|host| (h, host))
        })?;
        let (first, token) = host.dynamic(n)?;
        self.dynamic.push((title, first, n, token, true));
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { list.list.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        Some((host, first))
    }

    /// The host list's root signatures and pipeline were replaced by a pass
    /// of the host's own; its descriptor heaps were not.
    pub(crate) fn compute_bindings_lost(&mut self) {
        self.host_roots = [0; 2];
        self.host_pipeline = 0;
    }

    pub(crate) fn host_bindings_lost(&mut self) {
        self.host_heaps = [usize::MAX; 2];
        self.host_roots = [0; 2];
        self.host_pipeline = 0;
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        #[cfg(feature = "diagnostics")]
        self.target_views.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            self.targets.push(record.resource as usize);
            let desc = record.desc();
            #[cfg(feature = "diagnostics")]
            {
                if crate::scene_color::enabled() {
                    self.target_views.push(desc);
                }
                crate::capture::note_target(record.resource);
            }
            rtvs.push(host.rtv(texture, desc.as_ref()));
        }
        self.depth = 0;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            #[cfg(feature = "diagnostics")]
            {
                crate::capture::note_target(record.resource);
                crate::capture::note_scene_depth(record.resource);
            }
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return false;
        }
        if !self.bind(list, p, false) {
            return false;
        }
        rec_timed!(RecIaUs);
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            crate::list::ensure_state(list, self.depth as This, DEPTH_WRITE);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False when a stage's tables cannot be bound safely (see
    /// [`State::tables`]): the draw or dispatch is dropped.
    fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        rec_timed!(RecBindUs);
        let l = &list.list;
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        // Shaders that index the heap from data: the submission brings every
        // buffer view's range up to date (`descriptor::bindless`).
        let bindless = p.stages.iter().fold(0, |b, s| b | s.translated().host.bindless);
        if bindless != 0 {
            list.note_bindless(bindless);
        }
        for stage in &p.stages {
            let t = stage.translated();
            let made = {
                rec_timed!(RecTablesUs);
                self.tables(list, &t.used, compute)
            };
            let Some(made) = made else {
                return false;
            };
            #[cfg(feature = "diagnostics")]
            let views_started = std::time::Instant::now();
            // Root views first: the user data holds their lengths.
            let mut lengths: Vec<(u8, u32)> = Vec::new();
            for (views, srv, uav) in [
                (&stage.root_cbvs, false, false),
                (&stage.root_srvs, true, false),
                (&stage.root_uavs, false, true),
            ] {
                for &(slot, param) in views {
                    let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        continue;
                    };
                    let Some((gpu, len)) = root_view(list, *address, srv, uav) else {
                        continue;
                    };
                    lengths.push((slot, len));
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv, uav) {
                            (false, _, true) => l.SetGraphicsRootUnorderedAccessView(param, gpu),
                            (true, _, true) => l.SetComputeRootUnorderedAccessView(param, gpu),
                            (false, false, _) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true, _) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false, _) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true, _) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::since(crate::copystat::C::RecRootViewsUs, views_started);
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        // A root view's length (0, reading nothing, when
                        // none is bound).
                        None if u.root == RootType::Descriptor => {
                            lengths.iter().find(|v| v.0 == u.slot).map_or(0, |v| v.1)
                        }
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
        }
        true
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        #[cfg(feature = "diagnostics")]
        let _recording = crate::copystat::recording();
        #[cfg(feature = "diagnostics")]
        self.pass_log("dispatch", x * y * z, true);
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            // Root view lengths show with the root views' host addresses.
            for u in t
                .host
                .user_data
                .iter()
                .filter(|u| u.root != RootType::Descriptor)
            {
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs).chain(&stage.root_uavs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        let _recording = crate::copystat::recording();
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    #[cfg(feature = "diagnostics")]
                    crate::scene_color::note_draw(self.depth, &self.targets, &self.target_views);
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Whether the bound pipeline completes rectangles in a geometry stage.
    fn rect_gs(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rect_gs)
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        let view = {
            rec_timed!(RecMirrorUs);
            host.mirror(index, handle, true)
        };
        if let Some(view) = view {
            readable(list, u, &view);
        }
    }

    /// The descriptors a shader reads through tables. Tables reaching past
    /// the title slots the host mirrors are made afresh for the draw in
    /// dynamic slots. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    /// Constant buffers in tables are raw views of their host copy, read
    /// as raw buffers by translated shaders and copied at submission as
    /// other buffer views are (they were made afresh per draw with their
    /// data copied, the host wanting constant buffer views 256-byte
    /// aligned where the console wants 16: ~50,000 dynamic slots per world
    /// frame, and the world's first frames dropped up to ~1,500 draws a
    /// frame for want of slots).
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once, and lists share such tables (see `HostHeap::acquire`).
    ///
    /// `None` when a table to make afresh could not be made (no free
    /// dynamic slots): the shader would index the title's slots instead,
    /// which the host does not have (past the host heap: the host's
    /// dynamic slots, or memory after the heap). Constants read from there bound the world's loops
    /// (CS_Visibility hung the GPU, Xid 109); descriptors read from there
    /// fault (Xid 13). The draw must not run.
    fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        compute: bool,
    ) -> Option<Vec<(u8, u32)>> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        // Constant buffers in tables are raw views of their host copy (made
        // from the title's slots like other views), not made afresh.
        let mut fresh: Vec<u8> = Vec::new();
        // Tables without host slots of their own (for the count).
        let mut past: Vec<u8> = Vec::new();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
                past.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8 + 1)
                .max()
                .unwrap_or(1);
            // The table and what its entries hold (record words).
            #[cfg(feature = "diagnostics")]
            let keyed = std::time::Instant::now();
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, h.finish())
            };
            #[cfg(feature = "diagnostics")]
            crate::copystat::since(crate::copystat::C::RecKeyUs, keyed);
            if let Some(&first) = self.fresh_tables.get(&key) {
                rec_timed!(RecHitUs);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::RecTableHits, 1);
                for u in &entries {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            // Made by another list (a table without constant buffers:
            // what it holds comes from the title's records alone).
            let shareable = share_tables();
            let shared_key = (key.0, key.1, n);
            if shareable && let Some((first, n, token)) = host.acquire(&shared_key) {
                SHARED_HITS.fetch_add(1, Ordering::Relaxed);
                self.fresh_tables.insert(key, first);
                self.dynamic
                    .push((std::ptr::from_ref(heap) as usize, first, n, token, true));
                for u in &entries {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let first = {
                rec_timed!(RecDynamicUs);
                let mut first = host.dynamic(n);
                // Out of slots: a submission under way gives its back when
                // it finishes (the world's first frames take more than
                // there are, held by lists recorded and not yet run).
                while first.is_none() && wait_for_lent_slots() {
                    first = host.dynamic(n);
                }
                first
            };
            let Some((first, token)) = first else {
                unmade_table(past.contains(&slot));
                if keep_unmade() {
                    continue;
                }
                return None;
            };
            #[cfg(feature = "diagnostics")]
            let making = {
                use crate::copystat::{C, add};
                add(C::RecTableMisses, 1);
                add(C::RecEntriesMade, u64::from(n));
                crate::copystat::timed(C::RecMakeUs)
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n, token, true));
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(view) => {
                        host.make_at(index, &view);
                        if let Some(u) = used {
                            readable(list, u, &view);
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            #[cfg(feature = "diagnostics")]
            drop(making);
            let counts = if shareable {
                host.share(shared_key, first, n, token);
                &MADE_SHARED
            } else {
                &MADE_OWN
            };
            counts[0].fetch_add(1, Ordering::Relaxed);
            counts[1].fetch_add(u64::from(n), Ordering::Relaxed);
            made.push((slot, first * 32));
        }
        Some(made)
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        self.pass_log(kind, count, false);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// Frames in `VICTORY_CAPTURE_FRAMES` with `VICTORY_PASS_LOG=1`: every
    /// draw and dispatch at info level, with its shaders (and why it is
    /// dropped), its targets and what it reads, to follow a frame's passes
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn pass_log(&self, kind: &str, count: u32, compute: bool) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_PASS_LOG").is_some())
            || !crate::capture::listed(crate::queue::presented() + 1)
        {
            return;
        }
        let texture = |r: usize| {
            crate::resource::host_resource(r as This).map_or_else(
                || format!("{r:#x} no host"),
                |t| {
                    // SAFETY: a live resource.
                    let d = unsafe { t.GetDesc() };
                    format!("{r:#x} {}x{} f{}", d.Width, d.Height, d.Format.0)
                },
            )
        };
        let targets: Vec<String> = if compute {
            Vec::new()
        } else {
            self.targets.iter().map(|&t| texture(t)).collect()
        };
        let depth = (!compute && self.depth != 0).then(|| texture(self.depth));
        let reads = if crate::pipeline::host_pipeline(self.pipeline as This).is_some() {
            self.reads(compute)
        } else {
            Vec::new()
        };
        tracing::info!(
            "pass {kind} {count} [{}] targets {targets:?} depth {depth:?} reads {reads:?}",
            crate::pipeline::describe(self.pipeline as This)
        );
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        let _recording = crate::copystat::recording();
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        // A pipeline with a rectangle geometry stage takes the title's
        // vertices as they are.
        let rects = self.topology == RECTANGLES && !self.rect_gs();
        let ran = self.prepare(list, false) && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            #[cfg(feature = "diagnostics")]
            crate::scene_color::note_draw(self.depth, &self.targets, &self.target_views);
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        let _recording = crate::copystat::recording();
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        if self.topology == RECTANGLES {
            once!("indexed rectangles not drawn");
        }
        let ran = self.topology != RECTANGLES && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            #[cfg(feature = "diagnostics")]
            crate::scene_color::note_draw(self.depth, &self.targets, &self.target_views);
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        #[cfg(feature = "diagnostics")]
        let _recording = crate::copystat::recording();
        #[cfg(feature = "diagnostics")]
        self.pass_log(&format!("indirect{kind}"), max, kind == 2);
        // What the title's memory holds there now (what the GPU last wrote
        // was copied back after its submission): the first record and the
        // count (diagnostics).
        #[cfg(feature = "diagnostics")]
        if crate::capture::listed(crate::queue::presented() + 1)
            && std::env::var_os("VICTORY_PASS_LOG").is_some()
        {
            let words = |at: u64, n: usize| -> Option<Vec<u32>> {
                crate::texture::committed(at as *const u8, n * 4).then(|| {
                    // SAFETY: committed title memory.
                    (0..n).map(|k| unsafe { (at as *const u32).add(k).read_unaligned() }).collect()
                })
            };
            tracing::info!(
                "pass indirect args at {arguments:#x} stride {stride} max {max}: {:?} count at {:?}: {:?}",
                words(arguments, (stride as usize / 4).clamp(1, 5)),
                count.map(|c| format!("{c:#x}")),
                count.and_then(|c| words(c, 1))
            );
        }
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        if kind == INDEXED && self.topology == RECTANGLES {
            once!("indirect indexed rectangles not drawn");
            why("indexed rectangles not supported");
            return false;
        }
        if kind == DRAW && self.topology == RECTANGLES {
            let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
                #[cfg(feature = "diagnostics")]
                crate::pipeline::note_drop(self.pipeline as This);
                return false;
            };
            if p.compute || p.mesh {
                why(&format!(
                    "pipeline mismatch (compute {}, mesh {}, topology {})",
                    p.compute, p.mesh, self.topology
                ));
                return false;
            }
            let Some(signature) = host_signature(DRAW, 16) else {
                why("no host signature");
                return false;
            };
            if max == 0 {
                DRAWN.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            let prepared = match crate::rectangles::expand(
                list,
                p,
                &self.vertex,
                (&args, arguments, stride, max),
                counter.as_ref().map(|(s, at)| (s, *at)),
            ) {
                Ok(p) => p,
                Err(reason) => {
                    why(reason);
                    return false;
                }
            };
            self.host_pipeline = 0;
            self.host_roots[1] = 0;
            if !self.prepare(list, false) {
                why("prepare failed for indirect rectangles");
                return false;
            }
            prepared.before_draw(list);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| {
                self.op_text(
                    format!("indirect rect draw max {max} at {arguments:#x}"),
                    false,
                )
            });
            // SAFETY: an open list with everything set; `prepared.args_buf` is
            // list-owned (records past the title's count are empty draws, so
            // no count buffer), the signature cached.
            unsafe {
                list.list.ExecuteIndirect(
                    &signature,
                    max,
                    &prepared.args_buf,
                    0,
                    None::<&ID3D12Resource>,
                    0,
                )
            };
            prepared.after_draw(list);
            DRAWN.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        // Bounded first: the clamp binds a compute pipeline of its own.
        if !crate::indirect::bound(list, &args.buffer, args.gpu(arguments), (kind, stride, max)) {
            why("not bounded (no clamp, or arguments not 4-byte aligned)");
            return false;
        }
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        let ran = match kind {
            // Rectangles run only where a geometry stage completes them:
            // an indirect draw's vertex range is not known here to expand.
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || (kind == DRAW && self.rect_gs()))
                    && self.prepare(list, kind == INDEXED)
            }
            MESH => self.prepare_for(list, false, true),
            DISPATCH => match crate::pipeline::host_pipeline(self.pipeline as This) {
                Some(p) if p.compute => self.bind(list, p, true),
                _ => false,
            },
            _ => false,
        };
        if !ran {
            crate::indirect::done(list, &args.buffer);
            #[cfg(feature = "diagnostics")]
            match crate::pipeline::host_pipeline(self.pipeline as This) {
                None => crate::pipeline::note_drop(self.pipeline as This),
                Some(p) => why(&format!("prepare failed (compute {}, mesh {})", p.compute, p.mesh)),
            }
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        // SAFETY: an open list with everything set; the buffers are host
        // copies the list keeps alive (refresh), the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &args.buffer,
                args.offset(arguments),
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        #[cfg(feature = "diagnostics")]
        args_probe::take(list, &args.buffer, args.offset(arguments), kind, &self.targets, self.pipeline);
        crate::indirect::done(list, &args.buffer);
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            #[cfg(feature = "diagnostics")]
            crate::scene_color::note_draw(self.depth, &self.targets, &self.target_views);
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// With `VICTORY_PASS_LOG` (diagnostics): the first record of each indirect
/// execution of listed frames as the GPU read it (after the clamp), copied
/// from the host copy into readback memory in the title's list, and
/// summarized a few frames later per kind, G-buffer or not, zero or not.
#[cfg(feature = "diagnostics")]
mod args_probe {
    use super::*;

    const SLOTS: u64 = 16384;

    struct Probe {
        buffer: ID3D12Resource,
        frame: u64,
        notes: Vec<(u32, bool, String)>,
        done: bool,
    }
    // SAFETY: D3D12 objects are free-threaded.
    unsafe impl Send for Probe {}

    static PROBE: std::sync::Mutex<Option<Probe>> = std::sync::Mutex::new(None);

    fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_PASS_LOG").is_some())
    }

    pub(super) fn take(
        list: &HostList,
        buffer: &ID3D12Resource,
        offset: u64,
        kind: u32,
        targets: &[usize],
        pipeline: usize,
    ) {
        if !on() {
            return;
        }
        let frame = crate::queue::presented() + 1;
        let mut probe = PROBE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = probe.as_mut()
            && !p.done
            && frame > p.frame + 4
        {
            summarize(p);
            p.done = true;
        }
        if !crate::capture::listed(frame) {
            return;
        }
        if probe.is_none() {
            let Some(host) = crate::host::host() else { return };
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: SLOTS * 16,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            let ok = unsafe {
                host.device.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut made,
                )
            };
            let (Ok(()), Some(buffer)) = (ok, made) else { return };
            *probe = Some(Probe { buffer, frame, notes: Vec::new(), done: false });
        }
        let Some(p) = probe.as_mut() else { return };
        if p.done || p.frame != frame || p.notes.len() as u64 >= SLOTS {
            return;
        }
        let at = p.notes.len() as u64 * 16;
        let gbuffer = targets.len() >= 5;
        p.notes.push((kind, gbuffer, crate::pipeline::describe(pipeline as This)));
        list.keep(&p.buffer);
        // SAFETY: an open list; the arguments buffer is in the indirect
        // argument state here (after the clamp), our readback buffer in
        // copy-dest; both live.
        unsafe {
            list.list.ResourceBarrier(&[crate::host::transition(
                buffer,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyBufferRegion(&p.buffer, at, buffer, offset, 12);
            list.list.ResourceBarrier(&[crate::host::transition(
                buffer,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
    }

    fn summarize(p: &Probe) {
        let mut data = std::ptr::null_mut();
        // SAFETY: readback memory its copies wrote frames ago.
        unsafe {
            if p.buffer.Map(0, None, Some(&mut data)).is_err() {
                return;
            }
        }
        let mut counts: std::collections::BTreeMap<(u32, bool, bool), u32> = Default::default();
        let mut groups: std::collections::BTreeMap<String, (u32, u32, u64)> = Default::default();
        for (k, (kind, gbuffer, name)) in p.notes.iter().enumerate() {
            // SAFETY: inside the mapped buffer.
            let w: [u32; 3] = unsafe { (data.cast::<u8>().add(k * 16) as *const [u32; 3]).read_unaligned() };
            let work = match kind {
                0 | 1 => u64::from(w[0]) * u64::from(w[1]),
                _ => u64::from(w[0]) * u64::from(w[1]) * u64::from(w[2]),
            };
            *counts.entry((*kind, *gbuffer, work != 0)).or_default() += 1;
            if *gbuffer {
                let g = groups.entry(name.clone()).or_default();
                g.0 += 1;
                g.1 += u32::from(work != 0);
                g.2 += work;
            }
        }
        tracing::info!(frame = p.frame, records = p.notes.len(), ?counts, "args probe: (kind, G-buffer, nonzero) -> executions");
        for (name, (n, nonzero, work)) in groups {
            tracing::info!(frame = p.frame, "args probe G-buffer {name}: {n} executions, {nonzero} nonzero, {work} groups/vertices");
        }
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn readable(list: &HostList, u: &UsedResource, view: &View) {
    rec_timed!(RecReadableUs);
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((address, len)) = crate::descriptor::host::buffer_range(view)
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        shadow.refresh(list, address, len);
        if u.kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            crate::list::ensure_state(list, *resource as This, SHADER_RESOURCE);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{r:?} {}x{} f{} flags {:#x}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || format!("uav buffer {:?}", crate::descriptor::host::buffer_range(v)),
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

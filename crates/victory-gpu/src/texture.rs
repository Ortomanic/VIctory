//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// The title's format, and its texel bytes and the host's, when the host
    /// texture is made in a stand-in format (`resource::widening`), so
    /// detiled texels expand into the staging buffer.
    widened_from: Option<(u32, usize, usize)>,
    last: Mutex<Last>,
}

/// What a texture's staging buffer holds: the hash of the memory detiled,
/// and (memory watched) the write-watch asking it was read after, with
/// room to read the memory into.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
    read: Vec<u8>,
    /// The memory was found not write-watched (views): it is hashed from
    /// then on without asking the kernel, as a buffer's copy stops asking
    /// (`Shadow::watched`). Hashing reads all of it, so this stays right
    /// if the memory is remapped as anything else.
    unwatched: bool,
}

impl Last {
    /// Whether `memory` (the texture's, as now) differs from what was last
    /// detiled; it is noted as detiled.
    fn changed(&mut self, memory: &[u8]) -> bool {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let hash = blake3::hash(memory);
        #[cfg(feature = "diagnostics")]
        tex_us(crate::copystat::C::TexHashUs, started);
        self.hash.replace(hash) != Some(hash)
    }
}

thread_local! {
    /// A subresource's rows before they go to the staging mapping, which
    /// is upload heap memory (write-combined, often across the bus): the
    /// detiling's small scattered writes land in cached memory, and the
    /// mapping gets whole rows in order. One per thread, not per texture,
    /// so it costs the largest subresource once.
    static DETILED: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    set.get_or_insert_with(Default::default)
        .insert(resource as usize);
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

/// Textures due together of at least this many bytes are refreshed on
/// several threads ([`refresh_all`]).
const PARALLEL_BYTES: u64 = 1 << 20;

/// Detiles each texture's memory into its staging buffer if it changed
/// since the last time (and is mapped at all): the textures a submission
/// samples between two title writes, on several threads when they are
/// big. One after another, their hashing (and querying) took most of the
/// world's texture time, inside the title's submissions.
///
/// From xcl 2cefc88 (1984742628): a big batch of title textures is
/// detiled on several threads.
pub(crate) fn refresh_all(due: Vec<Arc<Staging>>) {
    use std::sync::atomic::Ordering::Relaxed;
    if due.is_empty() {
        return;
    }
    let started = std::time::Instant::now();
    let bytes: u64 = due.iter().map(|s| s.layout.size).sum();
    let threads = if bytes < PARALLEL_BYTES { 1 } else { threads() };
    #[cfg(feature = "diagnostics")]
    {
        stat(Stat::Checks, due.len() as u64);
        stat(Stat::Batches, 1);
        stat(Stat::Parallel, u64::from(threads > 1 && due.len() > 1));
    }
    let tasks = due
        .into_iter()
        .map(|s| Box::new(move || s.refresh_now()) as Task)
        .collect();
    let read = run_tasks(tasks, threads);
    let preparing = &crate::host::PREPARING;
    preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    preparing[3].fetch_add(read as u64, Relaxed);
}

/// Threads a big batch of textures is refreshed on, the submitting one
/// among them: half the cores, at most 8 (`VICTORY_TEXTURE_THREADS`).
fn threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_THREADS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).clamp(1, 8))
            })
            .max(1)
    })
}

/// A texture's refresh, returning the bytes of title memory read.
type Task = Box<dyn Fn() -> usize + Send + Sync>;

/// Tasks shared by the submitting thread and the pool's.
struct Job {
    tasks: Vec<Task>,
    /// The next task to take.
    next: std::sync::atomic::AtomicUsize,
    /// Tasks not finished; the submitting thread waits for none.
    left: Mutex<usize>,
    finished: std::sync::Condvar,
    read: std::sync::atomic::AtomicUsize,
}

impl Job {
    /// Takes tasks until none are left to take.
    fn work(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        loop {
            let k = self.next.fetch_add(1, Relaxed);
            let Some(task) = self.tasks.get(k) else {
                return;
            };
            // A panicking task must not leave the submission waiting.
            let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task)).unwrap_or(0);
            self.read.fetch_add(read, Relaxed);
            let mut left = self.left.lock().unwrap_or_else(|e| e.into_inner());
            *left -= 1;
            if *left == 0 {
                self.finished.notify_all();
            }
        }
    }
}

/// Threads that help with [`Job`]s, made once.
struct Pool {
    jobs: Mutex<std::sync::mpsc::Sender<Arc<Job>>>,
    workers: usize,
}

fn pool() -> Option<&'static Pool> {
    static POOL: std::sync::OnceLock<Option<Pool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let (jobs, taken) = std::sync::mpsc::channel::<Arc<Job>>();
        let taken = Arc::new(Mutex::new(taken));
        let workers = (0..threads().saturating_sub(1))
            .filter(|i| {
                let taken = taken.clone();
                std::thread::Builder::new()
                    .name(format!("victory-texture-{i}"))
                    .spawn(move || {
                        loop {
                            let job = taken.lock().unwrap_or_else(|e| e.into_inner()).recv();
                            match job {
                                Ok(job) => job.work(),
                                Err(_) => return,
                            }
                        }
                    })
                    .is_ok()
            })
            .count();
        (workers > 0).then(|| Pool {
            jobs: Mutex::new(jobs),
            workers,
        })
    })
    .as_ref()
}

/// Runs `tasks` on up to `threads` threads, this one among them, and
/// returns the sum of what they return. Pool threads busy elsewhere leave
/// their share to the others.
fn run_tasks(tasks: Vec<Task>, threads: usize) -> usize {
    let pool = if threads > 1 && tasks.len() > 1 {
        pool()
    } else {
        None
    };
    let Some(pool) = pool else {
        return tasks.iter().map(|task| task()).sum();
    };
    let helpers = (threads - 1).min(tasks.len() - 1).min(pool.workers);
    let job = Arc::new(Job {
        left: Mutex::new(tasks.len()),
        tasks,
        next: Default::default(),
        finished: std::sync::Condvar::new(),
        read: Default::default(),
    });
    {
        let jobs = pool.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for _ in 0..helpers {
            let _ = jobs.send(job.clone());
        }
    }
    job.work();
    let mut left = job.left.lock().unwrap_or_else(|e| e.into_inner());
    while *left > 0 {
        left = job.finished.wait(left).unwrap_or_else(|e| e.into_inner());
    }
    job.read.load(std::sync::atomic::Ordering::Relaxed)
}

/// Texture counters (diagnostics), logged with the frame rate.
#[cfg(feature = "diagnostics")]
#[derive(Clone, Copy)]
enum Stat {
    /// Textures checked.
    Checks,
    /// Checks that asked `VirtualQuery`, and that knew the memory
    /// committed from the view regions kept (`watch::committed`).
    Queried,
    Known,
    /// Batches of due textures, and those run on several threads.
    Batches,
    Parallel,
}

#[cfg(feature = "diagnostics")]
static STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

#[cfg(feature = "diagnostics")]
fn stat(s: Stat, n: u64) {
    STATS[s as usize].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// Logs the texture counters per frame over `frames` frames.
#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64) {
    let [checks, queried, known, batches, parallel] =
        std::array::from_fn(|i| STATS[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames);
    tracing::info!(
        "textures per frame: {checks:.0} checked in {batches:.1} batches ({parallel:.1} on {} \
         threads); {queried:.0} asked VirtualQuery, {known:.0} known committed",
        threads()
    );
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all); returns the bytes of
    /// title memory read.
    fn refresh_now(&self) -> usize {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        let watched = if last.unwatched {
            None
        } else {
            let watched = crate::watch::writes(start, start + size, false);
            last.unwatched = watched.is_none();
            watched
        };
        if let Some((now, written)) = watched {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            last.read.resize(size, 0);
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
            #[cfg(feature = "diagnostics")]
            tex_us(crate::copystat::C::TexReadUs, started);
            // The watch is per page, and a title writes next to its
            // textures and writes again what they hold: the same bytes as
            // at the last detiling are not detiled again (from xcl 69f0635
            // by theaustinayers).
            let read = std::mem::take(&mut last.read);
            if last.changed(&read) {
                self.detile(&read);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            }
            last.read = read;
            return size;
        }
        // View regions found committed are known until the runtime remaps
        // them (`watch::committed`): asking `VirtualQuery` region by region
        // (a view per 64 KiB page in physical-page memory) took most of the
        // refreshes' time.
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let (is_committed, asked) = crate::watch::committed_asking(start, size);
        #[cfg(feature = "diagnostics")]
        {
            tex_us(crate::copystat::C::TexQueryUs, started);
            stat(if asked { Stat::Queried } else { Stat::Known }, 1);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = asked;
        if !is_committed {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return 0;
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if !last.changed(tiled) {
            return size;
        }
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer,
    /// each subresource through [`DETILED`].
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        DETILED.with_borrow_mut(|scratch| self.detile_through(tiled, scratch));
        #[cfg(feature = "diagnostics")]
        tex_us(crate::copystat::C::TexDetileUs, started);
    }

    /// [`Staging::detile`] with the thread's scratch borrowed.
    fn detile_through(&self, tiled: &[u8], scratch: &mut Vec<u8>) {
        // A widened texture's 2-byte texels, detiled before they expand.
        let mut packed = Vec::new();
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let depth_pitch = pitch * rows as usize;
            let len = depth_pitch * f.Depth as usize;
            if scratch.len() < len {
                scratch.resize(len, 0);
            }
            let linear = &mut scratch[..len];
            if let Err(e) = detile_subresource(
                &self.layout,
                tiled,
                mip,
                slice,
                linear,
                pitch,
                rows as usize,
                f.Depth as usize,
                self.widened_from,
                &mut packed,
            ) {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
            // The bytes the detiling wrote: each row's elements (4-byte
            // ones in a texture widened from a packed 16-bit format).
            let m = &self.layout.mips[mip as usize];
            let element = match self.widened_from {
                Some((_, _, host)) => host,
                None => self.layout.desc.element_bytes as usize,
            };
            let bytes = m.width as usize * element;
            for z in 0..m.depth as usize {
                for y in 0..m.height as usize {
                    let at = z * depth_pitch + y * pitch;
                    // SAFETY: the detiling checked the row lies inside the
                    // subresource's `len` bytes, in the scratch and in its
                    // part of the mapped staging buffer.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            scratch.as_ptr().add(at),
                            self.data.add(footprint.Offset as usize + at),
                            bytes,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(feature = "diagnostics")]
fn tex_us(c: crate::copystat::C, started: std::time::Instant) {
    crate::copystat::add(c, started.elapsed().as_micros() as u64);
}

/// Detiles a subresource of `layout` into `linear` (`row_pitch` bytes a
/// row, `rows` rows a depth slice); a texture made in a stand-in format
/// detiles its title texels into `packed` first and expands them.
#[allow(clippy::too_many_arguments)]
fn detile_subresource(
    layout: &victory_formats::texture::Layout,
    tiled: &[u8],
    mip: u32,
    slice: u32,
    linear: &mut [u8],
    row_pitch: usize,
    rows: usize,
    depth: usize,
    widened_from: Option<(u32, usize, usize)>,
    packed: &mut Vec<u8>,
) -> Result<(), victory_formats::texture::Error> {
    let Some((packed_format, texel, _)) = widened_from else {
        return layout.detile(tiled, mip, slice, linear, row_pitch, row_pitch * rows);
    };
    let width = layout
        .mips
        .get(mip as usize)
        .map_or(0, |m| m.width as usize);
    let packed_pitch = width * texel;
    let packed_plane = packed_pitch * rows;
    packed.resize(packed_plane * depth, 0);
    layout.detile(tiled, mip, slice, packed, packed_pitch, packed_plane)?;
    crate::resource::expand_widened_rows(
        packed_format,
        width,
        rows * depth,
        packed,
        packed_pitch,
        linear,
        row_pitch,
    );
    Ok(())
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive), as [`crate::watch::committed`] keeps
/// it for views.
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    crate::watch::committed(at as usize, len)
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let title_format = crate::resource::pc_format(desc.format);
    let widened_from = crate::resource::widening(title_format, d.Format.0 as u32)
        .map(|(title, host)| (title_format, title, host));
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        buffer,
        #[cfg(feature = "diagnostics")]
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        widened_from,
        last: Mutex::default(),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};

    /// Memory is detiled when it differs from what was last detiled, the
    /// first time, and when it changes back.
    #[test]
    fn memory_as_last_detiled_is_not_detiled_again() {
        let mut last = Last::default();
        let (a, mut b) = (vec![7u8; 3 << 16], vec![7u8; 3 << 16]);
        assert!(last.changed(&a), "first");
        assert!(!last.changed(&a), "the same bytes");
        assert!(!last.changed(&b), "the same bytes elsewhere");
        b[(3 << 16) - 1] = 8;
        assert!(last.changed(&b), "its last byte");
        assert!(last.changed(&a), "back as it was");
        assert!(last.changed(&a[1..]), "a byte shorter");
    }

    /// Tasks run on several threads run once each and their results add
    /// up as on one, a panicking one among them, from several submitting
    /// threads at once.
    #[test]
    fn tasks_on_several_threads_run_once_each() {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
        let submit = || {
            let runs: Arc<Vec<AtomicUsize>> = Arc::new((0..97).map(|_| AtomicUsize::new(0)).collect());
            let tasks: Vec<Task> = (0..97)
                .map(|k| {
                    let runs = runs.clone();
                    Box::new(move || {
                        runs[k].fetch_add(1, Relaxed);
                        if k == 13 {
                            panic!("a task that panics");
                        }
                        std::thread::yield_now();
                        k
                    }) as Task
                })
                .collect();
            let read = run_tasks(tasks, 4);
            assert_eq!(read, (0..97).sum::<usize>() - 13);
            assert!(runs.iter().all(|r| r.load(Relaxed) == 1));
        };
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(submit);
            }
        });
        // One thread, or one task: on the caller.
        assert_eq!(run_tasks(vec![Box::new(|| 5usize) as Task], 8), 5);
        assert_eq!(run_tasks((0..3usize).map(|k| Box::new(move || k) as Task).collect(), 1), 3);
    }

    /// Hashing a submission's due textures in physical-page memory, one
    /// after another and on the pool.
    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn texture_hashing_timing() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 2 << 20;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let tasks = || -> Vec<Task> {
            textures
                .iter()
                .map(|t| {
                    let t = t.clone();
                    Box::new(move || {
                        std::hint::black_box(blake3::hash(&t));
                        t.len()
                    }) as Task
                })
                .collect()
        };
        run_tasks(tasks(), threads()); // warm up the pool
        for n in [1, 2, 4, threads()] {
            let started = std::time::Instant::now();
            for _ in 0..10 {
                assert_eq!(run_tasks(tasks(), n), TEXTURES * SIZE);
            }
            let took = started.elapsed() / 10;
            println!(
                "{TEXTURES} textures of {} MiB on {n} threads: {took:?} ({:.1} GB/s)",
                SIZE >> 20,
                (TEXTURES * SIZE) as f64 / took.as_secs_f64() / 1e9
            );
        }
    }

    #[test]
    fn detile_widened_packed16_expands_into_rgba8_staging() {
        for format in crate::resource::PACKED_16 {
            let desc = TextureDesc {
                dimension: Dimension::Tex2d,
                width: 17,
                height: 9,
                depth_or_array: 1,
                mips: 2,
                format,
                samples: 1,
                swizzle: Swizzle::SW_4KB_S,
            };
            let tex = Texture::new(&desc).unwrap();
            let layout = &tex.planes[0].layout;
            let mut tiled = vec![0u8; layout.size as usize];
            let mut expected_mips = Vec::new();

            for mip in 0..2u32 {
                let m = layout.mips[mip as usize];
                let row_bytes = m.width as usize * 2;
                let mut src_mip = vec![0u8; row_bytes * m.height as usize];
                for y in 0..m.height as usize {
                    for x in 0..m.width as usize {
                        let px = ((mip as u16 + 1) * 0x1111)
                            .wrapping_add((y as u16) * 0x0103 + x as u16);
                        let at = y * row_bytes + x * 2;
                        src_mip[at..at + 2].copy_from_slice(&px.to_le_bytes());
                    }
                }
                layout
                    .tile(&mut tiled, mip, 0, &src_mip, row_bytes, src_mip.len())
                    .unwrap();
                expected_mips.push(src_mip);
            }

            let mut packed = Vec::new();
            for (mip, src_mip) in expected_mips.iter().enumerate() {
                let m = layout.mips[mip];
                let width = m.width as usize;
                let rows = m.height as usize;
                let row_pitch = (width * 4).next_multiple_of(256);
                let exact_len = row_pitch * (rows - 1) + width * 4;
                let mut staging = vec![0x5au8; exact_len];
                detile_subresource(
                    layout,
                    &tiled,
                    mip as u32,
                    0,
                    &mut staging,
                    row_pitch,
                    rows,
                    1,
                    Some((format, 2, 4)),
                    &mut packed,
                )
                .unwrap();
                for y in 0..rows {
                    let mut expected = vec![0u8; width * 4];
                    crate::resource::expand_packed16(
                        format,
                        &src_mip[y * width * 2..][..width * 2],
                        &mut expected,
                    );
                    assert_eq!(
                        &staging[y * row_pitch..][..width * 4],
                        &expected[..],
                        "format {format} mip {mip} row {y}"
                    );
                }
            }
        }
    }
}

//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! The ranges draws are known to read are brought up to date (vertex and
//! index buffer views, constant buffers in tables, root descriptors,
//! buffer views in tables), and every buffer view in a shader-visible heap
//! at each submission (`descriptor::bindless`: shaders may index the heap
//! from data).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, covering the ranges its
    /// shaders write: what the GPU wrote differs. By offset, disjoint, ranges
    /// that overlap or touch merged into one.
    before: Mutex<Vec<(usize, Vec<u8>)>>,
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 16 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 16 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(16);
    }
}

/// Makes the snapshots `before` (by offset, disjoint, not touching) cover
/// `[from, to)`: the snapshots it overlaps or touches become one, the parts
/// none covered taken from `host` (offsets `a..b`). Returns the bytes newly
/// taken.
fn cover<'a>(
    before: &mut Vec<(usize, Vec<u8>)>,
    from: usize,
    to: usize,
    host: impl Fn(usize, usize) -> &'a [u8],
) -> usize {
    // The snapshots the range overlaps or touches: `first..last`.
    let first = before.partition_point(|(at, bytes)| at + bytes.len() < from);
    let last = first + before[first..].partition_point(|&(at, _)| at <= to);
    if last == first + 1 {
        let (at, bytes) = &before[first];
        if *at <= from && to <= at + bytes.len() {
            return 0;
        }
    }
    let (start, end) = if last > first {
        let (at, bytes) = &before[last - 1];
        (before[first].0.min(from), (at + bytes.len()).max(to))
    } else {
        (from, to)
    };
    let merged: Vec<(usize, Vec<u8>)> = before.drain(first..last).collect();
    let kept: usize = merged.iter().map(|(_, b)| b.len()).sum();
    let mut grown = pooled(end - start);
    let mut at = start;
    for (taken, bytes) in merged {
        extend_par(&mut grown, host(at, taken));
        extend_par(&mut grown, &bytes);
        at = taken + bytes.len();
        give_back(bytes);
    }
    extend_par(&mut grown, host(at, end));
    before.insert(first, (start, grown));
    end - start - kept
}

/// What copy-back compares and writes at a time: title memory's 64 KiB
/// pieces.
const PIECE: usize = 64 << 10;

/// Bytes from which a copy or compare is split over threads, and the bytes
/// of each part.
const PARALLEL: usize = 8 << 20;
const PART: usize = 4 << 20;

/// Runs `part(offset, n)` over `[0, len)` in parts of `step`, on the
/// texture refresh threads when `len` is `big` or more (the submitting
/// thread is one of them; all are done when it returns).
fn in_parts(len: usize, step: usize, big: usize, part: impl Fn(usize, usize) + Send + Sync + 'static) {
    if len < big || step == 0 {
        part(0, len);
        return;
    }
    let part = Arc::new(part);
    let tasks = (0..len.div_ceil(step))
        .map(|i| {
            let part = part.clone();
            let at = i * step;
            Box::new(move || {
                part(at, step.min(len - at));
                0usize
            }) as crate::texture::Task
        })
        .collect();
    crate::texture::run_tasks(tasks, crate::texture::threads());
}

/// `v.extend_from_slice(src)`, big ones on several threads.
fn extend_par(v: &mut Vec<u8>, src: &[u8]) {
    let n = src.len();
    if n < PARALLEL {
        v.extend_from_slice(src);
        return;
    }
    v.reserve(n);
    let (dst, from) = (v.as_mut_ptr() as usize + v.len(), src.as_ptr() as usize);
    in_parts(n, PART, PARALLEL, move |at, len| {
        // SAFETY: `[at, at + len)` lies in `src` and in the reserved room
        // after `v`'s bytes; the parts do not overlap, and all are done
        // before `in_parts` returns.
        unsafe { std::ptr::copy_nonoverlapping((from + at) as *const u8, (dst + at) as *mut u8, len) };
    });
    // SAFETY: the `n` bytes after the old length were written above.
    unsafe { v.set_len(v.len() + n) };
}

/// The pieces (offset, length) where `host` differs from `old` (same
/// length), cut at title memory's [`PIECE`] boundaries (`phase`: the first
/// byte's offset in its piece), big ranges compared on several threads.
fn differing_pieces(host: &[u8], old: &[u8], phase: usize) -> Vec<(usize, usize)> {
    let len = host.len().min(old.len());
    let first = (PIECE - phase % PIECE).min(len);
    // Piece `k` starts at `first + (k - 1) * PIECE` (piece 0 at 0).
    let pieces = if len == 0 { 0 } else { 1 + (len - first).div_ceil(PIECE) };
    let bounds = move |k: usize| {
        let a = if k == 0 { 0 } else { first + (k - 1) * PIECE };
        let b = if k == 0 { first } else { (a + PIECE).min(len) };
        (a, b)
    };
    let differs: Arc<Vec<std::sync::atomic::AtomicBool>> =
        Arc::new((0..pieces).map(|_| Default::default()).collect());
    let (h, o) = (host.as_ptr() as usize, old.as_ptr() as usize);
    let marks = differs.clone();
    // Parts in whole pieces: piece `k` is compared by the part its start
    // falls in.
    in_parts(pieces, PART / PIECE, PARALLEL / PIECE, move |k0, n| {
        for k in k0..k0 + n {
            let (a, b) = bounds(k);
            // SAFETY: `[a, b)` lies in both slices, which outlive
            // `in_parts`; only read.
            let (hs, os) = unsafe {
                (
                    std::slice::from_raw_parts((h + a) as *const u8, b - a),
                    std::slice::from_raw_parts((o + a) as *const u8, b - a),
                )
            };
            if hs != os {
                marks[k].store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    });
    (0..pieces)
        .filter(|&k| differs[k].load(std::sync::atomic::Ordering::Relaxed))
        .map(|k| {
            let (a, b) = bounds(k);
            (a, b - a)
        })
        .collect()
}

/// Merges the GPU's writes into title memory, 8 bytes at a time: where
/// `host` differs from `old` (the GPU wrote) `title` takes the host's
/// bytes, and `host` then takes `title`'s. Whether `title` changed.
fn merge_written(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    const LOW: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let mut changed = false;
    let words = title.len() / 8 * 8;
    for ((t, h), o) in title[..words]
        .chunks_exact_mut(8)
        .zip(host[..words].chunks_exact_mut(8))
        .zip(old[..words].chunks_exact(8))
    {
        let tw = u64::from_ne_bytes(t.try_into().unwrap_or_default());
        let hw = u64::from_ne_bytes(h.try_into().unwrap_or_default());
        let ow = u64::from_ne_bytes(o.try_into().unwrap_or_default());
        let diff = hw ^ ow;
        if diff == 0 {
            if hw != tw {
                h.copy_from_slice(t);
            }
            continue;
        }
        // 0xff in each byte the GPU wrote.
        let mask = ((((diff & LOW) + LOW) | diff) & HIGH) >> 7;
        let mask = mask * 0xff;
        let merged = (tw & !mask) | (hw & mask);
        changed |= merged != tw;
        t.copy_from_slice(&merged.to_ne_bytes());
        h.copy_from_slice(&merged.to_ne_bytes());
    }
    for ((t, h), o) in title[words..]
        .iter_mut()
        .zip(host[words..].iter_mut())
        .zip(&old[words..])
    {
        if *h != *o && *t != *h {
            *t = *h;
            changed = true;
        }
        *h = *t;
    }
    changed
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. From now on no refresh copies title memory over
    /// it, so the copy's bytes there stay as they are until the submission
    /// runs; [`Shadow::snapshot`] keeps them once preparing is done.
    pub(crate) fn writing(&self, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
    }

    /// When the submission is prepared, before it runs: the bytes of the
    /// ranges its shaders write (`ranges`, title start and length, each
    /// declared by [`Shadow::writing`] before) are kept to tell later what
    /// the GPU wrote. The world declares overlapping views of its 174 MB
    /// heap dozens of times per submission: they are merged first and each
    /// byte is taken once (taken range by range as declared, a snapshot
    /// that grew was copied again whole each time), big ones on several
    /// threads, in reused memory (fresh snapshots cost ~2.3 million page
    /// faults per world frame).
    ///
    /// Taking them here instead of at each declaration takes the same
    /// bytes: a declared range is pending from its declaration on, and
    /// nothing but refreshes, which skip pending ranges, writes the copy
    /// before the submission runs.
    pub(crate) fn snapshot(&self, ranges: &[(usize, usize)]) {
        let mut inside: Vec<(usize, usize)> = ranges
            .iter()
            .map(|&(address, len)| self.inside(address, len))
            .filter(|(from, to)| from < to)
            .collect();
        inside.sort_unstable();
        let mut union: Vec<(usize, usize)> = Vec::with_capacity(inside.len());
        for (from, to) in inside {
            match union.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => union.push((from, to)),
            }
        }
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        for (from, to) in union {
            let _taken = cover(&mut before, from, to, host);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SnapshotBytes, _taken as u64);
        }
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(
        &self,
        address: usize,
        len: usize,
        regions: &mut crate::host::Regions,
    ) -> usize {
        let snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = snapshot
            .iter()
            .find(|(taken, before)| *taken <= from && to <= taken + before.len())
            && from < to
        {
            let before = &before[from - taken..to - taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission (big ranges on several threads), title memory read
            // only where the GPU wrote.
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackCompared, (to - from) as u64);
            #[cfg(feature = "diagnostics")]
            let compared = std::time::Instant::now();
            // SAFETY: inside the mapping; nothing writes it while compared.
            let host_all = unsafe { std::slice::from_raw_parts(self.data.add(from), to - from) };
            let differing = differing_pieces(host_all, before, (start + from) % PIECE);
            #[cfg(feature = "diagnostics")]
            crate::copystat::since(crate::copystat::C::BackCompareUs, compared);
            for (at, n) in differing {
                let at = from + at;
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let title = &mut title[..n];
                if crate::host::read_into((start + at) as *const u8, title) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::BackPieces, 1);
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    if merge_written(title, host, old) {
                        #[cfg(feature = "diagnostics")]
                        let _writing = crate::copystat::timed(crate::copystat::C::BackWriteUs);
                        crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n, regions);
                        written += n;
                    }
                    synced.push((start + at) / PAGE);
                }
            }
            #[cfg(feature = "diagnostics")]
            let _fresh = crate::copystat::timed(crate::copystat::C::BackFreshUs);
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            //
            // The asking also takes the bits of title writes made since a
            // page was merged: a page is marked current only if title
            // memory, read again after the asking, still holds what the
            // copy does (a title write after the asking shows at the next
            // one). From Marvisak/VIctory#722 (Dufftwt).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes_fresh(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                let title = &mut title[..PAGE];
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    let at = page * PAGE - start;
                    // SAFETY: a whole page inside the range, in the mapping.
                    let host = unsafe { std::slice::from_raw_parts(self.data.add(at), PAGE) };
                    if crate::host::read_into((page * PAGE) as *const u8, title) && *title == *host {
                        copied[page - base] = now;
                    }
                }
            }
        } else if from < to {
            // Every declared range was snapshotted when submitted
            // (`writing`), merged as the union is: a range found in none
            // would lose what the GPU wrote there.
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackMissed, 1);
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    address = format_args!("{address:#x}"),
                    len,
                    "a range shaders wrote has no snapshot: not copied back"
                );
            });
        }
        written
    }

    /// After the submission's copy-back: the steps that declared `ranges`
    /// (title start, end) are done with them (one submission runs at a
    /// time), and the snapshots with the last ranges.
    pub(crate) fn written_back(&self, ranges: &[(usize, usize)]) {
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|r| !ranges.contains(r));
        if pending.is_empty() {
            let taken = std::mem::take(&mut *self.before.lock().unwrap_or_else(|e| e.into_inner()));
            for (_, bytes) in taken {
                give_back(bytes);
            }
        }
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        self.update_if(address, len, &mut |a, b, _| Some((a, b))).0
    }

    /// [`Shadow::update`] copying of a run of stale pages (title `[from,
    /// to)`) only what `copy_ok` gives (bindless ranges: memory all
    /// committed, asked only of what is to be copied); `None`: none of it.
    /// Its third argument: every page of the run was copied before and is
    /// stale only because a write to it was reported since (so it was
    /// committed when written).
    /// Pages left out stay stale. Returns the bytes copied and the runs
    /// left out (in whole or in part).
    pub(crate) fn update_if(
        &self,
        address: usize,
        len: usize,
        copy_ok: &mut dyn FnMut(usize, usize, bool) -> Option<(usize, usize)>,
    ) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return (0, 0);
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes, mut left) = (from, 0, 0);
            for (a, e) in busy {
                if a > at {
                    let (b, l) = self.update_free(at, a, copy_ok);
                    (bytes, left) = (bytes + b, left + l);
                }
                at = at.max(e);
            }
            if at < to {
                let (b, l) = self.update_free(at, to, copy_ok);
                (bytes, left) = (bytes + b, left + l);
            }
            return (bytes, left);
        }
        self.update_free(from, to, copy_ok)
    }

    /// Whether writes to its title memory are known to be tracked (by the
    /// kernel's write watch, or guards on view memory).
    pub(crate) fn write_watched(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == 1
    }

    /// [`Shadow::update_if`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(
        &self,
        from: usize,
        to: usize,
        copy_ok: &mut dyn FnMut(usize, usize, bool) -> Option<(usize, usize)>,
    ) -> (usize, usize) {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let (copied, left) = self.copy_stale(from, to, now, &written, copy_ok);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return (copied, left);
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        match copy_ok(from, to, false) {
            Some((a, b)) if a < b => (self.copy(a, b), usize::from((a, b) != (from, to))),
            _ => (0, 1),
        }
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer, of each run of them what
    /// `copy_ok` gives (the rest stays stale). Returns the bytes copied
    /// and the runs left out (in whole or in part).
    fn copy_stale(
        &self,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
        copy_ok: &mut dyn FnMut(usize, usize, bool) -> Option<(usize, usize)>,
    ) -> (usize, usize) {
        let base = self.start as usize / PAGE;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        // Runs of stale pages: [first, past), and whether one was never
        // copied.
        let mut runs = Vec::new();
        let mut run: Option<(usize, bool)> = None;
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = copied[page - base];
            if at == 0 || write > at {
                run.get_or_insert((page, false)).1 |= at == 0;
            } else if let Some((first, fresh)) = run.take() {
                runs.push((first, page, fresh));
            }
        }
        if let Some((first, fresh)) = run {
            runs.push((first, last + 1, fresh));
        }
        let (mut bytes, mut left) = (0, 0);
        for (first, past, fresh) in runs {
            // Cut to the range: next to it shaders may be writing the copy.
            let (a, b) = ((first * PAGE).max(from), (past * PAGE).min(to));
            let Some((a2, b2)) = copy_ok(a, b, !fresh)
                .map(|(c, d)| (c.max(a), d.min(b)))
                .filter(|(c, d)| c < d)
            else {
                left += 1;
                continue;
            };
            left += usize::from((a2, b2) != (a, b));
            let (a, b) = (a2, b2);
            for page in a / PAGE..b.div_ceil(PAGE) {
                // A page only partly copied stays stale for the rest.
                if (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= b {
                    copied[page - base] = now;
                }
            }
            bytes += self.copy(a, b);
        }
        (bytes, left)
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    #[cfg(feature = "diagnostics")]
    let _finding = {
        crate::copystat::add(crate::copystat::C::RecFinds, 1);
        crate::copystat::timed(crate::copystat::C::RecFindUs)
    };
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_cover_ranges_without_the_gaps_between() {
        let host: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let at = |a: usize, b: usize| &host[a..b];
        let mut before = Vec::new();
        assert_eq!(cover(&mut before, 100, 200, at), 100);
        assert_eq!(cover(&mut before, 1000, 1100, at), 100);
        // Inside one: nothing new.
        assert_eq!(cover(&mut before, 120, 180, at), 0);
        assert_eq!(before.len(), 2);
        // Touching the first: merged with it.
        assert_eq!(cover(&mut before, 200, 300, at), 100);
        assert_eq!(before.len(), 2);
        assert_eq!((before[0].0, before[0].1.len()), (100, 200));
        // Bridging both: one snapshot, the old bytes kept as taken.
        before[1].1[0] = 0xee;
        assert_eq!(cover(&mut before, 250, 1050, at), 700);
        assert_eq!(before.len(), 1);
        let (start, bytes) = &before[0];
        assert_eq!((*start, bytes.len()), (100, 1000));
        assert_eq!(bytes[1000 - 100], 0xee);
        assert_eq!(&bytes[..900], &host[100..1000]);
        assert_eq!(&bytes[901..], &host[1001..1100]);
        // Before all: a new first snapshot.
        assert_eq!(cover(&mut before, 0, 50, at), 50);
        assert_eq!(before.iter().map(|b| b.0).collect::<Vec<_>>(), [0, 100]);
    }

    #[test]
    fn merged_writes_take_the_gpu_bytes_and_keep_the_title_elsewhere() {
        for len in [0, 5, 8, 64, 67] {
            let old: Vec<u8> = (0..len).map(|i| i as u8).collect();
            let mut host = old.clone();
            let mut title: Vec<u8> = old.iter().map(|b| b ^ 0x55).collect();
            // The GPU writes every third byte, some to the title's value.
            for i in (0..len).step_by(3) {
                host[i] = if i % 2 == 0 { 0xa0 } else { title[i] };
            }
            let mut want = title.clone();
            let mut changed = false;
            for i in (0..len).step_by(3) {
                changed |= want[i] != host[i];
                want[i] = host[i];
            }
            assert_eq!(merge_written(&mut title, &mut host, &old), changed, "len {len}");
            assert_eq!(title, want, "len {len}");
            assert_eq!(host, want, "len {len}");
        }
    }
}

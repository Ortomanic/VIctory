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
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

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
    /// The copy's bytes as last kept, by page of title memory (its part
    /// inside the buffer), for the pages of ranges shaders write: what the
    /// GPU wrote differs. Kept across submissions while the copy's page is
    /// unchanged: a refresh copying into a page drops it, copy-back keeps
    /// the pieces it merges equal to the copy ([`Shadow::snapshot`]).
    kept: Mutex<BTreeMap<usize, Vec<u8>>>,
}

/// Bytes from which a copy or compare is split over threads, and the bytes
/// of each part.
const PARALLEL: usize = 2 << 20;
const PART: usize = 1 << 20;

/// Copies of (source, destination, length), not overlapping, as one job on
/// several threads when they come to [`PARALLEL`] bytes or more.
fn copy_all(copies: &[(usize, usize, usize)]) {
    let total: usize = copies.iter().map(|c| c.2).sum();
    if total < PARALLEL {
        for &(src, dst, len) in copies {
            // SAFETY: the caller's ranges, valid and apart.
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
        }
        return;
    }
    let tasks = copies
        .iter()
        .flat_map(|&(src, dst, len)| {
            (0..len.div_ceil(PART)).map(move |i| {
                let (at, n) = (i * PART, PART.min(len - i * PART));
                Box::new(move || {
                    // SAFETY: a part of the caller's ranges; all parts are
                    // done before `run_tasks` returns.
                    unsafe { std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n) };
                    n
                }) as crate::texture::Task
            })
        })
        .collect();
    crate::texture::run_tasks(tasks, crate::texture::threads());
}

/// Pieces per task of [`differ_all`] (64 KiB each).
const PIECES_PER_PART: usize = PART / PAGE;

/// For each (copy, kept, length): whether the two differ, as one job on
/// several threads when they come to [`PARALLEL`] bytes or more.
fn differ_all(compares: &[(usize, usize, usize)]) -> Vec<bool> {
    // SAFETY (both closures): the caller's ranges, valid and only read
    // until this returns.
    let one = |&(a, b, n): &(usize, usize, usize)| unsafe {
        b == 0
            || std::slice::from_raw_parts(a as *const u8, n) != std::slice::from_raw_parts(b as *const u8, n)
    };
    let total: usize = compares.iter().map(|c| c.2).sum();
    if total < PARALLEL {
        return compares.iter().map(one).collect();
    }
    let compares: Arc<Vec<(usize, usize, usize)>> = Arc::new(compares.to_vec());
    let marks: Arc<Vec<std::sync::atomic::AtomicBool>> =
        Arc::new((0..compares.len()).map(|_| Default::default()).collect());
    let tasks = (0..compares.len().div_ceil(PIECES_PER_PART))
        .map(|i| {
            let (compares, marks) = (compares.clone(), marks.clone());
            Box::new(move || {
                let k0 = i * PIECES_PER_PART;
                for k in k0..(k0 + PIECES_PER_PART).min(compares.len()) {
                    if one(&compares[k]) {
                        marks[k].store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                0usize
            }) as crate::texture::Task
        })
        .collect();
    crate::texture::run_tasks(tasks, crate::texture::threads());
    marks.iter().map(|m| m.load(std::sync::atomic::Ordering::Relaxed)).collect()
}

/// Bounds the pages kept per copy (256 MiB, more than the world's
/// 174 MB heap): past it all are dropped and taken again.
const KEPT_PAGES: usize = 4096;

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

    /// The title range the copy covers: start and length (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
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
            kept: Mutex::default(),
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
    /// the GPU wrote, by page, only pages not kept already: a kept page is
    /// equal to the copy (the world declares the same views of its 174 MB
    /// heap in most of a frame's ~17 submissions, and copying them all at
    /// each, ~600 MB a frame, cost ~50-130 ms), taken as one job on several
    /// threads.
    ///
    /// Taking them here instead of at each declaration takes the same
    /// bytes: a declared range is pending from its declaration on, and
    /// nothing but refreshes, which skip pending ranges, writes the copy
    /// before the submission runs.
    pub(crate) fn snapshot(&self, ranges: &[(usize, usize)]) {
        let start = self.start as usize;
        let mut pages: Vec<usize> = Vec::new();
        for &(address, len) in ranges {
            let (from, to) = self.inside(address, len);
            if from < to {
                pages.extend((start + from) / PAGE..=(start + to - 1) / PAGE);
            }
        }
        pages.sort_unstable();
        pages.dedup();
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        pages.retain(|p| !kept.contains_key(p));
        if kept.len() + pages.len() > KEPT_PAGES {
            // Bounded: dropped all, taken again as needed.
            kept.clear();
            pages.clear();
            for &(address, len) in ranges {
                let (from, to) = self.inside(address, len);
                if from < to {
                    pages.extend((start + from) / PAGE..=(start + to - 1) / PAGE);
                }
            }
            pages.sort_unstable();
            pages.dedup();
        }
        let mut copies = Vec::with_capacity(pages.len());
        let mut taken = Vec::with_capacity(pages.len());
        for page in pages {
            let (a, b) = self.page_span(page);
            let mut bytes = Vec::with_capacity(b - a);
            copies.push((self.data as usize + a, bytes.as_mut_ptr() as usize, b - a));
            // SAFETY: written by `copy_all` below before anything reads it.
            unsafe { bytes.set_len(b - a) };
            taken.push((page, bytes));
        }
        copy_all(&copies);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(
            crate::copystat::C::SnapshotBytes,
            copies.iter().map(|c| c.2 as u64).sum(),
        );
        kept.extend(taken);
    }

    /// Page `page`'s part inside the buffer, as offsets in the copy.
    fn page_span(&self, page: usize) -> (usize, usize) {
        self.inside(page * PAGE, PAGE)
    }

    /// The copy's pages holding `[from, to)` (title addresses) changed
    /// other than through copy-back: not kept any more.
    fn changed(&self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.is_empty() {
            return;
        }
        let pages: Vec<usize> = kept
            .range(from / PAGE..to.div_ceil(PAGE))
            .map(|(&p, _)| p)
            .collect();
        for page in pages {
            kept.remove(&page);
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
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if from < to {
            // By 64 KiB piece (a page's part in the range): the copy
            // compared with its bytes kept before the submission (on
            // several threads), title memory read only where the GPU wrote.
            let mut title = vec![0u8; PAGE];
            let mut synced = Vec::new();
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            // (offset in the copy, length, page, offset in its kept bytes)
            let mut pieces = Vec::new();
            let mut missed = 0;
            for page in (start + from) / PAGE..=(start + to - 1) / PAGE {
                let (a, b) = self.page_span(page);
                let (a2, b2) = (a.max(from), b.min(to));
                match kept.get(&page) {
                    Some(bytes) if bytes.len() == b - a && a2 < b2 => {
                        pieces.push((a2, b2 - a2, page, a2 - a));
                    }
                    _ => missed += 1,
                }
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackCompared, (to - from) as u64);
            #[cfg(feature = "diagnostics")]
            let compared = std::time::Instant::now();
            let compares: Vec<(usize, usize, usize)> = pieces
                .iter()
                .map(|&(at, n, page, o)| {
                    let old = kept.get(&page).map_or(0, |b| b.as_ptr() as usize + o);
                    (self.data as usize + at, old, n)
                })
                .collect();
            let differs = differ_all(&compares);
            #[cfg(feature = "diagnostics")]
            crate::copystat::since(crate::copystat::C::BackCompareUs, compared);
            for (&(at, n, page, o), _) in pieces.iter().zip(&differs).filter(|(_, d)| **d) {
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let Some(old) = kept.get_mut(&page).map(|b| &mut b[o..o + n]) else {
                    continue;
                };
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
                // Kept equal to the copy for the next submission.
                old.copy_from_slice(host);
            }
            drop(kept);
            if missed > 0 {
                // Every declared range's pages were kept when submitted
                // (`snapshot`): a page found in none would lose what the
                // GPU wrote there.
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::BackMissed, missed);
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        address = format_args!("{address:#x}"),
                        len,
                        missed,
                        "pages shaders wrote have no snapshot: not copied back"
                    );
                });
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
        }
        written
    }

    /// After the submission's copy-back: the steps that declared `ranges`
    /// (title start, end) are done with them (one submission runs at a
    /// time).
    pub(crate) fn written_back(&self, ranges: &[(usize, usize)]) {
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|r| !ranges.contains(r));
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
        self.changed(from, to);
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
/// Read for every buffer a draw or a heap slot names (some 180,000 times a
/// world frame of Grand Theft Auto VI, from all recording threads at
/// once), so lookups share the lock and never walk far.
struct Buffers {
    all: BTreeMap<(u64, usize), Entry>,
    /// The same buffers by length class (`class`), each with its end: a
    /// buffer holding an address far past where it starts (a heap's buffer
    /// with thousands of placed buffers before the address) is found by its
    /// class, where few buffers start close enough below the address to
    /// reach it.
    classes: [BTreeMap<(u64, usize), u64>; 64],
    /// A bit per class with buffers.
    used: u64,
}

/// The entries starting closest below an address that a lookup tries
/// before it asks the length classes: nearly every lookup's buffer is the
/// first of them.
const NEAREST: usize = 8;

/// A buffer length's class: lengths in `[2^class, 2^(class + 1))`.
fn class(len: u64) -> usize {
    len.max(1).ilog2() as usize
}

impl Buffers {
    const fn new() -> Self {
        Self {
            all: BTreeMap::new(),
            classes: [const { BTreeMap::new() }; 64],
            used: 0,
        }
    }

    fn insert(&mut self, key: (u64, usize), end: u64) {
        self.remove(key);
        let k = class(end - key.0);
        self.classes[k].insert(key, end);
        self.used |= 1 << k;
        self.all.insert(key, Entry { end, shadow: None });
    }

    fn remove(&mut self, key: (u64, usize)) {
        if let Some(entry) = self.all.remove(&key) {
            let k = class(entry.end - key.0);
            self.classes[k].remove(&key);
            if self.classes[k].is_empty() {
                self.used &= !(1 << k);
            }
        }
    }

    /// The innermost entry holding `[address, end)`: the one starting
    /// closest below `address` that reaches `end`. Big buffers hold many
    /// smaller ones (a heap's placed buffers), so the one holding an address
    /// past those can be any number of entries back: after the `NEAREST`
    /// entries, the ones before them are asked per length class, where only
    /// those starting less than the class's longest length below `end` can
    /// reach it (two if they do not overlap).
    fn enclosing(&self, address: u64, end: u64) -> Option<(u64, usize)> {
        let mut before = (address, usize::MAX);
        let mut left = NEAREST;
        for (key, entry) in self.all.range(..=before).rev() {
            if entry.end >= end {
                return Some(*key);
            }
            before = *key;
            left -= 1;
            if left == 0 {
                break;
            }
        }
        if left > 0 {
            // Every entry starting at or below the address was tried.
            return None;
        }
        // Entries before `before`: they start at or below its start, so
        // only the classes long enough to reach `end` from there can hold
        // the range.
        let need = end - before.0;
        let mut found: Option<(u64, usize)> = None;
        let mut classes = self.used & (u64::MAX << class(need));
        while classes != 0 {
            let k = classes.trailing_zeros() as usize;
            classes &= classes - 1;
            // Longest length in the class: 2^(k + 1) - 1.
            let longest = if k == 63 { u64::MAX } else { (2u64 << k) - 1 };
            let lowest = (end.saturating_sub(longest), 0);
            // Only one starting after the innermost so far is more inner.
            let from = match found {
                Some(f) if f >= lowest => std::ops::Bound::Excluded(f),
                _ => std::ops::Bound::Included(lowest),
            };
            if lowest >= before {
                continue;
            }
            if let Some((key, _)) = self.classes[k]
                .range((from, std::ops::Bound::Excluded(before)))
                .rev()
                .find(|&(_, &e)| e >= end)
            {
                found = Some(*key);
            }
        }
        found
    }
}

static BUFFERS: RwLock<Buffers> = RwLock::new(Buffers::new());

fn buffers() -> std::sync::RwLockReadGuard<'static, Buffers> {
    BUFFERS.read().unwrap_or_else(|e| e.into_inner())
}

fn buffers_mut() -> std::sync::RwLockWriteGuard<'static, Buffers> {
    BUFFERS.write().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers_mut().insert((start, resource), start.saturating_add(len));
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers_mut().remove((start, resource));
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
    {
        // Nearly always the copy is there: readers do not wait for each
        // other.
        let map = buffers();
        let Some(key) = map.enclosing(address, end) else {
            drop(map);
            return loose(address, end);
        };
        if let Some(shadow) = map.all.get(&key).and_then(|e| e.shadow.clone()) {
            return Some(shadow);
        }
    }
    // Its first use; the buffers may have changed since.
    let mut map = buffers_mut();
    let key = map.enclosing(address, end);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.all.get_mut(&key)?;
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
static LOOSE: RwLock<BTreeMap<(u64, u64), Arc<Shadow>>> = RwLock::new(BTreeMap::new());

/// The longest window in `LOOSE`. Changed under its lock.
static LOOSE_SPAN: AtomicU64 = AtomicU64::new(0);

/// Below this nothing is ever mapped (the null page and its guard area).
const LOWEST_MAPPED: u64 = 0x10000;

/// The window copy holding `[address, end)` among `windows` (by start and
/// length, none longer than `span`), the innermost if several do.
fn enclosing_window<V>(windows: &BTreeMap<(u64, u64), V>, span: u64, address: u64, end: u64) -> Option<&V> {
    let lowest = end.saturating_sub(span);
    if lowest > address {
        return None;
    }
    windows
        .range((lowest, 0)..=(address, u64::MAX))
        .rev()
        .find(|&(&(s, l), _)| s.saturating_add(l) >= end)
        .map(|(_, v)| v)
}

/// The window copy for `[address, end)` outside buffers: one already made
/// that holds it (a range spanning two windows has one of its own, which
/// later ranges in it share), else a new one. None for memory that is not
/// committed (a stale or null address): 4 MiB of zeros would not help.
fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    if address < LOWEST_MAPPED {
        return None;
    }
    let start = address & !(WINDOW - 1);
    let len = end.checked_next_multiple_of(WINDOW)? - start;
    {
        // Nearly always the window is there: readers do not wait for each
        // other.
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        if let Some(shadow) = enclosing_window(&map, LOOSE_SPAN.load(Ordering::Relaxed), address, end) {
            return Some(shadow.clone());
        }
    }
    let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = enclosing_window(&map, LOOSE_SPAN.load(Ordering::Relaxed), address, end) {
        return Some(shadow.clone());
    }
    if !crate::texture::committed(address as *const u8, 1) {
        return None;
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
    LOOSE_SPAN.fetch_max(len, Ordering::Relaxed);
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .all
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .read()
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
    fn copies_and_compares_small_and_big() {
        // Below and above the size split over threads.
        for n in [100, 3 * PAGE + 5, PARALLEL + PART / 2 + 3] {
            let src: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
            let mut dst = vec![0u8; n + 16];
            copy_all(&[
                (src.as_ptr() as usize, dst.as_mut_ptr() as usize + 8, n - 8),
                (src.as_ptr() as usize + n - 8, dst.as_mut_ptr() as usize, 8),
            ]);
            assert_eq!(&dst[8..n], &src[..n - 8], "n {n}");
            assert_eq!(&dst[..8], &src[n - 8..], "n {n}");
            // Pieces of a page, one in three changed.
            let mut changed = src.clone();
            let mut compares = Vec::new();
            let mut want = Vec::new();
            for (k, at) in (0..n).step_by(PAGE).enumerate() {
                let len = PAGE.min(n - at);
                if k % 3 == 1 {
                    changed[at + len / 2] ^= 1;
                }
                want.push(k % 3 == 1);
                compares.push((changed.as_ptr() as usize + at, src.as_ptr() as usize + at, len));
            }
            assert_eq!(differ_all(&compares), want, "n {n}");
        }
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

    const MIB: u64 = 1 << 20;

    /// A 256 MiB parent with `children` 64 KiB buffers placed in its first
    /// part, as a heap's placed buffers.
    fn heap(base: u64, children: u64) -> Buffers {
        let mut map = Buffers::new();
        map.insert((base, 1), base + 256 * MIB);
        for i in 0..children {
            map.insert((base + i * 0x10000, 100 + i as usize), base + (i + 1) * 0x10000);
        }
        map
    }

    /// Every entry at or below the address, the closest first.
    fn enclosing_of_all(map: &Buffers, address: u64, end: u64) -> Option<(u64, usize)> {
        map.all
            .range(..=(address, usize::MAX))
            .rev()
            .find(|(_, e)| e.end >= end)
            .map(|(k, _)| *k)
    }

    #[test]
    fn a_parent_is_found_past_many_children() {
        let base = 0x10_0000_0000;
        let map = heap(base, 200);
        // Past all 200 children (and the nearest entries tried first).
        let at = base + 200 * 0x10000 + 0x100;
        assert_eq!(map.enclosing(at, at + 16), Some((base, 1)));
        let nearest = map.all.range(..=(at, usize::MAX)).rev().take(NEAREST).find(|(_, e)| e.end >= at + 16);
        assert!(nearest.is_none(), "the parent is further back than the nearest entries");
        // Near the parent's end.
        let at = base + 256 * MIB - 16;
        assert_eq!(map.enclosing(at, at + 16), Some((base, 1)));
        // Inside a child: the child, the innermost.
        let at = base + 150 * 0x10000 + 8;
        assert_eq!(map.enclosing(at, at + 8), Some((base + 150 * 0x10000, 250)));
        // Across two children: only the parent holds it.
        let at = base + 151 * 0x10000 - 8;
        assert_eq!(map.enclosing(at, at + 16), Some((base, 1)));
        // Past the parent, and before it.
        let at = base + 256 * MIB;
        assert_eq!(map.enclosing(at, at + 1), None);
        assert_eq!(map.enclosing(base - 16, base + 16), None);
        // Longer than any buffer.
        assert_eq!(map.enclosing(base, base + 256 * MIB + 1), None);
        assert_eq!(Buffers::new().enclosing(base, base + 1), None);
    }

    #[test]
    fn lookups_past_many_buffers_do_not_walk_them() {
        // A heap's buffer with 20,000 placed buffers, and memory past it
        // that no buffer holds (the title's own allocations): neither
        // lookup may try the placed buffers one by one.
        let base = 0x10_0000_0000;
        let mut map = Buffers::new();
        map.insert((base, 1), base + 256 * MIB);
        for i in 0..20_000u64 {
            map.insert((base + i * 0x1000, 100 + i as usize), base + (i + 1) * 0x1000);
        }
        let started = std::time::Instant::now();
        for i in 0..20_000u64 {
            let at = base + 20_000 * 0x1000 + i * 0x100;
            assert_eq!(map.enclosing(at, at + 64), Some((base, 1)));
            let at = base + 256 * MIB + i * 0x100;
            assert_eq!(map.enclosing(at, at + 64), None);
        }
        // Walking would be 2 x 20,000 x 20,000 entries (seconds).
        assert!(started.elapsed() < std::time::Duration::from_millis(500), "{:?}", started.elapsed());
    }

    #[test]
    fn the_innermost_buffer_is_found_among_any() {
        // Buffers of all lengths, nested, overlapping, at the same start
        // and removed again: the same answer as trying every entry.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move |below: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % below
        };
        let base = 0x30_0000_0000u64;
        let space = 64 * MIB;
        for round in 0..4 {
            let mut map = Buffers::new();
            let mut keys = Vec::new();
            for i in 0..3000usize {
                let len = match next(8) {
                    0 => 1 + next(64),
                    1..=4 => 0x100 << next(8),
                    5 | 6 => 0x10000 << next(6),
                    _ => MIB << next(6),
                };
                // Round 0 packs them at 4 KiB steps (many at one start).
                let start = base + if round == 0 { next(256) * 0x1000 } else { next(space) };
                map.insert((start, i), start + len);
                keys.push((start, i));
                if next(4) == 0 {
                    let gone = keys.swap_remove(next(keys.len() as u64) as usize);
                    map.remove(gone);
                }
            }
            assert_eq!(map.all.len(), keys.len());
            assert_eq!(map.classes.iter().map(|c| c.len()).sum::<usize>(), keys.len());
            for k in 0..64 {
                assert_eq!(map.used >> k & 1 == 1, !map.classes[k].is_empty());
            }
            for _ in 0..20_000 {
                let at = base - MIB + next(space + 2 * MIB);
                let len = if next(3) == 0 { 1 + next(4 * MIB) } else { 1 + next(0x400) };
                assert_eq!(
                    map.enclosing(at, at + len),
                    enclosing_of_all(&map, at, at + len),
                    "round {round} at {at:#x}+{len:#x}"
                );
            }
        }
    }

    #[test]
    fn registered_buffers_are_found_and_forgotten() {
        // High above anything a test maps; resource numbers no live
        // resource has.
        let base = 0x7f00_0000_0000;
        let len = 512 * MIB;
        register(1, base, len);
        for i in 0..100u64 {
            register(2 + i as usize, base + i * 0x1000, 0x1000);
        }
        let at = base + len - 64;
        let found = buffers().enclosing(at, at + 64);
        for i in 0..100u64 {
            unregister(2 + i as usize, base + i * 0x1000);
        }
        unregister(1, base);
        assert_eq!(found, Some((base, 1)));
        let map = buffers();
        assert!(map.all.range((base, 0)..=(base + len, usize::MAX)).next().is_none());
        assert!(map.classes.iter().all(|c| c.range((base, 0)..=(base + len, usize::MAX)).next().is_none()));
    }

    #[test]
    fn loose_windows_are_reused_for_ranges_inside_them() {
        let w = WINDOW;
        let base = 0x20_0000_0000;
        let mut windows = BTreeMap::new();
        windows.insert((base, w), "first");
        // A range across the next two windows got one of its own.
        windows.insert((base + w, 2 * w), "spanning");
        let span = 2 * w;
        let find = |address: u64, len: u64| enclosing_window(&windows, span, address, address + len).copied();
        assert_eq!(find(base, 1), Some("first"));
        assert_eq!(find(base + w - 16, 16), Some("first"));
        // Inside the spanning window, also in its second half.
        assert_eq!(find(base + w + 8, 64), Some("spanning"));
        assert_eq!(find(base + 2 * w + 8, 64), Some("spanning"));
        assert_eq!(find(base + 3 * w - 8, 8), Some("spanning"));
        // Across the first window's end: none holds it whole.
        assert_eq!(find(base + w - 8, 16), None);
        // Past every window, before them, longer than any.
        assert_eq!(find(base + 3 * w, 1), None);
        assert_eq!(find(base - 8, 4), None);
        assert_eq!(find(base, 3 * w), None);
    }

    #[test]
    fn loose_memory_must_be_committed() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // The null page and its guard area.
        assert!(loose(0, 16).is_none());
        assert!(loose(0x1000, 0x1010).is_none());
        assert!(loose(LOWEST_MAPPED - 1, LOWEST_MAPPED + 1).is_none());
        // Reserved, not committed.
        // SAFETY: a fresh reservation, released below.
        let p = unsafe { VirtualAlloc(std::ptr::null(), WINDOW as usize, MEM_RESERVE, PAGE_READWRITE) } as u64;
        assert_ne!(p, 0);
        assert!(loose(p + 64, p + 128).is_none());
        // SAFETY: our reservation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        // Too close to the top to round up to a window.
        assert!(loose(u64::MAX - 8, u64::MAX).is_none());
    }
}

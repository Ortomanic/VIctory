//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::host;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Dynamic slots past the mirrored ones (view heaps; 0 for samplers).
    dynamic: u32,
    /// Which dynamic slots are taken (by lists not yet run, or running).
    next: Mutex<Slots>,
}

/// Dynamic slots: which are taken (a bit each), how many are free, and
/// where the search for free ones starts. Searched a word at a time: a
/// world frame takes thousands of runs, and when the slots ran short a
/// slot-by-slot search of all 65,536 under the heap's lock cost recording
/// threads up to ~0.1 ms per table.
///
/// A list's slots are taken while it is recorded and while it runs, and
/// lent back once its submission has finished (`lend`): a title may run a
/// closed list again without recording it, so a lent run keeps its
/// descriptors (and its owner token) until another list needs the slots.
/// Free slots nobody lent are taken first; lent ones only when those run
/// short. A list run again takes its lent slots back (`reclaim`) if nobody
/// took them since.
struct Slots {
    next: usize,
    taken: Vec<u64>,
    /// Free slots whose descriptors a list that has run still owns.
    lent: Vec<u64>,
    /// The token of the run last taken at each slot (0: none).
    owner: Vec<u32>,
    /// The last token handed out.
    token: u32,
    len: usize,
    free: usize,
    /// Diagnostics: the most slots taken at once, since the last ask.
    high: usize,
    /// Runs taken from lent slots (the lender can no longer run again).
    stolen: u64,
}

impl Slots {
    fn new(n: usize) -> Slots {
        let words = n.div_ceil(64);
        Slots {
            next: 0,
            taken: vec![0; words],
            lent: vec![0; words],
            owner: vec![0; n],
            token: 0,
            len: n,
            free: n,
            high: 0,
            stolen: 0,
        }
    }

    #[cfg(test)]
    fn is_taken(&self, at: usize) -> bool {
        self.taken[at / 64] >> (at % 64) & 1 == 1
    }

    #[cfg(test)]
    fn is_lent(&self, at: usize) -> bool {
        self.lent[at / 64] >> (at % 64) & 1 == 1
    }

    fn set_lent(&mut self, at: usize, lent: bool) {
        let bit = 1u64 << (at % 64);
        if lent {
            self.lent[at / 64] |= bit;
        } else {
            self.lent[at / 64] &= !bit;
        }
    }

    /// The slots of `[first, first + n)` that run `token` still owns, as
    /// runs (the rest were taken by other lists since).
    fn owned(&self, first: usize, n: usize, token: u32) -> impl Iterator<Item = usize> + '_ {
        let end = (first + n).min(self.len);
        (first.min(end)..end).filter(move |&at| self.owner[at] == token)
    }

    /// After the run's submission finished: its slots are free, but keep
    /// its descriptors until another list needs them. Slots counted.
    fn lend(&mut self, first: usize, n: usize, token: u32) -> usize {
        let mine: Vec<usize> = self.owned(first, n, token).collect();
        for &at in &mine {
            self.mark(at, at + 1, false);
            self.set_lent(at, true);
        }
        mine.len()
    }

    /// A lent run taken back for another run of its list: false if
    /// another list took any of its slots since (their descriptors are
    /// that list's now).
    fn reclaim(&mut self, first: usize, n: usize, token: u32) -> bool {
        let end = first + n;
        if end > self.len || self.owned(first, n, token).count() != n {
            return false;
        }
        if (first..end).any(|at| self.taken[at / 64] >> (at % 64) & 1 == 1) {
            return false;
        }
        self.mark(first, end, true);
        (first..end).for_each(|at| self.set_lent(at, false));
        self.high = self.high.max(self.len - self.free);
        true
    }

    /// The run's list was reset or released: whatever of it is still its
    /// own is free for good.
    fn release(&mut self, first: usize, n: usize, token: u32) {
        let mine: Vec<usize> = self.owned(first, n, token).collect();
        for at in mine {
            self.mark(at, at + 1, false);
            self.set_lent(at, false);
            self.owner[at] = 0;
        }
    }

    /// The first run of `n` free slots inside `[lo, hi)`: its start. Runs
    /// of taken and free slots are stepped over whole (bit counts), so a
    /// search costs about a step per run, not per slot: when the slots
    /// ran short and were fragmented, a slot-by-slot search of all of them
    /// at every table cost the world's recording ~0.7 s per frame.
    fn find(&self, lo: usize, hi: usize, n: usize, spare_lent: bool) -> Option<usize> {
        let (mut run, mut start, mut at) = (0, lo, lo);
        while at < hi {
            let bit = at % 64;
            let blocked = if spare_lent {
                self.taken[at / 64] | self.lent[at / 64]
            } else {
                self.taken[at / 64]
            };
            let word = blocked >> bit;
            let avail = (64 - bit).min(hi - at);
            if word & 1 == 1 {
                // Taken slots from `at` on.
                run = 0;
                at += (word.trailing_ones() as usize).min(avail);
            } else {
                let free = if word == 0 {
                    avail
                } else {
                    (word.trailing_zeros() as usize).min(avail)
                };
                if run == 0 {
                    start = at;
                }
                run += free;
                if run >= n {
                    return Some(start);
                }
                at += free;
            }
        }
        None
    }

    /// Marks `[from, to)` taken or free.
    fn mark(&mut self, from: usize, to: usize, taken: bool) {
        for at in from..to {
            let (word, bit) = (at / 64, 1u64 << (at % 64));
            if (self.taken[word] & bit != 0) != taken {
                self.taken[word] ^= bit;
                if taken {
                    self.free -= 1;
                } else {
                    self.free += 1;
                }
            }
        }
    }

    /// `n` consecutive free slots, taken: the first's index. Searched from
    /// where the last search ended on, then from the start; a run does not
    /// wrap past the end.
    #[cfg(test)]
    fn take(&mut self, n: usize) -> Option<usize> {
        self.take_run(n).map(|(first, _)| first)
    }

    /// [`Slots::take`], with the run's owner token. Slots nobody lent
    /// first; lent ones (their lenders' descriptors overwritten) only when
    /// no run of those fits.
    fn take_run(&mut self, n: usize) -> Option<(usize, u32)> {
        let end = self.len;
        if n == 0 || n > self.free {
            return None;
        }
        let next = self.next;
        let first = match self
            .find(next, end, n, true)
            .or_else(|| self.find(0, next, n, true))
        {
            Some(first) => first,
            None => {
                let first = self
                    .find(next, end, n, false)
                    .or_else(|| self.find(0, next, n, false))?;
                self.stolen += 1;
                first
            }
        };
        self.mark(first, first + n, true);
        self.token = self.token.wrapping_add(1).max(1);
        for at in first..first + n {
            self.set_lent(at, false);
            self.owner[at] = self.token;
        }
        self.next = (first + n) % end;
        self.high = self.high.max(self.len - self.free);
        Some((first, self.token))
    }
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
/// Translated shaders clamp view heap indices below it.
const PORTABLE_VIEWS: u32 = victory_shader::emit::HEAP_VIEWS;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until its submission has run (then lends them back, see
/// [`Slots`]), and gives them back for good when it is reset or released.
/// At least this many; when the host's heap limit is above the portable
/// one, all the room past the mirrored title slots.
const DYNAMIC: u32 = 1 << 16;

/// The host's largest shader-visible view heap (D3D12 options 19), or the
/// portable limit if the host does not say. vkd3d-proton with
/// `VK_EXT_descriptor_heap` reports what the driver's heap holds (RTX
/// 5080, driver 620: 1,041,451); without it, 1,000,000.
fn max_views(device: &ID3D12Device) -> u32 {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS19::default();
    // SAFETY: the query writes its own structure of the given size.
    let ok = unsafe {
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS19,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .is_ok()
    };
    let max = if ok { options.MaxViewDescriptorHeapSize } else { 0 };
    tracing::info!(max, "host view heap limit");
    max.max(PORTABLE_VIEWS)
}

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        let mut extra = if sampler { 0 } else { DYNAMIC };
        let max = if sampler { u32::MAX } else { max_views(&host.device) };
        let (heap, count) = if count.saturating_add(extra) <= max {
            match make(count + extra) {
                Ok(heap) => (heap, count),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else {
            // shortcut: descriptors past the host's limit are left out (the
            // tables reaching them are made afresh in dynamic slots); a title
            // that indexes them directly needs remapping. The title slots
            // mirrored stay those below the portable limit less the minimum
            // of dynamic slots (translated shaders clamp direct indices below
            // the portable limit); a larger host heap gives the rest to
            // dynamic slots.
            let mirrored = count.min(PORTABLE_VIEWS - DYNAMIC);
            let mut total = max;
            let heap = loop {
                match make(total) {
                    Ok(heap) => break heap,
                    Err(e) if total > PORTABLE_VIEWS => {
                        tracing::warn!(error = %e, total, "host view heap refused; the portable size instead");
                        total = PORTABLE_VIEWS;
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, count, "no host view heap");
                        return None;
                    }
                }
            };
            extra = total - mirrored;
            tracing::warn!(
                count,
                mirrored,
                dynamic = extra,
                "host view heap capped at {total}"
            );
            (heap, mirrored)
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(count + extra) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Mutex::new(vec![0; count as usize]),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            next: Mutex::new(Slots::new(extra as usize)),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, count + extra) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::lend`] or [`HostHeap::release`]: the first's index and
    /// the run's owner token.
    pub(crate) fn dynamic(&self, n: u32) -> Option<(u32, u32)> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let found = self.next.lock().unwrap_or_else(|e| e.into_inner()).take_run(n as usize);
        let Some((first, token)) = found else {
            static MISSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let missed = MISSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if missed.is_power_of_two() {
                tracing::warn!(n, missed, "no free dynamic descriptor slots: a table is not made");
            }
            return None;
        };
        Some((self.count + first as u32, token))
    }

    /// The dynamic slots: in all, taken now, the most taken at once since
    /// the last call, and runs taken from lent slots since the last call.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn dynamic_use(&self) -> (u32, u32, u32, u64) {
        let mut slots = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let now = slots.len - slots.free;
        let high = std::mem::replace(&mut slots.high, now);
        let stolen = std::mem::take(&mut slots.stolen);
        (self.dynamic, now as u32, high as u32, stolen)
    }

    /// The run of `n` from `first` (an index [`dynamic`] returned with
    /// `token`) has finished running: lent back, its descriptors kept
    /// until another list needs the slots. The slots lent.
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn lend(&self, first: u32, n: u32, token: u32) -> u32 {
        let Some(at) = first.checked_sub(self.count) else {
            return 0;
        };
        let mut slots = self.next.lock().unwrap_or_else(|e| e.into_inner());
        slots.lend(at as usize, n as usize, token) as u32
    }

    /// A lent run taken back to run its list again; false if another list
    /// took any of its slots.
    pub(crate) fn reclaim(&self, first: u32, n: u32, token: u32) -> bool {
        let Some(at) = first.checked_sub(self.count) else {
            return false;
        };
        let mut slots = self.next.lock().unwrap_or_else(|e| e.into_inner());
        slots.reclaim(at as usize, n as usize, token)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned with `token`) for good: those another list took since it
    /// was lent are left to that list.
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32, token: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut slots = self.next.lock().unwrap_or_else(|e| e.into_inner());
        slots.release(at as usize, n as usize, token);
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            // A running submission may still read the old resource through
            // this slot.
            if let Some(old) = std::mem::replace(slot, resource) {
                crate::host::retire(old);
            }
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.next_multiple_of(256),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    /// Slot `index` (a dynamic one) is made by the caller, which writes
    /// its view at [`HostHeap::slot_handles`] and holds `resource` there.
    pub(crate) fn hold(&self, index: u32, resource: ID3D12Resource) {
        self.keep(index, Some(resource));
    }

    /// Slot `index`'s CPU and GPU handles.
    pub(crate) fn slot_handles(
        &self,
        index: u32,
    ) -> (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE) {
        // SAFETY: a live shader-visible heap of ours.
        let gpu = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        (
            self.cpu(index),
            D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu.ptr + u64::from(index) * u64::from(self.step),
            },
        )
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made[index as usize] != 0 {
                made[index as usize] = 0;
                self.make_null(index);
            }
            return None;
        };
        if made[index as usize] != key {
            let later = !for_draw && view.is_buffer();
            made[index as usize] = if later { 0 } else { key };
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(index))
            };
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            self.keep(index, kept.flatten());
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// A structured view's bytes as a raw view of its host copy: the first
/// dword and the dword count, for a view `offset` bytes into the copy whose
/// description starts at element `first` and has `count` elements of
/// `stride` bytes.
///
/// The host places structured element `i` at byte `i * stride` of the
/// resource, so a structured view whose start in the copy is not a multiple
/// of its stride (a title buffer's view 16 bytes past a 32-byte element)
/// cannot be made there. A raw view of the same bytes can if its start is
/// 16-byte aligned (`D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT`; the host removes the
/// device for one that is not), and it reads and writes the copy every other
/// view of that memory uses. Translated shaders read buffer views raw unless
/// they index them (`victory-shader`, "Buffer views in the heap").
fn raw_window(offset: u64, first: u64, count: u32, stride: u32) -> Option<(u64, u32)> {
    if stride == 0 || !stride.is_multiple_of(4) {
        return None;
    }
    let start = offset.checked_add(first.checked_mul(u64::from(stride))?)?;
    if !start.is_multiple_of(u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT)) {
        return None;
    }
    let dwords = u64::from(count) * u64::from(stride / 4);
    Some((start / 4, u32::try_from(dwords).ok()?))
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let offset = shadow.offset(address);
    if element != 0 && offset % element == 0 {
        buffer.FirstElement += offset / element;
    } else if let Some((first, count)) = raw_window(
        offset,
        buffer.FirstElement,
        buffer.NumElements,
        buffer.StructureByteStride,
    ) {
        // shortcut: a shader that indexes the view declares it structured,
        // and D3D12 leaves structured reads of a raw view undefined (NVIDIA
        // and WARP read the right bytes).
        once!(
            offset,
            element,
            "structured buffer view off its host copy's elements made raw"
        );
        *buffer = D3D12_BUFFER_SRV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
        };
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    } else {
        once!(
            offset,
            element,
            "buffer view not on an element of its host copy"
        );
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    buffer.CounterOffsetInBytes = 0;
    if element != 0 && offset % element == 0 {
        buffer.FirstElement += offset / element;
    } else if let Some((first, count)) = raw_window(
        offset,
        buffer.FirstElement,
        buffer.NumElements,
        buffer.StructureByteStride,
    ) {
        // As in `buffer_srv`; what shaders write lands in the shared copy,
        // in the range `buffer_range` gives (copied back to title memory).
        once!(
            offset,
            element,
            "structured unordered buffer view off its host copy's elements made raw"
        );
        *buffer = D3D12_BUFFER_UAV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            CounterOffsetInBytes: 0,
            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
        };
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    } else {
        once!(offset, element, "unordered buffer view not on an element of its host copy");
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless;
    }
    // A target made in a stand-in format on this GPU (packed 16-bit as
    // R8G8B8A8_UNORM, R9G9B9E5 as R16G16B16A16_FLOAT;
    // `resource::target_format`): views name the format it has.
    if view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: size.next_multiple_of(256),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let mut desc = desc.as_ref().map(uav_desc);
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(dimension = resource[0], "placed unordered texture view not mirrored");
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// The word-at-a-time search finds what a slot-by-slot first fit does.
    #[test]
    fn slot_search_is_first_fit_from_the_last_end() {
        let mut slots = Slots::new(1000);
        assert_eq!(slots.take(130), Some(0));
        assert_eq!(slots.take(3), Some(130));
        slots.mark(10, 20, false);
        slots.mark(64, 200, false);
        // From 133 on first, then from the start.
        assert_eq!(slots.take(850), Some(133));
        assert_eq!(slots.take(70), None);
        assert_eq!(slots.take(67), Some(64));
        assert_eq!(slots.take(10), Some(983));
        assert_eq!(slots.take(10), Some(10));
        assert_eq!(slots.free, 2 + 7);
        assert!(!slots.is_taken(131) && !slots.is_taken(132) && slots.is_taken(133));
        // A run does not wrap past the end.
        let mut slots = Slots::new(256);
        slots.mark(0, 256, true);
        slots.mark(250, 256, false);
        slots.mark(0, 4, false);
        assert_eq!(slots.take(8), None);
        assert_eq!(slots.take(6), Some(250));
        assert_eq!(slots.take(4), Some(0));
        assert_eq!(slots.free, 0);
    }

    /// Against a slot-by-slot first fit, over random takes and gives.
    #[test]
    fn slot_search_matches_a_slot_by_slot_search() {
        let len = 3000;
        let mut slots = Slots::new(len);
        let mut naive = vec![false; len];
        let mut next = 0;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut random = move |below: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % below as u64) as usize
        };
        let mut held: Vec<(usize, usize)> = Vec::new();
        for _ in 0..4000 {
            if random(3) > 0 || held.is_empty() {
                let n = 1 + random(90);
                let find = |lo: usize, hi: usize, naive: &[bool]| {
                    let mut run = 0;
                    (lo..hi).find(|&at| {
                        run = if naive[at] { 0 } else { run + 1 };
                        run == n
                    })
                    .map(|at| at + 1 - n)
                };
                let want = find(next, len, &naive).or_else(|| find(0, next, &naive));
                assert_eq!(slots.take(n), want);
                if let Some(first) = want {
                    naive[first..first + n].fill(true);
                    next = (first + n) % len;
                    held.push((first, n));
                }
            } else {
                let (first, n) = held.swap_remove(random(held.len()));
                slots.mark(first, first + n, false);
                naive[first..first + n].fill(false);
            }
            assert_eq!(slots.free, naive.iter().filter(|t| !**t).count());
        }
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let all = heap.dynamic;
        let (a, ta) = heap.dynamic(all - 100).unwrap();
        let (b, _) = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + all - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50, ta);
        assert_eq!(heap.dynamic(50).map(|d| d.0), Some(4));
        assert_eq!(heap.dynamic(40).map(|d| d.0), Some(4 + all - 40));
    }

    /// A run lent back keeps its slots for its list until another list
    /// needs them: free slots nobody lent go first, then lent ones, and a
    /// list whose lent run was taken cannot take it back.
    #[test]
    fn lent_slots_are_taken_last_and_reclaimed_if_untouched() {
        let mut slots = Slots::new(256);
        let (a, ta) = slots.take_run(100).unwrap();
        let (b, tb) = slots.take_run(100).unwrap();
        assert_eq!((a, b), (0, 100));
        assert_ne!(ta, tb);
        // Run a finished: lent, free, but not handed out while others are.
        assert_eq!(slots.lend(a, 100, ta), 100);
        assert_eq!(slots.free, 156);
        assert!(slots.is_lent(0) && !slots.is_taken(0));
        let (c, _) = slots.take_run(50).unwrap();
        assert_eq!(c, 200);
        assert_eq!(slots.stolen, 0);
        // Run a again: taken back whole.
        assert!(slots.reclaim(a, 100, ta));
        assert!(!slots.is_lent(0) && slots.is_taken(99));
        assert_eq!(slots.free, 6);
        // A taken run cannot be reclaimed (it is not lent).
        assert!(!slots.reclaim(a, 100, ta));
        // Lent again, then needed: the 6 unlent slots are too few, so a
        // lent run is taken.
        slots.lend(a, 100, ta);
        let (d, td) = slots.take_run(30).unwrap();
        assert_eq!(slots.stolen, 1);
        assert!(d < 100);
        assert!(!slots.reclaim(a, 100, ta), "part of it is another run's now");
        // Released for good: what is still a's is free and not lent; d
        // keeps its own.
        slots.release(a, 100, ta);
        assert!((0..100).all(|at| !slots.is_lent(at)));
        assert!((d..d + 30).all(|at| slots.is_taken(at) && slots.owner[at] == td));
        assert_eq!(slots.free, 256 - 100 - 50 - 30);
        // Lending a run another list took part of lends only its own.
        assert_eq!(slots.lend(d, 30, td), 30);
        assert_eq!(slots.lend(b, 100, tb), 100);
        assert_eq!(slots.free, 256 - 50);
        assert_eq!(slots.high, 256 - 6);
    }

    /// A structured view whose start in its host copy is not on an element
    /// becomes the raw view of the same bytes, where those start 16-byte
    /// aligned.
    #[test]
    fn structured_views_off_an_element_are_raw_views_of_the_same_bytes() {
        // (offset in the copy, first element, elements, stride)
        for (offset, first, count, stride) in [
            (16, 0, 4, 32),
            (48, 3, 7, 32),
            (4096 + 16, 1, 1, 48),
            (32, 0, 9, 12),
        ] {
            let (dword, dwords) = raw_window(offset, first, count, stride).unwrap();
            assert_eq!(dword * 4, offset + first * u64::from(stride));
            assert_eq!(u64::from(dwords) * 4, u64::from(count) * u64::from(stride));
        }
        // Not 16-byte aligned, strides raw views cannot cover, too long.
        assert_eq!(raw_window(8, 0, 4, 32), None);
        assert_eq!(raw_window(16, 1, 4, 12), None);
        assert_eq!(raw_window(16, 0, 4, 6), None);
        assert_eq!(raw_window(16, 0, 4, 0), None);
        assert_eq!(raw_window(16, 0, u32::MAX, 32), None);
    }

    /// A structured view (32-byte elements) 16 bytes into its host copy,
    /// read and another written by a shader: it reads the title's bytes from
    /// there, and what it writes lands at its own bytes of title memory.
    #[test]
    fn structured_views_off_an_element_read_and_write_their_bytes() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;
        let Some(host) = host() else {
            return; // no host GPU
        };
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0), UAV(u0, numDescriptors = 2))"
ByteAddressBuffer view : register(t0);
RWByteAddressBuffer window : register(u0);
RWByteAddressBuffer seen : register(u1);
[RootSignature(RS)]
[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    seen.Store(id.x * 4, view.Load(id.x * 4));
    window.Store(id.x * 4, 0xc0de0000u | id.x);
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointer ours.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("views"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .unwrap();
        let code = code.unwrap();
        let device = &host.device;
        // SAFETY: the blob's own bytes, with the root signature inside;
        // plain object creation.
        let (root, pso, views, step) = unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            let views: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 3,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            let step =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            (root, pso, views, step as usize)
        };
        // SAFETY: a live heap.
        let first = unsafe { views.GetCPUDescriptorHandleForHeapStart() }.ptr;
        let at = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: first + i * step,
        };

        // A title buffer of 1 KiB, numbered dwords; four 32-byte elements
        // from byte 16 are read, four from byte 272 written.
        let before: Vec<u32> = (0..256).map(|i| 0x7171_0000 | i).collect();
        let mut title = before.clone();
        let start = title.as_mut_ptr() as u64;
        crate::shadow::register(start as usize, start, 1024);
        let shadow = crate::shadow::find(start, 1024).unwrap();
        let structured = D3D12_BUFFER_SRV {
            NumElements: 4,
            StructureByteStride: 32,
            ..Default::default()
        };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: structured },
        };
        let uav = |buffer| D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: buffer },
        };
        let written = D3D12_BUFFER_UAV {
            NumElements: 4,
            StructureByteStride: 32,
            ..Default::default()
        };
        // Views the host refuses are left null, as `HostHeap::mirror` leaves them.
        let mut placed = [0; 14];
        placed[0] = 1; // a buffer
        if buffer_srv(start + 16, 128, srv, at(0)).is_none() {
            null_for(&View::Placed(false, placed, [0; 16]), at(0));
        }
        if buffer_uav(start + 272, 128, uav(written), at(1)).is_none() {
            null_for(&View::Placed(true, placed, [0; 16]), at(1));
        }
        // What the shader read, as a raw view of 32 dwords of our own.
        let (seen, out) = host.shared_buffer(128).unwrap();
        let mut raw = uav(D3D12_BUFFER_UAV {
            NumElements: 32,
            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
            ..Default::default()
        });
        raw.Format = DXGI_FORMAT_R32_TYPELESS;
        // SAFETY: a live buffer and a slot of our heap.
        unsafe {
            device.CreateUnorderedAccessView(&seen, None::<&ID3D12Resource>, Some(&raw), at(2))
        };

        let list = host.new_list().unwrap();
        shadow.refresh(&list, start, 1024);
        shadow.gpu_writes(&list, start + 272, 128);
        // SAFETY: an open list; our heap, root signature and pipeline.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(views.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRootDescriptorTable(0, views.GetGPUDescriptorHandleForHeapStart());
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        crate::shadow::unregister(start as usize, start);

        // SAFETY: 32 dwords the GPU wrote, read after it finished.
        let seen = unsafe { std::slice::from_raw_parts(out.cast::<u32>(), 32) };
        assert_eq!(seen, &before[4..36], "what the read view gave");
        let marks: Vec<u32> = (0..32).map(|i| 0xc0de_0000 | i).collect();
        assert_eq!(&title[68..100], &marks[..], "what the written view left");
        assert_eq!(
            (&title[..68], &title[100..]),
            (&before[..68], &before[100..])
        );
    }
}

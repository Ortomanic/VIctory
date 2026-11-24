//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
});

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // View memory cannot be write-watched: where it is known to be a view,
    // the kernel is not asked. Under Wine every query takes the process's
    // virtual-memory lock, which the title's mapping threads hold across
    // wineserver calls: a microsecond alone, tens to thousands of them
    // with four threads mapping views (/tmp/wwbench2 `contend`).
    let view = guarded && known_view(first * PAGE);
    #[cfg(feature = "diagnostics")]
    if view {
        crate::copystat::add(crate::copystat::C::KernelSkips, 1);
    }
    // SAFETY: the answer buffer holds `count` entries.
    let failed = view || unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::WatchKernelUs, started.elapsed().as_micros() as u64);
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        guard(first * PAGE, last * PAGE)?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    let pages = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded);
    for page in pages.collect::<Vec<_>>() {
        j.written.insert(page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Blocks that could not be guarded because nothing is committed there
    /// (placeholders between the tiled heap's page views), with the frame
    /// ([`FRAME`]) they were last found so: nothing to write, so not
    /// reported written until the runtime maps them
    /// ([`title_memory_remapped`]). Once a frame a hole is looked at again
    /// (one region query for its run), which bounds a commit the runtime
    /// does not report to a frame. Reported written at every asking before
    /// round 1, each was copied (reading as zeros through the unmapped-page
    /// fallback) at every submission; reported again at each frame's first
    /// asking after it, ~800 such fallback copies a world frame.
    holes: BTreeMap<usize, u64>,
}

/// Counts presented frames ([`new_frame`]).
static FRAME: AtomicU64 = AtomicU64::new(1);

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    holes: BTreeMap::new(),
});

/// What [`mapped`] found, by the start of the allocation (or region) asked
/// about: its end and whether it is a view. Forgotten where the runtime
/// maps or unmaps title memory. Asking the host costs: under Wine 11 the
/// allocation query walks the whole allocation (~0.2-0.4 ms for a GiB
/// section view) and reports no type, and `VirtualQuery` then walks the
/// region (~0.1 ms), at every guarded asking (thousands per world frame).
static KINDS: Mutex<BTreeMap<usize, (usize, bool)>> = Mutex::new(BTreeMap::new());

/// Forgets the entries of `kinds` overlapping `[from, to)` (they do not
/// overlap each other).
fn forget_kinds(kinds: &mut BTreeMap<usize, (usize, bool)>, from: usize, to: usize) {
    forget(kinds, from, to, |&(end, _)| end);
}

/// Forgets the extents of `map` (start → value holding the end, as `end`
/// reads it; they do not overlap each other) overlapping `[from, to)`.
fn forget<V>(map: &mut BTreeMap<usize, V>, from: usize, to: usize, end: impl Fn(&V) -> usize) {
    let hit: Vec<usize> = map
        .range(..to)
        .rev()
        .take_while(|(_, v)| end(v) > from)
        .map(|(&start, _)| start)
        .collect();
    for start in hit {
        map.remove(&start);
    }
}

/// View regions found committed, start → end, until the runtime remaps
/// them ([`title_memory_remapped`]): a view's pages stay committed until it
/// is unmapped, and the runtime reports that before it unmaps. Private
/// memory is asked every time (it may be decommitted unreported).
static COMMITTED: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

/// Adjacent view regions found committed are kept as one run of up to this
/// many bytes: physical-page memory is a view per 64 KiB block, and a
/// texture's check then reads a few runs, not an entry per block. A
/// remapping of any part forgets the whole run (asked again), so runs stay
/// short where streaming remaps blocks one by one.
const COMMITTED_RUN: usize = 2 << 20;

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive; copies into title memory must not
/// meet a hole). Asked of the host once per view region: textures are
/// checked at every submission that samples them, and under Wine each
/// query takes the process's virtual-memory lock and walks the region
/// (~80 us for a 4 GiB view of uniform protection).
pub(crate) fn committed(at: usize, len: usize) -> bool {
    committed_asking(at, len).0
}

/// [`committed`], and whether the host was asked (some of the range was not
/// known committed views).
pub(crate) fn committed_asking(at: usize, len: usize) -> (bool, bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let end = at.saturating_add(len);
    let mut p = at;
    let mut asked = false;
    while p < end {
        {
            // The known runs from `p` on, under one lock.
            let known = COMMITTED.lock().unwrap_or_else(|e| e.into_inner());
            while p < end {
                let Some(e) = known
                    .range(..=p)
                    .next_back()
                    .map(|(_, &e)| e)
                    .filter(|&e| p < e)
                else {
                    break;
                };
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::CommitHits, 1);
                p = e;
            }
        }
        if p >= end {
            break;
        }
        asked = true;
        let remappings = REMAPPINGS.load(Ordering::Acquire);
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::CommitQueries, 1);
        if n == 0 || info.State != MEM_COMMIT || info.RegionSize == 0 {
            return (false, true);
        }
        let (from, to) = (
            info.BaseAddress as usize,
            (info.BaseAddress as usize).saturating_add(info.RegionSize),
        );
        if info.Type == MEM_MAPPED {
            let mut known = COMMITTED.lock().unwrap_or_else(|e| e.into_inner());
            // A remapping since the query may have changed what it found.
            if REMAPPINGS.load(Ordering::Acquire) == remappings {
                forget(&mut known, from, to, |&e| e);
                keep_committed(&mut known, from, to);
            }
        }
        p = to;
    }
    (true, asked)
}

/// Notes `[from, to)` (overlapping nothing in `known`) committed, joined
/// with the runs it adjoins while the run stays within [`COMMITTED_RUN`].
fn keep_committed(known: &mut BTreeMap<usize, usize>, mut from: usize, mut to: usize) {
    if let Some((&start, _)) = known
        .range(..from)
        .next_back()
        .filter(|&(&start, &e)| e == from && to - start <= COMMITTED_RUN)
    {
        known.remove(&start);
        from = start;
    }
    if let Some(&e) = known.get(&to).filter(|&&e| e - from <= COMMITTED_RUN) {
        known.remove(&to);
        to = e;
    }
    known.insert(from, to);
}

/// A frame was presented: holes are looked at again (once a frame, not at
/// every submission), in case memory was committed there unreported.
pub(crate) fn new_frame() {
    FRAME.fetch_add(1, Ordering::AcqRel);
}

/// Whether `block` is a hole: found so this frame, or found uncommitted
/// now (then so are the holes after it in the same region, noted with it).
/// A hole found committed is a hole no longer.
fn still_hole(g: &mut Guards, block: usize, frame: u64) -> bool {
    match g.holes.get(&block) {
        None => false,
        Some(&seen) if seen == frame => true,
        Some(_) => match uncommitted(block * BLOCK) {
            Some(end) => {
                for (_, seen) in g.holes.range_mut(block..(end / BLOCK).max(block + 1)) {
                    *seen = frame;
                }
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::HoleChecks, 1);
                true
            }
            None => {
                g.holes.remove(&block);
                false
            }
        },
    }
}

/// Counts the runtime's mappings and unmappings of title memory
/// ([`title_memory_remapped`]): what [`crate::host::Regions`] learnt of the
/// memory before one is stale after it.
pub(crate) static REMAPPINGS: AtomicU64 = AtomicU64::new(0);

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    if !mapped(start) {
        return None;
    }
    install();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let frame = FRAME.load(Ordering::Acquire);
    for block in first..=last {
        let open = block < last && !g.clean.contains(&block);
        if open && still_hole(&mut g, block, frame) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::HoleSkips, 1);
        } else if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Whether [`mapped`] found the allocation at `at` to be a view (and the
/// runtime has not remapped it since); false if not or not known.
fn known_view(at: usize) -> bool {
    let kinds = KINDS.lock().unwrap_or_else(|e| e.into_inner());
    kinds
        .range(..=at)
        .next_back()
        .is_some_and(|(_, &(end, view))| view && at < end)
}

/// Whether the current allocation at `at` is a mapped view, as last found
/// for its allocation ([`KINDS`]) since the runtime remapped it.
fn mapped(at: usize) -> bool {
    {
        let kinds = KINDS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, &(end, view))) = kinds.range(..=at).next_back()
            && at < end
        {
            return view;
        }
    }
    let remappings = REMAPPINGS.load(Ordering::Acquire);
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let (view, extent) = query_kind(at);
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::RegionQueries, 1);
        add(C::RegionQueryUs, started.elapsed().as_micros() as u64);
    }
    if let Some((from, to)) = extent
        && from <= at
        && at < to
    {
        let mut kinds = KINDS.lock().unwrap_or_else(|e| e.into_inner());
        // A remapping since the query may have changed what it found
        // (the runtime reports an unmapping before it unmaps).
        if REMAPPINGS.load(Ordering::Acquire) == remappings {
            forget_kinds(&mut kinds, from, to);
            kinds.insert(from, (to, view));
        }
    }
    view
}

/// Whether the allocation at `at` is a mapped view, and the extent the
/// answer holds for (the allocation, or the region asked about), unless
/// the address is free. Asking only for the allocation type avoids
/// enumerating pages with identical protection, which can span an entire
/// large view after its blocks were guarded (on Windows; Wine reports no
/// type there, so the region query follows).
fn query_kind(at: usize) -> (bool, Option<(usize, usize)>) {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_MAPPED, MEMORY_BASIC_INFORMATION, MemoryRegionInfo, QueryVirtualMemoryInformation,
        VirtualQuery, WIN32_MEMORY_REGION_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut region = WIN32_MEMORY_REGION_INFORMATION::default();
    // SAFETY: both queries describe an address without dereferencing it. The
    // output buffers match their information classes.
    let queried = unsafe {
        QueryVirtualMemoryInformation(
            GetCurrentProcess(),
            at as *const _,
            MemoryRegionInfo,
            (&raw mut region).cast(),
            size_of_val(&region),
            std::ptr::null_mut(),
        )
    };
    let base = region.AllocationBase as usize;
    let allocation = (queried != 0 && region.RegionSize != 0)
        .then(|| (base, base.saturating_add(region.RegionSize)));
    if queried != 0 {
        // These are the Windows SDK's named bitfields in declaration order.
        // windows-sys exposes only the union's raw bits.
        const PRIVATE: u32 = 1 << 0;
        const MAPPED_DATA_FILE: u32 = 1 << 1;
        const MAPPED_IMAGE: u32 = 1 << 2;
        const MAPPED_PAGE_FILE: u32 = 1 << 3;
        const MAPPED_PHYSICAL: u32 = 1 << 4;
        // SAFETY: this union was filled by the successful query above.
        match unsafe { region.Anonymous.Flags } {
            MAPPED_DATA_FILE | MAPPED_PAGE_FILE | MAPPED_PHYSICAL => return (true, allocation),
            PRIVATE | MAPPED_IMAGE => return (false, allocation),
            _ => {}
        }
    }
    // Unknown flag combinations and failed allocation queries keep the
    // VirtualQuery type check, including for a free address.
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: the output buffer has the required layout and size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.State == MEM_FREE {
        return (false, None);
    }
    let extent = match allocation {
        // The same allocation as the allocation query's.
        Some(a) if info.AllocationBase as usize == a.0 => Some(a),
        _ => {
            let region = info.BaseAddress as usize;
            Some((region, region.saturating_add(info.RegionSize)))
        }
    };
    (info.Type == MEM_MAPPED, extent)
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail. Those
/// with nothing committed are noted as holes.
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        // SAFETY: title view memory; only its protection changes.
        let done = unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        };
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Protects, 1);
            add(C::ProtectUs, started.elapsed().as_micros() as u64);
        }
        done
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    let mut block = from;
    while block < to {
        if set(block, block + 1) {
            note(g, block, block + 1);
            block += 1;
            continue;
        }
        // Not committed: neither are the blocks wholly in the rest of its
        // region, which are not tried one by one. A block committed in
        // part stays as it was (reported written at every asking).
        match uncommitted(block * BLOCK).map(|end| (end / BLOCK).min(to)) {
            Some(past) if past > block => {
                let frame = FRAME.load(Ordering::Acquire);
                g.holes.extend((block..past).map(|b| (b, frame)));
                block = past;
            }
            _ => block += 1,
        }
    }
}

/// If nothing is committed at `at` (reserved or free memory, a
/// placeholder): where that region ends.
fn uncommitted(at: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    (n != 0 && info.State != MEM_COMMIT && info.RegionSize != 0)
        .then(|| (info.BaseAddress as usize).saturating_add(info.RegionSize))
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded.
pub fn title_memory_remapped(at: usize, len: usize) {
    REMAPPINGS.fetch_add(1, Ordering::AcqRel);
    let to = at.saturating_add(len.max(1));
    forget_kinds(&mut KINDS.lock().unwrap_or_else(|e| e.into_inner()), at, to);
    forget(&mut COMMITTED.lock().unwrap_or_else(|e| e.into_inner()), at, to, |&e| e);
    let mut g = guards();
    let span = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let blocks: Vec<usize> = g.protected.range(span.clone()).copied().collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
    let holes: Vec<usize> = g.holes.range(span).map(|(&b, _)| b).collect();
    for block in holes {
        g.holes.remove(&block);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn allocation_query_recognizes_file_views_and_rejects_nonviews() {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, MapViewOfFile, PAGE_READONLY, UnmapViewOfFile,
        };
        let file = tempfile::tempfile().unwrap();
        file.set_len(PAGE as u64).unwrap();
        // SAFETY: a valid, open file and a read-only mapping of its contents.
        let section = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: an open section containing at least one page.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_READ, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        assert!(mapped(view.Value as usize + 13));
        // SAFETY: our successful mapping and its section handle.
        unsafe {
            UnmapViewOfFile(view);
            CloseHandle(section);
        }
        title_memory_remapped(view.Value as usize, PAGE);
        assert!(!mapped((&raw const file) as usize)); // private stack allocation
        assert!(!mapped(mapped as *const () as usize)); // executable image
        assert!(!mapped(0)); // no allocation to query
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// What the allocation queries found is kept for the allocation and
    /// forgotten where the runtime remaps memory.
    #[test]
    fn allocation_kinds_are_kept_until_remapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of four blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (4 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 4 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let known = |at: usize| {
            let kinds = KINDS.lock().unwrap();
            kinds
                .range(..=at)
                .next_back()
                .filter(|(_, (end, _))| at < *end)
                .map(|(&start, &(end, view))| (start, end, view))
        };
        assert!(mapped(view + 3 * BLOCK + 5));
        let (start, end, is_view) = known(view + BLOCK).expect("kept");
        assert!(is_view && start <= view && view + 4 * BLOCK <= end);
        // Another address of the allocation is answered from it.
        assert!(mapped(view + 1));
        title_memory_remapped(view + 2 * BLOCK, BLOCK);
        assert_eq!(known(view + BLOCK), None);
        assert!(mapped(view));
        assert!(known(view).is_some());
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 4 * BLOCK);
        assert_eq!(known(view), None);
    }

    /// Blocks of a guarded range with nothing committed (placeholders
    /// between page views, as xmem maps them) are reported written once,
    /// then not until the runtime maps them, or a later frame finds them
    /// committed without a report.
    #[test]
    fn uncommitted_blocks_are_holes_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a pagefile section and a placeholder of four blocks.
        let (section, base) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (4 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                4 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            (section, base)
        };
        assert_ne!(base, 0);
        // Block `i` of the placeholder shows section block `i`; `report`:
        // as the runtime does.
        let map = |i: usize, report: bool| {
            let at = base + i * BLOCK;
            // SAFETY: a block of our placeholder, split off and mapped.
            let view = unsafe {
                VirtualFree(at as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    at as *const _,
                    (i * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, at);
            if report {
                title_memory_remapped(at, BLOCK);
            }
        };
        map(0, true);
        let pages = |blocks: &[usize]| -> Vec<usize> {
            blocks.iter().flat_map(|&b| (base + b * BLOCK) / PAGE..(base + (b + 1) * BLOCK) / PAGE).collect()
        };
        let (start, end) = (base, base + 4 * BLOCK);
        assert_eq!(guard(start, end), Some(pages(&[0, 1, 2, 3])));
        assert_eq!(guard(start, end), Some(Vec::new()));
        map(2, true);
        assert_eq!(guard(start, end), Some(pages(&[2])));
        assert_eq!(guard(start, end), Some(Vec::new()));
        // SAFETY: a byte of a mapped block; the handler opens it.
        unsafe { ((base + 2 * BLOCK + 9) as *mut u8).write_volatile(7) };
        assert_eq!(guard(start, end), Some(pages(&[2])));
        // Still uncommitted a frame later: looked at, not reported.
        new_frame();
        assert_eq!(guard(start, end), Some(Vec::new()));
        // Mapped without a report: seen at the next frame.
        map(3, false);
        assert_eq!(guard(start, end), Some(Vec::new()));
        new_frame();
        assert_eq!(guard(start, end), Some(pages(&[3])));
        assert_eq!(guard(start, end), Some(Vec::new()));
        assert!(guards().holes.contains_key(&(base / BLOCK + 1)));
        for i in [0, 2, 3] {
            // SAFETY: our views.
            unsafe {
                windows_sys::Win32::System::Memory::UnmapViewOfFile(
                    windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: (base + i * BLOCK) as *mut _,
                    },
                )
            };
        }
        title_memory_remapped(base, 4 * BLOCK);
    }

    /// View regions found committed are kept until remapped; private
    /// memory is asked each time.
    #[test]
    fn committed_views_are_kept_until_remapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_DECOMMIT, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let kept = |at: usize| {
            COMMITTED
                .lock()
                .unwrap()
                .range(..=at)
                .next_back()
                .is_some_and(|(_, &end)| at < end)
        };
        assert!(committed(view + 5, BLOCK));
        assert!(kept(view + BLOCK));
        assert!(committed(view, 2 * BLOCK));
        title_memory_remapped(view, 2 * BLOCK);
        assert!(!kept(view));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        assert!(!committed(view, BLOCK));
        // SAFETY: a fresh private allocation, partly decommitted, freed.
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), 2 * BLOCK, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert_ne!(p, 0);
            assert!(committed(p, 2 * BLOCK));
            assert!(!kept(p));
            VirtualFree((p + BLOCK) as *mut _, BLOCK, MEM_DECOMMIT);
            assert!(!committed(p, 2 * BLOCK));
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
        }
    }

    /// `blocks` contiguous 64 KiB views of a pagefile section in one
    /// placeholder, as xmem maps physical pages: their base. They live
    /// until the test process ends.
    fn page_views(blocks: usize) -> usize {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a pagefile section and a placeholder, split and mapped
        // block by block.
        unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (blocks * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                blocks * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(base, 0);
            for i in 0..blocks {
                let at = base + i * BLOCK;
                if i + 1 < blocks {
                    VirtualFree(at as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                }
                let view = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    at as *const _,
                    (i * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(view.Value as usize, at);
            }
            base
        }
    }

    /// Page views found committed are kept as runs of up to
    /// [`COMMITTED_RUN`]; a remapping forgets only the run it touches.
    #[test]
    fn committed_page_views_are_kept_as_runs() {
        let per_run = COMMITTED_RUN / BLOCK;
        let blocks = per_run * 2 + per_run / 2;
        let base = page_views(blocks);
        let len = blocks * BLOCK;
        let runs = || {
            COMMITTED
                .lock()
                .unwrap()
                .range(base..base + len)
                .map(|(&s, &e)| (s - base, e - base))
                .collect::<Vec<_>>()
        };
        assert_eq!(committed_asking(base, len), (true, true));
        // A remapping elsewhere (another test) during a query keeps
        // nothing; asked again, it is kept.
        while committed_asking(base, len).1 {}
        assert_eq!(
            runs(),
            [
                (0, COMMITTED_RUN),
                (COMMITTED_RUN, 2 * COMMITTED_RUN),
                (2 * COMMITTED_RUN, len)
            ]
        );
        assert_eq!(committed_asking(base + 100, BLOCK), (true, false));
        title_memory_remapped(base + COMMITTED_RUN + 3 * BLOCK, BLOCK);
        assert_eq!(runs(), [(0, COMMITTED_RUN), (2 * COMMITTED_RUN, len)]);
        assert_eq!(committed_asking(base, COMMITTED_RUN), (true, false));
        assert!(committed_asking(base, len).0);
    }

    /// What a texture's committed check costs in physical-page memory:
    /// asking `VirtualQuery` view by view, and reading the kept runs.
    #[test]
    #[ignore = "timing; run with --ignored --nocapture"]
    fn committed_timing() {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        const BLOCKS: usize = 512;
        const ROUNDS: u32 = 200;
        let base = page_views(BLOCKS);
        let len = BLOCKS * BLOCK;
        let started = std::time::Instant::now();
        for _ in 0..ROUNDS {
            let mut p = base;
            while p < base + len {
                // SAFETY: plain data filled in by the query.
                let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                // SAFETY: a valid out pointer of the right size.
                assert_ne!(
                    unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) },
                    0
                );
                p = info.BaseAddress as usize + info.RegionSize;
            }
        }
        let asked = started.elapsed() / ROUNDS;
        while committed_asking(base, len).1 {}
        let started = std::time::Instant::now();
        for _ in 0..ROUNDS {
            assert_eq!(committed_asking(base, len), (true, false));
        }
        let known = started.elapsed() / ROUNDS;
        println!(
            "{} MiB in {BLOCKS} views: VirtualQuery {asked:?}, kept runs {known:?} a check",
            len >> 20
        );
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK), Some((view / PAGE..view / PAGE + 1).collect()));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    /// A direct write into a guarded block (what `host::write_mapped` does
    /// when a block was guarded after it asked) faults into the handler,
    /// which opens the block and notes it written.
    #[test]
    fn direct_writes_into_guarded_blocks_open_them() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        // SAFETY: a byte of our view; the handler gives write access back.
        unsafe { ((view + BLOCK + 8) as *mut u8).write_volatile(5) };
        // SAFETY: the byte just written.
        assert_eq!(unsafe { ((view + BLOCK + 8) as *const u8).read_volatile() }, 5);
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE + 1..view / PAGE + 2).collect())
        );
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }
}

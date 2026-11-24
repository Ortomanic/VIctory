//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
#[derive(Clone, Copy)]
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Microseconds of texture refreshes reading watched memory through the
    /// kernel, asking whether hashed memory is committed, hashing, and
    /// detiling into the staging buffers.
    TexReadUs,
    TexQueryUs,
    TexHashUs,
    TexDetileUs,
    /// Of the guarded askings: allocation-type queries made (the others
    /// were known, `watch::KINDS`) and their time; guarded blocks not
    /// reported written because nothing is committed there.
    RegionQueries,
    RegionQueryUs,
    HoleSkips,
    /// Of the watch time, in the kernel's write-watch query; askings of
    /// known view memory, which skip that query.
    WatchKernelUs,
    KernelSkips,
    /// Guarding: `VirtualProtect` calls and their time.
    Protects,
    ProtectUs,
    /// Hole runs checked again (once a frame) and found still uncommitted.
    HoleChecks,
    /// Time in copies that met an unmapped page, after the failed read.
    FallbackUs,
    /// `committed` answers: regions asked of the host, and ranges answered
    /// from what was asked before (view memory, until remapped).
    CommitQueries,
    CommitHits,
    /// Where a frame's time goes (round 3), all microseconds unless named
    /// otherwise: inside `Host::execute` (any thread, the present's copy
    /// included), of that waiting for another submission's lock,
    /// preparing, on the GPU, copying back, and the deferred writes after.
    ExecUs,
    LockUs,
    PrepUs,
    GpuUs,
    BackUs,
    AfterUs,
    /// Of the preparing: Copy steps (title destinations) and their count,
    /// Fill steps (into upload memory), refresh flushes
    /// (`Shadow::update`), texture batches (wall time), snapshots of
    /// ranges shaders write (`Shadow::writing`) and their bytes.
    TitleCopyUs,
    TitleCopySteps,
    FillUs,
    FlushUs,
    TexBatchUs,
    SnapshotUs,
    SnapshotBytes,
    /// Inside the title's present (`queue::present`, the copy's execute
    /// included) and of it the swap chain's `Present`.
    PresentUs,
    PresentExecUs,
    SwapUs,
    /// Of the present (frame-perf2): the presenter's lock, swap chain
    /// checks and back buffer; the GPU part of its execute; recording the
    /// composition; the frame meter (and the log lines it writes).
    PresentReadyUs,
    PresentGpuUs,
    PresentComposeUs,
    PresentMeterUs,
    /// Of the preparing: the lists' preludes (texture state transitions).
    PreludeUs,
    /// Of the host slots: waiting for the heap's made-slot lock.
    RecMirrorLockUs,
    /// Recording draws and dispatches (title threads, summed) and their count.
    RecordUs,
    Records,
    /// The runtime's view maps and unmaps reported.
    Remaps,
    /// Of the watch: waiting for the journal's lock; write-watch kernel
    /// queries of unguarded askings (textures) and how many; kernel
    /// queries in all; time in `guard()`.
    JournalWaitUs,
    TexKernelUs,
    TexKernelCalls,
    KernelCalls,
    GuardUs,
    /// Askings answered (in whole, in part) from the submission's earlier
    /// askings of the same pages (`watch::Epoch`).
    EpochSkips,
    EpochPartial,
    /// Region queries in fallbacks, and those answered from known
    /// committed views.
    FallbackQueries,
    FallbackKnown,
    /// Of the recording (frame-perf): waiting for host pipelines being made
    /// (`pipeline::make::wait`), binding (`State::bind`: heaps, root
    /// signature, pipeline, user data), of that the tables and the root
    /// views, the table descriptors' host slots (`HostHeap::mirror`), what
    /// they read (`readable`: buffer ranges, texture uploads and states),
    /// vertex and index buffers, the breadcrumb notes, and the buffer-copy
    /// lookups (`shadow::find`, any thread) and their count.
    RecWaitUs,
    RecBindUs,
    RecTablesUs,
    RecRootViewsUs,
    RecMirrorUs,
    RecReadableUs,
    RecIaUs,
    RecNoteUs,
    RecFindUs,
    RecFinds,
    /// Copy-back: bytes compared with their snapshot, and 64 KiB pieces
    /// the GPU wrote (title memory read and merged).
    BackCompared,
    BackPieces,
    /// Of the recording's tables made afresh (`State::tables`): hashing
    /// their contents, those found made in the list and their time, slots
    /// taken (`HostHeap::dynamic`), the tables made and their entries, and
    /// the time making them.
    RecKeyUs,
    RecTableHits,
    RecHitUs,
    RecDynamicUs,
    RecTableMisses,
    RecEntriesMade,
    RecMakeUs,
    /// Of the copy-back: comparing with the snapshots, writing title
    /// memory (opening guards included), asking about the pages after.
    BackCompareUs,
    BackWriteUs,
    BackFreshUs,
    /// Copy-back ranges found in no snapshot (should stay 0).
    BackMissed,
    /// Bindless buffer views (`descriptor::bindless`): the registry's
    /// ranges merged again and the time, and preparing them at submission
    /// (refresh and snapshot; their copy-back is in `BackUs`).
    BindlessRebuilds,
    BindlessMergeUs,
    BindlessPrepUs,
    /// Submissions with shaders that read (only) or write through heap
    /// descriptors at slots from data.
    BindlessReadSubmissions,
    BindlessWriteSubmissions,
    /// Bindless ranges not refreshed: not all committed.
    BindlessUnmapped,
    /// The once-a-frame sweep of the bindless ranges read (all of them;
    /// `prepared` also has the submission's own flush): its time and
    /// count, the ranges asked about, the bytes copied, and of its time
    /// what the watch, the committed checks (and their region queries),
    /// the kernel askings ahead per allocation (and their count) and the
    /// fallback copies took.
    BindlessSweepUs,
    BindlessSweeps,
    BindlessRanges,
    BindlessCopied,
    BindlessWatchUs,
    BindlessCommitUs,
    BindlessCommitQueries,
    BindlessAheadCalls,
    BindlessAheadUs,
    BindlessKernelCalls,
    BindlessFallbackUs,
    /// Copies of write-watched memory with ranges read, and of those the
    /// ones in an allocation asked about ahead.
    BindlessWatchedCopies,
    BindlessAheadCopies,
    /// Committed checks of the sweep answered from its own earlier region
    /// queries, and runs left out on blocks the guards found holes.
    BindlessPassHits,
    BindlessHoleRuns,
}
const COUNTERS: usize = C::BindlessHoleRuns as usize + 1;
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

pub(crate) fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

/// A counter's value since the last line (for differences around a step).
pub(crate) fn get(c: C) -> u64 {
    COUNT[c as usize].load(Relaxed)
}

/// Adds the time until it is dropped to a counter.
pub(crate) struct Timed(C, std::time::Instant);

impl Drop for Timed {
    fn drop(&mut self) {
        add(self.0, self.1.elapsed().as_micros() as u64);
    }
}

pub(crate) fn timed(c: C) -> Timed {
    Timed(c, std::time::Instant::now())
}

/// Times a draw or dispatch being recorded (until dropped), and counts it.
pub(crate) fn recording() -> Timed {
    add(C::Records, 1);
    timed(C::RecordUs)
}

/// Adds the time since `started` to `c`.
pub(crate) fn since(c: C, started: std::time::Instant) {
    add(c, started.elapsed().as_micros() as u64);
}

/// The slowest fallback copy since the last line: microseconds, source, length.
static SLOWEST_FALLBACK: Mutex<(u64, usize, usize)> = Mutex::new((0, 0, 0));

/// A copy that met an unmapped page took `us` after its failed read.
pub(crate) fn fallback(src: usize, len: usize, us: u64) {
    add(C::FallbackUs, us);
    let mut s = SLOWEST_FALLBACK.lock().unwrap_or_else(|e| e.into_inner());
    if us > s.0 {
        *s = (us, src, len);
    }
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        _ => 0,
    })
}

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, kind));
    e.0 += 1;
    e.1 += len as u64;
    e.2 += us;
}

fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame over `frames` frames (all since the last
/// line), and the ranges that took most bytes.
pub(crate) fn log(frames: f64) {
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks; {:.0} guard faults; {:.1} region queries {:.2} ms, {:.0} hole \
         blocks skipped; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed; texture read {:.2} ms, committed \
         {:.2} ms, hash {:.2} ms, detile {:.2} ms; over {frames:.0} frames: watch kernel {:.2} ms, \
         {:.0} kernel askings skipped (views); {:.0} protects {:.2} ms; {:.0} hole runs rechecked; \
         fallbacks {:.2} ms; committed {:.0} queried, {:.0} known",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        c[C::GuardFaults as usize],
        c[C::RegionQueries as usize],
        ms(c[C::RegionQueryUs as usize]),
        c[C::HoleSkips as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
        ms(c[C::TexReadUs as usize]),
        ms(c[C::TexQueryUs as usize]),
        ms(c[C::TexHashUs as usize]),
        ms(c[C::TexDetileUs as usize]),
        ms(c[C::WatchKernelUs as usize]),
        c[C::KernelSkips as usize],
        c[C::Protects as usize],
        ms(c[C::ProtectUs as usize]),
        c[C::HoleChecks as usize],
        ms(c[C::FallbackUs as usize]),
        c[C::CommitQueries as usize],
        c[C::CommitHits as usize],
    );
    // Where the frame's time went, over the same frames.
    static LINE: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let wall = {
        let mut line = LINE.lock().unwrap_or_else(|e| e.into_inner());
        let wall = line.map_or(0.0, |at| at.elapsed().as_secs_f64() * 1e3 / frames);
        *line = Some(std::time::Instant::now());
        wall
    };
    let at = |k: C| c[k as usize];
    let us = |k: C| at(k) / 1e3;
    let (slow_us, slow_src, slow_len) =
        std::mem::take(&mut *SLOWEST_FALLBACK.lock().unwrap_or_else(|e| e.into_inner()));
    tracing::info!(
        "frame time per frame: {wall:.1} ms wall; executing {:.1} ms (lock wait {:.1}; \
         preparing {:.1}: title copies {:.1} in {:.0} steps, fills {:.1}, refresh flushes {:.1} \
         [watch {:.1}: kernel {:.1} in {:.0} calls ({:.0} for textures {:.1} ms), guard {:.1}, \
         journal wait {:.1}; {:.0} askings from the submission's earlier ones, {:.0} in part], \
         texture batches {:.1}, bindless {:.1} (sweep {:.1}), preludes {:.1}, snapshots {:.1} ms {:.1} MB; on the GPU {:.1}; copying back \
         {:.1} ({:.1} MB compared in {:.1} ms, {:.0} pieces written by the GPU, written in {:.1} ms, \
         asked after {:.1} ms, {:.0} ranges without a snapshot); after {:.1}); presenting {:.1} (ready {:.1}, execute {:.1} of it GPU {:.1}, compose {:.1}, meter {:.1}, swap chain {:.1}); recording {:.1} ms in {:.0} \
         draws and dispatches (title threads); {:.0} remaps; fallback queries {:.0}, {:.0} \
         known views; slowest fallback {:.2} ms at {slow_src:#x}+{slow_len:#x}; outside executing \
         and presenting {:.1} ms",
        us(C::ExecUs),
        us(C::LockUs),
        us(C::PrepUs),
        us(C::TitleCopyUs),
        at(C::TitleCopySteps),
        us(C::FillUs),
        us(C::FlushUs),
        us(C::WatchUs),
        us(C::WatchKernelUs),
        at(C::KernelCalls),
        at(C::TexKernelCalls),
        us(C::TexKernelUs),
        us(C::GuardUs),
        us(C::JournalWaitUs),
        at(C::EpochSkips),
        at(C::EpochPartial),
        us(C::TexBatchUs),
        us(C::BindlessPrepUs),
        us(C::BindlessSweepUs),
        us(C::PreludeUs),
        us(C::SnapshotUs),
        mb(at(C::SnapshotBytes)),
        us(C::GpuUs),
        us(C::BackUs),
        mb(at(C::BackCompared)),
        us(C::BackCompareUs),
        at(C::BackPieces),
        us(C::BackWriteUs),
        us(C::BackFreshUs),
        at(C::BackMissed),
        us(C::AfterUs),
        us(C::PresentUs),
        us(C::PresentReadyUs),
        us(C::PresentExecUs),
        us(C::PresentGpuUs),
        us(C::PresentComposeUs),
        us(C::PresentMeterUs),
        us(C::SwapUs),
        us(C::RecordUs),
        at(C::Records),
        at(C::Remaps),
        at(C::FallbackQueries),
        at(C::FallbackKnown),
        slow_us as f64 / 1e3,
        // The present's own execute counts in both.
        (wall - us(C::ExecUs) - us(C::PresentUs) + us(C::PresentExecUs)).max(0.0),
    );
    tracing::info!(
        "recording per frame: {:.1} ms in {:.0} draws and dispatches: pipeline waits {:.1}, bind \
         {:.1} (tables {:.1}: host slots {:.1} (lock wait {:.1}), what they read {:.1}, fresh tables keyed {:.1}, \
         {:.0} found made {:.1}, slots taken {:.1}, {:.0} made with {:.0} entries {:.1}; root \
         views {:.1}), vertex \
         and index {:.1}, notes {:.1}; buffer lookups {:.0} in {:.1} ms (all threads)",
        us(C::RecordUs),
        at(C::Records),
        us(C::RecWaitUs),
        us(C::RecBindUs),
        us(C::RecTablesUs),
        us(C::RecMirrorUs),
        us(C::RecMirrorLockUs),
        us(C::RecReadableUs),
        us(C::RecKeyUs),
        at(C::RecTableHits),
        us(C::RecHitUs),
        us(C::RecDynamicUs),
        at(C::RecTableMisses),
        at(C::RecEntriesMade),
        us(C::RecMakeUs),
        us(C::RecRootViewsUs),
        us(C::RecIaUs),
        us(C::RecNoteUs),
        at(C::RecFinds),
        us(C::RecFindUs),
    );
    // Dynamic descriptor slots (tables made afresh): how many are taken
    // and how they come back.
    let [waits, wait_us, made_shared, slots_shared, made_own, slots_own, shared, lent, reclaimed, lost, unmade] =
        crate::draw::take_slot_stats();
    let (all, now, high, stolen) = crate::descriptor::dynamic_use();
    tracing::info!(
        "dynamic descriptor slots: {all} in all, {now} taken now, at most {high} at once; per \
         frame {:.0} tables made shareable ({:.0} slots), {:.0} with constants ({:.0} slots), {:.0} \
         found made by another list, {:.0} slots lent back after runs, {:.1} lent runs taken back, {:.1} runs taken from lent \
         slots, {:.1} lists not run (lent slots taken), {:.1} waits for slots lent back {:.1} ms, \
         {:.1} draws and dispatches dropped (a table not made)",
        made_shared as f64 / frames,
        slots_shared as f64 / frames,
        made_own as f64 / frames,
        slots_own as f64 / frames,
        shared as f64 / frames,
        lent as f64 / frames,
        reclaimed as f64 / frames,
        stolen as f64 / frames,
        lost as f64 / frames,
        waits as f64 / frames,
        wait_us as f64 / 1e3 / frames,
        unmade as f64 / frames,
    );
    // Buffer views in the heaps by what the host made of them, and what
    // their bindless reads cost per frame.
    let [_, _, table, bindless, deferred, failed] = crate::descriptor::slot_states();
    let (slots, copies, span, written) = crate::descriptor::bindless::sizes();
    tracing::info!(
        "heap buffer views: {table} named by tables, {bindless} made for bindless reads only, \
         {deferred} null until a table names them, {failed} null (not made); bindless ranges: {slots} \
         slots over {copies} copies, {:.1} MB read, {:.1} MB of it written by shaders; per frame \
         {:.1} submissions reading bindlessly, {:.1} writing, {:.1} ranges skipped (not all mapped), \
         {:.1} merges {:.2} ms, prepared {:.2} ms \
         (copy-back of all ranges {:.2} ms, {:.1} MB compared); sweep {:.2} ms in {:.1} \
         ({:.0} ranges, {:.1} MB copied; watch {:.2} ms, {:.0} kernel calls, {:.0} ahead \
         {:.2} ms for {:.0} of {:.0} tracked copies; committed {:.2} ms, {:.0} queries, {:.0} answered by the sweep's, {:.0} runs on holes; \
         fallbacks {:.2} ms)",
        span as f64 / 1e6,
        written as f64 / 1e6,
        c[C::BindlessReadSubmissions as usize],
        c[C::BindlessWriteSubmissions as usize],
        c[C::BindlessUnmapped as usize],
        c[C::BindlessRebuilds as usize],
        ms(c[C::BindlessMergeUs as usize]),
        ms(c[C::BindlessPrepUs as usize]),
        ms(c[C::BackUs as usize]),
        mb(c[C::BackCompared as usize]),
        ms(c[C::BindlessSweepUs as usize]),
        c[C::BindlessSweeps as usize],
        c[C::BindlessRanges as usize],
        mb(c[C::BindlessCopied as usize]),
        ms(c[C::BindlessWatchUs as usize]),
        c[C::BindlessKernelCalls as usize],
        c[C::BindlessAheadCalls as usize],
        ms(c[C::BindlessAheadUs as usize]),
        c[C::BindlessAheadCopies as usize],
        c[C::BindlessWatchedCopies as usize],
        ms(c[C::BindlessCommitUs as usize]),
        c[C::BindlessCommitQueries as usize],
        c[C::BindlessPassHits as usize],
        c[C::BindlessHoleRuns as usize],
        ms(c[C::BindlessFallbackUs as usize]),
    );
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
        let kind = info.Type;
        tracing::info!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}

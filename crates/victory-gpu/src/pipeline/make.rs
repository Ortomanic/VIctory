//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and Grand
//! Theft Auto VI's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! shortcut: the budget (half a second of waiting per title frame,
//! `VICTORY_PIPELINE_WAIT_MS`) is a guess; a pass the title runs only once
//! and whose pipeline takes longer is lost.

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queued: AtomicBool,
    /// The host pipeline once made (null before; `None` inside: none can
    /// be made). Live controls replace it: a replaced one is released a
    /// few frames later, once no draw being recorded or submission running
    /// can use it (`super::sweep`), so a reference to it stays valid for
    /// the recording of a command.
    made: AtomicPtr<Option<HostPipeline>>,
    /// Held while making: a remake waits for a make in progress.
    making: Mutex<()>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    key: String,
    used: AtomicBool,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Drop for Pending {
    fn drop(&mut self) {
        let made = *self.made.get_mut();
        if !made.is_null() {
            // SAFETY: our box; the title released the pipeline.
            drop(unsafe { Box::from_raw(made) });
        }
    }
}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        Pending {
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            made: AtomicPtr::new(std::ptr::null_mut()),
            making: Mutex::new(()),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
        }
    }

    /// What was made, if it was.
    fn made(&self) -> Option<&Option<HostPipeline>> {
        let made = self.made.load(Ordering::Acquire);
        // SAFETY: set from a box that lives until the pending is dropped or
        // (replaced) until no recording or submission can use it.
        unsafe { made.as_ref() }
    }

    /// Makes the host pipeline from the shaders' translations of now,
    /// replacing one made before.
    fn make(&self) {
        {
            let _making = self.making.lock().unwrap_or_else(|e| e.into_inner());
            let made = Box::new(HostPipeline::new(&self.shaders, self.graphics.as_deref()));
            let old = self.made.swap(Box::into_raw(made), Ordering::AcqRel);
            if !old.is_null() {
                // SAFETY: a box of ours, no longer reachable from `made`.
                super::replaced(unsafe { Box::from_raw(old) });
            }
        }
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| s.failed())
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads: 500 ms unless
/// `VICTORY_PIPELINE_WAIT_MS` says otherwise (0: never wait, a draw whose
/// pipeline is not made yet is dropped until it is; one-off passes such as
/// a colour-grading bake may then be lost).
/// Live controls change it (`set_budget`).
fn budget() -> Duration {
    let ms = match BUDGET_MS.load(Ordering::Relaxed) {
        u64::MAX => {
            let ms = parse_budget(std::env::var("VICTORY_PIPELINE_WAIT_MS").ok().as_deref());
            let _ = BUDGET_MS.compare_exchange(u64::MAX, ms, Ordering::Relaxed, Ordering::Relaxed);
            BUDGET_MS.load(Ordering::Relaxed)
        }
        ms => ms,
    };
    Duration::from_millis(ms)
}

/// [`budget`] in milliseconds; `u64::MAX`: not read yet.
static BUDGET_MS: AtomicU64 = AtomicU64::new(u64::MAX);

/// `VICTORY_PIPELINE_WAIT_MS`'s value in milliseconds (unset or unreadable:
/// 500).
fn parse_budget(value: Option<&str>) -> u64 {
    value
        .and_then(|v| v.trim().parse().ok())
        .filter(|&ms| ms != u64::MAX)
        .unwrap_or(500)
}

/// Sets [`budget`] from now on as `VICTORY_PIPELINE_WAIT_MS=value` would
/// (live controls) and returns it in milliseconds.
#[cfg(feature = "diagnostics")]
pub(crate) fn set_budget(value: Option<&str>) -> u64 {
    let ms = parse_budget(value);
    BUDGET_MS.store(ms, Ordering::Relaxed);
    ms
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.made().is_none() {
        start(p);
        wait(p, crate::frame::current());
    }
    let made = p.made()?.as_ref();
    if made.is_some() && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    made
}

/// Queues `p` for the workers, once.
fn start(p: &Arc<Pending>) {
    if !p.queued.swap(true, Ordering::Relaxed) {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        queue(p.clone());
    }
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p);
    }
}

/// The frame's waiting so far: which frame, the time spent while some
/// thread waited, since when threads are waiting now and how many.
struct Spent {
    frame: u64,
    spent: Duration,
    since: Option<Instant>,
    waiters: u32,
}

static SPENT: Mutex<Spent> = Mutex::new(Spent {
    frame: u64::MAX,
    spent: Duration::ZERO,
    since: None,
    waiters: 0,
});

impl Spent {
    /// The budget's wall time used up: threads waiting at once spend it
    /// together (summed per thread, five recording threads waited 2.5 s
    /// of a world frame's 1.1).
    fn used(&self, now: Instant) -> Duration {
        self.spent + self.since.map_or(Duration::ZERO, |at| now - at)
    }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    wait_in(p, frame, &SPENT);
}

/// [`wait`], with the frame's waiting so far in `spent`.
fn wait_in(p: &Pending, frame: u64, spent: &Mutex<Spent>) {
    // The console's title thread never waits here ([`crate::held`]).
    let _held = crate::held::hold();
    let spent = || spent.lock().unwrap_or_else(|e| e.into_inner());
    let started = Instant::now();
    {
        let mut s = spent();
        if s.frame != frame {
            // Waits still running into the new frame count for it.
            *s = Spent {
                frame,
                spent: Duration::ZERO,
                since: (s.waiters > 0).then_some(started),
                waiters: s.waiters,
            };
        }
        if s.waiters == 0 {
            s.since = Some(started);
        }
        s.waiters += 1;
    }
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made().is_none() {
        let rest = budget().saturating_sub(spent().used(Instant::now()));
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    let mut s = spent();
    s.waiters -= 1;
    if s.waiters == 0
        && let Some(at) = s.since.take()
    {
        s.spent += at.elapsed();
    }
    drop(s);
    #[cfg(feature = "diagnostics")]
    crate::copystat::since(crate::copystat::C::RecWaitUs, started);
}

/// Hands `p` to the workers (half the logical processors, 2 to 8), started
/// on first use.
fn queue(p: Arc<Pending>) {
    queue_job(p, false);
}

/// [`queue`]; `remake`: a live controls remake (counted apart).
fn queue_job(p: Arc<Pending>, remake: bool) {
    static SENDER: OnceLock<std::sync::mpsc::Sender<(Arc<Pending>, bool)>> = OnceLock::new();
    let sender = SENDER.get_or_init(|| {
        let (sender, receiver) = std::sync::mpsc::channel::<(Arc<Pending>, bool)>();
        let receiver = Arc::new(Mutex::new(receiver));
        let workers = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
        for i in 0..workers {
            let receiver = receiver.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(move || {
                    // Below the title's own threads (loading, recording).
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                    }
                    loop {
                        let next = receiver.lock().unwrap_or_else(|e| e.into_inner()).recv();
                        let Ok((p, remake)) = next else { return };
                        p.make();
                        #[cfg(feature = "diagnostics")]
                        if remake {
                            remade();
                        }
                        #[cfg(not(feature = "diagnostics"))]
                        let _ = remake;
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save(false);
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            }
        }
        sender
    });
    if let Err(e) = sender.send((p, remake)) {
        // No worker: made here instead.
        e.0.0.make();
        WAITING.fetch_sub(1, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        if remake {
            remade();
        }
    }
}

/// Every title pipeline (diagnostics: for live controls), pruned of
/// released ones as it grows.
#[cfg(feature = "diagnostics")]
static ALL: Mutex<Vec<std::sync::Weak<Pending>>> = Mutex::new(Vec::new());

/// Notes a title pipeline created (diagnostics).
pub(crate) fn register(p: &Arc<Pending>) {
    #[cfg(feature = "diagnostics")]
    {
        let mut all = ALL.lock().unwrap_or_else(|e| e.into_inner());
        if all.len() == all.capacity() {
            all.retain(|w| w.strong_count() > 0);
        }
        all.push(Arc::downgrade(p));
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = p;
}

/// Live controls: queues a remake of every title pipeline whose host
/// pipeline was started (made or being made) and that `which` picks; how
/// many.
#[cfg(feature = "diagnostics")]
pub(crate) fn remake_where(which: impl Fn(&Pending, Option<&Option<HostPipeline>>) -> bool) -> usize {
    let picked: Vec<Arc<Pending>> = {
        let mut all = ALL.lock().unwrap_or_else(|e| e.into_inner());
        all.retain(|w| w.strong_count() > 0);
        all.iter()
            .filter_map(std::sync::Weak::upgrade)
            .filter(|p| p.queued.load(Ordering::Relaxed) && which(p, p.made()))
            .collect()
    };
    let n = picked.len();
    if n > 0 {
        REMAKING.fetch_add(n as u64, Ordering::Relaxed);
    }
    for p in picked {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        queue_job(p, true);
    }
    n
}

#[cfg(feature = "diagnostics")]
impl Pending {
    pub(crate) fn shaders(&self) -> &[Arc<Shader>] {
        &self.shaders
    }
}

/// Live controls remakes queued and not done.
#[cfg(feature = "diagnostics")]
static REMAKING: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "diagnostics")]
static REMADE: AtomicU64 = AtomicU64::new(0);

/// A remake is done: the last of a batch is logged.
#[cfg(feature = "diagnostics")]
fn remade() {
    let total = REMADE.fetch_add(1, Ordering::Relaxed) + 1;
    if REMAKING.fetch_sub(1, Ordering::Relaxed) == 1 {
        tracing::info!(
            "live: host pipelines remade (all queued done) at frame {}, {total} remade so far",
            crate::queue::presented()
        );
    }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made(), Some(None)));
    }

    /// Threads waiting at once share the frame's budget: two waits for
    /// pipelines never made end together after it, and a later one in
    /// the same frame does not wait.
    #[test]
    fn waits_at_once_share_the_budget() {
        static OWN: Mutex<Spent> = Mutex::new(Spent {
            frame: u64::MAX,
            spent: Duration::ZERO,
            since: None,
            waiters: 0,
        });
        let frame = 7;
        let started = Instant::now();
        let waits: Vec<_> = (0..2)
            .map(|_| {
                std::thread::spawn(move || {
                    let p = Pending::new(Vec::new(), None);
                    wait_in(&p, frame, &OWN);
                })
            })
            .collect();
        for w in waits {
            w.join().unwrap();
        }
        let both = started.elapsed();
        let budget = budget();
        assert!(both >= budget && both < budget + budget / 2, "{both:?}");
        let later = Instant::now();
        wait_in(&Pending::new(Vec::new(), None), frame, &OWN);
        assert!(later.elapsed() < budget / 10);
    }
}

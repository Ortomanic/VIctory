//! Time a title thread spends held inside VIctory for work the console does
//! not make it wait for: the host GPU running a submission to the end
//! (`ExecuteCommandLists` waits for it, decision 0007) and a first use
//! waiting for a host pipeline. On the console those calls return at once.
//!
//! Grand Theft Auto VI's stall check (a thread at `+0x2215a60`) sleeps
//! `Sleep(1000)` and ends the process with a write of `0x4d8bd421` to
//! address 0 (`+0x23cf683`) after 30 such ticks without the frame thread's
//! heartbeat. Time we hold its threads would count as the title being
//! stuck, so the title's `Sleep` of a second or more (`victory-hle`'s
//! `synch` module) is made to last until that much time has passed with no
//! title thread held here.
//!
//! Only the union of the holds counts: two threads held at once for a
//! second count one second. A hold older than [`STUCK`] (a submission the
//! host GPU never finishes) counts as the title being stuck again, so a
//! real hang still ends in the title's own report.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A hold this old is a hang, not work: its time counts again.
pub const STUCK: Duration = Duration::from_secs(60);

/// The holds now and the time some hold was active.
#[derive(Debug, Default)]
pub struct Holds {
    /// Start of every hold not yet over, by id.
    active: Vec<(u64, Instant)>,
    next_id: u64,
    /// Since when some hold is active (the union's current stretch).
    busy_since: Option<Instant>,
    /// Union time of the stretches already over.
    busy: Duration,
}

impl Holds {
    pub const fn new() -> Holds {
        Holds {
            active: Vec::new(),
            next_id: 0,
            busy_since: None,
            busy: Duration::ZERO,
        }
    }

    /// A hold begins at `now`; returns its id for [`Holds::end`].
    pub fn begin(&mut self, now: Instant) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        if self.active.is_empty() {
            self.busy_since = Some(now);
        }
        self.active.push((id, now));
        id
    }

    /// Hold `id` ends at `now`.
    pub fn end(&mut self, id: u64, now: Instant) {
        self.active.retain(|&(i, _)| i != id);
        if self.active.is_empty()
            && let Some(since) = self.busy_since.take()
        {
            self.busy += now.saturating_duration_since(since);
        }
    }

    /// Time some hold was active, up to `now` (never decreases).
    pub fn total(&self, now: Instant) -> Duration {
        self.busy
            + self
                .busy_since
                .map_or(Duration::ZERO, |since| now.saturating_duration_since(since))
    }

    /// Whether a hold has lasted longer than [`STUCK`] at `now`.
    pub fn stuck(&self, now: Instant) -> bool {
        self.active
            .iter()
            .any(|&(_, at)| now.saturating_duration_since(at) > STUCK)
    }
}

static HOLDS: Mutex<Holds> = Mutex::new(Holds::new());

fn holds() -> std::sync::MutexGuard<'static, Holds> {
    HOLDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title thread held here until this is dropped.
#[must_use]
pub struct Held(u64);

/// Marks the calling (title) thread as held here until the guard drops.
pub fn hold() -> Held {
    Held(holds().begin(Instant::now()))
}

impl Drop for Held {
    fn drop(&mut self) {
        holds().end(self.0, Instant::now());
    }
}

/// Time some title thread was held here, up to `now`.
pub fn total(now: Instant) -> Duration {
    holds().total(now)
}

/// Whether a title thread has been held longer than [`STUCK`].
pub fn stuck(now: Instant) -> bool {
    holds().stuck(now)
}

/// How long a sleep of `want` begun at `start`, with `held_at_start` held
/// time then, still has to go at `now`: time held since does not count,
/// unless a hold is stuck (then plain wall time counts). `None` when done.
pub fn sleep_left(
    want: Duration,
    start: Instant,
    held_at_start: Duration,
    held_now: Duration,
    stuck: bool,
    now: Instant,
) -> Option<Duration> {
    let wall = now.saturating_duration_since(start);
    let counted = if stuck {
        wall
    } else {
        wall.saturating_sub(held_now.saturating_sub(held_at_start))
    };
    (counted < want).then(|| want - counted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    /// Overlapping holds count once; time between holds does not count.
    #[test]
    fn holds_count_their_union() {
        let t = Instant::now();
        let mut h = Holds::new();
        let a = h.begin(t);
        let b = h.begin(t + s(1));
        h.end(a, t + s(2));
        assert_eq!(h.total(t + s(2)), s(2));
        h.end(b, t + s(3));
        assert_eq!(h.total(t + s(10)), s(3));
        let c = h.begin(t + s(10));
        assert_eq!(h.total(t + s(12)), s(5));
        h.end(c, t + s(12));
        assert_eq!(h.total(t + s(20)), s(5));
    }

    /// Only a hold older than the limit is stuck.
    #[test]
    fn a_long_hold_is_stuck() {
        let t = Instant::now();
        let mut h = Holds::new();
        let a = h.begin(t);
        assert!(!h.stuck(t + STUCK));
        assert!(h.stuck(t + STUCK + s(1)));
        h.end(a, t + STUCK + s(1));
        assert!(!h.stuck(t + STUCK + s(2)));
    }

    /// A one-second sleep with 3 s held during it lasts 4 s; with the hold
    /// stuck it lasts one second of wall time.
    #[test]
    fn held_time_does_not_count_unless_stuck() {
        let t = Instant::now();
        let want = s(1);
        assert_eq!(sleep_left(want, t, s(7), s(7), false, t), Some(s(1)));
        // Held the first 3 s: none of it counted yet.
        assert_eq!(
            sleep_left(want, t, s(7), s(10), false, t + s(3)),
            Some(s(1))
        );
        assert_eq!(
            sleep_left(want, t, s(7), s(10), false, t + Duration::from_millis(3500)),
            Some(Duration::from_millis(500))
        );
        assert_eq!(sleep_left(want, t, s(7), s(10), false, t + s(4)), None);
        // Stuck: wall time counts.
        assert_eq!(sleep_left(want, t, s(7), s(10), true, t + s(1)), None);
    }

    /// The shared guard: a hold shows in the total while it lasts.
    #[test]
    fn the_guard_holds_while_alive() {
        let before = total(Instant::now());
        let held = hold();
        std::thread::sleep(Duration::from_millis(20));
        assert!(total(Instant::now()) >= before + Duration::from_millis(20));
        drop(held);
    }
}

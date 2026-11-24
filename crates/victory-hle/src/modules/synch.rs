//! `Sleep` that does not count the time VIctory holds title threads.
//!
//! Grand Theft Auto VI's stall check (thread at `+0x2215a60` in 1.0.2.1)
//! loops `QueryPerformanceCounter; Sleep(1000); QueryPerformanceCounter`
//! and ends the process ("GPU Loc Sig", a write of `0x4d8bd421` to address
//! 0 at `+0x23cf683`) once the frame thread's heartbeat has not changed
//! for 30 ticks. On the PC its threads also wait for work the console does
//! not make them wait for (`victory_gpu::held`: host GPU submissions run
//! to the end, first uses of a host pipeline), which the check would count
//! as the title being stuck.
//!
//! So a title `Sleep` of a second or more returns once that much time has
//! passed with no title thread held. A tick that took longer than 1.57 s
//! makes the check start counting again (it takes the process as having
//! been suspended), so a second with half a second or more held does not
//! count towards the 30. A hold older than `victory_gpu::held::STUCK` (a
//! hung GPU) counts again, so the title's report still comes.
//!
//! Shorter sleeps go to the host's `Sleep` unchanged.
//! `VICTORY_TITLE_WATCHDOG=1` passes every `Sleep` through (the title's own
//! timing).
#![allow(non_snake_case)]

use std::time::{Duration, Instant};

use victory_gpu::held;

use crate::{HleModule, hle_export};

/// Sleeps this long or longer do not count time held.
const FROM_MS: u32 = 1000;
const INFINITE: u32 = u32::MAX;

/// `VICTORY_TITLE_WATCHDOG=1`: the title's own `Sleep` timing.
fn title_timing() -> bool {
    static TITLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TITLE.get_or_init(|| {
        let title = std::env::var("VICTORY_TITLE_WATCHDOG").is_ok_and(|v| v.trim() == "1");
        if title {
            tracing::info!("VICTORY_TITLE_WATCHDOG=1: the title's Sleep counts time held");
        }
        title
    })
}

pub extern "system" fn Sleep(ms: u32) {
    if ms < FROM_MS || ms == INFINITE || title_timing() {
        host_sleep(ms);
        return;
    }
    let want = Duration::from_millis(u64::from(ms));
    let start = Instant::now();
    let held_at_start = held::total(start);
    loop {
        let now = Instant::now();
        let left = held::sleep_left(
            want,
            start,
            held_at_start,
            held::total(now),
            held::stuck(now),
            now,
        );
        let Some(left) = left else { break };
        // Whole milliseconds, at least one, so the loop always sleeps.
        host_sleep(left.as_millis().clamp(1, u128::from(ms)) as u32);
    }
    note_extended(start.elapsed().saturating_sub(want));
}

/// Logs sleeps lengthened by a second or more, at most once a minute.
fn note_extended(extra: Duration) {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNT: AtomicU64 = AtomicU64::new(0);
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    if extra < Duration::from_secs(1) {
        return;
    }
    let count = COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if last.is_none_or(|at| at.elapsed() >= Duration::from_secs(60)) {
        *last = Some(Instant::now());
        tracing::info!(
            extra_ms = extra.as_millis() as u64,
            count,
            "a title Sleep lengthened by the time title threads were held (GPU, pipelines)"
        );
    }
}

fn host_sleep(ms: u32) {
    // SAFETY: a plain call.
    unsafe { windows_sys::Win32::System::Threading::Sleep(ms) }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "synch",
        dlls: &["api-ms-win-core-synch-*"],
        exports: vec![hle_export!(Sleep)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With nothing held a one-second sleep takes about a second; a hold
    /// during it lengthens it by the hold.
    #[test]
    fn held_time_lengthens_a_long_sleep() {
        let started = Instant::now();
        Sleep(1000);
        let plain = started.elapsed();
        assert!(plain >= Duration::from_millis(990), "{plain:?}");

        let holder = std::thread::spawn(|| {
            let _held = held::hold();
            std::thread::sleep(Duration::from_millis(600));
        });
        let started = Instant::now();
        Sleep(1000);
        let lengthened = started.elapsed();
        holder.join().unwrap();
        assert!(lengthened >= Duration::from_millis(1500), "{lengthened:?}");
    }

    /// Short sleeps are the host's.
    #[test]
    fn short_sleeps_pass_through() {
        let _held = held::hold();
        let started = Instant::now();
        Sleep(50);
        assert!(started.elapsed() < Duration::from_millis(900));
    }
}

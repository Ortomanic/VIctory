//! Buffer views in the host heaps that shaders may read bindlessly: by an
//! index from data (Grand Theft Auto VI's skinning job records name their
//! vertex buffers' heap slots), not through a table the draw names. The
//! host cannot know which slots a shader reads that way, so every buffer
//! view mirrored from a title heap is made when the title writes it, and
//! the ranges they view are brought up to date before every submission,
//! those of unordered access views copied back after it, as the ranges of
//! views in tables are ([`crate::draw`], `readable`).
//!
//! Ranges are kept per host copy of title memory ([`Shadow`]), merged where
//! they overlap or lie within a page of each other (what is refreshed: the
//! gaps between views may be unmapped, which costs the copy a slow
//! fallback), and the unordered ones merged where they overlap or touch
//! (what the snapshot and the copy-back after the submission cost).
//!
//! Only submissions whose shaders index the heap from data
//! (`victory_shader::emit::bindless`) do this, and the ranges read are
//! brought up to date once per frame, not per submission: a world frame has
//! ~28 such submissions over ~4,000 copies (~2 GB of ranges), and asking
//! each time which pages the title wrote cost ~1.8 s per frame.
//! shortcut: a bindless read sees what the title wrote in the same frame
//! only if that submission is the frame's first bindless one or writes
//! bindlessly (unordered ranges are brought up to date at each such
//! submission); `VICTORY_BINDLESS_REFRESH=submission` refreshes all at
//! every one (A/B).
//!
//! `VICTORY_BINDLESS=0`: buffer views are made only for tables again (the
//! old behaviour, for an A/B).

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use crate::shadow::Shadow;

/// Whether buffer views outside tables are made (on unless
/// `VICTORY_BINDLESS=0`).
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = !std::env::var("VICTORY_BINDLESS").is_ok_and(|v| v.trim() == "0");
        tracing::info!(on, "buffer views in heaps made for bindless reads");
        on
    })
}

/// The ranges of one host copy that heap slots view: (start, end) of title
/// memory, with how many slots view each.
struct PerShadow {
    shadow: Arc<Shadow>,
    reads: BTreeMap<(u64, u64), u32>,
    writes: BTreeMap<(u64, u64), u32>,
}

/// A slot's range: its copy (by address), title start and end, unordered.
type Slot = (usize, u64, u64, bool);

#[derive(Default)]
struct Registry {
    /// By (host heap, slot index).
    slots: HashMap<(usize, u32), Slot>,
    shadows: HashMap<usize, PerShadow>,
    /// Bumped at every change; `merged` was built at `built`.
    generation: u64,
    built: u64,
    merged: Option<Arc<Merged>>,
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Registry) -> R) -> R {
    let mut r = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    f(r.get_or_insert_with(Registry::default))
}

/// Whether every bindless submission brings every range up to date
/// (`VICTORY_BINDLESS_REFRESH=submission`), not once per frame.
pub(crate) fn every_submission() -> bool {
    static EVERY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        std::env::var("VICTORY_BINDLESS_REFRESH").is_ok_and(|v| v.trim() == "submission")
    })
}

/// What one submission brings up to date for bindless reads.
#[derive(Default)]
pub(crate) struct Merged {
    /// Per copy: the ranges read or written (title start, end), merged
    /// within a page of each other, and the unordered ranges merged
    /// (start, end).
    pub(crate) shadows: Vec<(Arc<Shadow>, Vec<(u64, u64)>, Vec<(u64, u64)>)>,
    /// Slots, bytes of the ranges read, bytes of the unordered ranges
    /// (diagnostics).
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    pub(crate) slots: usize,
    pub(crate) span_bytes: u64,
    pub(crate) write_bytes: u64,
}

fn count(map: &mut BTreeMap<(u64, u64), u32>, key: (u64, u64), add: bool) {
    if add {
        *map.entry(key).or_insert(0) += 1;
    } else if let Some(n) = map.get_mut(&key) {
        *n -= 1;
        if *n == 0 {
            map.remove(&key);
        }
    }
}

impl Registry {
    fn remove(&mut self, key: (usize, u32)) {
        let Some((at, from, to, unordered)) = self.slots.remove(&key) else {
            return;
        };
        self.generation += 1;
        if let Some(per) = self.shadows.get_mut(&at) {
            count(&mut per.reads, (from, to), false);
            if unordered {
                count(&mut per.writes, (from, to), false);
            }
            if per.reads.is_empty() {
                self.shadows.remove(&at);
            }
        }
    }
}

/// Slot `index` of host heap `heap` views `[from, to)` of title memory in
/// `shadow` (written by shaders if `unordered`).
pub(crate) fn set(
    heap: usize,
    index: u32,
    shadow: &Arc<Shadow>,
    from: u64,
    to: u64,
    unordered: bool,
) {
    let at = Arc::as_ptr(shadow) as usize;
    with(|r| {
        if r.slots.get(&(heap, index)) == Some(&(at, from, to, unordered)) {
            return;
        }
        r.remove((heap, index));
        r.generation += 1;
        r.slots.insert((heap, index), (at, from, to, unordered));
        let per = r.shadows.entry(at).or_insert_with(|| PerShadow {
            shadow: shadow.clone(),
            reads: BTreeMap::new(),
            writes: BTreeMap::new(),
        });
        count(&mut per.reads, (from, to), true);
        if unordered {
            count(&mut per.writes, (from, to), true);
        }
    });
}

/// Slot `index` of host heap `heap` views no title buffer memory.
pub(crate) fn clear(heap: usize, index: u32) {
    let mut r = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = r.as_mut() {
        r.remove((heap, index));
    }
}

/// Host heap `heap` is gone.
pub(crate) fn forget_heap(heap: usize) {
    with(|r| {
        let keys: Vec<_> = r.slots.keys().filter(|k| k.0 == heap).copied().collect();
        for k in keys {
            r.remove(k);
        }
    });
}

/// The ranges as of now, merged (rebuilt only after a change).
pub(crate) fn merged() -> Arc<Merged> {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let merged = with(|r| {
        if let Some(m) = &r.merged
            && r.built == r.generation
        {
            return m.clone();
        }
        let mut m = Merged {
            slots: r.slots.len(),
            ..Merged::default()
        };
        for per in r.shadows.values() {
            let reads = merge(per.reads.keys(), crate::watch::PAGE as u64);
            if reads.is_empty() {
                continue;
            }
            let writes = merge(per.writes.keys(), 0);
            m.span_bytes += reads.iter().map(|(a, b)| b - a).sum::<u64>();
            m.write_bytes += writes.iter().map(|(a, b)| b - a).sum::<u64>();
            m.shadows.push((per.shadow.clone(), reads, writes));
        }
        let m = Arc::new(m);
        r.merged = Some(m.clone());
        r.built = r.generation;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BindlessRebuilds, 1);
        m
    });
    #[cfg(feature = "diagnostics")]
    crate::copystat::since(crate::copystat::C::BindlessMergeUs, started);
    merged
}

/// At a bindless submission: brings the ranges read (`all`, once a frame)
/// or those shaders write (`writes`) up to date.
///
/// What costs is not the bytes (almost none changed) but the host's
/// virtual-memory calls, each a wait for Wine's lock while the title's
/// streaming maps and unmaps views (~0.3 ms each at ~3,000 remaps a
/// frame): a region query per range to tell it was all committed, and a
/// write-watch query per range of private memory, ~4,000 + ~500 a world
/// frame (1.2-1.8 s). So write-watched memory is asked about once per
/// allocation ([`crate::watch::ask_ahead`]; the ranges' askings are then
/// answered from the journal), and only runs of pages with something to
/// copy are asked whether they are committed (views of memory the title
/// has unmapped, released buffers and streamed pages, are left as last
/// copied: a copy meeting a hole falls back to slow reads).
pub(crate) fn refresh(m: &Merged, all: bool, writes: bool) {
    use std::sync::atomic::Ordering;
    #[cfg(feature = "diagnostics")]
    let (started, before) = {
        use crate::copystat::{C, get};
        (
            std::time::Instant::now(),
            [C::WatchUs, C::CommitQueries, C::KernelCalls, C::FallbackUs].map(get),
        )
    };
    if all {
        // Write-watched copies' ranges by allocation: one asking each.
        let mut hulls: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
        for (shadow, reads, _) in &m.shadows {
            let (Some(&(from, _)), Some(&(_, to))) = (reads.first(), reads.last()) else {
                continue;
            };
            let (from, to) = (from as usize, to as usize);
            if !shadow.write_watched() {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BindlessWatchedCopies, 1);
            if let Some((start, end, false)) = crate::watch::allocation(from)
                && to <= end
            {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::BindlessAheadCopies, 1);
                let hull = hulls.entry(start).or_insert((from, to));
                *hull = (hull.0.min(from), hull.1.max(to));
            }
        }
        for (start, (from, to)) in hulls {
            if !crate::watch::ask_ahead(from, to) {
                // Not one write-watched allocation (any more): asked again
                // next time, its ranges one by one meanwhile.
                crate::watch::forget_allocation(start);
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    let mut commit_us = 0u64;
    #[cfg(feature = "diagnostics")]
    let mut ranges_asked = 0u64;
    let mut copied_all = 0usize;
    for (shadow, reads, unordered) in &m.shadows {
        let ranges = match (all, writes) {
            (true, _) => reads,
            (false, true) => unordered,
            (false, false) => continue,
        };
        // Private memory the kernel reported written was committed then:
        // not asked again (if it was decommitted since, the copy's slow
        // fallback reads it as it is). Views are asked (a guard reports a
        // block remapped as written), mostly answered from what is known.
        let private = ranges
            .first()
            .and_then(|&(from, _)| crate::watch::allocation(from as usize))
            .is_some_and(|(_, _, view)| !view)
            && shadow.write_watched();
        for &(from, to) in ranges {
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            {
                ranges_asked += 1;
            }
            let (from, to) = (from as usize, to as usize);
            // Whole pages (within the copy): a page copied only in part
            // stays stale, so each range's end pages were copied, and asked
            // whether committed, at every sweep.
            let page = crate::watch::PAGE;
            let (start, end) = (from / page * page, to.next_multiple_of(page));
            let (copied, _left) = shadow.update_if(start, end - start, &mut |a, b, reported| {
                if reported && private {
                    return Some((a, b));
                }
                #[cfg(feature = "diagnostics")]
                let checking = std::time::Instant::now();
                // Else the part in the range itself: next to it may lie a
                // hole.
                let (c, d) = (a.max(from), b.min(to));
                let ok = if crate::watch::committed(a, b - a) {
                    Some((a, b))
                } else if c < d && (c, d) != (a, b) && crate::watch::committed(c, d - c) {
                    Some((c, d))
                } else {
                    None
                };
                #[cfg(feature = "diagnostics")]
                {
                    commit_us += checking.elapsed().as_micros() as u64;
                }
                ok
            });
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BindlessUnmapped, _left as u64);
            copied_all += copied;
            crate::host::PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            crate::host::PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
    let _ = copied_all;
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add, get};
        let after = [C::WatchUs, C::CommitQueries, C::KernelCalls, C::FallbackUs].map(get);
        let d = |i: usize| after[i].saturating_sub(before[i]);
        add(C::BindlessSweepUs, started.elapsed().as_micros() as u64);
        add(C::BindlessSweeps, 1);
        add(C::BindlessRanges, ranges_asked);
        add(C::BindlessCopied, copied_all as u64);
        add(C::BindlessWatchUs, d(0));
        add(C::BindlessCommitQueries, d(1));
        add(C::BindlessKernelCalls, d(2));
        add(C::BindlessFallbackUs, d(3));
        add(C::BindlessCommitUs, commit_us);
    }
}

/// Ranges in start order merged where they overlap or lie within `gap`
/// bytes of each other.
fn merge<'a>(ranges: impl Iterator<Item = &'a (u64, u64)>, gap: u64) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    for &(a, b) in ranges {
        match out.last_mut() {
            Some(last) if a <= last.1.saturating_add(gap) => last.1 = last.1.max(b),
            _ if a < b => out.push((a, b)),
            _ => {}
        }
    }
    out
}

/// Diagnostics: slots, copies, bytes of the ranges read and of the
/// unordered ones now.
#[cfg(feature = "diagnostics")]
pub(crate) fn sizes() -> (usize, usize, u64, u64) {
    let m = merged();
    (m.slots, m.shadows.len(), m.span_bytes, m.write_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counted_ranges_leave_with_their_last_slot() {
        let mut map = BTreeMap::new();
        count(&mut map, (0, 16), true);
        count(&mut map, (0, 16), true);
        count(&mut map, (0, 16), false);
        assert_eq!(map.get(&(0, 16)), Some(&1));
        count(&mut map, (0, 16), false);
        assert!(map.is_empty());
        // Removing what is not there is no change.
        count(&mut map, (8, 16), false);
        assert!(map.is_empty());
    }

    #[test]
    fn ranges_merge_where_they_overlap_or_lie_close() {
        let ranges = [(0, 16), (8, 32), (40, 48), (100, 120), (130, 130)];
        assert_eq!(merge(ranges.iter(), 0), [(0, 32), (40, 48), (100, 120)]);
        assert_eq!(merge(ranges.iter(), 8), [(0, 48), (100, 120)]);
        assert_eq!(merge(ranges.iter(), 10), [(0, 48), (100, 130)]);
    }
}

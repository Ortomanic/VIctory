//! Live controls: diagnostics switches changed while the title runs, so
//! experiments share one long session instead of a game launch each
//! (a launch costs minutes of boot for a minute of world).
//!
//! `VICTORY_CONTROL=<file>` (diagnostics builds): the file is read about
//! once per presented frame (a size and modification-time check, at most
//! every 200 ms); lines appended since the last read are applied in order:
//!
//! - `KEY=value`: a switch as if `KEY` had been set to `value` at start
//!   (`VICTORY_` may be left out, any case: `debug_view=normals`). An
//!   empty value, `off` or `unset` unsets it; for switches whose mere
//!   presence turns them on (`PASS_LOG`, `DUMP_TARGETS`, `DEBUG_PS`,
//!   `GPU_BREADCRUMBS`, `GPU_TIMING`) `0` turns them off too.
//! - `mark=<text>`: only logged, to split a log by phase.
//! - `snapshot=<n>`: the next `n` frames (default 1) are listed frames
//!   with their targets dumped (as `VICTORY_CAPTURE_FRAMES` plus
//!   `VICTORY_DUMP_TARGETS`).
//!
//! Every applied line logs `live: key=value applied at frame N`; unknown
//! keys and bad values are logged and ignored. A file that shrinks is read
//! from its start again.
//!
//! Switches that change translations or pipelines (`PATCH_CODE`,
//! `WAVE64_UNIFORM`, `SHADER_LOOP_LIMIT`, `BVH_SKINNING`, `DEBUG_PS`,
//! `PS_FALLBACK`) translate again only the shaders they affect and remake
//! only the host pipelines made from those; draws keep the old pipelines
//! until the new ones are made, and the old ones are released once
//! submissions that may use them have finished (see `pipeline`).

// Without diagnostics (or outside Windows) only the switch types are used.
#![cfg_attr(not(all(windows, feature = "diagnostics")), allow(dead_code))]

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, RwLock};

/// A boolean switch read at first use (`init`), replaceable at run time.
pub(crate) struct Flag(AtomicU8);

impl Flag {
    pub(crate) const fn new() -> Flag {
        Flag(AtomicU8::new(0))
    }

    /// The value; `init` gives it on the first read if it was not set.
    pub(crate) fn get(&self, init: impl FnOnce() -> bool) -> bool {
        // 0: not read yet, 1: off, 2: on.
        match self.0.load(Relaxed) {
            0 => {
                let _ = self
                    .0
                    .compare_exchange(0, 1 + u8::from(init()), Relaxed, Relaxed);
                self.0.load(Relaxed) == 2
            }
            v => v == 2,
        }
    }

    pub(crate) fn set(&self, on: bool) {
        self.0.store(1 + u8::from(on), Relaxed);
    }
}

/// A number switch read at first use, replaceable at run time.
pub(crate) struct Number(AtomicU64);

impl Number {
    const UNREAD: u64 = u64::MAX;

    pub(crate) const fn new() -> Number {
        Number(AtomicU64::new(Self::UNREAD))
    }

    /// The value; `init` gives it on the first read if it was not set
    /// (`u64::MAX` is not a value).
    pub(crate) fn get(&self, init: impl FnOnce() -> u64) -> u64 {
        match self.0.load(Relaxed) {
            Self::UNREAD => {
                let _ = self.0.compare_exchange(
                    Self::UNREAD,
                    init().min(Self::UNREAD - 1),
                    Relaxed,
                    Relaxed,
                );
                self.0.load(Relaxed)
            }
            v => v,
        }
    }

    pub(crate) fn set(&self, value: u64) {
        self.0.store(value.min(Self::UNREAD - 1), Relaxed);
    }
}

/// A switch of any type read at first use, replaceable at run time.
pub(crate) struct Value<T>(RwLock<Option<Arc<T>>>);

impl<T> Value<T> {
    pub(crate) const fn new() -> Value<T> {
        Value(RwLock::new(None))
    }

    pub(crate) fn get(&self, init: impl FnOnce() -> T) -> Arc<T> {
        if let Some(v) = &*self.0.read().unwrap_or_else(|e| e.into_inner()) {
            return v.clone();
        }
        let mut v = self.0.write().unwrap_or_else(|e| e.into_inner());
        v.get_or_insert_with(|| Arc::new(init())).clone()
    }

    /// Sets the value; the one before (if it was read).
    pub(crate) fn set(&self, value: T) -> Option<Arc<T>> {
        self.0
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .replace(Arc::new(value))
    }
}

/// One line of the control file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Command {
    /// A switch by its environment name (`VICTORY_...`), `None`: unset.
    Set {
        key: String,
        value: Option<String>,
    },
    Mark(String),
    /// Frames to list and dump, from the next one.
    Snapshot(u64),
}

/// A control file line: `None` for blank lines and `#` comments.
pub(crate) fn parse(line: &str) -> Result<Option<Command>, String> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return Ok(None);
    }
    let Some((key, value)) = line.split_once('=') else {
        return Err(format!("expected key=value: {line:?}"));
    };
    let (key, value) = (key.trim(), value.trim());
    if key.is_empty() || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
        return Err(format!("not a key: {key:?}"));
    }
    match key.to_ascii_lowercase().as_str() {
        "mark" => return Ok(Some(Command::Mark(value.to_owned()))),
        "snapshot" => {
            return match value {
                "" => Ok(Some(Command::Snapshot(1))),
                v => match v.parse::<u64>() {
                    Ok(n) if (1..=1000).contains(&n) => Ok(Some(Command::Snapshot(n))),
                    _ => Err(format!("snapshot expects 1 to 1000 frames: {v:?}")),
                },
            };
        }
        _ => {}
    }
    let mut key = key.to_ascii_uppercase();
    if !key.starts_with("VICTORY_") {
        key.insert_str(0, "VICTORY_");
    }
    let value = match value.to_ascii_lowercase().as_str() {
        "" | "off" | "unset" => None,
        _ => Some(value.to_owned()),
    };
    Ok(Some(Command::Set { key, value }))
}

/// For switches whose presence turns them on: also `0` is off.
pub(crate) fn present(value: Option<&str>) -> bool {
    value.is_some_and(|v| v != "0")
}

/// Switches only read at start (made live: none of them).
const STARTUP_ONLY: [&str; 23] = [
    "VICTORY_BINDLESS",
    "VICTORY_LEND_SLOTS",
    "VICTORY_SHARE_TABLES",
    "VICTORY_KEEP_UNMADE_TABLES",
    "VICTORY_HOST_VIEWS",
    "VICTORY_D3D12_DEBUG",
    "VICTORY_D3D12_GBV",
    "VICTORY_D3D12_WARP",
    "VICTORY_D3D12_DRED",
    "VICTORY_PRESENT_PLANE",
    "VICTORY_ROOT_BOUNDS",
    "VICTORY_STRICT_PHIS",
    "VICTORY_HOST_WAVE32",
    "VICTORY_TEXTURE_THREADS",
    "VICTORY_COPY_MODE",
    "VICTORY_DIRECT_WRITES",
    "VICTORY_INDIRECT",
    "VICTORY_PREVIEW_SCENE_DEPTH",
    "VICTORY_CAPTURE",
    "VICTORY_FFX_OWN_HEAP",
    // Whether host pipelines go into a library: decided when the cache
    // loads (pipeline/library.rs), before the first pipeline.
    "VICTORY_PIPELINE_LIBRARY",
    "VICTORY_VIEW_PROBE",
    "VICTORY_ARGS_PROBE_LOG",
];

/// Reads the lines appended to a file since the last read.
#[derive(Default)]
pub(crate) struct Tail {
    offset: u64,
    stamp: Option<(u64, std::time::SystemTime)>,
}

impl Tail {
    /// The whole lines appended since the last call (none if the file is
    /// unchanged or missing); a file shorter than what was read is read
    /// from its start.
    pub(crate) fn read(&mut self, path: &std::path::Path) -> Vec<String> {
        use std::io::{Read, Seek, SeekFrom};
        let Ok(meta) = std::fs::metadata(path) else {
            return Vec::new();
        };
        let stamp = (meta.len(), meta.modified().unwrap_or(std::time::UNIX_EPOCH));
        if self.stamp == Some(stamp) {
            return Vec::new();
        }
        if meta.len() < self.offset {
            self.offset = 0;
        }
        let Ok(mut file) = std::fs::File::open(path) else {
            return Vec::new();
        };
        let mut bytes = Vec::new();
        if file.seek(SeekFrom::Start(self.offset)).is_err() || file.read_to_end(&mut bytes).is_err()
        {
            return Vec::new();
        }
        // A line still being written waits for its end.
        let Some(end) = bytes.iter().rposition(|&b| b == b'\n') else {
            self.stamp = None;
            return Vec::new();
        };
        self.offset += end as u64 + 1;
        // Unchanged since: until the size or time moves on.
        self.stamp = (self.offset == meta.len()).then_some(stamp);
        String::from_utf8_lossy(&bytes[..end])
            .lines()
            .map(str::to_owned)
            .collect()
    }
}

/// Live controls once per presented frame `frame` (cheap when nothing
/// changed): applies new control file lines, and releases host pipelines
/// replaced a few frames ago.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn poll(frame: u64) {
    use std::sync::{Mutex, OnceLock};
    crate::pipeline::sweep(frame);
    static PATH: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    let Some(path) = PATH
        .get_or_init(|| {
            let path = std::env::var_os("VICTORY_CONTROL").map(std::path::PathBuf::from)?;
            tracing::info!(path = %path.display(), "live controls: reading the control file once per frame");
            Some(path)
        })
        .as_ref()
    else {
        return;
    };
    static STATE: Mutex<Option<(Tail, std::time::Instant)>> = Mutex::new(None);
    let lines = {
        let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let now = std::time::Instant::now();
        let (tail, last) =
            state.get_or_insert_with(|| (Tail::default(), now - std::time::Duration::from_secs(1)));
        if now - *last < std::time::Duration::from_millis(200) {
            return;
        }
        *last = now;
        tail.read(path)
    };
    for line in lines {
        match parse(&line) {
            Ok(Some(command)) => {
                if let Err(e) = apply(&command, frame) {
                    tracing::warn!("live: {} ignored at frame {frame}: {e}", line.trim());
                }
            }
            Ok(None) => {}
            Err(e) => tracing::warn!("live: line ignored at frame {frame}: {e}"),
        }
    }
}

/// Applies `command` at presented frame `frame` and logs it.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn apply(command: &Command, frame: u64) -> Result<(), String> {
    use victory_shader::emit;
    let (key, value) = match command {
        Command::Mark(text) => {
            tracing::info!("live: mark={text} applied at frame {frame}");
            return Ok(());
        }
        Command::Snapshot(n) => {
            crate::capture::snapshot(frame + 1, frame + n);
            tracing::info!(
                "live: snapshot={n} applied at frame {frame} (frames {} to {} listed, targets dumped)",
                frame + 1,
                frame + n
            );
            return Ok(());
        }
        Command::Set { key, value } => (key.as_str(), value.as_deref()),
    };
    let detail: String = match key {
        "VICTORY_SKIP_PIPELINES" => format!(
            "{} shader hashes skipped",
            crate::pipeline::set_skip_pipelines(value)
        ),
        "VICTORY_PATCH_CODE" => crate::pipeline::set_patch_code(value),
        "VICTORY_BVH_SKINNING" => crate::pipeline::set_bvh_skinning(value),
        "VICTORY_WAVE64_UNIFORM" => {
            emit::set_wave64_uniform(value.is_none_or(|v| v != "0"));
            crate::pipeline::retranslate_asking(emit::SETTING_WAVE64_UNIFORM)
        }
        "VICTORY_SMEM_HEAP_RAW" => {
            emit::set_smem_heap_raw(value.is_none_or(|v| v != "0"));
            crate::pipeline::retranslate_asking(emit::SETTING_SMEM_HEAP_RAW)
        }
        "VICTORY_PIPELINE_WAIT_MS" => {
            format!("{} ms per frame", crate::pipeline::set_pipeline_wait(value))
        }
        "VICTORY_SHADER_LOOP_LIMIT" => {
            let limit = emit::set_loop_limit(value);
            format!(
                "limit {limit}, {}",
                crate::pipeline::retranslate_asking(emit::SETTING_LOOP_LIMIT)
            )
        }
        "VICTORY_DEBUG_PS" => {
            crate::pipeline::host::set_debug_ps(present(value).then_some("1"));
            crate::pipeline::remake_for_debug_ps(false)
        }
        "VICTORY_PS_FALLBACK" => {
            crate::pipeline::host::set_ps_fallback(value);
            crate::pipeline::remake_for_debug_ps(true)
        }
        "VICTORY_PASS_LOG" => {
            crate::draw::set_pass_log(present(value).then_some(value.unwrap_or_default()));
            String::new()
        }
        "VICTORY_ARGS_PROBE" => {
            crate::draw::set_args_probe(value);
            String::new()
        }
        "VICTORY_CAPTURE_FRAMES" => {
            crate::capture::set_frames(value);
            String::new()
        }
        "VICTORY_CAPTURE_AFTER_SCENE" => {
            crate::capture::set_after_scene(value);
            String::new()
        }
        "VICTORY_DUMP_TARGETS" => {
            crate::capture::set_dump(present(value));
            String::new()
        }
        "VICTORY_CAPTURE_FAST" => {
            crate::capture::set_fast(value == Some("1"));
            String::new()
        }
        "VICTORY_SHOW_TARGET" => crate::capture::set_show(value)?,
        "VICTORY_DEBUG_VIEW" => crate::debug_view::set(value)?,
        "VICTORY_GPU_BREADCRUMBS" => crate::host::set_breadcrumbs(present(value)),
        "VICTORY_GPU_TIMING" => {
            crate::host::set_timing(present(value));
            String::new()
        }
        "VICTORY_SYNC_HEAPS" => {
            crate::descriptor::set_sync_heaps(value);
            String::new()
        }
        "VICTORY_BINDLESS_REFRESH" => {
            crate::descriptor::bindless::set_every_submission(value == Some("submission"));
            String::new()
        }
        key if STARTUP_ONLY.contains(&key) => return Err(format!("{key} is read at start only")),
        key => return Err(format!("unknown key {key}")),
    };
    let value = value.unwrap_or("");
    if detail.is_empty() {
        tracing::info!("live: {key}={value} applied at frame {frame}");
    } else {
        tracing::info!("live: {key}={value} applied at frame {frame} ({detail})");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(key: &str, value: Option<&str>) -> Command {
        Command::Set {
            key: key.into(),
            value: value.map(Into::into),
        }
    }

    #[test]
    fn lines_parse() {
        assert_eq!(parse("  "), Ok(None));
        assert_eq!(parse("# a comment = x"), Ok(None));
        assert_eq!(
            parse("debug_view=normals"),
            Ok(Some(set("VICTORY_DEBUG_VIEW", Some("normals"))))
        );
        assert_eq!(
            parse(" VICTORY_SKIP_PIPELINES = 0123456789abcdef "),
            Ok(Some(set(
                "VICTORY_SKIP_PIPELINES",
                Some("0123456789abcdef")
            )))
        );
        assert_eq!(parse("Pass_Log="), Ok(Some(set("VICTORY_PASS_LOG", None))));
        assert_eq!(
            parse("pass_log=off"),
            Ok(Some(set("VICTORY_PASS_LOG", None)))
        );
        assert_eq!(
            parse("sync_heaps=0"),
            Ok(Some(set("VICTORY_SYNC_HEAPS", Some("0"))))
        );
        // Patch entries keep their own `=` and `>`.
        assert_eq!(
            parse("patch_code=ab@1c=0>1"),
            Ok(Some(set("VICTORY_PATCH_CODE", Some("ab@1c=0>1"))))
        );
        assert_eq!(
            parse("mark=phase 2: albedo = on"),
            Ok(Some(Command::Mark("phase 2: albedo = on".into())))
        );
        assert_eq!(parse("snapshot="), Ok(Some(Command::Snapshot(1))));
        assert_eq!(parse("SNAPSHOT=3"), Ok(Some(Command::Snapshot(3))));
        assert!(parse("snapshot=0").is_err());
        assert!(parse("snapshot=x").is_err());
        assert!(parse("no equals sign").is_err());
        assert!(parse("bad key=1").is_err());
        assert!(parse("=1").is_err());
    }

    #[test]
    fn presence_switches_take_zero_as_off() {
        assert!(!present(None));
        assert!(!present(Some("0")));
        assert!(present(Some("1")));
        assert!(present(Some("all")));
    }

    #[test]
    fn switches_read_once_then_follow_sets() {
        let flag = Flag::new();
        assert!(flag.get(|| true));
        assert!(flag.get(|| false), "read once");
        flag.set(false);
        assert!(!flag.get(|| true));
        let unread = Flag::new();
        unread.set(true);
        assert!(unread.get(|| false), "a set before the first read wins");

        let n = Number::new();
        assert_eq!(n.get(|| 7), 7);
        n.set(0);
        assert_eq!(n.get(|| 7), 0);
        n.set(u64::MAX);
        assert_eq!(n.get(|| 7), u64::MAX - 1);

        let v: Value<Vec<u32>> = Value::new();
        assert_eq!(*v.get(|| vec![1]), [1]);
        assert_eq!(v.set(vec![2]).as_deref().map(Vec::as_slice), Some(&[1][..]));
        assert_eq!(*v.get(|| vec![3]), [2]);
    }

    #[test]
    fn the_tail_reads_whole_appended_lines_once() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("control");
        let mut tail = Tail::default();
        assert!(tail.read(&path).is_empty(), "no file yet");
        std::fs::write(&path, "a=1\nb=2\nc=").unwrap();
        assert_eq!(tail.read(&path), ["a=1", "b=2"]);
        assert!(tail.read(&path).is_empty());
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(b"3\nd=4\n").unwrap();
        drop(f);
        assert_eq!(tail.read(&path), ["c=3", "d=4"]);
        assert!(tail.read(&path).is_empty());
        // Truncated and written again: read from the start.
        std::fs::write(&path, "e=5\n").unwrap();
        assert_eq!(tail.read(&path), ["e=5"]);
    }

    /// Applying lines changes the switches the code reads (here the
    /// pure run-time ones), logs, and refuses unknown keys.
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn applied_lines_change_switches() {
        let apply_line = |line: &str| apply(&parse(line).unwrap().unwrap(), 10);
        apply_line("skip_pipelines=0123456789ABCDEF,bad").unwrap();
        assert_eq!(*crate::pipeline::skip_list(), ["0123456789abcdef"]);
        apply_line("skip_pipelines=").unwrap();
        assert!(crate::pipeline::skip_list().is_empty());
        apply_line("snapshot=2").unwrap();
        assert!(!crate::capture::listed(10));
        assert!(crate::capture::listed(11) && crate::capture::listed(12));
        assert!(!crate::capture::listed(13));
        apply_line("args_probe=8").unwrap();
        assert_eq!(crate::draw::args_probe_every(), 8);
        apply_line("args_probe=").unwrap();
        assert_eq!(crate::draw::args_probe_every(), 0);
        apply_line("pass_log=4").unwrap();
        assert_eq!(crate::draw::pass_log(), (true, 4));
        apply_line("pass_log=0").unwrap();
        assert_eq!(crate::draw::pass_log(), (false, 0));
        apply_line("debug_view=normals").unwrap();
        assert!(crate::debug_view::enabled());
        assert!(apply_line("debug_view=sideways").is_err());
        assert!(crate::debug_view::enabled(), "a bad value changes nothing");
        apply_line("debug_view=").unwrap();
        assert!(!crate::debug_view::enabled());
        assert!(apply_line("no_such_switch=1").is_err());
        assert!(apply_line("bindless=0").is_err(), "start-only");
        assert!(apply_line("pipeline_library=1").is_err(), "start-only");
        // The default again (make's tests read it).
        apply_line("pipeline_wait_ms=500").unwrap();
        apply_line("mark=phase two").unwrap();
    }
}

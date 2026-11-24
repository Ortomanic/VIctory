//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are made on a pipeline's first draw, not at creation:
//! Grand Theft Auto VI creates some 60,000 pipelines in its first 20
//! seconds and draws with a few dozen (decision 0007, step 3). Shader
//! containers are shared by content hash and translated once, on first
//! use (decision 0003).
//!
//! shortcut: only vertex plus pixel pipelines from streams and compute
//! pipelines get host pipelines; mesh, tessellation and geometry stages,
//! and the plain graphics description, keep their containers only. The 32-byte
//! packet after the header stays zero: titles only copy it inline, which
//! our null command-list limits keep from running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::{Arc, Mutex};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const PIPELINE_IIDS: [Iid; 4] = [
    guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged.
    pub(crate) hash: String,
    container: Vec<u8>,
    /// The translation once made (`None` inside: it does not translate).
    /// Live controls clear it to have the shader translated again; host
    /// pipelines keep the translation they were made from.
    #[cfg(windows)]
    translated: std::sync::RwLock<Option<Option<Arc<Translated>>>>,
    /// Held while translating, so a shader is translated once at a time.
    #[cfg(windows)]
    translating: Mutex<()>,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)
    }
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// The shared container for `bytes`.
fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let full = blake3::hash(bytes);
    let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    shaders
        .get_or_insert_with(HashMap::new)
        .entry(*full.as_bytes())
        .or_insert_with(|| {
            Arc::new(Shader {
                stage,
                hash: full.to_hex()[..16].to_owned(),
                container: bytes.to_vec(),
                #[cfg(windows)]
                translated: std::sync::RwLock::new(None),
                #[cfg(windows)]
                translating: Mutex::new(()),
            })
        })
        .clone()
}

/// A translated shader and the console resources its code reads.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) host: victory_shader::emit::HostShader,
    pub(crate) used: Vec<victory_shader::UsedResource>,
}

#[cfg(windows)]
impl Shader {
    /// The translation, made on first use; a shader that does not
    /// translate is logged once (per translation: live controls may have
    /// it translated again).
    pub(crate) fn translated(&self) -> Option<Arc<Translated>> {
        if let Some(done) = &*self.translated.read().unwrap_or_else(|e| e.into_inner()) {
            return done.clone();
        }
        let _translating = self.translating.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(done) = &*self.translated.read().unwrap_or_else(|e| e.into_inner()) {
            return done.clone();
        }
        let done = self.translate().map(Arc::new);
        *self.translated.write().unwrap_or_else(|e| e.into_inner()) = Some(done.clone());
        done
    }

    /// Whether the shader was translated and failed (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn failed(&self) -> bool {
        matches!(&*self.translated.read().unwrap_or_else(|e| e.into_inner()), Some(None))
    }

    /// Live controls: forgets the translation (after one in progress), so
    /// the next use translates again with the settings of now. Whether
    /// there was one.
    #[cfg(feature = "diagnostics")]
    fn forget(&self) -> bool {
        let _translating = self.translating.lock().unwrap_or_else(|e| e.into_inner());
        self.translated.write().unwrap_or_else(|e| e.into_inner()).take().is_some()
    }

    /// Whether it has no translation (none yet, or one in progress).
    #[cfg(feature = "diagnostics")]
    fn untranslated(&self) -> bool {
        self.translated.read().unwrap_or_else(|e| e.into_inner()).is_none()
    }

    /// The run-time translation settings its translation asked
    /// (`victory_shader::emit::SETTING_*`; 0 if not translated).
    #[cfg(feature = "diagnostics")]
    fn settings(&self) -> u8 {
        match &*self.translated.read().unwrap_or_else(|e| e.into_inner()) {
            Some(Some(t)) => t.host.settings,
            _ => 0,
        }
    }

    fn translate(&self) -> Option<Translated> {
        let patches = code_patches(&self.hash);
        let done = if bvh_skinning(&self.hash) {
            Err(BVH_SKINNING.to_owned())
        } else if !patches.is_empty() {
            tracing::warn!(hash = %self.hash, ?patches, "VICTORY_PATCH_CODE: console words replaced");
            translate_patched(&self.container, &patches)
        } else {
            translate(&self.container)
        };
        match &done {
            Ok(t) => {
                tracing::info!(
                    stage = self.stage,
                    hash = %self.hash,
                    user_data = t.host.user_data.len(),
                    samplers = t.host.samplers.len(),
                    "shader translated"
                );
                if t.host.zero_filled_phis {
                    tracing::info!(
                        stage = self.stage,
                        hash = %self.hash,
                        "#612 fallback: zero-filled live phi"
                    );
                }
            }
            Err(e) => tracing::warn!(
                stage = self.stage,
                hash = %self.hash,
                error = %e,
                "shader does not translate; its pipelines' draws are dropped"
            ),
        }
        done.ok()
    }
}

/// Grand Theft Auto VI's skinning for its ray-tracing structures
/// (`bvh_skinning` and `bvh_skinning_smh` in shaders.rpf: skinned, rope,
/// cable, tree and hair vertices and indices that `XboxBvhBuilder.dll`
/// builds the BVH from), by shader hash.
///
/// Nothing on the host reads what they write: VIctory does no ray tracing
/// (the BVH builder's own shaders do not translate either). Every vertex
/// buffer a job record names is a buffer view in the heap read bindlessly;
/// until those were made outside tables (`descriptor::bindless`) they all
/// read zeros through null slots (probe, bisect-fault, 2026-11-24: 6,144
/// records, every slot named unmade), and run that way they faulted or
/// hung the GPU (Xid 13 at CSComputeSkinningDefaultVelocity after
/// CSComputeSkinningRigidVelocity, Xid 109 at CSGenerateHairIB).
///
/// With those views made they still fault the GPU (bindless, 2026-11-24:
/// Xid 13 in 2 of 3 world runs within seconds of their first dispatch, 1
/// run clean for 5.5 min), so they still do not run by default;
/// `VICTORY_BVH_SKINNING=1` runs them.
#[cfg(windows)]
const BVH_SKINNING_SHADERS: [&str; 43] = [
    "81dc252c4d183e30", // CSComputeCable
    "6b39336022e5e663", // CSComputeCableVelocity
    "ced45910ad577dcb", // CSComputeCrossCardTree
    "207bf7d2f14a87e6", // CSComputeRope
    "2cb480030989538a", // CSComputeRopeVelocity
    "c7aef9c89f12051f", // CSComputeSkinningDefault
    "3cf39f4495b908d0", // CSComputeSkinningDefaultVelocity
    "9db486cee396e315", // CSComputeSkinningDeformable
    "96d1ef386a0ddcfb", // CSComputeSkinningDeformableRigid
    "cb90a66c7fe122aa", // CSComputeSkinningDeformableRigidVelocity
    "d28b4922384831fe", // CSComputeSkinningDeformableSecMotion
    "822e62ded676ef45", // CSComputeSkinningDeformableSecMotionRigid
    "ca2c77616107b128", // CSComputeSkinningDeformableSecMotionRigidVelocity
    "c7213dc64c67638b", // CSComputeSkinningDeformableSecMotionVelocity
    "cd34b8dd52ee5c0c", // CSComputeSkinningDeformableVelocity
    "60b09004fef7f2f1", // CSComputeSkinningPed
    "9ffcaff533995118", // CSComputeSkinningPedBlendShapes
    "827b309e779552d7", // CSComputeSkinningPedBlendShapesSecMotion
    "82a2ad6d07417d36", // CSComputeSkinningPedBlendShapesSecMotionVelocity
    "90a0692af4320129", // CSComputeSkinningPedBlendShapesVelocity
    "b8e420aac4197681", // CSComputeSkinningPedSecMotion
    "904e7a51e3fab26c", // CSComputeSkinningPedSecMotionVelocity
    "dd24ff46af3ae6ee", // CSComputeSkinningPedVelocity
    "71ab19b0a4669299", // CSComputeSkinningRigid
    "8f9eb68d0c6dd5cf", // CSComputeSkinningRigidVelocity
    "c968414adaaeb93d", // CSComputeSkinningSecMotion
    "77edaab306206e9a", // CSComputeSkinningSecMotionRigid
    "d3022ba8427bc72c", // CSComputeSkinningSecMotionRigidVelocity
    "4518151afe8c6c09", // CSComputeSkinningSecMotionVelocity
    "e06a21f3cea46973", // CSComputeSkinningSlodPed
    "b5a672a450b321e2", // CSComputeSkinningStaticPedProp
    "878933b499cfa14d", // CSComputeSkinningTreeAnim
    "f16876929fa64f9f", // CSComputeSkinningTreeAnimVelocity
    "023439f44041b689", // CSComputeSkinningUmWind
    "75bd00da58a1615c", // CSComputeSkinningUmWindVelocity
    "a567b9ef845fb680", // CSComputeTreeAnim
    "aff8936db59182b7", // CSComputeTreeAnimVelocity
    "409e2cddc90b6d0d", // CSComputeVSplitTree
    "952fd529dd779b27", // CSUnpackIB_CCW
    "1a77e876755eff12", // CSUnpackIB
    "64fe9fa3cb71d57f", // CSUnpackVB
    "48426bae580a080c", // CSGenerateHairIB (bvh_skinning_smh)
    "72f552f4553fb5f9", // CSGenerateHairVB (bvh_skinning_smh)
];

#[cfg(windows)]
const BVH_SKINNING: &str =
    "BVH skinning: nothing on the host reads it (no ray tracing) and it faults the GPU; VICTORY_BVH_SKINNING=1 runs it";

/// Whether the shader with `hash` is one of [`BVH_SKINNING_SHADERS`] and
/// they are not to run (unless `VICTORY_BVH_SKINNING=1`).
#[cfg(windows)]
fn bvh_skinning(hash: &str) -> bool {
    !BVH_RUN.get(|| std::env::var("VICTORY_BVH_SKINNING").ok().as_deref() == Some("1"))
        && BVH_SKINNING_SHADERS.contains(&hash)
}

#[cfg(windows)]
static BVH_RUN: crate::live::Flag = crate::live::Flag::new();

/// With the `diagnostics` feature, `VICTORY_PATCH_CODE` replaces console
/// code words before translation: `hash@offset=old>new` entries (hex: the
/// 16-digit shader hash, the byte offset in its code as shader-lab's `.s`
/// listing gives it, the words), separated by commas. E.g. a culling
/// shader's visibility test forced true, to see what culling removes.
/// Patched shaders bypass the shader cache.
#[cfg(windows)]
fn code_patches(hash: &str) -> Vec<(usize, u32, u32)> {
    let all = PATCHES.get(|| parse_patches(&std::env::var("VICTORY_PATCH_CODE").unwrap_or_default()));
    all.iter()
        .filter(|(h, ..)| h == hash)
        .map(|&(_, at, old, new)| (at, old, new))
        .collect()
}

/// `VICTORY_PATCH_CODE`'s entries (none without the `diagnostics` feature).
#[cfg(windows)]
type Patches = Vec<(String, usize, u32, u32)>;

#[cfg(windows)]
static PATCHES: crate::live::Value<Patches> = crate::live::Value::new();

#[cfg(windows)]
fn parse_patches(value: &str) -> Patches {
    if !cfg!(feature = "diagnostics") {
        return Vec::new();
    }
    value
        .split(',')
        .filter(|e| !e.trim().is_empty())
        .filter_map(|e| {
            let parse = || {
                let (hash, rest) = e.trim().split_once('@')?;
                let (at, words) = rest.split_once('=')?;
                let (old, new) = words.split_once('>')?;
                Some((
                    hash.to_ascii_lowercase(),
                    usize::from_str_radix(at.trim_start_matches("0x"), 16).ok()?,
                    u32::from_str_radix(old.trim_start_matches("0x"), 16).ok()?,
                    u32::from_str_radix(new.trim_start_matches("0x"), 16).ok()?,
                ))
            };
            let p = parse();
            if p.is_none() {
                tracing::warn!(entry = e, "VICTORY_PATCH_CODE expects hash@offset=old>new");
            }
            p
        })
        .collect()
}

#[cfg(windows)]
fn translate_patched(container: &[u8], patches: &[(usize, u32, u32)]) -> Result<Translated, String> {
    let host = victory_shader::emit::translate_patched(
        container,
        crate::shader_cache::options(),
        patches,
    )
    .map_err(|e| e.to_string())?;
    let mut t = translate(container)?;
    t.host = host;
    Ok(t)
}

#[cfg(windows)]
fn translate(container: &[u8]) -> Result<Translated, String> {
    use victory_shader::{Container, HwStage};
    let host = crate::shader_cache::host_shader(container)?;
    let c = Container::parse(container).map_err(|e| e.to_string())?;
    let used = match victory_shader::program_part(&c) {
        Some(part) => victory_shader::native_programs(part)
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|p| {
                matches!(
                    p.stage,
                    HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs
                )
            })
            .map(|p| p.used)
            .unwrap_or_default(),
        None => Vec::new(),
    };
    Ok(Translated { host, used })
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container.
    shaders: Vec<Arc<Shader>>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// What its host pipeline is made from, and the host pipeline once made
    /// (by a worker: the title may release the pipeline meanwhile).
    #[cfg(windows)]
    host: Arc<make::Pending>,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
pub(crate) fn bound(this: This) {
    if this.is_null() {
        return;
    }
    let p = pipeline(this);
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    for shader in &p.shaders {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.container.len(), hash = %shader.hash, "first bind");
    }
}

/// The host pipeline of a title pipeline; `None` if one of its shaders does
/// not translate, the host refuses it, or it is still being made (its first
/// use starts that, see [`make`]).
#[cfg(windows)]
pub(crate) fn host_pipeline<'a>(this: This) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    make::ready(&pipeline(this).host)
}

/// Host pipelines queued and not made yet.
#[cfg(windows)]
pub(crate) fn making() -> u64 {
    make::waiting()
}

/// Draws and dispatches dropped since the last [`take_drops`], by reason
/// (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
static DROPS: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Shader hashes as printed in the pipeline diagnostics.
#[cfg(all(windows, feature = "diagnostics"))]
fn parse_skip_hashes(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(
                    hash,
                    "VICTORY_SKIP_PIPELINES expects 16-digit shader hashes"
                );
                None
            }
        })
        .collect()
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skipped_shader(p: &Pipeline) -> Option<&str> {
    // Per draw: the common case (nothing skipped) without the lock.
    if !SKIP_ANY.get(|| !skip_hashes().is_empty()) {
        return None;
    }
    let hashes = skip_hashes();
    p.shaders
        .iter()
        .find(|shader| hashes.contains(&shader.hash))
        .map(|shader| shader.hash.as_str())
}

#[cfg(all(windows, feature = "diagnostics"))]
static SKIP: crate::live::Value<Vec<String>> = crate::live::Value::new();
#[cfg(all(windows, feature = "diagnostics"))]
static SKIP_ANY: crate::live::Flag = crate::live::Flag::new();

#[cfg(all(windows, feature = "diagnostics"))]
fn skip_hashes() -> Arc<Vec<String>> {
    SKIP.get(|| parse_skip_hashes(&std::env::var("VICTORY_SKIP_PIPELINES").unwrap_or_default()))
}

/// The shader hashes skipped now (diagnostics).
#[cfg(all(windows, feature = "diagnostics", test))]
pub(crate) fn skip_list() -> Arc<Vec<String>> {
    skip_hashes()
}

/// Live controls: `VICTORY_SKIP_PIPELINES` from now on.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn set_skip_pipelines(value: Option<&str>) -> usize {
    let hashes = parse_skip_hashes(value.unwrap_or_default());
    let n = hashes.len();
    SKIP.set(hashes);
    SKIP_ANY.set(n > 0);
    n
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn is_skipped(this: This) -> bool {
    !this.is_null() && skipped_shader(pipeline(this)).is_some()
}

/// A pipeline's shaders and, if it has no host pipeline, why
/// (diagnostics: the pass log of listed frames).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn describe(this: This) -> String {
    if this.is_null() {
        return "no pipeline".into();
    }
    let p = pipeline(this);
    let hashes: Vec<String> = p
        .shaders
        .iter()
        .map(|s| format!("{}:{}", &s.stage[..2], s.hash))
        .collect();
    let why = skipped_shader(p)
        .map(|hash| format!("skipped shader {hash}"))
        .or_else(|| make::missing(&p.host));
    match why {
        Some(why) => format!("{} DROPPED({why})", hashes.join(",")),
        None => hashes.join(","),
    }
}

/// Counts a draw or dispatch of `this` intentionally skipped or dropped
/// for want of a host pipeline (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_drop(this: This) {
    let why = if this.is_null() {
        Some("no pipeline set".to_owned())
    } else {
        let p = pipeline(this);
        skipped_shader(p)
            .map(|hash| format!("skipped shader {hash}"))
            .or_else(|| make::missing(&p.host))
    };
    if let Some(why) = why {
        note_why(why);
    }
}

/// Counts a draw or dispatch dropped for `why` (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_why(why: String) {
    let mut drops = DROPS.lock().unwrap_or_else(|e| e.into_inner());
    *drops.get_or_insert_with(HashMap::new).entry(why).or_default() += 1;
}

/// The drops counted since the last call, most first (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn take_drops() -> Vec<(String, u64)> {
    let drops = DROPS.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut drops: Vec<(String, u64)> = drops.into_iter().flatten().collect();
    // Frame reports only show the first 40 reasons. Keep ties ordered so
    // a HashMap iteration order change cannot change which reasons appear.
    drops.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    drops
}

/// The stage's container, if it has one.
fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    if code.code.is_null() || code.size == 0 {
        return None;
    }
    // SAFETY: the title passes `size` bytes of shader code.
    Some(shader(stage, unsafe {
        std::slice::from_raw_parts(code.code, code.size)
    }))
}

fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Box<host::GraphicsState>>,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    #[cfg(windows)]
    let host = Arc::new(make::Pending::new(shaders.clone(), graphics));
    #[cfg(windows)]
    #[cfg(windows)]
    make::register(&host);
    make::early(&host);
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(windows)]
        host,
        shaders,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// Device method.
pub(crate) extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    let shaders = stages.filter_map(|(s, c)| copy(s, c)).collect();
    hand_out(
        shaders,
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method.
pub(crate) extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    hand_out(
        copy("compute", &desc.stage).into_iter().collect(),
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
pub(crate) extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_compute(this, desc, iid, out)
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
pub(crate) extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    let mut shaders = Vec::new();
    #[cfg(windows)]
    let mut graphics = host::GraphicsState::default();
    let mut at = 0;
    while at + 8 <= desc.size {
        // SAFETY: inside the stream the title passes.
        let kind = unsafe { desc.stream.add(at).cast::<u32>().read_unaligned() };
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(
                kind,
                at,
                "pipeline subobject of unknown size; the rest is ignored"
            );
            break;
        };
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8.
            let code = unsafe { &*desc.stream.add(at + 8).cast::<Bytecode>() };
            shaders.extend(copy(stage, code));
        }
        #[cfg(windows)]
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe {
            graphics.read(kind, desc.stream.add(at))
        };
        at += size;
    }
    hand_out(
        shaders,
        #[cfg(windows)]
        Some(Box::new(graphics)),
        iid,
        out,
    )
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
mod library;
#[cfg(windows)]
mod make;

/// A host pipeline replaced by a remake: released by [`sweep`] (without
/// diagnostics, where nothing is remade, once running submissions finish).
#[cfg(windows)]
fn replaced(old: Box<Option<host::HostPipeline>>) {
    #[cfg(feature = "diagnostics")]
    live::bury(old);
    #[cfg(not(feature = "diagnostics"))]
    crate::host::retire(Grave(old));
}

/// A replaced host pipeline on its way out.
#[cfg(windows)]
struct Grave(#[allow(dead_code)] Box<Option<host::HostPipeline>>);
// SAFETY: D3D12 objects are free-threaded; nothing else reaches it.
#[cfg(windows)]
unsafe impl Send for Grave {}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) use live::{
    remake_for_debug_ps, retranslate_asking, set_bvh_skinning, set_patch_code, set_pipeline_wait,
    sweep,
};

/// Live controls (`crate::live`): switches that change translations or
/// host pipelines.
#[cfg(all(windows, feature = "diagnostics"))]
mod live {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// Presented frames a replaced host pipeline is kept for: lists
    /// recorded with it before the remake are submitted by then (the title
    /// records and submits a frame's lists within it).
    const KEEP_FRAMES: u64 = 4;

    static GRAVES: Mutex<Vec<(u64, Grave)>> = Mutex::new(Vec::new());
    static REPLACED: AtomicU64 = AtomicU64::new(0);
    static RELEASED: AtomicU64 = AtomicU64::new(0);

    pub(super) fn bury(old: Box<Option<host::HostPipeline>>) {
        REPLACED.fetch_add(1, Relaxed);
        let due = crate::queue::presented() + KEEP_FRAMES;
        GRAVES.lock().unwrap_or_else(|e| e.into_inner()).push((due, Grave(old)));
    }

    /// At presented frame `frame`: replaced host pipelines kept long enough
    /// go to [`crate::host::retire`] (dropped once the submissions running
    /// now have finished).
    pub(crate) fn sweep(frame: u64) {
        let due: Vec<Grave> = {
            let mut graves = GRAVES.lock().unwrap_or_else(|e| e.into_inner());
            if graves.is_empty() {
                return;
            }
            let (due, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut *graves).into_iter().partition(|(at, _)| *at <= frame);
            *graves = keep;
            due.into_iter().map(|(_, g)| g).collect()
        };
        if due.is_empty() {
            return;
        }
        let released = RELEASED.fetch_add(due.len() as u64, Relaxed) + due.len() as u64;
        tracing::info!(
            "live: {} replaced host pipelines released at frame {frame} ({} replaced, {released} released so far)",
            due.len(),
            REPLACED.load(Relaxed)
        );
        crate::host::retire(due);
    }

    /// Forgets the translations of the shaders `which` picks and queues a
    /// remake of the host pipelines made from them: (shaders, pipelines).
    fn retranslate(which: impl Fn(&Shader) -> bool) -> (usize, usize) {
        let shaders: Vec<Arc<Shader>> = SHADERS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .flat_map(|m| m.values())
            .filter(|s| which(s))
            .cloned()
            .collect();
        let forgotten: std::collections::HashSet<*const Shader> =
            shaders.iter().filter(|s| s.forget()).map(Arc::as_ptr).collect();
        if forgotten.is_empty() {
            return (0, 0);
        }
        let pipelines = make::remake_where(|p, _| p.shaders().iter().any(|s| forgotten.contains(&Arc::as_ptr(s))));
        (forgotten.len(), pipelines)
    }

    fn summary((shaders, pipelines): (usize, usize)) -> String {
        format!("{shaders} shaders to translate again, {pipelines} host pipelines remade")
    }

    /// The translation setting `bit` (`victory_shader::emit::SETTING_*`)
    /// changed: shaders whose translation asked it are translated again.
    /// So are those being translated now (they may have read the old
    /// value): [`Shader::forget`] waits for them.
    pub(crate) fn retranslate_asking(bit: u8) -> String {
        summary(retranslate(|s| s.settings() & bit != 0 || s.untranslated()))
    }

    /// `VICTORY_PIPELINE_WAIT_MS` from now on: the per-frame pipeline wait
    /// in milliseconds.
    pub(crate) fn set_pipeline_wait(value: Option<&str>) -> u64 {
        make::set_budget(value)
    }

    /// `VICTORY_PATCH_CODE` from now on: shaders whose patches changed are
    /// translated again.
    pub(crate) fn set_patch_code(value: Option<&str>) -> String {
        let new = parse_patches(value.unwrap_or_default());
        // Read from the environment first if no translation has yet.
        code_patches("");
        let old = PATCHES.set(new.clone()).unwrap_or_default();
        let entries = |all: &Patches, h: &str| -> Vec<(usize, u32, u32)> {
            all.iter().filter(|(x, ..)| x == h).map(|&(_, a, o, n)| (a, o, n)).collect()
        };
        let mut changed: Vec<String> = old.iter().chain(&new).map(|(h, ..)| h.clone()).collect();
        changed.sort();
        changed.dedup();
        changed.retain(|h| entries(&old, h) != entries(&new, h));
        format!("{} patch entries; {}", new.len(), summary(retranslate(|s| changed.contains(&s.hash))))
    }

    /// `VICTORY_BVH_SKINNING` from now on: the BVH skinning shaders are
    /// translated again (or refused again).
    pub(crate) fn set_bvh_skinning(value: Option<&str>) -> String {
        BVH_RUN.set(value == Some("1"));
        summary(retranslate(|s| BVH_SKINNING_SHADERS.contains(&s.hash.as_str())))
    }

    /// `VICTORY_DEBUG_PS` (`fallback` false) or `VICTORY_PS_FALLBACK`
    /// changed: remakes the geometry pipelines whose pixel shader that
    /// changes (all of them; with `fallback`, those refused or drawing with
    /// the debug pixel shader though it is off).
    pub(crate) fn remake_for_debug_ps(fallback: bool) -> String {
        let debug = host::debug_ps();
        let n = make::remake_where(|p, made| {
            host::draws_geometry(p.shaders(), p.graphics.as_deref())
                && (!fallback
                    || match made {
                        Some(None) => true,
                        Some(Some(h)) => h.debug_ps != debug,
                        None => false,
                    })
        });
        format!("{n} host pipelines remade")
    }
}

/// Keeps compiled host pipelines, and the list of title pipelines to start
/// early, in `dir` (a folder of the cache) across runs.
#[cfg(windows)]
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    library::use_pipeline_cache(dir);
}

/// Saves the pipeline cache now if anything was added (at the end of a run).
#[cfg(windows)]
pub fn save_pipeline_cache() {
    library::save(true);
}

#[cfg(test)]
mod tests {
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn skip_hashes_accepts_full_hashes_only() {
        assert!(super::parse_skip_hashes(" , ").is_empty());
        assert_eq!(
            super::parse_skip_hashes(" 0123456789ABCDEF, 0123456789abcde, xyz, 0011223344556677,"),
            ["0123456789abcdef", "0011223344556677"]
        );
    }

    use super::*;
    use crate::object::{S_OK, slot};

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders
            .iter()
            .map(|s| (s.stage, s.container.clone()))
            .collect()
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode {
            code: std::ptr::null(),
            size: 0,
        };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode {
                    code: vs.as_ptr(),
                    size: vs.len(),
                },
                Bytecode {
                    code: ps.as_ptr(),
                    size: ps.len(),
                },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        assert_eq!(
            create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
            S_OK
        );
        assert_eq!(
            shaders(p),
            [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]
        );
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert!(Arc::ptr_eq(
            &pipeline(p).shaders[1],
            &pipeline(q).shaders[1]
        ));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
    }

    /// The stream layout Grand Theft Auto VI builds (title+0x86f090): root
    /// signature, five shader slots (the first a mesh shader), blend, sample
    /// mask, rasterizer, depth-stencil 1, sample description, input layout,
    /// topology, render target and depth formats: 0x2b0 bytes.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream = vec![0u8; 0x2b0];
        let mut put = |at: usize, v: u64| stream[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for (at, kind) in [
            (0x10, 24),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            put(at, kind);
        }
        put(0x18, ms.as_ptr() as u64);
        put(0x20, ms.len() as u64);
        put(0x30, ps.as_ptr() as u64);
        put(0x38, ps.len() as u64);
        // Topology type 3 (triangles), one render target of format 28.
        put(0x278, 14 | 3 << 32);
        put(0x280, 15 | 28 << 32);
        put(0x2a0, 1 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("mesh", ms.to_vec()), ("pixel", ps.to_vec())]);
        #[cfg(windows)]
        {
            let g = pipeline(p).host.graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 3);
            assert_eq!(
                (g.targets.NumRenderTargets, g.targets.RTFormats[0].0),
                (1, 28)
            );
        }
    }
}

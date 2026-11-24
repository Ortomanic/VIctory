//! Wave64 console programs run on the host GPU in the forms a 32-lane host
//! gets (`docs/wave64-on-wave32.md`): the results must be the console's.
//! Run under Wine with vkd3d-proton on NVIDIA (32-lane waves) these test
//! the split and uniform forms themselves; hosts with 64-lane waves skip.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// The world's instance culling append (InstanceCulling CS_Prop_BatchCull,
/// 17af15c4015cb4b6 at 0x27d0..0x28f4, words copied): a whole-wave sum of
/// the surviving lanes' counts (`s_or_saveexec -1`, DPP8, row_mirror,
/// permlanex16, `v_readlane` 32 and 0), one atomic add by the first
/// surviving lane (`s_ff1` of EXEC against `v_mbcnt`), its result taken by
/// `v_readfirstlane`, then each lane's exclusive prefix (row_shr scan,
/// permlanex16 `fi`, `v_readlane` 31 added under an upper-half EXEC).
/// Group g's survivors are the lanes from 8g on; it writes, at dword 72g,
/// the atomic's sum, at 72g+1+lane each survivor's slot (base + prefix),
/// at 72g+65 the wave sum and at 72g+66 the base. Non-elected lanes hold
/// 1000 where the atomic's result goes, so a base from another lane shows.
const APPEND: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0402,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[4:5], 0x0
    0x8F03_8302, // s_lshl_b32 s3, s2, 3
    0x9306_FF02,
    0x0000_0120, // s_mul_i32 s6, s2, 0x120
    0xBEBE_0383, // s_mov_b32 s62, 3
    0x3420_0082, // v_lshlrev_b32_e32 v16, 2, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7DA6_0003, // v_cmpx_le_u32_e32 s3, v0
    0xBF88_004E, // s_cbranch_execz END (+78)
    0x7E1C_023E, // v_mov_b32_e32 v14, s62
    0xBEAA_25C1, // s_or_saveexec_b64 s[42:43], -1
    0xD501_0001,
    0x00AA_1C80, // v_cndmask_b32_e64 v1, 0, v14, s[42:43]
    0x4A02_02E9,
    0xDE54_C101, // v_add_nc_u32_dpp v1, v1, v1 dpp8:[1,0,3,2,5,4,7,6]
    0x4A02_02E9,
    0x9360_1201, // v_add_nc_u32_dpp v1, v1, v1 dpp8:[2,2,0,0,6,6,4,4]
    0x4A02_02E9,
    0x0009_2401, // v_add_nc_u32_dpp v1, v1, v1 dpp8:[4,4,4,4,0,0,0,0]
    0x4A02_02FA,
    0xFF09_4001, // v_add_nc_u32_dpp v1, v1, v1 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
    0xD778_0002,
    0x0201_0101, // v_permlanex16_b32 v2, v1, 0, 0
    0x4A02_0302, // v_add_nc_u32_e32 v1, v2, v1
    0xD760_0031,
    0x0201_4101, // v_readlane_b32 s49, v1, 32
    0x4A02_0231, // v_add_nc_u32_e32 v1, s49, v1
    0xD760_0031,
    0x0201_0101, // v_readlane_b32 s49, v1, 0
    0xBEFE_042A, // s_mov_b64 exec, s[42:43]
    0xD765_000E,
    0x0201_00C1, // v_mbcnt_lo_u32_b32 v14, -1, 0
    0xBEB2_147E, // s_ff1_i32_b64 s50, exec
    0xBEAA_047E, // s_mov_b64 s[42:43], exec
    0xD766_000F,
    0x0202_1CC1, // v_mbcnt_hi_u32_b32 v15, -1, v14
    0x7E1C_02FF,
    0x0000_03E8, // v_mov_b32_e32 v14, 0x3e8
    0x7D24_1E32, // v_cmpx_eq_i32_e32 s50, v15
    0xBF88_0005, // s_cbranch_execz RESTORE (+5)
    0x7E1C_0231, // v_mov_b32_e32 v14, s49
    0xBF80_0000, // s_nop 0
    0xBF80_0000, // s_nop 0
    0xE0C8_4000,
    0x0604_0E00, // buffer_atomic_add v14, off, s[16:19], s6 glc
    // RESTORE:
    0xBEFE_042A, // s_mov_b64 exec, s[42:43]
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E64_050E, // v_readfirstlane_b32 s50, v14
    0x7E1E_023E, // v_mov_b32_e32 v15, s62
    0xBEAA_25C1, // s_or_saveexec_b64 s[42:43], -1
    0xD501_0001,
    0x00AA_1E80, // v_cndmask_b32_e64 v1, 0, v15, s[42:43]
    0x7E04_0301, // v_mov_b32_e32 v2, v1
    0x4A04_04FA,
    0xFF09_1102, // v_add_nc_u32_dpp v2, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
    0x4A04_04FA,
    0xFF01_1202, // v_add_nc_u32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf
    0x4A04_04FA,
    0xFF01_1402, // v_add_nc_u32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf
    0x4A04_04FA,
    0xFF01_1802, // v_add_nc_u32_dpp v2, v2, v2 row_shr:8 row_mask:0xf bank_mask:0xf
    0xBEFE_03FF,
    0xFFFF_0000, // s_mov_b32 exec_lo, 0xffff0000
    0xBEFF_03FF,
    0xFFFF_0000, // s_mov_b32 exec_hi, 0xffff0000
    0xD778_0803,
    0x0305_8302, // v_permlanex16_b32 v3, v2, -1, -1 op_sel:[1,0]
    0x4A04_0503, // v_add_nc_u32_e32 v2, v3, v2
    0xD760_0030,
    0x0201_3F02, // v_readlane_b32 s48, v2, 31
    0xBEFE_0380, // s_mov_b32 exec_lo, 0
    0xBEFF_03C1, // s_mov_b32 exec_hi, -1
    0x4A04_0430, // v_add_nc_u32_e32 v2, s48, v2
    0xBEFE_03C1, // s_mov_b32 exec_lo, -1
    0xBEFF_03C1, // s_mov_b32 exec_hi, -1
    0x4C04_0302, // v_sub_nc_u32_e32 v2, v2, v1
    0xBEFE_042A, // s_mov_b64 exec, s[42:43]
    0x4A1C_0432, // v_add_nc_u32_e32 v14, s50, v2
    0xE070_1004,
    0x0604_0E10, // buffer_store_dword v14, v16, s[16:19], s6 offen offset:4
    0x7E22_0231, // v_mov_b32_e32 v17, s49
    0x7E24_0232, // v_mov_b32_e32 v18, s50
    0x7E26_0280, // v_mov_b32_e32 v19, 0
    0xE070_1104,
    0x0604_1113, // buffer_store_dword v17, v19, s[16:19], s6 offen offset:260
    0xE070_1108,
    0x0604_1213, // buffer_store_dword v18, v19, s[16:19], s6 offen offset:264
    // END:
    0xBF81_0000, // s_endpgm
];

/// [`APPEND`] twice in a scalar loop (as in CS_Prop_BatchCull, which
/// appends per batch in a loop): no block runs once, so the split form
/// refuses and the uniform form runs it. The second pass's base is the
/// first's sum.
const APPEND_LOOPED: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0402,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[4:5], 0x0
    0x8F03_8302, // s_lshl_b32 s3, s2, 3
    0x9306_FF02,
    0x0000_0120, // s_mul_i32 s6, s2, 0x120
    0xBEBE_0383, // s_mov_b32 s62, 3
    0x3420_0082, // v_lshlrev_b32_e32 v16, 2, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7DA6_0003, // v_cmpx_le_u32_e32 s3, v0
    0xBF88_0052, // s_cbranch_execz END (+82)
    0xBE87_0380, // s_mov_b32 s7, 0
    // LOOP:
    0x7E1C_023E, // v_mov_b32_e32 v14, s62
    0xBEAA_25C1, // s_or_saveexec_b64 s[42:43], -1
    0xD501_0001,
    0x00AA_1C80, // v_cndmask_b32_e64 v1, 0, v14, s[42:43]
    0x4A02_02E9,
    0xDE54_C101, // v_add_nc_u32_dpp v1, v1, v1 dpp8:[1,0,3,2,5,4,7,6]
    0x4A02_02E9,
    0x9360_1201, // v_add_nc_u32_dpp v1, v1, v1 dpp8:[2,2,0,0,6,6,4,4]
    0x4A02_02E9,
    0x0009_2401, // v_add_nc_u32_dpp v1, v1, v1 dpp8:[4,4,4,4,0,0,0,0]
    0x4A02_02FA,
    0xFF09_4001, // v_add_nc_u32_dpp v1, v1, v1 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
    0xD778_0002,
    0x0201_0101, // v_permlanex16_b32 v2, v1, 0, 0
    0x4A02_0302, // v_add_nc_u32_e32 v1, v2, v1
    0xD760_0031,
    0x0201_4101, // v_readlane_b32 s49, v1, 32
    0x4A02_0231, // v_add_nc_u32_e32 v1, s49, v1
    0xD760_0031,
    0x0201_0101, // v_readlane_b32 s49, v1, 0
    0xBEFE_042A, // s_mov_b64 exec, s[42:43]
    0xD765_000E,
    0x0201_00C1, // v_mbcnt_lo_u32_b32 v14, -1, 0
    0xBEB2_147E, // s_ff1_i32_b64 s50, exec
    0xBEAA_047E, // s_mov_b64 s[42:43], exec
    0xD766_000F,
    0x0202_1CC1, // v_mbcnt_hi_u32_b32 v15, -1, v14
    0x7E1C_02FF,
    0x0000_03E8, // v_mov_b32_e32 v14, 0x3e8
    0x7D24_1E32, // v_cmpx_eq_i32_e32 s50, v15
    0xBF88_0005, // s_cbranch_execz RESTORE (+5)
    0x7E1C_0231, // v_mov_b32_e32 v14, s49
    0xBF80_0000, // s_nop 0
    0xBF80_0000, // s_nop 0
    0xE0C8_4000,
    0x0604_0E00, // buffer_atomic_add v14, off, s[16:19], s6 glc
    // RESTORE:
    0xBEFE_042A, // s_mov_b64 exec, s[42:43]
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E64_050E, // v_readfirstlane_b32 s50, v14
    0x7E1E_023E, // v_mov_b32_e32 v15, s62
    0xBEAA_25C1, // s_or_saveexec_b64 s[42:43], -1
    0xD501_0001,
    0x00AA_1E80, // v_cndmask_b32_e64 v1, 0, v15, s[42:43]
    0x7E04_0301, // v_mov_b32_e32 v2, v1
    0x4A04_04FA,
    0xFF09_1102, // v_add_nc_u32_dpp v2, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
    0x4A04_04FA,
    0xFF01_1202, // v_add_nc_u32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf
    0x4A04_04FA,
    0xFF01_1402, // v_add_nc_u32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf
    0x4A04_04FA,
    0xFF01_1802, // v_add_nc_u32_dpp v2, v2, v2 row_shr:8 row_mask:0xf bank_mask:0xf
    0xBEFE_03FF,
    0xFFFF_0000, // s_mov_b32 exec_lo, 0xffff0000
    0xBEFF_03FF,
    0xFFFF_0000, // s_mov_b32 exec_hi, 0xffff0000
    0xD778_0803,
    0x0305_8302, // v_permlanex16_b32 v3, v2, -1, -1 op_sel:[1,0]
    0x4A04_0503, // v_add_nc_u32_e32 v2, v3, v2
    0xD760_0030,
    0x0201_3F02, // v_readlane_b32 s48, v2, 31
    0xBEFE_0380, // s_mov_b32 exec_lo, 0
    0xBEFF_03C1, // s_mov_b32 exec_hi, -1
    0x4A04_0430, // v_add_nc_u32_e32 v2, s48, v2
    0xBEFE_03C1, // s_mov_b32 exec_lo, -1
    0xBEFF_03C1, // s_mov_b32 exec_hi, -1
    0x4C04_0302, // v_sub_nc_u32_e32 v2, v2, v1
    0xBEFE_042A, // s_mov_b64 exec, s[42:43]
    0x4A1C_0432, // v_add_nc_u32_e32 v14, s50, v2
    0xE070_1004,
    0x0604_0E10, // buffer_store_dword v14, v16, s[16:19], s6 offen offset:4
    0x7E22_0231, // v_mov_b32_e32 v17, s49
    0x7E24_0232, // v_mov_b32_e32 v18, s50
    0x7E26_0280, // v_mov_b32_e32 v19, 0
    0xE070_1104,
    0x0604_1113, // buffer_store_dword v17, v19, s[16:19], s6 offen offset:260
    0xE070_1108,
    0x0604_1213, // buffer_store_dword v18, v19, s[16:19], s6 offen offset:264
    0x8107_8107, // s_add_i32 s7, s7, 1
    0xBF0A_8207, // s_cmp_lt_u32 s7, 2
    0xBF85_FFAF, // s_cbranch_scc1 LOOP (-81)
    // END:
    0xBF81_0000, // s_endpgm
];

fn build(words: &[u32], threads: [u32; 3], host_wave32: bool) -> (HostShader, lower::Lowered) {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        // Two user SGPRs, then the group's x id in s2.
        io: StageIo::Cs(ComputeIo {
            rsrc2: (2 << 1) | (1 << 7),
            threads,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track_lanes(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        host_wave32,
        root_bounds: true,
        heap_views: crate::emit::HEAP_VIEWS,
    })
    .expect("lower");
    if let Some(dir) = std::env::var_os("VICTORY_WAVE64_DUMP") {
        let _ = std::fs::write(
            std::path::Path::new(&dir).join("wave64_append.dxil"),
            &lowered.dxil,
        );
    }
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil.clone(),
        user_data: lowered.user_data.clone(),
        samplers: lowered.samplers.clone(),
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs.clone(),
        root_srvs: lowered.root_srvs.clone(),
        root_uavs: lowered.root_uavs.clone(),
        bindless: 0,
        zero_filled_phis: false,
    };
    (shader, lowered)
}

/// Runs `shader` in `groups` groups on the hardware adapter with a zeroed
/// raw UAV of `dwords` at heap index 0; `None` without a device.
fn run(shader: &HostShader, groups: u32, dwords: u32) -> Option<Vec<u32>> {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return None;
    };
    let (root, pso) = host
        .compute_pipeline(shader)
        .unwrap_or_else(|e| panic!("compute_pipeline failed: {e}"));
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let views = [View::RawUav(dwords)];
    let words = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &views,
                samplers: &[],
                groups: [groups, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(words[0].clone())
}

/// What the console writes for [`APPEND`] run `passes` times in `groups`
/// groups.
fn append_expected(groups: u32, passes: u32) -> Vec<u32> {
    let mut want = vec![0u32; 72 * groups as usize];
    for g in 0..groups as usize {
        let first = 8 * g;
        let count = 64usize.saturating_sub(first) as u32;
        if count == 0 {
            continue;
        }
        let base = 3 * count * (passes - 1);
        want[72 * g] = 3 * count * passes;
        for lane in first..64 {
            want[72 * g + 1 + lane] = base + 3 * (lane - first) as u32;
        }
        want[72 * g + 65] = 3 * count;
        want[72 * g + 66] = base;
    }
    want
}

fn check(words: &[u32], passes: u32, uniform: bool) {
    let (shader, l) = build(words, [64, 1, 1], true);
    assert_eq!((l.wave_size, l.split, l.uniform), (Some(32), true, uniform));
    let groups = 9;
    let Some(got) = run(&shader, groups, 72 * groups) else {
        return;
    };
    let want = append_expected(groups, passes);
    for g in 0..groups as usize {
        let (a, b) = (&got[72 * g..72 * g + 72], &want[72 * g..72 * g + 72]);
        if a != b {
            eprintln!("group {g} (survivors from lane {}):", 8 * g);
            eprintln!(
                "  got  sum {} wave {} base {} slots {:?}",
                a[0],
                a[65],
                a[66],
                &a[1..65]
            );
            eprintln!(
                "  want sum {} wave {} base {} slots {:?}",
                b[0],
                b[65],
                b[66],
                &b[1..65]
            );
        }
    }
    assert_eq!(got, want);
}

/// The append once (uniform form too: its EXEC region is entered by
/// some lanes only).
#[test]
fn the_culling_append_counts_and_places_every_survivor() {
    check(APPEND, 1, true);
}

/// The append in a loop, in the uniform form.
#[test]
fn the_looped_culling_append_counts_and_places_every_survivor() {
    check(APPEND_LOOPED, 2, true);
}

from visualizer.bundle import blocks_per_sm, family, idle_ns, label_streams, split_frames

THOR = {
    "sms": 20,
    "max_warps_per_sm": 48,
    "max_registers_per_sm": 65536,
    "max_shmem_per_sm": 233472,
    "max_blocks_per_sm": 32,
    "threads_per_warp": 32,
}


def kernel(start, end, name="k", stream=1):
    # (start, end, short, full, stream, graph, blocks, threads, regs, smem)
    return (start, end, name, name, stream, None, 1, 32, 32, 0)


def test_kernel_families_follow_the_name():
    assert family("flash_attn_tile") == "attention"
    assert family("soft_max_f32") == "attention"
    assert family("mul_mat_q") == "matmul"
    assert family("cutlass3x_sm100_tensorop_h256x") == "matmul"
    assert family("rms_norm_mul_rope_f32") == "normalization"
    assert family("convert_unary_cont_vec4") == "copy / convert"
    assert family("quantize_q8_1") == "quantize"
    assert family("k_bin_bcast") == "elementwise"
    assert family("something_new") == "other"


def test_blocks_per_sm_takes_the_tightest_limit():
    # flash_attn_tile on the Thor: 256 threads, 106 registers, 19 KB shared memory.
    # Registers allow 2 blocks (8 warps x 3584 registers each), the tightest limit.
    assert blocks_per_sm(THOR, threads=256, regs=106, smem=19168) == 2
    # A tiny kernel is limited by the 32-blocks-per-SM cap.
    assert blocks_per_sm(THOR, threads=32, regs=16, smem=0) == 32


def test_frames_split_on_long_gaps_and_idle_counts_only_when_nothing_runs():
    ms = 1_000_000
    kernels = [kernel(0, 10 * ms), kernel(5 * ms, 12 * ms, stream=2), kernel(20 * ms, 30 * ms),
               kernel(2000 * ms, 2010 * ms)]  # fmt: skip
    frames = split_frames(kernels)
    assert [len(f) for f in frames] == [3, 1]
    # 0-12 ms busy (two streams overlap), 12-20 ms idle, 20-30 ms busy.
    assert idle_ns(frames[0]) == 8 * ms


def test_the_stream_that_embeds_patches_is_the_vision_encoder():
    kernels = [kernel(0, 1, "mul_mat_q", stream=17), kernel(2, 3, "im2col_kernel", stream=18)]
    assert label_streams(kernels) == {17: "language model", 18: "vision encoder"}

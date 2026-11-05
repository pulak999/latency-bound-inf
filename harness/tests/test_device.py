from harness.device import DeviceSample, read_device_samples


def test_old_runs_without_clock_columns_still_load(tmp_path):
    (tmp_path / "power.csv").write_text("t,gpu_w,board_w\n1.0,20.0,40.0\n")
    assert read_device_samples(tmp_path) == [DeviceSample(t=1.0, gpu_w=20.0, board_w=40.0)]


def test_new_runs_load_clocks(tmp_path):
    (tmp_path / "device.csv").write_text(
        "t,gpu_w,board_w,gpu_mhz,memory_mhz\n1.0,20.0,40.0,1386.0,4266.0\n"
    )
    (sample,) = read_device_samples(tmp_path)
    assert sample.gpu_mhz == 1386.0 and sample.memory_mhz == 4266.0

import numpy as np

from harness.synthetic import synthetic_observation

SMOLVLA_CAMERAS = {
    "observation.images.camera1": (3, 256, 256),
    "observation.images.camera2": (3, 256, 256),
    "observation.images.camera3": (3, 256, 256),
}


def test_observation_matches_the_policy_config():
    observation = synthetic_observation(0, SMOLVLA_CAMERAS, state_dim=6)

    assert set(observation) == set(SMOLVLA_CAMERAS) | {"observation.state"}
    for key in SMOLVLA_CAMERAS:
        # Config shapes are (channels, height, width); a camera produces (height, width, channels).
        assert observation[key].shape == (256, 256, 3)
        assert observation[key].dtype == np.uint8
    assert observation["observation.state"].shape == (6,)
    assert observation["observation.state"].dtype == np.float32


def test_non_square_cameras_keep_height_and_width_apart():
    observation = synthetic_observation(0, {"observation.images.wide": (3, 224, 320)}, state_dim=8)
    assert observation["observation.images.wide"].shape == (224, 320, 3)


def test_observations_are_reproducible_and_distinct():
    a = synthetic_observation(1, SMOLVLA_CAMERAS, state_dim=6)
    b = synthetic_observation(1, SMOLVLA_CAMERAS, state_dim=6)
    c = synthetic_observation(2, SMOLVLA_CAMERAS, state_dim=6)
    assert all(np.array_equal(a[key], b[key]) for key in a)
    assert not np.array_equal(a["observation.images.camera1"], c["observation.images.camera1"])

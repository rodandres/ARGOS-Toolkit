import json
from pathlib import Path

import numpy as np
import pytest

from argos.general.data_save import (
    SimulationHistory,
    SimulationHistoryMetadata,
    SpacecraftHistory,
    TransitionEventInfo,
    save_simulation_history_csv,
)
from argos.general.dataclasses import (
    ControlOutput,
    GuidanceOutput,
    NavigationOutput,
    SpacecraftData,
    StateVariables,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def test_simulation_history_metadata_stores_values():
    metadata = SimulationHistoryMetadata(
        data_file_path="sim_data/test",
        unique_id=123,
        spacecraft_names=["Chaser", "Target"],
        chunk_size={"Chaser": 100},
        chunk_count={"Chaser": 2},
    )

    assert metadata.data_file_path == "sim_data/test"
    assert metadata.unique_id == 123
    assert metadata.spacecraft_names == ["Chaser", "Target"]
    assert metadata.chunk_size == {"Chaser": 100}
    assert metadata.chunk_count == {"Chaser": 2}
    assert metadata.csv_folder_path == ""
    assert metadata.targets_info is None


def test_transition_event_info_stores_values():
    event = TransitionEventInfo(
        time=10.0,
        tick=100,
        from_phase="phase_1",
        to_phase="phase_2",
        transition_name="test_transition",
        dt_guidance=0.1,
        dt_navigation=0.2,
        dt_control=0.3,
        dt_propagation=0.4,
        dt_master=0.5,
    )

    assert event.time == 10.0
    assert event.tick == 100
    assert event.from_phase == "phase_1"
    assert event.to_phase == "phase_2"
    assert event.transition_name == "test_transition"
    assert event.dt_guidance == 0.1
    assert event.dt_navigation == 0.2
    assert event.dt_control == 0.3
    assert event.dt_propagation == 0.4
    assert event.dt_master == 0.5


# ---------------------------------------------------------------------------
# SimulationHistory
# ---------------------------------------------------------------------------


def test_simulation_history_initializes(tmp_path):
    history = SimulationHistory(
        chunk_size=10,
        data_file_path=str(tmp_path),
        auto_convert_to_csv=False,
    )

    assert history.chunk_size == 10
    assert history.auto_convert_to_csv is False
    assert history.csv_folder_path == "sim_data"
    assert history.spacecraft_names == []
    assert history.spacecrafts_history == {}
    assert isinstance(history.unique_id, int)

    assert history.data_file_path.startswith(str(tmp_path))
    assert history.data_file_path.endswith(
        f"simulation_{history.unique_id}"
    )


def test_simulation_history_creates_directory(tmp_path):
    history = SimulationHistory(
        chunk_size=10,
        data_file_path=str(tmp_path),
        auto_convert_to_csv=False,
    )

    assert Path(history.data_file_path).exists()


def test_add_spacecraft_history(tmp_path):
    history = SimulationHistory(
        chunk_size=10,
        data_file_path=str(tmp_path),
        auto_convert_to_csv=False,
    )

    history.add_spacecraft_history("Chaser")

    assert history.spacecraft_names == ["Chaser"]
    assert "Chaser" in history.spacecrafts_history

    spacecraft_history = history.spacecrafts_history["Chaser"]

    assert isinstance(spacecraft_history, SpacecraftHistory)
    assert spacecraft_history.spacecraft_name == "Chaser"
    assert spacecraft_history.chunk_size == 10


def test_add_duplicate_spacecraft_raises_error(tmp_path):
    history = SimulationHistory(
        chunk_size=10,
        data_file_path=str(tmp_path),
        auto_convert_to_csv=False,
    )

    history.add_spacecraft_history("Chaser")

    with pytest.raises(ValueError):
        history.add_spacecraft_history("Chaser")


def test_record_spacecraft_without_history_raises_error(tmp_path):
    history = SimulationHistory(
        chunk_size=10,
        data_file_path=str(tmp_path),
        auto_convert_to_csv=False,
    )

    data = SpacecraftData()

    with pytest.raises(ValueError):
        history.record_spacecraft("Chaser", data)


def test_record_event_without_history_raises_error(tmp_path):
    history = SimulationHistory(
        chunk_size=10,
        data_file_path=str(tmp_path),
        auto_convert_to_csv=False,
    )

    event = TransitionEventInfo(
        time=1.0,
        tick=1,
        from_phase="phase_1",
        to_phase="phase_2",
        transition_name="transition",
        dt_guidance=0.1,
        dt_navigation=0.1,
        dt_control=0.1,
        dt_propagation=0.1,
        dt_master=0.1,
    )

    with pytest.raises(ValueError):
        history.record_event("Chaser", event)


def test_record_target_without_history_raises_error(tmp_path):
    history = SimulationHistory(
        chunk_size=10,
        data_file_path=str(tmp_path),
        auto_convert_to_csv=False,
    )

    with pytest.raises(ValueError):
        history.record_target("Chaser", "Target")


# ---------------------------------------------------------------------------
# SpacecraftHistory
# ---------------------------------------------------------------------------


def test_spacecraft_history_initializes(tmp_path):
    history = SpacecraftHistory(
        spacecraft_name="Chaser",
        chunk_size=5,
        data_file_path=str(tmp_path),
    )

    assert history.spacecraft_name == "Chaser"
    assert history.chunk_size == 5
    assert history.chunk_index == 0
    assert history.index == 0
    assert history.targets == []
    assert history.events == []

    assert history.t.shape == (5,)
    assert history.tick.shape == (5,)
    assert history.true_position.shape == (5, 3)
    assert history.true_velocity.shape == (5, 3)
    assert history.true_acceleration.shape == (5, 3)
    assert history.true_attitude.shape == (5, 4)
    assert history.true_angular_velocity.shape == (5, 3)
    assert history.true_angular_acceleration.shape == (5, 3)


def test_spacecraft_history_flush_without_data_does_nothing(tmp_path):
    history = SpacecraftHistory(
        spacecraft_name="Chaser",
        chunk_size=5,
        data_file_path=str(tmp_path),
    )

    history.flush()

    files = list(tmp_path.glob("*.npz"))

    assert files == []
    assert history.chunk_index == 0
    assert history.index == 0


def test_record_spacecraft_data(tmp_path):
    history = SpacecraftHistory(
        spacecraft_name="Chaser",
        chunk_size=5,
        data_file_path=str(tmp_path),
    )

    data = SpacecraftData()
    data.t = 12.5
    data.tick = 125

    data.true_state.position = np.array([1.0, 2.0, 3.0])
    data.true_state.velocity = np.array([4.0, 5.0, 6.0])
    data.true_state.attitude = np.array([0.1, 0.2, 0.3, 0.4])

    data.navigation_data.spacecraft_state.position = np.array(
        [10.0, 20.0, 30.0]
    )

    data.guidance_data.state.position = np.array(
        [100.0, 200.0, 300.0]
    )

    data.control_data.force = np.array([7.0, 8.0, 9.0])
    data.current_force_exerted = np.array([11.0, 12.0, 13.0])

    history.record(data)

    assert history.index == 1

    assert history.t[0] == 12.5
    assert history.tick[0] == 125

    assert np.array_equal(
        history.true_position[0],
        [1.0, 2.0, 3.0],
    )

    assert np.array_equal(
        history.true_velocity[0],
        [4.0, 5.0, 6.0],
    )

    assert np.array_equal(
        history.true_attitude[0],
        [0.1, 0.2, 0.3, 0.4],
    )

    assert np.array_equal(
        history.estimated_sc_position[0],
        [10.0, 20.0, 30.0],
    )

    assert np.array_equal(
        history.guidance_position[0],
        [100.0, 200.0, 300.0],
    )

    assert np.array_equal(
        history.control_force[0],
        [7.0, 8.0, 9.0],
    )

    assert np.array_equal(
        history.current_force_exerted[0],
        [11.0, 12.0, 13.0],
    )


def test_record_flushes_when_chunk_is_full(tmp_path):
    history = SpacecraftHistory(
        spacecraft_name="Chaser",
        chunk_size=2,
        data_file_path=str(tmp_path),
    )

    data = SpacecraftData()

    data.t = 1.0
    history.record(data)

    assert history.index == 1
    assert history.chunk_index == 0

    data.t = 2.0
    history.record(data)

    assert history.index == 0
    assert history.chunk_index == 1

    chunk_file = (
        tmp_path / "Chaser_history_chunk_0.npz"
    )

    assert chunk_file.exists()


def test_flush_saves_recorded_data(tmp_path):
    history = SpacecraftHistory(
        spacecraft_name="Chaser",
        chunk_size=5,
        data_file_path=str(tmp_path),
    )

    data = SpacecraftData()
    data.t = 42.0
    data.tick = 420
    data.true_state.position = np.array([1.0, 2.0, 3.0])

    history.record(data)
    history.flush()

    chunk_file = (
        tmp_path / "Chaser_history_chunk_0.npz"
    )

    assert chunk_file.exists()

    saved_data = np.load(chunk_file)

    assert saved_data["t"][0] == 42.0
    assert saved_data["tick"][0] == 420
    assert np.array_equal(
        saved_data["true_position"][0],
        [1.0, 2.0, 3.0],
    )


# ---------------------------------------------------------------------------
# Events and targets
# ---------------------------------------------------------------------------


def test_record_event(tmp_path):
    history = SpacecraftHistory(
        spacecraft_name="Chaser",
        chunk_size=5,
        data_file_path=str(tmp_path),
    )

    event = TransitionEventInfo(
        time=5.0,
        tick=50,
        from_phase="phase_1",
        to_phase="phase_2",
        transition_name="test",
        dt_guidance=0.1,
        dt_navigation=0.2,
        dt_control=0.3,
        dt_propagation=0.4,
        dt_master=0.5,
    )

    history.record_event(event)

    assert len(history.events) == 1
    assert history.events[0] == event


def test_record_target_does_not_duplicate_targets(tmp_path):
    history = SpacecraftHistory(
        spacecraft_name="Chaser",
        chunk_size=5,
        data_file_path=str(tmp_path),
    )

    history.record_target("Target")
    history.record_target("Target")
    history.record_target("Target_2")

    assert history.targets == ["Target", "Target_2"]


# ---------------------------------------------------------------------------
# SimulationHistory.finalize
# ---------------------------------------------------------------------------


def test_finalize_writes_metadata(tmp_path):
    history = SimulationHistory(
        chunk_size=5,
        data_file_path=str(tmp_path),
        auto_convert_to_csv=False,
    )

    history.add_spacecraft_history("Chaser")

    data = SpacecraftData()
    data.t = 10.0

    history.record_spacecraft("Chaser", data)
    history.record_target("Chaser", "Target")

    metadata = history.finalize()

    metadata_file = (
        Path(history.data_file_path) / "simulation_history_metadata.json"
    )

    assert metadata_file.exists()
    assert isinstance(metadata, SimulationHistoryMetadata)

    with open(metadata_file, "r") as file:
        saved_metadata = json.load(file)

    assert saved_metadata["unique_id"] == history.unique_id
    assert saved_metadata["spacecraft_names"] == ["Chaser"]
    assert saved_metadata["targets_info"]["Chaser"] == ["Target"]


def test_finalize_flushes_remaining_data(tmp_path):
    history = SimulationHistory(
        chunk_size=5,
        data_file_path=str(tmp_path),
        auto_convert_to_csv=False,
    )

    history.add_spacecraft_history("Chaser")

    data = SpacecraftData()
    data.t = 10.0

    history.record_spacecraft("Chaser", data)
    history.finalize()

    chunk_file = (
        Path(history.data_file_path)
        / "Chaser_history_chunk_0.npz"
    )

    assert chunk_file.exists()


def test_finalize_saves_transition_events(tmp_path):
    history = SimulationHistory(
        chunk_size=5,
        data_file_path=str(tmp_path),
        auto_convert_to_csv=False,
    )

    history.add_spacecraft_history("Chaser")

    event = TransitionEventInfo(
        time=5.0,
        tick=50,
        from_phase="phase_1",
        to_phase="phase_2",
        transition_name="test",
        dt_guidance=0.1,
        dt_navigation=0.2,
        dt_control=0.3,
        dt_propagation=0.4,
        dt_master=0.5,
    )

    history.record_event("Chaser", event)
    history.finalize()

    event_file = (
        Path(history.data_file_path)
        / "Chaser_transition_events.json"
    )

    assert event_file.exists()

    with open(event_file, "r") as file:
        saved_events = json.load(file)

    assert len(saved_events) == 1
    assert saved_events[0]["time"] == 5.0
    assert saved_events[0]["transition_name"] == "test"


# ---------------------------------------------------------------------------
# CSV conversion
# ---------------------------------------------------------------------------

def test_save_simulation_history_csv(tmp_path):
    history = SimulationHistory(
        chunk_size=5,
        data_file_path=str(tmp_path),
        auto_convert_to_csv=False,
    )

    history.add_spacecraft_history("Chaser")

    data = SpacecraftData()
    data.t = 10.0
    data.tick = 100
    data.true_state.position = np.array([1.0, 2.0, 3.0])

    history.record_spacecraft("Chaser", data)

    history.finalize()

    csv_directory = Path(
        save_simulation_history_csv(
            str(
                Path(history.data_file_path)
                / "simulation_history_metadata.json"
            ),
            main_directory=str(tmp_path),
        )
    )

    csv_file = csv_directory / "Chaser.csv"

    assert csv_file.exists()

    csv_data = np.genfromtxt(
        csv_file,
        delimiter=",",
        names=True,
        ndmin=1,
    )

    assert np.isclose(csv_data["t"][0], 10.0)
    assert np.isclose(csv_data["tick"][0], 100.0)
    assert np.isclose(csv_data["true_position_0"][0], 1.0)
    assert np.isclose(csv_data["true_position_1"][0], 2.0)
    assert np.isclose(csv_data["true_position_2"][0], 3.0)


def test_save_simulation_history_csv_missing_chunk_raises_error(
    tmp_path,
):
    history = SimulationHistory(
        chunk_size=5,
        data_file_path=str(tmp_path),
        auto_convert_to_csv=False,
    )

    history.add_spacecraft_history("Chaser")

    data = SpacecraftData()
    history.record_spacecraft("Chaser", data)
    history.finalize()

    chunk_file = (
        Path(history.data_file_path)
        / "Chaser_history_chunk_0.npz"
    )

    chunk_file.unlink()

    metadata_file = (
        Path(history.data_file_path)
        / "simulation_history_metadata.json"
    )

    with pytest.raises(FileNotFoundError):
        save_simulation_history_csv(
            str(metadata_file),
            main_directory=str(tmp_path),
        )
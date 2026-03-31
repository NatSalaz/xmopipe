import os
import json
import time
import datetime
import threading
import fcntl
import logging
from pathlib import Path
import contextlib
from contextlib import contextmanager
import tempfile
import functools


# Definition of Log class for logging
class Log:
    @staticmethod
    def info(message):
        logging.info(message)

    @staticmethod
    def warning(message):
        logging.warning(message)

    @staticmethod
    def error(message):
        logging.error(message)

    @staticmethod
    def debug(message):
        logging.debug(message)


# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to lock a file during access


def safe_write_json_atomic(path: Path, data: dict):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as tf:
        json.dump(data, tf, indent=2)
        tf.flush()
        os.fsync(tf.fileno())
        temp_path = Path(tf.name)
    os.replace(temp_path, path)  # Atomic on POSIX


@contextmanager
def locked_file(file_path, mode="r+", max_retries=100, initial_delay=0.1):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        with open(file_path, "w") as f:
            json.dump({"processed": {}, "in_progress": {}, "failed": {}, "heartbeats": {}}, f, indent=2)

    retry_count = 0
    delay = initial_delay

    while retry_count < max_retries:
        try:
            f = open(file_path, mode)
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield f
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            return
        except (IOError, BlockingIOError):
            time.sleep(delay)
            delay *= 1.5
            retry_count += 1
        except Exception:
            if not f.closed:
                f.close()
            raise

    raise IOError(f"Could not acquire lock on {file_path} after {max_retries} retries")


def load_tracking_data_safely(tracking_file):
    tracking_file = Path(tracking_file)
    tracking_file.parent.mkdir(parents=True, exist_ok=True)

    if not tracking_file.exists():
        with open(tracking_file, "w") as f:
            json.dump(
                {"processed": {}, "in_progress": {}, "failed": {}, "preprocessed": {}, "heartbeats": {}}, f, indent=2
            )
        return {"processed": {}, "in_progress": {}, "failed": {}, "preprocessed": {}, "heartbeats": {}}

    try:
        with open(tracking_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                Log.error(f"Corrupt tracking file {tracking_file}: {e}")
                f.seek(0)
                Log.error("Recovering with empty tracking data")
                data = {"processed": {}, "in_progress": {}, "failed": {}, "preprocessed": {}, "heartbeats": {}}

            # Ensure all required keys exist
            data.setdefault("processed", {})
            data.setdefault("in_progress", {})
            data.setdefault("failed", {})
            data.setdefault("preprocessed", {})
            data.setdefault("heartbeats", {})
            return data
    except (json.JSONDecodeError, IOError) as e:
        Log.error(f"Error loading tracking file {tracking_file}: {e}. Initializing new tracking data.")
        with open(tracking_file, "w") as f:
            data = {"processed": {}, "in_progress": {}, "failed": {}, "preprocessed": {}, "heartbeats": {}}
            json.dump(data, f, indent=2)
        return data


class VideoTracker:
    """
    Video tracker that maintains a separate tracking file for each subfolder.
    Works with a hierarchical folder structure:
    main_folder/
        subfolder_1/
            video1.mp4
            video2.mp4
            metadata.txt
        subfolder_2/
            video3.mp4
            metadata.txt
    """

    def __init__(self, main_folder_path, timeout_seconds=9000, heartbeat_interval=300, tracking_file="tracking"):
        self.main_folder_path = Path(main_folder_path).resolve()
        self.tracking_base_dir = self.main_folder_path / tracking_file
        self.tracking_base_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.heartbeat_interval = heartbeat_interval
        self._pid = os.getpid()
        self._heartbeat_thread = None
        self._stop_heartbeat = threading.Event()
        self._tracked_videos = set()

        # Register cleanup handler
        import atexit

        atexit.register(self._cleanup_on_exit)

        # Verify main folder structure
        if not self.main_folder_path.is_dir():
            raise ValueError(f"Main folder path {self.main_folder_path} is not a directory")

        # Log.info(f"VideoTracker initialized with main folder: {self.main_folder_path}")

    def _cleanup_on_exit(self):
        self.stop_heartbeat()
        self.clear_local_reservations()

    def get_subfolders(self):
        return [f for f in self.main_folder_path.iterdir() if f.is_dir() and not f.name.startswith(".")]

    def _get_tracking_file_for_subfolder(self, subfolder_path):
        subfolder_path = Path(subfolder_path).resolve()

        # Make sure it's a subfolder of the main folder
        if self.main_folder_path not in subfolder_path.parents and subfolder_path != self.main_folder_path:
            raise ValueError(f"Subfolder {subfolder_path} is not within main folder {self.main_folder_path}")

        # Generate a safe filename for the tracking file
        tracking_filename = f"tracking_{subfolder_path.name}.json"

        return self.tracking_base_dir / tracking_filename

    def _get_tracking_file_for_video(self, video_path):
        video_path = Path(video_path).resolve()
        subfolder_path = video_path.parent

        return self._get_tracking_file_for_subfolder(subfolder_path)

    def _get_all_tracking_files(self):
        return list(self.tracking_base_dir.glob("tracking_*.json"))

    def is_video_processed(self, video_path):
        tracking_file = self._get_tracking_file_for_video(video_path)
        data = load_tracking_data_safely(tracking_file)
        video_path_str = str(Path(video_path).resolve())
        return video_path_str in data["processed"]

    def is_video_failed(self, video_path):
        tracking_file = self._get_tracking_file_for_video(video_path)
        data = load_tracking_data_safely(tracking_file)
        video_path_str = str(Path(video_path).resolve())
        return video_path_str in data["failed"]

    def is_video_in_progress(self, video_path):
        tracking_file = self._get_tracking_file_for_video(video_path)
        data = load_tracking_data_safely(tracking_file)
        video_path_str = str(Path(video_path).resolve())
        return video_path_str in data["in_progress"]

    def is_subfolder_fully_processed(self, subfolder_path):
        subfolder_path = Path(subfolder_path).resolve()
        video_files = [f for f in subfolder_path.iterdir() if f.suffix.lower() in (".mp4", ".avi", ".mov")]

        if not video_files:
            return True

        tracking_file = self._get_tracking_file_for_subfolder(subfolder_path)

        data = load_tracking_data_safely(tracking_file)
        for file_path in video_files:
            if str(file_path) not in data["processed"]:
                return False
        return True

    def get_unprocessed_subfolders(self):
        unprocessed_subfolders = []

        for subfolder in self.get_subfolders():
            if not self.is_subfolder_fully_processed(subfolder):
                unprocessed_subfolders.append(subfolder)

        return unprocessed_subfolders

    def mark_video_in_progress(self, video_path):
        video_path_str = str(Path(video_path).resolve().as_posix())
        tracking_file = self._get_tracking_file_for_video(video_path)

        try:
            with locked_file(tracking_file, "r+") as f:
                f.seek(0)
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    Log.warning(f"Corrupted JSON detected in {tracking_file}.")
                    data = {"processed": {}, "in_progress": {}, "failed": {}, "heartbeats": {}}

                if video_path_str in data["in_progress"]:
                    Log.info(f"Video {video_path_str} already in progress.")
                    return False

                data["in_progress"][video_path_str] = {
                    "start_time": time.time(),
                    "pid": self._pid,
                    "date_started": datetime.datetime.now().isoformat(),
                }

                safe_write_json_atomic(tracking_file, data)

            self._tracked_videos.add(video_path_str)
            Log.info(f"Video {video_path_str} marked as in progress.")
            return True

        except Exception as e:
            Log.error(f"Failed to mark video in progress: {e}")
            return False

    def mark_video_processed(self, video_path, metadata=None):
        video_path_str = str(Path(video_path).resolve().as_posix())
        tracking_file = self._get_tracking_file_for_video(video_path)

        try:
            with locked_file(tracking_file, "r+") as f:
                f.seek(0)
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    Log.warning(f"Corrupted JSON detected in {tracking_file}, reinitializing.")
                    data = {"processed": {}, "in_progress": {}, "failed": {}, "heartbeats": {}}

                # Ajout dans 'processed'
                if os.path.exists(video_path_str):
                    timestamp = os.path.getmtime(video_path_str)
                else:
                    timestamp = time.time()

                data["processed"][video_path_str] = {
                    "processed_timestamp": timestamp,
                    "processing_time": metadata.get("processing_time") if metadata else None,
                    "tracks": metadata.get("tracks") if metadata else None,
                    "date_processed": metadata.get("date_processed", datetime.datetime.now().isoformat()),
                }

                # Suppression de 'in_progress'
                data["in_progress"].pop(video_path_str, None)

                # Écriture atomique sécurisée
                safe_write_json_atomic(tracking_file, data)

                # Nettoyage local
                if video_path_str in self._tracked_videos:
                    self._tracked_videos.remove(video_path_str)

                Log.info(f"Video treated : {video_path_str}")

                # Vérification du dossier
                subfolder_path = Path(video_path).parent
                if self.is_subfolder_fully_processed(subfolder_path):
                    Log.info(f"Dossier complet : {subfolder_path}")
                    return True

                return False
        except Exception as e:
            Log.error(f"Erreur dans mark_video_processed pour {video_path_str} : {e}")
            return False

    def mark_video_failed(self, video_path, error_message=None):
        video_path_str = str(Path(video_path).resolve())
        tracking_file = self._get_tracking_file_for_video(video_path)

        with locked_file(tracking_file, "r+") as f:
            f.seek(0)
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                Log.error(f"Corrupt tracking file {tracking_file}: {e}")
                f.seek(0)
                Log.error("Recovering with empty tracking data")
                data = {"processed": {}, "in_progress": {}, "failed": {}, "heartbeats": {}}

            # Add to failed
            data["failed"][video_path_str] = {
                "date_failed": datetime.datetime.now().isoformat(),
                "error_message": error_message,
                "pid": self._pid,
            }

            # Remove from in_progress
            data["in_progress"].pop(video_path_str, None)

            # Update file

            safe_write_json_atomic(tracking_file, data)

            # Remove from local tracking
            if video_path_str in self._tracked_videos:
                self._tracked_videos.remove(video_path_str)

            Log.info(f"Video {video_path_str} marked as failed with error: {error_message}")

    def mark_video_preprocessed(self, video_path):
        video_path_str = str(Path(video_path).resolve())
        tracking_file = self._get_tracking_file_for_video(video_path)

        with locked_file(tracking_file, "r+") as f:
            f.seek(0)
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                Log.warning(f"Corrupted JSON detected in {tracking_file}, reinitializing.")
                data = {"processed": {}, "in_progress": {}, "failed": {}, "preprocessed": {}, "heartbeats": {}}

            data["preprocessed"][video_path_str] = {
                "date_preprocessed": datetime.datetime.now().isoformat(),
                "pid": self._pid,
            }

            # Écriture atomique sécurisée
            safe_write_json_atomic(tracking_file, data)

            Log.info(f"⚙️ Vidéo marquée comme prétraitée : {video_path_str}")

    def clear_local_reservations(self):
        tracking_files = self._get_all_tracking_files()

        for tracking_file in tracking_files:
            try:
                with locked_file(tracking_file, "r+") as f:
                    f.seek(0)
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        Log.error(f"Corrupt tracking file {tracking_file}: {e}")
                        f.seek(0)
                        Log.error("Recovering with empty tracking data")
                        data = {"processed": {}, "in_progress": {}, "failed": {}, "heartbeats": {}}

                    cleared = False
                    for key in list(data["in_progress"].keys()):
                        if data["in_progress"][key].get("pid") == self._pid:
                            Log.info(f"Clearing local reservation: {key}")
                            if key in data["in_progress"]:
                                del data["in_progress"][key]
                            cleared = True

                    # Also clean heartbeats
                    if str(self._pid) in data.get("heartbeats", {}):
                        del data["heartbeats"][str(self._pid)]
                        cleared = True

                    if cleared:
                        safe_write_json_atomic(tracking_file, data)
            except Exception as e:
                Log.warning(f"Failed to clear reservations in {tracking_file}: {e}")

        # Clear local tracking
        self._tracked_videos.clear()

    def reserve_subfolder_videos(self, subfolder_path, max_workers, batch_size=5):
        subfolder_path = Path(subfolder_path).resolve()
        video_files = [str(f) for f in subfolder_path.iterdir() if f.suffix.lower() in (".mp4", ".avi", ".mov")]

        return self.reserve_videos(video_files, max_workers, batch_size)

    def reserve_videos(self, video_list, max_workers, batch_size=5, local=True):
        heartbeat_multiplier = 12
        if not video_list:
            return []

        # Group videos by their subfolder to reduce contention
        videos_by_folder = {}
        for video in video_list:
            video_path = Path(video).resolve()
            folder = video_path.parent
            if folder not in videos_by_folder:
                videos_by_folder[folder] = []
            videos_by_folder[folder].append(str(video_path))

        reserved = []
        available_slots = max_workers

        # First, check local reservation count if needed
        if local:
            local_count = len(self._tracked_videos)
            available_slots = max(0, max_workers - local_count)
            if available_slots <= 0:
                return []

        # Process one folder at a time to minimize contention
        for folder, folder_videos in videos_by_folder.items():
            # Skip if we've filled our quota
            if len(reserved) >= batch_size or len(reserved) >= available_slots:
                break

            # Get sample video to determine tracking file
            if not folder_videos:
                continue

            sample_video = folder_videos[0]
            tracking_file = self._get_tracking_file_for_video(sample_video)

            # Use locked_file to ensure exclusive access while checking AND reserving
            with locked_file(tracking_file, "r+") as f:
                f.seek(0)
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    Log.warning(f"Tracker file {tracking_file} corrupted in reservation. Reinitializing.")
                    data = {"processed": {}, "in_progress": {}, "failed": {}, "heartbeats": {}}

                current_time = time.time()

                # Clean stale reservations
                stale_keys = []
                for vid, info in data.get("in_progress", {}).items():
                    if current_time - info.get("start_time", 0) > self.timeout_seconds:
                        stale_keys.append(vid)

                    # Also check heartbeats
                    pid = info.get("pid")
                    if pid and str(pid) in data.get("heartbeats", {}):
                        last_heartbeat = data["heartbeats"][str(pid)].get("timestamp", 0)
                        if current_time - last_heartbeat > self.heartbeat_interval * heartbeat_multiplier:
                            stale_keys.append(vid)
                            Log.warning(f"Freeing reservation with stale heartbeat: {vid} (PID: {pid})")

                for key in stale_keys:
                    Log.info(f"Freeing stale reservation: {key}")
                    if key in data["in_progress"]:
                        del data["in_progress"][key]

                # Calculate available folder slots
                if local:
                    local_folder_reservations = sum(
                        1 for _, info in data["in_progress"].items() if info.get("pid") == self._pid
                    )
                    folder_available_slots = max_workers - local_folder_reservations
                else:
                    folder_available_slots = max_workers - len(data["in_progress"])

                # Further limit by global slots we have left
                folder_available_slots = min(folder_available_slots, available_slots - len(reserved))
                if folder_available_slots <= 0:
                    safe_write_json_atomic(tracking_file, data)
                    continue

                # Limit by the batch size that's left
                folder_batch_size = min(batch_size - len(reserved), folder_available_slots)

                # IMPORTANT: Filter videos that are not already reserved or processed WITHIN THE LOCK
                available_videos = [
                    video
                    for video in folder_videos
                    if video not in data["processed"] and video not in data["in_progress"]
                ]

                # Reserve videos in this folder
                for video in available_videos:
                    data["in_progress"][video] = {
                        "start_time": current_time,
                        "pid": self._pid,
                        "date_started": datetime.datetime.now().isoformat(),
                    }
                    reserved.append(video)
                    # Add to local tracking
                    self._tracked_videos.add(video)

                    if len(reserved) >= folder_batch_size:
                        break

                # Update heartbeat
                data.setdefault("heartbeats", {})
                folder_tracked_videos = [v for v in self._tracked_videos if Path(v).parent == folder]
                existing_heartbeat = data.get("heartbeats", {}).get(str(self._pid), {})
                existing_videos = set(existing_heartbeat.get("videos", []))

                updated_videos = existing_videos.union(set(folder_tracked_videos))

                data["heartbeats"][str(self._pid)] = {"timestamp": time.time(), "videos": list(updated_videos)}

                # Save changes
                safe_write_json_atomic(tracking_file, data)

            # Stop if we've reached our quota
            if len(reserved) >= batch_size or len(reserved) >= available_slots:
                break

        Log.info(f"Reserved {len(reserved)} videos across {len(videos_by_folder)} folders")
        return reserved

    def is_video_available(self, video_path):
        video_path_str = str(Path(video_path).resolve())
        tracking_file = self._get_tracking_file_for_video(video_path)
        heartbeat_multiplier = 12
        with locked_file(tracking_file, "r") as f:
            f.seek(0)
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                Log.warning(f"Corrupted JSON detected in {tracking_file}, reinitializing.")
                return True  # Consider it available if we can't read the status

            if video_path_str in data["processed"]:
                # Log.info(f"Video {video_path_str} already processed.")
                return False

            if video_path_str in data["in_progress"]:
                info = data["in_progress"][video_path_str]
                elapsed = time.time() - info.get("start_time", 0)

                # Check heartbeat if available
                pid = info.get("pid")
                heartbeat_expired = False

                if pid and str(pid) in data.get("heartbeats", {}):
                    last_heartbeat = data["heartbeats"][str(pid)].get("timestamp", 0)
                    heartbeat_expired = (time.time() - last_heartbeat) > (
                        self.heartbeat_interval * heartbeat_multiplier
                    )

                if elapsed > self.timeout_seconds or heartbeat_expired:
                    Log.warning(f"Video {video_path_str} lock expired after {elapsed:.1f}s or heartbeat expired.")
                    return True
                else:
                    Log.info(f"Video {video_path_str} still in progress ({elapsed:.1f}s).")
                    return False

            return True

    def _send_heartbeat(self):
        while not self._stop_heartbeat.is_set():
            try:
                heartbeat_pid = str(self._pid)
                timestamp_now = time.time()

                # Group tracked videos by folder
                videos_by_folder = {}
                for video in self._tracked_videos:
                    video_path = Path(video).resolve()
                    folder = video_path.parent
                    videos_by_folder.setdefault(folder, []).append(str(video_path))

                for folder, local_videos in videos_by_folder.items():
                    tracking_file = self._get_tracking_file_for_subfolder(folder)

                    with locked_file(tracking_file, "r+", max_retries=100, initial_delay=0.1) as f:
                        f.seek(0)
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError as e:
                            Log.error(f"Corrupt tracking file {tracking_file}: {e}")
                            f.seek(0)
                            Log.error("Recovering with empty tracking data")
                            data = {"processed": {}, "in_progress": {}, "failed": {}, "heartbeats": {}}

                        data.setdefault("heartbeats", {})
                        data["heartbeats"][heartbeat_pid] = {
                            "timestamp": timestamp_now,
                            "videos": local_videos,  # ✅ uniquement les vidéos de CE dossier
                        }

                        safe_write_json_atomic(tracking_file, data)

                if videos_by_folder:
                    Log.debug(f"Heartbeat updated for PID {self._pid} in {len(videos_by_folder)} folders")

            except Exception as e:
                Log.warning(f"Failed to send heartbeat: {e}")

            self._stop_heartbeat.wait(self.heartbeat_interval)

    def start_heartbeat(self):
        try:
            if self._heartbeat_thread is None or not self._heartbeat_thread.is_alive():
                self._stop_heartbeat.clear()
                self._heartbeat_thread = threading.Thread(target=self._send_heartbeat, daemon=True)
                self._heartbeat_thread.start()
                Log.info(f"Heartbeat thread started for PID {self._pid}")
        except Exception as e:
            Log.error(f"Failed to start heartbeat thread: {e}")

    def stop_heartbeat(self):
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._stop_heartbeat.set()
            self._heartbeat_thread.join(timeout=2)

    def cleanup_processed(self):
        tracking_files = self._get_all_tracking_files()

        for tracking_file in tracking_files:
            try:
                with locked_file(tracking_file, "r+") as f:
                    f.seek(0)
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        Log.error(f"Corrupt tracking file {tracking_file}: {e}")
                        f.seek(0)
                        Log.error("Recovering with empty tracking data")
                        data = {"processed": {}, "in_progress": {}, "failed": {}, "heartbeats": {}}

                    processed_set = set(data.get("processed", {}).keys())
                    to_remove = [key for key in data.get("in_progress", {}) if key in processed_set]

                    for key in to_remove:
                        Log.info(f"Cleaning reservation already treated: {key}")
                        if key in data["in_progress"]:
                            del data["in_progress"][key]

                        # Remove from local tracking if applicable
                        if key in self._tracked_videos:
                            self._tracked_videos.remove(key)

                    if to_remove:
                        safe_write_json_atomic(tracking_file, data)
            except Exception as e:
                Log.warning(f"Failed to clean up processed videos in {tracking_file}: {e}")

    def get_metadata_path(self, subfolder_path):
        subfolder_path = Path(subfolder_path).resolve()
        return subfolder_path / "metadata.txt"

    def has_metadata(self, subfolder_path):
        metadata_path = self.get_metadata_path(subfolder_path)
        return metadata_path.exists()

import os
import re
import sys
import cv2
import subprocess
import datetime
import logging
import fcntl
import time
import random
import requests
import yt_dlp
import concurrent.futures
import time
import yaml
from pathlib import Path
_cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yml"))["download"]

# OK, since we request with a Google API, we need to limit our requests.
global_request_count = 0
REQUEST_LIMIT = _cfg["request_limit"]
VERBOSE = False
TREATED_FILE = _cfg["treated_file"]
API_KEY = _cfg["youtube_api_key"]


# Utils functions
def debug_print(msg):
    if VERBOSE:
        print(msg)


def load_treated_entries(file_path):
    """
    We charge treated.txt in order to not download twice the same video
    """
    treated_ids = set()
    treated_titles = set()
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if i == 0 and line.startswith("counter:"):
                    continue
                if not line:
                    continue
                if " | " in line:
                    parts = line.split(" | ", 1)
                    treated_ids.add(parts[0])
                    treated_titles.add(parts[1].lower())
                else:
                    treated_ids.add(line)
    return treated_ids, treated_titles


def get_next_video_number_and_record(file_path, video_id, video_title):
    """
    Gets the next video number and saved the name/id in treated.txt
    We lock the file
    """
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("counter: 0\n")
    with open(file_path, "r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        lines = f.readlines()
        if lines and lines[0].startswith("counter:"):
            try:
                counter = int(lines[0].split("counter:")[1].strip())
            except ValueError:
                counter = 0
        else:
            counter = 0
        new_counter = counter + 1
        f.seek(0)
        f.write(f"counter: {new_counter}\n")
        if len(lines) > 1:
            f.writelines(lines[1:])
        # get rid of "|" char for metadatas
        video_title_clean = video_title.replace("|", "")
        f.write(f"{video_id} | {video_title_clean.lower()}\n")
        f.truncate()
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return new_counter


def get_video_folder_with_number(video_number, base_path):
    """
    makes a folder for the file with the number
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    folder_name = f"video_{video_number}"
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    video_name = f"video_{video_number}"
    return folder_path, video_name


def save_video_metadata(folder_path, video_title, video_url, video_framerate, query):
    """
    saves the metadata of the video in a file metadata.txt in video folder
    """
    metadata_file = os.path.join(folder_path, "metadata.txt")
    now = datetime.datetime.now()
    # get rid of "|" char for metadatas
    video_title_clean = video_title.replace("|", "")
    with open(metadata_file, "a", encoding="utf-8") as file:
        file.write(
            f"{video_title_clean} | {video_url} | fps:{video_framerate} | accessed:{now.strftime('%d/%m/%Y %H:%M:%S')}\n | request: {query}"
        )


def sanitize_filename(filename):
    filename = filename.replace(" ", "_")
    return re.sub(r"[^\w\-.]", "", filename)


# Can be changed if you download via another API ofc
def check_if_banned():
    """
    checks if we are banned from downloading with yt-dlp. If Youtube thinks we are a bot we stop here.
    """
    test_url = "https://www.youtube.com/watch?v=L6bx26mcQyM"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.100 Safari/537.36",
        "retries": 3,
        "fragment_retries": 3,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(test_url, download=False)
        return False
    except Exception as e:
        if "Sign in to confirm" in str(e):
            return True
        return False


def has_video_only_720p_plus(info):
    """
    Returns True if a video-only format >=720p exists
    """
    formats = info.get("formats", [])
    for f in formats:
        if (
            f.get("vcodec") != "none"
            and f.get("acodec") == "none"
            and f.get("height", 0) >= 720
        ):
            return True
    return False


# Download with yt-dlp
def download_video(url, folder_path, video_title):
    try:
        probe_opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            debug_print("No info extracted")
            return None, None

        if not has_video_only_720p_plus(info):
            debug_print("No video-only >=720p format available, skipping")
            return None, None

        ydl_opts = {
            "format": "bestvideo[height>=720][vcodec^=avc][acodec=none]/bestvideo[height>=720][vcodec!=av1][acodec=none]",
            "outtmpl": os.path.join(
                folder_path, sanitize_filename(video_title) + ".%(ext)s"
            ),
            "quiet": not VERBOSE,
            "no_warnings": not VERBOSE,
            "retries": 3,
            "fragment_retries": 3,
            "merge_output_format": "mp4",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        if not info:
            return None, None

        filepath = ydl.prepare_filename(info)
        fps = info.get("fps", "unknown")

        return filepath, fps

    except Exception as e:
        debug_print(f"Download error: {e}")
        return None, None


def get_video_duration(filepath):
    """
    gets the video duration
    """
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise Exception("Impossible to open video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration
    except Exception as e:
        debug_print(f"Error when opening {filepath}: {e}")
        return None


def cut_video(input_path, duration):
    """
    cuts the video at given duration
    """
    temp_output_path = f"{os.path.splitext(input_path)[0]}_temp.mp4"
    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-t",
        str(duration),
        "-c",
        "copy",
        temp_output_path,
    ]
    try:
        if not VERBOSE:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(command, check=True)
        os.replace(temp_output_path, input_path)
        return input_path
    except subprocess.CalledProcessError:
        return input_path


# Search via YT data v3 API
def search_videos_api(query, api_key, max_results=20):
    global global_request_count
    if global_request_count >= REQUEST_LIMIT:
        raise Exception("Request limit")

    search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoLicense": "creativeCommon",
        "maxResults": max_results,
        "key": api_key,
    }

    response = requests.get(search_url, params=params)
    global_request_count += 1
    print("Request number", global_request_count)
    if response.status_code != 200:
        debug_print(f"Erreur API: {response.status_code} {response.text}")
        return []
    data = response.json()
    videos = []
    for item in data.get("items", []):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        watch_url = f"https://youtu.be/{video_id}"
        videos.append({"video_id": video_id, "title": title, "watch_url": watch_url})
    return videos


# Main function of the code, search and downloads the videos
def search_and_download(
    query,
    treated_ids,
    treated_titles,
    treated_file,
    duration_limit=300,
    max_results=20,
    api_key=None,
    output_dir="scrapedVideos",
):
    """
    search videos via the API and downloads them while the duration limit is not reached (in seconds)
    """
    try:
        videos = search_videos_api(query, api_key, max_results)
    except Exception as e:
        debug_print(str(e))
        return 0
    total_duration = 0

    for video in videos:
        if video["video_id"] in treated_ids or video["title"].lower() in treated_titles:
            debug_print(f"Video {video['video_id']} already treated, pass.")
            continue

        if check_if_banned():
            debug_print("You seem to be banned...")
            sleep_time = random.uniform(1800, 5400)
            time.sleep(sleep_time)
            print("Banned, sleeping for:", sleep_time, "s")
            return total_duration

        video_number = get_next_video_number_and_record(
            treated_file, video["video_id"], video["title"]
        )
        treated_ids.add(video["video_id"])
        treated_titles.add(video["title"].lower())

        folder_path, video_name = get_video_folder_with_number(
            video_number, base_path=output_dir
        )

        sleep_time = random.uniform(6, 12)
        print("Before download, just in case: sleeping for ", round(sleep_time), "s")
        time.sleep(sleep_time)
        filepath, fps = download_video(video["watch_url"], folder_path, video_name)
        if not filepath:
            continue

        save_video_metadata(folder_path, video["title"], video["watch_url"], fps, query)
        video_duration = get_video_duration(filepath)
        if video_duration is None:
            continue

        remaining_time = duration_limit - total_duration
        if video_duration > remaining_time:
            filepath = cut_video(filepath, remaining_time)
            total_duration += remaining_time
        else:
            total_duration += video_duration

        # Random pauses to limit requests frequency. Google API is quite tolerant but still
        sleep_time = random.uniform(6, 12)
        print("Just in case, sleeping for:", round(sleep_time, 2), "s")
        time.sleep(sleep_time)
        if total_duration >= duration_limit:
            break

    return total_duration


# main(), launches the queries and downloads via other funcitons
def main():
    global VERBOSE
    if "--verbose" in sys.argv:
        VERBOSE = True
        sys.argv.remove("--verbose")

    # Positional args are optional: values from config.yml are used as defaults
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    output_dir = positional[0] if len(positional) >= 1 else _cfg["scraped_dir"]
    duration_limit_minutes = int(positional[1]) if len(positional) >= 2 else _cfg["duration_minutes"]

    if check_if_banned():
        print("Your IP seems banned...")
        sys.exit(1)

    start_time = time.time()
    print(f"Output dir: {output_dir} | Duration: {duration_limit_minutes} min")
    remaining_time = duration_limit_minutes * 60
    total_downloaded_time = 0

    treated_ids, treated_titles = load_treated_entries(TREATED_FILE)

    # Uses the requests from an external file. (video_ideas.txt here)
    with open("video_ideas.txt", "r", encoding="utf-8") as file:
        queries = file.readlines()

    for query in queries:
        query = query.strip()
        if query and remaining_time > 0:
            time_used = search_and_download(
                query,
                treated_ids,
                treated_titles,
                TREATED_FILE,
                duration_limit=remaining_time,
                max_results=_cfg["max_results_per_query"],
                api_key=API_KEY,
                output_dir=output_dir,
            )
            # If no video treated, and requests count exceeded, we stop here
            if time_used == 0 and global_request_count >= REQUEST_LIMIT:
                print("Request limit exceeded. Script stop")
                break
            remaining_time -= time_used
            total_downloaded_time += time_used

    print(
        f"Total downloaded video time: {total_downloaded_time} seconds in {round(time.time()-start_time,2)}s"
    )


if __name__ == "__main__":
    main()

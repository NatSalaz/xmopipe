from ultralytics import YOLO
from hmr4d import PROJ_ROOT

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from hmr4d.utils.seq_utils import (
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    frame_id_to_mask,
    rearrange_by_mask,
)
from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.net_utils import moving_average_smooth


class Tracker:
    def __init__(self) -> None:
        # https://docs.ultralytics.com/modes/predict/
        self.yolo = YOLO(PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")

    def track(self, video_path):
        track_history = []
        cfg = {
            "device": "cuda",
            "conf": 0.5,  # default 0.25, wham 0.5
            "classes": 0,  # human
            "verbose": False,
            "stream": True,
        }
        results = self.yolo.track(video_path, **cfg)
        # frame-by-frame tracking
        track_history = []
        for result in tqdm(results, total=get_video_lwh(video_path)[0], desc="YoloV8 Tracking", disable=True):
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()  # (N)
                bbx_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                result_frame = [{"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]} for i in range(len(track_ids))]
            else:
                result_frame = []
            track_history.append(result_frame)

        return track_history

    @staticmethod
    def sort_track_length(track_history, video_path):
        """This handles the track history from YOLO tracker."""
        id_to_frame_ids = defaultdict(list)
        id_to_bbx_xyxys = defaultdict(list)
        # parse to {det_id : [frame_id]}
        for frame_id, frame in enumerate(track_history):
            for det in frame:
                id_to_frame_ids[det["id"]].append(frame_id)
                id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
        for k, v in id_to_bbx_xyxys.items():
            id_to_bbx_xyxys[k] = np.array(v)

        # Sort by length of each track (max to min)
        id_length = {k: len(v) for k, v in id_to_frame_ids.items()}
        id2length = dict(sorted(id_length.items(), key=lambda item: item[1], reverse=True))

        # Sort by area sum (max to min)
        id_area_sum = {}
        l, w, h = get_video_lwh(video_path)
        for k, v in id_to_bbx_xyxys.items():
            bbx_wh = v[:, 2:] - v[:, :2]
            id_area_sum[k] = (bbx_wh[:, 0] * bbx_wh[:, 1] / w / h).sum()
        id2area_sum = dict(sorted(id_area_sum.items(), key=lambda item: item[1], reverse=True))
        id_sorted = list(id2area_sum.keys())

        return id_to_frame_ids, id_to_bbx_xyxys, id_sorted

    def get_one_track(self, video_path, idx):
        # track
        track_history = self.track(video_path)

        # parse track_history & use top1 track
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, video_path)
        track_id = id_sorted[idx]
        frame_ids = torch.tensor(id_to_frame_ids[track_id])  # (N,)
        bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])  # (N, 4)

        # interpolate missing frames
        mask = frame_id_to_mask(frame_ids, get_video_lwh(video_path)[0])
        bbx_xyxy_one_track = rearrange_by_mask(bbx_xyxys, mask)  # (F, 4), missing filled with 0
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)  # list of list
        bbx_xyxy_one_track = linear_interpolate_frame_ids(bbx_xyxy_one_track, missing_frame_id_list)
        assert (bbx_xyxy_one_track.sum(1) != 0).all()

        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
        
    def get_tracks(self, video_path, indices=None, max_gap_threshold=60, min_duration=30):
        """
        Get all the tracks histories from yolo and verify gaps and min duration
        """
        
        track_history = self.track(video_path)
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, video_path)
        
        # If no index is given, we treat everything
        if indices is None:
            indices = range(len(id_sorted))
        bbx_xyxy_tracks = []
        flagged_frames_tracks = []
        total_frames = get_video_lwh(video_path)[0]
        
        # For each track
        for idx in indices:
            track_id = id_sorted[idx]
            frame_ids = torch.tensor(id_to_frame_ids[track_id])   # (N_detected,)
            bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])   # (N_detected, 4)
            mask = frame_id_to_mask(frame_ids, total_frames) 
            interpolated_frames = [i for i, present in enumerate(mask.tolist()) if not present]

            duration = int(frame_ids[-1] - frame_ids[0] + 1)
            if duration < min_duration:
                # Track too short, next one
                continue
            
            # Calculate the gaps
            mask_np = mask.cpu().numpy().astype(np.bool_)
            max_gap = 0
            current_gap = 0
            for present in mask_np:
                if not present:
                    current_gap += 1
                else:
                    if current_gap > max_gap:
                        max_gap = current_gap
                    current_gap = 0
            # In case last gap was longer
            if current_gap > max_gap:
                max_gap = current_gap
            
            if max_gap > max_gap_threshold:
                # If the tracking is unknown for too long, we get rid of it
                continue
            
            # interpolate missing frames
            bbx_xyxy_full = rearrange_by_mask(bbx_xyxys, mask)  # (total_frames, 4)
            missing_frame_id_list = get_frame_id_list_from_mask(~mask)
            bbx_xyxy_interp = linear_interpolate_frame_ids(bbx_xyxy_full, missing_frame_id_list)
            
            # Checking the interpolated boxes are correct
            if (bbx_xyxy_interp.sum(dim=1) == 0).any():
                continue
            
            # smoothing
            bbx_xyxy_interp = moving_average_smooth(bbx_xyxy_interp, window_size=5, dim=0)
            bbx_xyxy_interp = moving_average_smooth(bbx_xyxy_interp, window_size=5, dim=0)

            #flag the frames when we intepolated or the bbox moves a lot
            variation_anomalies = Tracker.detect_variation_anomalies(bbx_xyxy_interp, threshold_ratio=0.2)
            flagged_frames = sorted(set(int(x) for x in (interpolated_frames + variation_anomalies)))

            bbx_xyxy_tracks.append(bbx_xyxy_interp)
            flagged_frames_tracks.append(flagged_frames)
        return bbx_xyxy_tracks, flagged_frames_tracks
    
    def detect_variation_anomalies(bbx_interp, threshold_ratio=0.3):
        """
        bbx_interp : Tensor de taille (F, 4) contenant les bounding boxes interpolées.
        threshold_ratio : Ratio pour définir le seuil de variation anormale.
        Retourne : liste d'indices de frames avec forte variation.
        """
        anomalies = []
        for i in range(1, bbx_interp.shape[0]):
            diff = torch.norm(bbx_interp[i] - bbx_interp[i-1], p=2).item()
            box_size = torch.norm(bbx_interp[i][2:] - bbx_interp[i][:2], p=2).item()
            threshold = threshold_ratio * box_size
            if diff > threshold:
                anomalies.append(i)
        return anomalies

# tracking.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def cosine_distance(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 1.0
    emb1, emb2 = np.array(emb1), np.array(emb2)
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-6)
    return 1 - cos_sim

class KalmanBoxTracker:
    def __init__(self, box, tracker_id, dt=1.0, embedding=None):
        self.id = tracker_id
        self.dt = dt
        self.time_since_update = 0
        self.age = 0
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        x, y = x1 + w / 2.0, y1 + h / 2.0
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.x = np.array([x, y, s, r, 0, 0, 0]).reshape((7, 1))
        self.kf.F = np.array([[1, 0, 0, 0, self.dt, 0, 0],
                              [0, 1, 0, 0, 0, self.dt, 0],
                              [0, 0, 1, 0, 0, 0, self.dt],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        self.kf.P *= 10.0
        self.kf.R *= 1.0
        self.kf.Q = np.eye(7) * 0.01
        self.embedding = embedding

    def predict(self, dt=1.0):
        self.kf.F[0, 4] = dt
        self.kf.F[1, 5] = dt
        self.kf.F[2, 6] = dt
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_state()

    def update(self, box, detection_embedding=None):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        x, y = x1 + w / 2.0, y1 + h / 2.0
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        self.kf.update(np.array([x, y, s, r]).reshape((4, 1)))
        self.time_since_update = 0
        if detection_embedding is not None:
            if self.embedding is None:
                self.embedding = detection_embedding
            else:
                self.embedding = (np.array(self.embedding) + np.array(detection_embedding)) / 2.0

    def get_state(self):
        x, y, s, r = self.kf.x[:4].flatten()
        w = np.sqrt(np.maximum(s * r, 0)) if r > 0 else 0
        h = s / w if w > 0 else 0
        return (x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0)

class BetterFaceTracker:
    def __init__(self, max_miss=30, iou_threshold=0.25, appearance_weight=0.5):
        self.max_miss = max_miss
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.next_id = 1
        self.appearance_weight = appearance_weight

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA + 1)
        interH = max(0, yB - yA + 1)
        interArea = interW * interH
        areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / (areaA + areaB - interArea) if (areaA + areaB - interArea) > 0 else 0

    def update(self, boxes, dt=1.0):
        predicted = [trk.predict(dt=dt) for trk in self.trackers]
        cost_matrix = np.zeros((len(predicted), len(boxes)), dtype=np.float32)
        for i, pred_box in enumerate(predicted):
            for j, det in enumerate(boxes):
                iou_cost = 1 - self._iou(pred_box, det[:4])
                app_cost = cosine_distance(self.trackers[i].embedding, det[5])
                cost_matrix[i, j] = self.appearance_weight * app_cost + (1 - self.appearance_weight) * iou_cost
        assigned = set()
        if cost_matrix.size:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < (1 - self.iou_threshold):
                    assigned.add((i, j))
        assigned_trackers = {i for i, j in assigned}
        assigned_dets = {j for i, j in assigned}
        for i, j in assigned:
            self.trackers[i].update(boxes[j][:4], detection_embedding=boxes[j][5])
        for idx, trk in enumerate(self.trackers):
            if idx not in assigned_trackers:
                trk.time_since_update += 1
        self.trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_miss]
        for j, det in enumerate(boxes):
            if j not in assigned_dets:
                self.trackers.append(KalmanBoxTracker(det[:4], self.next_id, dt=dt, embedding=det[5]))
                self.next_id += 1
        return [(trk.id, trk.get_state()) for trk in self.trackers if trk.time_since_update == 0]
    def reset(self):
        self.trackers = []
        self.next_id = 1
"""
Auto-label Mapping Objects From Detections
One of the primary ways we find information about things around us in self-driving perception is through 3D object detection models. These models take in sensor data from the world around our car and output bounding boxes, an object type, and a confidence score. The models we're working with for this problem can only detect traffic signals, stop signs, and yield signs.
These models are usually trained using supervised learning, and require human labels. However, 3D objects are expensive to label by hand, so we want to use detections from our existing model to generate labels. Each individual model has some false positive rate. In order to reduce the chance of false positive labels, the team decides to employ an ensemble of object detection models. In machine learning domains, this means we run N different models on the same sensor data.
You are the engineer tasked with designing the algorithm for combining detections: given object detections in 3D space from N different models, each with some confidence score, how would you design an algorithm to produce object labels?

Solution - 

Our job is to fuse detections into final object labels by resolving 3 main sources of error:

    Localization error – Detectors may estimate the object position/size differently.

            Solution: Weighted average based on confidence.

    Classification error – Detectors may disagree on the object type.

            Solution: Majority voting.

    Detection error – Some models may hallucinate detections.

            Solution: Filter out low-confidence detections and require multiple models to agree via IoU clustering.

 # TODO,
    
    # sort the detections using confidence, filter out the lower confidence detections (threshold 0.5)
    # convert the detections into clusters using IoU matching 
    # Fuse the clusters - the output detection size and xyz is computed using weighted avg of uncertainty 
    #                   - the output label is calculated using majority voting        


"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List
from collections import Counter

class Type(Enum):
    STOP_SIGN = 0
    YIELD_SIGN = 1
    TRAFFIC_SIGNAL = 2

@dataclass
class Vector3:
    x: float
    y: float
    z: float

@dataclass
class ObjectDetection:
    object_type: Type
    confidence: float
    xyz: Vector3
    size: Vector3

@dataclass
class ObjectLabel:
    object_type: Type
    xyz: Vector3
    size: Vector3

def iou_3d(box1: ObjectDetection, box2: ObjectDetection) -> float:
    def get_bounds(center, size):
        min_pt = np.array([center.x - size.x / 2, center.y - size.y / 2, center.z - size.z / 2])
        max_pt = np.array([center.x + size.x / 2, center.y + size.y / 2, center.z + size.z / 2])
        return min_pt, max_pt

    min1, max1 = get_bounds(box1.xyz, box1.size)
    min2, max2 = get_bounds(box2.xyz, box2.size)

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_volume = np.prod(inter_size)

    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)

    union_volume = vol1 + vol2 - inter_volume
    return inter_volume / union_volume if union_volume > 0 else 0.0

def weighted_avg(cluster: List[ObjectDetection]):
    total_conf = sum(det.confidence for det in cluster)
    if total_conf == 0:
        total_conf = 1e-6  # avoid div by 0

    def avg_coord(attr):
        return sum(getattr(det.xyz, attr) * det.confidence for det in cluster) / total_conf

    def avg_size(attr):
        return sum(getattr(det.size, attr) * det.confidence for det in cluster) / total_conf

    output_xyz = Vector3(avg_coord('x'), avg_coord('y'), avg_coord('z'))
    output_size = Vector3(avg_size('x'), avg_size('y'), avg_size('z'))
    return output_xyz, output_size

def majority_voting(cluster: List[ObjectDetection]):
    votes = [det.object_type for det in cluster]
    return Counter(votes).most_common(1)[0][0]

def cluster_detections(detections: List[ObjectDetection], iou_threshold=0.5):
    clusters = []
    used = [False] * len(detections)

    for i, det in enumerate(detections):
        if used[i]:
            continue
        cluster = [det]
        used[i] = True
        for j in range(i+1, len(detections)):
            if not used[j] and iou_3d(det, detections[j]) >= iou_threshold:
                cluster.append(detections[j])
                used[j] = True
        clusters.append(cluster)
    return clusters

def run_autolabeler(detections: List[ObjectDetection]) -> List[ObjectLabel]:
    # Step 1: Filter low-confidence detections
    filtered = [det for det in detections if det.confidence >= 0.5]

    # Step 2: Cluster using IoU
    clusters = cluster_detections(filtered, iou_threshold=0.5)

    # Step 3: Fuse each cluster into a single ObjectLabel
    labels = []
    for cluster in clusters:
        fused_xyz, fused_size = weighted_avg(cluster)
        fused_type = majority_voting(cluster)
        labels.append(ObjectLabel(fused_type, fused_xyz, fused_size))
    
    return labels

detections = [
    ObjectDetection(Type.STOP_SIGN, 0.9, Vector3(1,2,3), Vector3(1,1,1)),
    ObjectDetection(Type.STOP_SIGN, 0.85, Vector3(1.1,2.05,3), Vector3(1.1,1,1)),
    ObjectDetection(Type.YIELD_SIGN, 0.7, Vector3(10,10,10), Vector3(2,2,2)),
    ObjectDetection(Type.YIELD_SIGN, 0.6, Vector3(10.2,10.1,10), Vector3(2.1,2,2)),
]

output = run_autolabeler(detections)
for label in output:
    print(label)
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:10:16 2026

@author: MUEF2360
"""

# ============================================================
# Programming Task: Dataset-Based Face Matching Using SIFT
# Dataset: LFW (Labeled Faces in the Wild)
# ============================================================

import cv2
import numpy as np

from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ------------------------------------------------------------
# 1. Load the LFW biometric face dataset
# ------------------------------------------------------------

lfw = fetch_lfw_people(
    min_faces_per_person=20,
    resize=0.5,
    color=False
)

images = lfw.images
labels = lfw.target
target_names = lfw.target_names

print("Number of images:", len(images))
print("Image dimensions:", images.shape[1:])
print("Number of subjects:", len(target_names))


# ------------------------------------------------------------
# 2. Create SIFT detector
# ------------------------------------------------------------

# TODO:
# Create a SIFT feature detector
#
sift = cv2.SIFT_create()


# ------------------------------------------------------------
# 3. Extract SIFT features from all images
# ------------------------------------------------------------

def extract_sift_features(images, sift):
    """
    Extract SIFT keypoints and descriptors
    from every image in the dataset.

    Return:
        all_descriptors: SIFT descriptors for each image
    """

    all_descriptors = []

    for image in images:

        # Convert normalized image to uint8
        image = (image * 255).astype(np.uint8)

        # Detect keypoints and compute SIFT descriptors
        keypoints, descriptors = sift.detectAndCompute(
            image, None
        )

        # Store descriptors
        if descriptors is None:
            all_descriptors.append([])
        else:
            all_descriptors.append(descriptors)

    return all_descriptors


# TODO:
descriptors = extract_sift_features(images, sift)


# ------------------------------------------------------------
# 4. Generate genuine and impostor pairs
# ------------------------------------------------------------

def generate_face_pairs(labels, descriptors, num_pairs=1000):
    """
    Generate face pairs from the dataset.

    Genuine pair:
        Two images belonging to the same person.

    Impostor pair:
        Two images belonging to different persons.

    Returns:
        pairs       : pairs of SIFT descriptors
        pair_labels : 1 = genuine, 0 = impostor
    """

    pairs = []
    pair_labels = []

    # Randomly select two different images until
    # the requested number of pairs is generated.
    while len(pairs) < num_pairs:
        i, j = np.random.choice(len(labels), size=2, replace=False)

        if labels[i] == labels[j]:
            pair_label = 1       # Genuine pair
        else:
            pair_label = 0       # Impostor pair

        # Skip pairs where either image has no SIFT descriptors
        if len(descriptors[i]) == 0 or len(descriptors[j]) == 0:
            continue

        pairs.append((descriptors[i], descriptors[j]))
        pair_labels.append(pair_label)

    return pairs, pair_labels


#TODO:
pairs, pair_labels = generate_face_pairs(labels, descriptors, num_pairs=1000)


# ------------------------------------------------------------
# 5. Match SIFT descriptors
# ------------------------------------------------------------

def calculate_sift_match_score(desc1, desc2):
    """
    Match two SIFT descriptor sets.

    Use:
        - BFMatcher or FLANN
        - KNN matching
        - Lowe's ratio test

    Return:
        Normalized matching score
    """

    # Handle cases where descriptors are unavailable
    if desc1 is None or desc2 is None:
        return 0

    if len(desc1) == 0 or len(desc2) == 0:
        return 0

    # Create BFMatcher
    matcher = cv2.BFMatcher()

    # Perform KNN matching
    matches = matcher.knnMatch(
        desc1, desc2, k=2
    )

    good_matches = []

    # Apply Lowe's ratio test
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Number of good matches is the matching score
    score = len(good_matches)

    return score


# ------------------------------------------------------------
# 6. Perform matching for all generated pairs
# ------------------------------------------------------------

scores = []

for desc1, desc2 in pairs:

    score = calculate_sift_match_score(
        desc1, desc2
    )

    scores.append(score)


# ------------------------------------------------------------
# 7. Determine matching threshold
# ------------------------------------------------------------

# TODO:
# Determine an appropriate threshold using
# genuine and impostor matching scores.
#
# A pair is classified as:
#
#     MATCH     -> score >= threshold
#     NON-MATCH -> score < threshold

threshold = 20


# ------------------------------------------------------------
# 8. Classify face pairs
# ------------------------------------------------------------

predictions = []

for score in scores:

    if score >= threshold:
        predictions.append(1)
    else:
        predictions.append(0)


# ------------------------------------------------------------
# 9. Evaluate the face-matching system
# ------------------------------------------------------------

# TODO:
# Calculate:
#
accuracy = accuracy_score(pair_labels, predictions)
# )
#
precision = precision_score(
    pair_labels,
    predictions
)
#
recall = recall_score(
    pair_labels,
    predictions
)
#
f1 = f1_score(
    pair_labels,
    predictions
)


# ------------------------------------------------------------
# 10. Display evaluation results
# ------------------------------------------------------------

# TODO:
print("--------------------------------")
print("SIFT Face Matching Performance")
print("--------------------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)


# ------------------------------------------------------------
# Expected Tasks
# ------------------------------------------------------------
#
# Complete the TODO sections to:
#
# 1. Load the LFW biometric face dataset.
# 2. Extract SIFT features from every face image.
# 3. Generate genuine and impostor face pairs.
# 4. Match SIFT descriptors using a feature matcher.
# 5. Apply Lowe's ratio test.
# 6. Determine a suitable matching threshold.
# 7. Classify face pairs as MATCH/NON-MATCH.
# 8. Evaluate the system using:
#       - Accuracy
#       - Precision
#       - Recall
#       - F1-score
#
# ------------------------------------------------------------
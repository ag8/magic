import pickle

with open('/data/affinity/2d/overlap_tiny/OVERLAP_AREAS') as fp:
    overlap_areas = pickle.load(fp)
    overlap_areas = overlap_areas[:100]

    with open('/data/affinity/2d/overlap_micro/OVERLAP_AREAS', 'wb') as f:
        pickle.dump(overlap_areas, f)
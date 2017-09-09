import pickle

with open('/data/affinity/2d/overlap_micro/OVERLAP_AREAS') as fp:
    overlap_areas = pickle.load(fp)
    print(overlap_areas)

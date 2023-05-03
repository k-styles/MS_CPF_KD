import numpy as np
import json

# Load the data from the individual files
class_map = json.load(open('ppi-class_map.json'))
feats = np.load('ppi-feats.npy')
G = json.load(open('ppi-G.json'))
id_map = json.load(open('ppi-id_map.json'))
walks = np.loadtxt('ppi-walks.txt')

# Save the data to a .npz file
np.savez('ppi.npz', class_map=class_map, feats=feats, G=G, id_map=id_map, walks=walks)


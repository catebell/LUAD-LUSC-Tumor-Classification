"""
Change the tumor name and the model name here
"""
tumor = "lung"
#tumor = "kidney"

model = "MultiModalGNN"
#model = "GAT"
#model = "BasicGraphConvGNN"
#model = "GINEConvGNN"
#model = "MLP"

'''
Predefined paths, name of the directories referenced through the code 
'''
DATASET = "original_dataset"  # folder where to upload the original dataset
FILES = "files"  # folder where the processed files will be saved and retrieved
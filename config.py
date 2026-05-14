"""
Change the tumor name, the model name, and the training mode here
"""
tumor = "lung"
#tumor = "kidney"

model = "MultiModalGNN"
#model = "GAT"
#model = "BasicGraphConvGNN"
#model = "GINEConvGNN"
#model = "MLP"
#model = "MoAGNN"

mode = "kfold"
#mode = "gridsearch"
#mode = "montecarlo"

'''
Predefined paths, name of the directories referenced through the code 
'''
DATASET = "original_dataset"  # folder where to upload the original dataset
FILES = "files"  # folder where the processed files will be saved and retrieved
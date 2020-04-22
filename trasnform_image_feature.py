import numpy as np 

print("loading image features")
train_arr = np.load("preprocessed_data/new_filtered_train_image_features_vgg.npy")
print("transforming features")
train_arr = train_arr.reshape(train_arr.shape[0],train_arr.shape[1],-1)
print("saving features")
np.save("preprocessed_data/transformed_train_image_features_vgg",train_arr)



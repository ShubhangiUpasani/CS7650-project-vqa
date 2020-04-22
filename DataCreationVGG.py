#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('cd', '"drive/My Drive"')


# In[ ]:


import json 
  
# JSON file
def read_json(filename): 
  f = open (filename, "r") 
    
  # Reading from file 
  data = json.loads(f.read()) 
  return data


# In[3]:


#get_ipython().run_line_magic('cd', '"VQA-master/PythonHelperTools/"')


# In[ ]:


# %cd ..


# In[5]:


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

# Load the pretrained model
# model = models.resnet18(pretrained=True)
model = models.vgg16(pretrained=True)
# Use the model object to select the desired layer
layer = model._modules.get('features')._modules.get('29')


# Set model to evaluation mode
model.eval()

scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


# In[6]:


print(model)


# In[ ]:


def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    if len(np.array(img).shape) == 2:
      img = np.array(img)
      tmp_img = np.zeros(shape=[img.shape[0], img.shape[1], 3])
      tmp_img[:,:,0] = img
      tmp_img[:,:,1] = img
      tmp_img[:,:,2] = img
      img = tmp_img
      img = Image.fromarray(np.uint8(img))
    # print(np.array(img).shape)
    # print(len(np.array(img).shape))
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512,14,14)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        # print(o.data.size())
        my_embedding.copy_(o.data.view(o.data.size(1),o.data.size(2),o.data.size(3)))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding


# In[9]:


#get_ipython().run_line_magic('cd', '..')


# In[ ]:


train_data = read_json("preprocessed_data/combined_filtered_train_dataset_32k.json")


# In[ ]:


import numpy as np
image_features = np.zeros(shape=[len(train_data), 512, 14, 14])
for i in range(len(train_data)):
  if i%10 == 0:
    print(i)
  image_loc = train_data[i]['image_loc']
  #print(image_loc)
  vector = get_vector("/srv/share/datasets/coco/"+image_loc)
  #vector = get_vector("Images/mscoco/"+image_loc)
  image_features[i,:] = vector
  # print(vector)
  # break


# In[ ]:


np.save("preprocessed_data/new_filtered_train_image_features_vgg.npy", image_features)


# In[ ]:





# In[ ]:


train_data = read_json("preprocessed_data/combined_filtered_val_dataset_19k.json")


# In[ ]:


import numpy as np
image_features = np.zeros(shape=[len(train_data), 512, 14, 14])
for i in range(len(train_data)):
  if i%10 == 0:
    print(i)
  image_loc = train_data[i]['image_loc']
  # print(image_loc)
  vector = get_vector("/srv/share/datasets/coco/"+image_loc)
  image_features[i,:] = vector
  # print(vector)
  # break


# In[ ]:


np.save("preprocessed_data/new_filtered_val_image_features_vgg.npy", image_features)


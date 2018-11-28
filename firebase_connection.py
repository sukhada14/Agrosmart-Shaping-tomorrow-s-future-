from keras.models import load_model
from PIL import Image
import os
from array import array

import h5py
import numpy as np
from keras.utils import np_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img_rows=200
img_cols=200



def potato(filename):
	model = load_model('Potato_model.h5')

	model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

	img=Image.open(filename)

	img_resized=img.resize((200,200))
	img_array=np.array(img_resized)

	transpose=img_array.transpose((1,0,2))

	nk=np.expand_dims(img_array,axis=0)
	nk=np.reshape(img_resized,[3,1,img_rows,img_cols])
	print("size:",nk.shape)

	gray=img_resized.convert('L')
	gray_array=np.array(gray)
	print("array shape:",gray_array.shape)
	nk=np.reshape(gray_array,[1,1,img_rows,img_cols])


	print(model.predict_classes(nk))
	ar=model.predict_classes(nk)
	print(ar[0])
	result={0:'PotatoEarlyBlight',1:'PotatoHealthy',2:'PotatoLateBlight'}
	result = result.get(ar[0])
	#print("result",result)
	return result







def grape(filename):
	model = load_model('Grape_model.h5')

	model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

	img=Image.open(filename)

	img_resized=img.resize((200,200))
	img_array=np.array(img_resized) 

	transpose=img_array.transpose((1,0,2))

	nk=np.expand_dims(img_array,axis=0)
	nk=np.reshape(img_resized,[3,1,img_rows,img_cols])
	print(nk.shape)

	gray=img_resized.convert('L')
	gray_array=np.array(gray)
	print(gray_array.shape)
	nk=np.reshape(gray_array,[1,1,img_rows,img_cols])


	print(model.predict_classes(nk))
	ar=model.predict_classes(nk)
	print(ar[0])
	result={0:'GrapeBlackRot',1:'GrapeEsca',2:'GrapeHealthy',3:'GrapeLeafBlight'}
	result = result.get(ar[0])
	#print("result",result)
	return result




from firebase import firebase
firebase = firebase.FirebaseApplication('https://agrosmart-91732.firebaseio.com/agrosmart-91732', None)
result = firebase.get('/Upload',None)
print(result)
#userID,value=result.items()
userID=next(iter(result))
#print('userID',userID)
#key=userID[0]
#print('key',key)
#print(result.items())
url=result.get(userID,{}).get('url')
model_name=result.get(userID,{}).get('name')
print('model_name',model_name)
print('url',url)


import urllib.request as urllib			
from PIL import Image
from io import BytesIO
			
img_file = urllib.urlopen(url)
im = BytesIO(img_file.read())
resized_image = Image.open(im)
print(resized_image)
name='image.jpg'
resized_image.save(name)






if(model_name=='Potato'):
	print(potato(name))
	result1 = firebase.post('/RESULT',{'result':potato(name)})
else :
	if(model_name=='Grape'):
		print(grape(name))
		result1 = firebase.post('/RESULT',{'result':grape(name)}       

print (result1)






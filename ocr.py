#import any plotting library
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import random
#from ._conv import register_converters as _register_converters
def rescale_array(given_array):
	#return
	 temp=(given_array/given_array.max())
	 return temp
def convert_files(given_array,train_x,train_y):
	a1=train_x[given_array].reshape(-1,28*28)
	a2=train_y[given_array]
	return (a1,a2)

batch_size=28
seed=10
l_rate=0.01
epochs=50	
cur_dir=os.getcwd()
train=pd.read_csv(cur_dir+'/train.csv')
print ("total data set="+str(len(train.filename)))
#print (train.head())
train_size=int(49000*0.75)
print("train_size="+str(train_size))
val_size=49000-train_size
print("val_size="+str(val_size))
#train_indx=np.random.randint(0,49000,train_size)
train_indx=random.sample(range(49000),train_size)
train_indx=np.array(train_indx)
print("train_indx size ="+str(train_indx.shape))
val_indx=list()
for i in range(49000):
	if i not in train_indx:
		val_indx.append(i)
val_indx=np.array(val_indx)
print("val_indx size="+str(val_indx.shape))
#train_x=train.filename[train_indx]
#val_x=train.filename[val_indx]
train_y=train.label[train_indx]
train_y=np.array(train_y)
val_y=train.label[val_indx]
val_y=np.array(val_y)
df=pd.DataFrame(val_y)
df.to_csv(cur_dir+'/val_y_values.csv',index=False,header=False)
#train_y=train_y[:train_size]
#print (train_y[:5])
temp_x=list()
for i in train.filename:
	temp_x.append(imread(cur_dir+'/Images/train/'+str(i),flatten=True))
temp_x=np.array(temp_x)
train_x=temp_x[train_indx]
val_x=temp_x[val_indx]
train_x=np.array(train_x)
val_x=np.array(val_x)
val_x=val_x.reshape(-1,28*28) # this should be commented if not saved as pandas
df=pd.DataFrame(val_x)
df.to_csv(cur_dir+'/val_x_values.csv',index=False,header=False)
#train_x=train_x[:train_size,:,:]
print ("train_x="+str(train_x.shape)) # 4900,28,28
print ("train_y="+str(train_y.shape)) #4900 rows
print("val_x"+str(val_x.shape))
print("val_y"+str(val_y.shape))
train_y=np.eye(10)[train_y] # changed to one-hot-encoding
val_y=np.eye(10)[val_y]
test=pd.read_csv(cur_dir+'/Test_fCbTej3.csv')
test_x=list()
for i in test.filename:
	test_x.append(imread(cur_dir+'/Images/test/'+str(i),flatten=True))
test_x=np.array(test_x)
print(test_x.shape[0]) #21000 training examples


inpfet=28*28;h1units=500;h2units=200;outputunits=10
x=tf.placeholder(tf.float32,[None,inpfet],name='X')
y=tf.placeholder(tf.float32,[None,outputunits],name='Y')
hidden1={'w1':tf.Variable(tf.random_normal([inpfet,h1units],seed=seed)),'b1':tf.Variable(tf.random_normal([h1units],seed=seed))}
#hidden2={'w2':tf.Variable(tf.random_normal([h1units,h2units],seed=seed)),'b2':tf.Variable(tf.random_normal([h2units],seed=seed))}
output={'w3':tf.Variable(tf.random_normal([h1units,outputunits],seed=seed)),'b3':tf.Variable(tf.random_normal([outputunits],seed=seed))}
z=tf.add(tf.matmul(x,hidden1['w1']),hidden1['b1'])
z=tf.nn.leaky_relu(z)
z1_dropout=tf.nn.dropout(z,0.5)
#z=tf.add(tf.matmul(z,hidden2['w2']),hidden2['b2'])
#z=tf.nn.leaky_relu(z)
#z2_dropout=tf.nn.dropout(z,0.45)
z=tf.add(tf.matmul(z,output['w3']),output['b3'],name='z')
#z=tf.nn.softmax(z)
cst=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z,labels=y))
optimiser=tf.train.AdamOptimizer(learning_rate=l_rate).minimize(cst)
saver=tf.train.Saver()
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for i in range(epochs):
	my_list=list()
	for j in range(train_x.shape[0]):
		my_list.append(j)
	start=0
	avg_cst=0.0
	while(start<train_x.shape[0]):
		cnt=0
		while(cnt<batch_size and start<train_x.shape[0]):
			batch_x=list()
			batch_x.append(my_list.pop())
			#else:
			#	batch_x.append(my_list.pop(0))
			cnt+=1;start+=1
		new_x,new_y=convert_files(batch_x,train_x,train_y)
		new_x=rescale_array(new_x)
		_,c=sess.run([optimiser,cst],feed_dict={x:new_x,y:new_y})
		avg_cst+=c/int(train_x.shape[0]/batch_size)
		del (batch_x)
	print ("epoch %d ,cost=%f"  %(i+1,avg_cst))


'''train_x=train_x.reshape(-1,28*28)
for i in range(epochs):
	avg_cst=0.0
	_,c=sess.run([optimiser,cst],feed_dict={x:train_x,y:train_y})
	print("epochs="+str(i)+"   cost="+str(c))
'''
print("Training complete")
cur_dir=os.getcwd()
saver.save(sess,cur_dir+'/ocr_val')
pred_temp = tf.equal(tf.argmax(z, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
print "Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, inpfet), y:val_y},session=sess)
sess.close()
#predict = tf.argmax(z, 1)
#pred = predict.eval({x: test_x.reshape(-1, input_num_units)})

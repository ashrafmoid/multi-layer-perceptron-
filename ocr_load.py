import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
cur_dir=os.getcwd()
new_g=tf.train.import_meta_graph(cur_dir+'/ocr_val.meta')
sess=tf.Session()
new_g.restore(sess,tf.train.latest_checkpoint('./'))
val_x=pd.read_csv(cur_dir+'/val_x_values.csv')
val_y=pd.read_csv(cur_dir+'/val_y_values.csv')
val_x=np.array(val_x)
val_y=np.array(val_y)
val_y=val_y.reshape(1,-1)
#val_y=np.array(val_y)
print val_y.shape
print val_x.shape
#print val_x[0]
#print val_y[:5]
val_y=np.eye(10)[val_y]
val_y=val_y[0]
print val_y.shape
#print val_y[:5]
cx=tf.get_default_graph()
x=cx.get_tensor_by_name('X:0')
y=cx.get_tensor_by_name('Y:0')
z=cx.get_tensor_by_name('z:0')
pred_temp = tf.equal(tf.argmax(z, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
print "Validation Accuracy:", accuracy.eval({x: val_x, y:val_y},session=sess)
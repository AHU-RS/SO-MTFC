import numpy as np
from utils import *
from model import *
from sklearn.metrics import r2_score

# The path of the input data
t1_path = 'F:/Day_S3/dataset5/train/t1/'  # T1 Time data path
t2_path = 'F:/Day_S3/dataset5/train/t2/'  # T2 Time data path
t3_path = 'F:/Day_S3/dataset5/train/t3/'  # T3 Time data path
label_path = 'F:/Day_S3/dataset5/train/label/'  # label path
# 样本保存路径
checkpoint_save_path = "F:/Day_S3/dataset5/checkpoint/DayS3_MSE_300.ckpt"



t1, t2, t3, label = get_data(t1_path, t2_path, t3_path, label_path)
concat_1 = tf.concat([t1, t3], 3)  # [1,24,24,2]
add_1 = tf.concat([concat_1, t2], 3)  # [1,24,24,2] + [1,24,24,1] =  [1,24,24,3]
add_2 = concat_1 + t2  # [1,24,24,2]
# 输入层 c
train = tf.concat([add_1, add_2], 3)  # [1,24,24,5] # Input the model size
train_img = np.array(train)
label_img = label


np.random.seed(5678)
np.random.shuffle(train_img)
np.random.seed(5678)
np.random.shuffle(label_img)
np.random.seed(5678)

model = MTFC_model()


# r2_score
def my_acc(y_true,y_pred):

    y_global = y_pred[:,:,:,0]
    y_global = tf.expand_dims(y_global, 3)
    mse = tf.math.reduce_mean(tf.square(y_true - y_global))
    var = tf.math.reduce_mean(tf.square(tf.math.reduce_mean(y_true) - y_global))
    r2 = 1 - mse/var
    return r2

# Calculate the number of specified values in the tensor
def tf_count(tensor, val):
    elements_equal_to_value = tf.equal(tensor, val)
    as_ints = tf.cast(elements_equal_to_value, tf.float32)
    count = tf.reduce_sum(as_ints)
    return count

def GLLoss(y_ture,y_pred):
    '''
    Global and local joint loss constraints, refer to Wu Da and Yuan Qiangqiang's "Deep Loss Reconstruction Model of Time-space Block Group"
    Global Local calculated the loss of the whole and the missing area respectively, and weighted the sum to obtain the final Loss
    :return:
    '''
    # weight
    a = 1
    b = 1
    c = 10
    mask = y_pred[:,:,:,1] # The real area is 1 and the reconstructed area is 0
    num1 = tf_count(mask, 0)
    num2 = tf_count(mask, 1)
    y_global = y_pred[:,:,:,0] # Global part
    mask = tf.expand_dims(mask, 3)
    y_global = tf.expand_dims(y_global, 3)

    y_local_p = (1-mask)*y_global # Reconstructed part
    y_local_r = mask * y_global # Real Part

    # Calculate mse by region
    global_loss = tf.reduce_mean(tf.square(tf.subtract(y_ture, y_global)))# Global Loss
    local_p_loss = tf.reduce_sum(tf.square(tf.subtract(y_ture*(1-mask), y_local_p)))/num1 # Reconstructed local losses without taking 0 into account in the calculation
    local_r_loss = tf.reduce_sum(tf.square(tf.subtract(y_ture * mask, y_local_r))) /num2  # Real local losses without taking 0 into account in the calculation

    # Make the mse of the two regions close
    err_loss = tf.abs(local_p_loss-local_r_loss) # MSE difference between the reconstructed region and the whole region

    # Total mse + reconstruction Partial mse + reconstruction difference from true mse
    gl_loss = a*local_r_loss+b*local_p_loss+c*err_loss
    gl_loss = gl_loss * 1000
    return gl_loss

def MSELoss(y_ture,y_pred):
    y_global = y_pred[:,:,:,0] # Global part
    y_global = tf.expand_dims(y_global, 3)


    global_loss = tf.reduce_mean(tf.square(tf.subtract(y_ture, y_global)))# Global Loss

    return global_loss*1000

model.compile(optimizer='adam',
              loss=[MSELoss],
              metrics=[my_acc])
# 断点续训
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True),
               tf.keras.callbacks.EarlyStopping(monitor='loss',mode='min',patience=20,restore_best_weights=True,
                                                verbose=2)]

history = model.fit(train_img, label_img, batch_size=64, epochs=300, shuffle=True, validation_split=0.1,validation_freq=1,
                    callbacks=cp_callback, verbose=2)


model.summary()

print(history.history)
###############################################    show   ###############################################
# Show the acc and loss curves for the training set and the validation set
acc = history.history['my_acc']
val_acc = history.history['val_my_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss-MSE')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
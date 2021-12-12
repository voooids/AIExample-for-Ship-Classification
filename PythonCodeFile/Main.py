# 2. Load File.
data = np.load("File Directory..")
files= data.files

# 3. The Shape of the Elements.
print('Shape - "Image" item: ' + str(data['image'].shape))
print('Shape - "Label" item: ' + str(data['label'].shape))

# 4. Let's Set a Random Index and Visualize it.
index = "Give Random Number for Prediction.."
plt.imshow(data['image'][index,:,:])
print('Ship class: ' + str(data['label'][index]))

# 5. Let's Set the Data as Input-Output..
X = data['image']
y = data['label']

# 6. Contouring According to Pictures.
inputs = tf.keras.layers.Input(shape=(128, 128, 1))

# 7. VGG16s Process.
vgg = VGG16(include_top=False,
            weights=None,
            input_tensor=inputs,
            pooling='avg')

inp = vgg.input

dense1 = tf.keras.layers.Dense(512, activation='relu')(vgg.output)
dropout1 = tf.keras.layers.Dropout(0.5)(dense1)
dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
dropout2 = tf.keras.layers.Dropout(0.5)(dense2)
pred = tf.keras.layers.Dense(3, activation='softmax')(dropout2)

model = tf.keras.Model(inp, pred)
model.summary()

opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True, name='SGD')
model.compile(optimizer=opt,
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=42)

# Learn Shape
print("X_train'in Boyutu...:",X_train.shape)
print("X_test'in Boyutu...:",X_test.shape)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

# 8. Save the Model.
model.save(' Give Directory..\\vgg.h5')

# 9. Callbacks..
check = tf.keras.callbacks.ModelCheckpoint('Give Directory..\\vgg.h5',
                                           monitor='val_accuracy',
                                           verbose=0,
                                           save_best_only=True, mode='auto')

log = tf.keras.callbacks.CSVLogger('Give Directory..\\vgg.txt')

# 10 Build Model

import multiprocessing
from timeit import default_timer as timer
start = timer()
cpu_count = multiprocessing.cpu_count()
print(f"cpu: {cpu_count} found")
model.fit(X_train, y_train,
          batch_size=12,
          epochs=2,
#          epochs=150,
          verbose=1,
          validation_data=(X_test, y_test),
          steps_per_epoch = 25,
          max_queue_size=10,
          workers=cpu_count,
          use_multiprocessing=cpu_count > 1,
          callbacks=[check, log])

end = timer()
print('Elapsed time: ' + str(end - start))


# 11. Test Acc 
_, test_acc = model.evaluate(X_test, y_test)
print('Test_acc: %.4f' % test_acc)

# 12.  Predict..
class_names =['Bulk Carrier', 'Container Ship', 'Tanker'] # Type of Ship in Data 
index = 70  # Ship we want to predict
pred = model.predict(np.array([X_test[index]]))
print(pred)

# 13. Predict our prediction..
print('Class to predict: ' + class_names[np.argmax(y_test[index])])
print('Predicted with label: ' +class_names[np.argmax(pred)])




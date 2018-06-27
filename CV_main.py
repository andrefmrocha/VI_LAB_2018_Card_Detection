from CV_Dataset import *
from CV_CNN import *
from sklearn.model_selection import train_test_split
from keras.layers import concatenate
from keras.utils import to_categorical
from keras import regularizers
x_Card, x_Camera = open_images()
list_card, list_camera, list_labels = gen_data(x_Card, x_Camera)
# 60% - train, 20%, 20% - test, validation
x_Train_Card, x_Test_Card, y_Train, y_Test= train_test_split(list_card, list_labels, test_size=0.4, random_state=1)
x_Test_Card, x_Val_Card, y_Test, y_Val = train_test_split(x_Test_Card, y_Test, test_size=0.5, random_state=1)
x_Train_Camera, x_Test_Camera= train_test_split(list_camera, test_size=0.4, random_state=1)
x_Test_Camera, x_Val_Camera= train_test_split(x_Test_Camera, test_size=0.5, random_state=1)

x_Train = {"card": x_Train_Card, "camera": x_Train_Camera}
x_Test = {"card": x_Test_Card, "camera": x_Test_Camera}
x_Val = {"card": x_Val_Card, "camera": x_Val_Camera}
input_shape = [160,160,3]
base_network, opt = create_base_network(input_shape,2,2)
input_card = Input(shape=input_shape)
input_camera = Input(shape=input_shape)

processed_card = base_network(input_card)
processed_camera = base_network(input_camera)

merged_network = concatenate([processed_card,processed_camera])
# distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_card,processed_camera])
merged_network = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(merged_network)
merged_network = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(merged_network)
merged_network = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(merged_network)
merged_network = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(merged_network)
merged_network = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(1e-3))(merged_network)

cnn_model = Model(inputs=[input_card,input_camera],outputs = merged_network)

xen_loss = losses.binary_crossentropy
y_Train = to_categorical(y_Train,2)
y_Val = to_categorical(y_Val, 2)
y_Test = to_categorical(y_Test, 2)
cnn_model.compile(loss=xen_loss, optimizer= opt, metrics=['accuracy'])


cnn_model.fit(x=[x_Train["card"], x_Train["camera"]], y=y_Train, epochs=20,
              batch_size=32, validation_data=([x_Val["card"], x_Val["camera"]], y_Val))


json_model = cnn_model.to_json()
with open("model.json",'w') as json:
    json.write(json_model)
scores = cnn_model.evaluate([x_Test["card"],x_Test["camera"]],y_Test)


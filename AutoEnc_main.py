from CV_Dataset import *
from CV_AutoEnc import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import losses, optimizers

x_Card, x_Camera = open_images()
list_card, list_camera, list_labels = gen_data(x_Card, x_Camera)
# 60% - train, 20%, 20% - test, validation))
x_Train_Card, x_Test_Card, y_Train, y_Test= train_test_split(list_card, list_labels, test_size=0.4, random_state=1)
x_Test_Card, x_Val_Card, y_Test, y_Val = train_test_split(x_Test_Card, y_Test, test_size=0.5, random_state=1)
x_Train_Camera, x_Test_Camera= train_test_split(list_camera, test_size=0.4, random_state=1)
x_Test_Camera, x_Val_Camera= train_test_split(x_Test_Camera, test_size=0.5, random_state=1)

x_Train = {"card": x_Train_Card, "camera": x_Train_Camera}
x_Test = {"card": x_Test_Card, "camera": x_Test_Camera}
x_Val = {"card": x_Val_Card, "camera": x_Val_Camera}

loss_function = losses.mean_squared_error
opt = optimizers.Adam(lr=1e-4)
encoder = create_enc_dense(input_shape)
decoder = create_dec_dense()

input= Input(shape=input_shape)
latent = encoder(input)
output = decoder(latent)
auto_encoder = Model(inputs=input, outputs= output)

auto_encoder.compile(loss=loss_function, optimizer=opt, metrics=['accuracy'])
x_Card = np.array(x_Train["card"])
x_Camera = np.array(x_Train["camera"])
x_Val["card"] = np.array(x_Val["card"])
x_Val["camera"] = np.array(x_Val["camera"])

auto_encoder.fit(x=x_Card, y=x_Camera, epochs=20, batch_size=32, validation_data=(x_Val["card"],x_Val["camera"]))

json_model = auto_encoder.to_json()
with open("auto_encoder.json",'w') as json:
    json.write(json_model)


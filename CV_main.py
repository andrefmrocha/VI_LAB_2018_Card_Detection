from CV_Dataset import *
from CV_CNN import *
from sklearn.model_selection import train_test_split
x_Card, x_Camera = open_images()
list_card, list_camera, list_labels = gen_data(x_Card, x_Camera)
# 60% - train, 20%, 20% - test, validation
x_Train_Card, x_Test_Card, y_Train, y_Test= train_test_split(list_card, list_labels, test_size=0.4, random_state=1)
x_Test_Card, x_Val_Card, y_Test, y_Val = train_test_split(x_Test_Card,y_Test, test_size=0.5, random_state=1)
x_Train_Camera, x_Test_Camera= train_test_split(list_camera, test_size=0.4, random_state=1)
x_Test_Camera, x_Val_Camera= train_test_split(x_Test_Camera, test_size=0.5, random_state=1)

x_Train = {"card": x_Train_Card, "camera": x_Train_Camera}
x_Test = {"card": x_Test_Card, "camera": x_Test_Camera}
x_Val = {"card": x_Val_Card, "camera": x_Val_Camera}

cnnmodel, opt = create_base_network([160,160,3],8,8)

cnnmodel.add(Flatten())
cnnmodel.add(Dense(512,activation='relu'))
cnnmodel.add(Dense(2,activation='softmax'))

xen_loss = losses.binary_crossentropy

cnnmodel.compile(loss=xen_loss, optimizer= opt, metrics=['acuracy'])
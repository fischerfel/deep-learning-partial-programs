from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
import keras
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense



def parse_xml(f_name='static-20171213_1938.xml'):
    tree = ET.parse(f_name)
    root = tree.getroot()
    cls = root.find("declarations").find("classDetails").findall("ce")
    class_lst = []
    method_lst = []
    for each in cls: 
        tmp = []
        class_id = each.get('id')
        class_lst.append(class_id)
        methods = each.findall("me")
    
        for item in methods:
            method_name = item.get('id')
            tmp.append(method_name)
        method_lst.append(tmp)
    return class_lst, method_lst

def prepare_data_and_label(class_lst, method_lst):
    # TODO: make sure class_lst and method_lst are the same dimension
    X = []
    y = []
    count = 0
    hash_table = {}
    for i in xrange(len(class_lst)):
        for j in xrange(len(method_lst[i])):
            y.append(i)
            X.append(count)
            hash_table[method_lst[i][j]] = count
            count += 1
            
    
    return X,y,hash_table


def get_embed(X_train, y_train, num_classes, output_dim = 64, train_epoch = 250,batch_size=256 ):
    model = Sequential()
    model.add(Dense(output_dim, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Activation('tanh'))
    model.add(Dense(num_classes, activation='softmax')) 
    model.summary()
    
    # we train it
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X_train, y_train, nb_epoch=train_epoch, batch_size=batch_size)
    model.save("embed_model.hdf5")
    model2 = Sequential()
    model2.add(Dense(output_dim, activation='relu', input_shape=(X_train.shape[1],), weights=model.layers[0].get_weights()))
    model2.add(Activation('tanh'))
    #model2.compile(loss='categorical_crossentropy', optimizer='sgd')
    activations = model2.predict(X_train)
    return activations

if __name__ == "__main__":
    class_lst, method_lst = parse_xml()
    
    X,y,h = prepare_data_and_label(class_lst, method_lst)
    
    onehot_X = keras.utils.to_categorical(X)
    onehot_y = keras.utils.to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(onehot_X, onehot_y,test_size=0, random_state=42,shuffle=True) 
    
    num_classes = len(class_lst)
    output_dim = 64
    
    embed_vec = get_embed(X_train, y_train, num_classes = num_classes,output_dim = output_dim)

    dictionary = {}
    X_train_id = np.argmax(X_train, axis=1)
    for i in xrange(embed_vec.shape[0]):
        dictionary[h.keys()[h.values().index(X_train_id[i])]] = embed_vec[i]
    
    # save dictionary     
    np.save('embedding.npy', dictionary) 
    print ("saved embedding vectors")
    
    # load dictionary 
    #read_dictionary = np.load('embedding.npy').item()
        
    

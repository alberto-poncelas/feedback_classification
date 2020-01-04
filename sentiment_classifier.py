

from sklearn import preprocessing

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.utils import np_utils
import pandas as pd
import numpy as np
import re


### Preprocess data

MAXLEN=30
#word index
tokenizer = Tokenizer(num_words=300)
#label categories
le = preprocessing.LabelEncoder()


def process_data(filepath,type_process):
	df = pd.read_csv(filepath,sep="\t")
	df.columns = ["sentence","sentiment"]
	x = df['sentence'].values
	x=[re.sub(r'[^a-z0-9\s]',"",b.lower()) for b in x]
	y = df['sentiment'].values
	if type_process=="train":
		tokenizer.fit_on_texts(x)
	vocab_size=len(tokenizer.word_index)+1
	#text to word index
	xdata= tokenizer.texts_to_sequences(x)
	#add padding to the sentences
	xdata=pad_sequences(xdata, padding='post', maxlen=MAXLEN)
	#preprocess class
	if type_process=="train":
		le.fit(y)
	encoded_y=le.transform(y)
	y_class = np_utils.to_categorical(encoded_y)
	return [xdata,y_class]


[xtrain,y_train]=process_data('data/train.txt',"train")
[xtest,y_test]=process_data('data/test.txt',"test")



### Create model

embedding_dim=100
vocab_size=len(tokenizer.word_index)+1

model=Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
         output_dim=embedding_dim,
         input_length=MAXLEN))
model.add(layers.LSTM(units=64,return_sequences=True))
model.add(layers.LSTM(units=16))
model.add(layers.Dense(3, activation="softmax"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
model.summary()


### Train model

num_epochs=10
history = model.fit(
    xtrain,
    y_train,
    epochs=num_epochs,
    batch_size=4,
    validation_split=0.1,
    verbose = True,
    shuffle=True
)


loss = history.history['loss']
val_loss = history.history['val_loss']


### Evaluate test set

ypred=model.predict(xtest)
actual_numeric_category=[np.argmax(x) for x in y_test]
predicted_numeric_category=[np.argmax(x) for x in ypred]


### Create confusion matrix
data = {'actual': actual_numeric_category,
        'predicted': predicted_numeric_category}

df = pd.DataFrame(data, columns=['actual','predicted'])
confusion_matrix = pd.crosstab(df['actual'], df['predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


predicted_categories=list(le.inverse_transform(predicted_numeric_category))
print(predicted_categories)


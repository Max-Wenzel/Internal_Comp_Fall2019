import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split as tts
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, Bidirectional
from keras.optimizers import SGD
import matplotlib.pyplot as plt
print("reading data")
train = pd.read_csv("training.csv")
treat = train["treatment_condition"].copy()
wells = train["well_id"].copy()
ttrac = pd.read_csv("normttrac.csv")
test = pd.read_csv("normtest.csv")
# ttrac["well"] = wells
# ttrac["treat"] = treat
onetreat = []
treat = treat[:len(treat) - 1]
print("filling out")
for ii in range(len(treat)):
    ent = treat.iloc[ii]#.iloc[0]
    ne = [0,0,0]
    ne[ent] = 1
    onetreat.append(ne)
print("making model")
model = Sequential()
model.add(Bidirectional(LSTM(3, input_shape=(600,1), return_sequences=True), merge_mode="ave"))
model.add(Bidirectional(LSTM(3), merge_mode="ave"))
model.add(Activation("softmax",input_shape=(3,)))

# com = list(zip(ttrac,onetreat))
# np.random.shuffle(com)
# ttrac[:], onetreat[:] = zip(*com)
X_train = np.array(ttrac)
y_train = np.array(onetreat)

#X_train, X_test, y_train, y_test = tts(np.array(ttrac),np.array(onetreat))

X_train = np.expand_dims(X_train,3)

X_test = np.expand_dims(np.array(test),3)


print("compile")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print("fit")
model.fit(X_train, y_train, verbose=1,epochs=8)
print("predict")
res = model.predict(X_test)
pd.DataFrame(res).to_csv("results.csv", header=None)
from keras.models import Sequential
from keras.layers import Dense
import numpy

numpy.random.seed(7)
# **** Load Data ****
dataset = numpy.loadtxt("Data/pima-indians-diabetes.csv",delimiter=",")
# **** Split Data into X And Output Y Variable ****
X = dataset[:,0:8]
Y = dataset[:,8]

# **** CREATE MODEL ****
model = Sequential()
model.add(Dense(12, input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# **** Compile Modal ****
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y,epochs=150 , batch_size=10, verbose=2)

# **** EVALUATES Result ****
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# **** Custom Data ****
data = numpy.array([[1,126,60,0,0,30.1,0.349,47],[1,85,66,29,0,26.6,0.351,31]])
# **** Prediction ****
predictions = model.predict(data)
rounded = [round(x[0]) for x in predictions]

print("PREDICTED DATA", rounded)

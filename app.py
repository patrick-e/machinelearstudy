from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy


#pelo longo?
#perna curta?
#faz au au?

pig = [0,1,0]
pig2 = [0,1,1]
pig3 = [1,1,0]

dog = [0,1,1]
dog2 = [1,0,1]
dog3 = [1,1,1]


training_x = [pig,pig2,pig3,dog,dog2,dog3]
training_y = [1,1,1,0,0,0] #dog 0 pig 1

modelo = LinearSVC()
print(modelo.fit(training_x,training_y))

        
elemento_X = [1,1,1] 
elemento_Y = [1,1,0]
elemento_Z = [0,1,1]
print(modelo.predict([elemento_X,elemento_Y,elemento_Z]))


prevision = modelo.predict(training_x)
prevision

print(accuracy_score(training_y,prevision))
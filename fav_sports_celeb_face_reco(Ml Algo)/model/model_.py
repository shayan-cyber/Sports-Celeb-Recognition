import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pywt


#importing haar cascade for face detection(We won't detect eyes)
face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')

def get_cropped_imgs_with_face(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting the img to gray
    faces_cropped = face_cascade.detectMultiScale(gray_img, 1.3,5) #returns an array of dimension of the crop..we use gray image for more efficiency
    #it can return mutiple cropped imgs so we will use for loop
    for (x,y,w,h) in faces_cropped:
        roi_gray= gray_img[y:y+h, x:x+w] # cropping for gray image return an array representing the img
        roi_color = img[y:y+h, x:x+w]
        # print(roi_gray)
        return roi_color


# print(get_cropped_imgs_with_face('./test_images/sharapova1.jpg'))

#generating cropped folder
path_to_data = './dataset/'
path_to_cropped = './dataset/cropped/'
#making a list out of raw imgs

raw_img_dirs = []

import os
for entry in os.scandir(path_to_data):
    if entry.is_dir(): #checking if it is directory or not
        if entry.path != './dataset/cropped/': #ignoring the produced cropped folder for multiple running
            raw_img_dirs.append(entry.path)

#making cropped folders
import shutil
if os.path.exists(path_to_cropped):
    shutil.rmtree(path_to_cropped)
os.mkdir(path_to_cropped)




cropped_img_dirs =[]
celeb_name_dict ={}

print(raw_img_dirs)
for img_dir in raw_img_dirs:
    if img_dir != '.dataset/cropped':
        print(img_dir)
        count =1
        celebrity_name = img_dir.split('/')[-1]
        print(celebrity_name)
        celeb_name_dict[celebrity_name] =[]
        for entry in os.scandir(img_dir):
            print(entry)
            roi_color = get_cropped_imgs_with_face(entry.path)
            if roi_color is not None:
                cropped_folder = path_to_cropped + celebrity_name
                if not os.path.exists(cropped_folder):
                    os.mkdir(cropped_folder)
                    cropped_img_dirs.append(cropped_folder)

                cropped_file_name = celebrity_name + str(count) + '.png'
                cropped_file_path = cropped_folder + '/' + cropped_file_name

                cv2.imwrite(cropped_file_path,roi_color)
                print(cropped_file_path)
                celeb_name_dict[celebrity_name].append(cropped_file_path)
                count = count +1
print(celeb_name_dict)



#converting to wavelet
import pywt
# we'll use both raw and wavelet transformed image and stack them together vertically to feed to the classifier
def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

#generating no for each celeb
class_dict ={}
cnt = 0

for entry in celeb_name_dict.keys():
    class_dict[entry] = cnt
    cnt = cnt+1


print(class_dict)

X = []
y =[]


for celeb_name , training_files in celeb_name_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image) #returns array
        print(img)
        if img is None:
            continue
        scalled_raw_img = cv2.resize(img, (32,32))
        img_har = w2d(img, 'db1',5)
        scalled_har_img = cv2.resize(img_har, (32,32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3, 1), scalled_har_img.reshape(32*32*1,1)))
        X.append(combined_img)
        y.append(class_dict[celeb_name])

print(len(X))
print(len(y))
print(len(X[0]))
#converting the X list to an array of (168,4096)

X= np.array(X).reshape(len(X),4096).astype(float)

#training model
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import  make_pipeline
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train,y_test = train_test_split(X,y, random_state=0,test_size=0.05)


#hypertuning

model_params ={
    'Logistic_regression':{
        'model':LogisticRegression(solver='liblinear', multi_class='auto'),
        'params':{
            'logisticregression__C':[1,5,10]
        }

    },
    'Random_forest':{
        'model':RandomForestClassifier(),
        'params':{
            'randomforestclassifier__n_estimators':[1,5,10,20]
        }
    },
    'SVM':{
        'model':SVC(gamma='auto', probability=True),
        'params':{
            'svc__C':[1,5,10,15],
            'svc__kernel':['linear', 'poly', 'rbf', ]
        }
    },
    'KNN':{
        'model':KNeighborsClassifier(),
        'params':{
            'kneighborsclassifier__n_neighbors':[5,10,20],
            'kneighborsclassifier__algorithm':['auto', 'ball_tree','kd_tree','brute']


        }
    }
}
scores =[]
best_estimators ={}
for algo_names, algo in model_params.items():
    pipe =make_pipeline(StandardScaler(), algo['model'])
    clf= GridSearchCV(pipe, algo['params'],cv=5, return_train_score=False)
    clf.fit(X_train,y_train)
    scores.append({
        'model':algo_names,
        'best_score':clf.best_score_,
        'best_param':clf.best_params_,

    })
    best_estimators[algo_names] = clf.best_estimator_
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
df = pd.DataFrame(scores,columns=['model','best_score','best_param' ])
print(df)
#X_test ,y_test not used cause these will be used in best estimator(trained model)

print(best_estimators['SVM'].score(X_test,y_test))#score on test set
print(best_estimators['Logistic_regression'].score(X_test,y_test))

#logistic performs best but shows less accuracy
best_clf = best_estimators['SVM']
from sklearn.metrics import confusion_matrix
import seaborn as sn
cm = confusion_matrix(y_test, best_clf.predict(X_test))

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()





import joblib
joblib.dump(best_clf,"model_svm.pkl")

import json
with open('class_dict.json','w') as  f:
    f.write(json.dumps(class_dict))
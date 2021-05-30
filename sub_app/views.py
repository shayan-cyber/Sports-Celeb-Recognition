from django.shortcuts import render
import cv2
import numpy as np
import base64
import joblib
import sklearn

# Create your views here.
def home(request):
    return render(request, 'home.html')


def classify_img(request):
    if request.method =='POST':
        print("Post")
        base_ = request.POST.get('img_string', '')
        pred= clf_img(base_)
        keys_celeb = ['Christiano Ronaldo', 'Lionel Messi','Mahendra Singh Dhoni',  'Maria Sharapova', 'Roger Federer','Sachin Tendulkar', 'Serena Williams', 'Virat Kohli']
        print(clf_img(base_))
        if len(pred) !=0:  
            pred_no = pred[0]['class']
            proba = int(float(pred[0]['class_proba'][pred_no])*100)
            pred = keys_celeb[pred_no]
            
            context ={
                'prediction':pred,
                'keys':keys_celeb,
                'probability':proba

            }
        else:
            context={
                'err':"Can't Detect Due To Either Small Image Or Failling To Detect Eyes "
            }
    return render(request, 'classify.html',context)

def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    return img


def get_cropped_images_if_2_eyes(image_base64_data):
    print("in cropped")
    face_cascade = cv2.CascadeClassifier('static/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('static/opencv/haarcascades/haarcascade_eye.xml')
   
    img = get_cv2_image_from_base64_string(image_base64_data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5) #detects face from the image returns dimension of the face
    print(img)
    print("faces")
    print(faces)

    cropped_faces =[]



    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        print("eyes")
        print(eyes)
        # if len(eyes) >=2:
        cropped_faces.append(roi_color)
    print("cropped faces")
    print(cropped_faces)
    return cropped_faces

def clf_img(image_base64_data):
    imgs = get_cropped_images_if_2_eyes(image_base64_data)
    print(imgs)
    result =[]
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32,32)) #scalling
        img_har = w2d(img,'db1', 5) #wavelet transform
        scalled_img_har = cv2.resize(img_har, (32,32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1) ))
        len_image_array= 32*32 + 32*32*3
        final = combined_img.reshape(1,len_image_array).astype(float)
        with open('static/model_svm.pkl','rb') as f :
            model_ = joblib.load(f)
        result.append({
            'class':model_.predict(final)[0],
            'class_proba': np.round(model_.predict_proba(final),2).tolist()[0],

        })
    return result
   
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

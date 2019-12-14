import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import pyzbar.pyzbar as pyzbar
from pyzbar.pyzbar import decode
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras.models import load_model

answer = ["1","2","3","4","5","6","7","8","9","0","1","2","3","4","5","6","7","8","9","0"]

def read_pred(img_read):
    gray_img = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
    
    #when the img is too dark, do this
    gamma =2
    gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    gray_img = cv2.LUT(gray_img,gamma_table)
    
    cv2.imwrite('final/gray_img.jpg',gray_img)

    
    blurred_img = cv2.GaussianBlur(gray_img, (5,5),0)
    barcodes = pyzbar.decode(blurred_img)
    model = load_model('model_mnist.h5')
    
    
    #the picture's number
    i=1
    #fl_1 = 0.00
    #fl_2 = 0.00
    
    ans_correct  = 0
    ans_false = []

    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
            
        #read the QRcode  and  make sure if that all of the Qrcode has been read 
        crop_qr = gray_img[y:y+h,x:x+w ] 
        cv2.imwrite("final/QR_{}.jpg".format(i), crop_qr)

        #cut the QRcode and save
        cv2.rectangle(img_read, (x , y ), (x + w , y + h ), (255, 255, 0), 2)        
        barcodeData = barcode.data.decode("utf-8")
        img_PIL = Image.fromarray(cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB))
        img_PIL.save('final/check the QRcode.jpg', 'jpeg')

        #binary image 
        crop_cut = gray_img[y:y+h ,x+w+5:x+w*2+5]
        #If the font is too small, do this
        kernel= cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        crop_cut = cv2.erode(crop_cut, kernel)
        th, binary = cv2.threshold(crop_cut,125, 255, cv2.THRESH_BINARY)
        cv2.imwrite('final/crop_cut_binary_{}.jpg'.format(i),binary)

        #center the number and padding 
        binary2black = 255 - binary 
        
        
        #Finding font outlines and padding
        _, thresh = cv2.threshold(binary, 125, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, 3, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        X, Y, W, H = cv2.boundingRect(cnt) 
        rectangle = binary2black[Y:Y+H,X:X+W]
        cv2.imwrite('final/rectangle_{}.jpg'.format(i),rectangle)
        
        #select the value of padding
        rec_shape = rectangle.shape[0]/rectangle.shape[1]
        if 0 <= rec_shape <= 1.3 :
            Padding = cv2.copyMakeBorder(rectangle,round(H*0.33),round(H*0.26),round(W*0.6),round(W*0.6),cv2.BORDER_CONSTANT,value=[0,0,0])
        elif rec_shape <= 4:
            Padding = cv2.copyMakeBorder(rectangle,round(H*0.33),round(H*0.26),round(W*0.8),round(W*0.8),cv2.BORDER_CONSTANT,value=[0,0,0])
        else :
            Padding = cv2.copyMakeBorder(rectangle,round(H*0.33),round(H*0.26),round(W*5),round(W*5),cv2.BORDER_CONSTANT,value=[0,0,0])
        #Padding = cv2.copyMakeBorder(rectangle,round(H*0.3),round(H*0.3),round(W*0.8),round(W*0.8),cv2.BORDER_CONSTANT,value=[0,0,0])
        #Padding = cv2.copyMakeBorder(rectangle,50,50,80,80,cv2.BORDER_CONSTANT,value=[0,0,0])
        cv2.imwrite('final/Padding_{}.jpg'.format(i),Padding)

        #Resize and sharpen image
        resized = cv2.resize(Padding,(28,28),interpolation=cv2.INTER_AREA)     
        cv2.imwrite('final/resized_{}.jpg'.format(i), resized)

        #increase the picture's number
        i += 1

        #Predict and show the result
        pred = model.predict(resized.reshape(1, 28, 28)).argmax()            
        print("No.{0} question's answer is {1}".format(int(barcodeData)+1,pred))

        qr_que = int(barcodeData)
        qr_ans = int(pred)
        true_ans = int(answer[qr_que])
                 
        if qr_ans == true_ans:
            #print('The answer is correct')
            ans_correct += 1
        else:
            #print('The answer is incorrect')
            ans_false.append(qr_que+1)
            
        #Check if the value of padding is appropriate
        #rec_1 = rectangle.shape[0]
        #rec_2 = rectangle.shape[1]
        #pad_1 = Padding.shape[0]
        #pad_2 = Padding.shape[1]
        #pad_shape = pad_1/pad_2
        #f_1 =float(pad_1/rec_1)
        #f_2 =float(pad_2/rec_2) 
        #fl_1 += f_1
        #fl_2 += f_2
        #print('f_1 = %.2f' % f_1)
        #print('f_2 = %.2f' % f_2)
        #print('fl_1 = %.2f' % float(fl_1/20))
        #print('fl_2 = %.2f' % float(fl_2/20))
        #print(rec_1/rec_2)
        #print('rec_shape = %.2f' % rec_shape)
        #print('pad_shape = %.2f\n' % pad_shape)

    
    x1 = (ans_correct / 20.0) * 100
    ans_false.sort()

    print('Predict false: {}'.format(ans_false))
    print('If the prediction is 100% correct. The correct rate of answer is {} %'.format(x1))
                     
img_read = cv2.imread('img/test.jpg')
read_pred(img_read)

import rawpy
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
#Converting the .nef files to .tiff processed files
raw1 = rawpy.imread('exposure1.nef')
pro1=raw1.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro1=cv2.cvtColor(pro1,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure1.tiff', pro1)

raw2 = rawpy.imread('exposure2.nef')
pro2=raw2.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro2=cv2.cvtColor(pro2,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure2.tiff', pro2)

raw3 = rawpy.imread('exposure3.nef')
pro3=raw3.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro3=cv2.cvtColor(pro3,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure3.tiff', pro3)

raw4 = rawpy.imread('exposure4.nef')
pro4=raw4.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro4=cv2.cvtColor(pro4,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure4.tiff', pro4)

raw5 = rawpy.imread('exposure5.nef')
pro5=raw5.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro5=cv2.cvtColor(pro5,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure5.tiff', pro5)

raw6 = rawpy.imread('exposure6.nef')
pro6=raw6.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro6=cv2.cvtColor(pro6,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure6.tiff', pro6)

raw7 = rawpy.imread('exposure7.nef')
pro7=raw7.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro7=cv2.cvtColor(pro7,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure7.tiff', pro7)

raw8 = rawpy.imread('exposure8.nef')
pro8=raw8.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro8=cv2.cvtColor(pro8,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure8.tiff', pro8)

raw9 = rawpy.imread('exposure9.nef')
pro9=raw9.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro9=cv2.cvtColor(pro9,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure9.tiff', pro9)

raw10 = rawpy.imread('exposure10.nef')
pro10=raw10.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro10=cv2.cvtColor(pro10,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure10.tiff', pro10)

raw11 = rawpy.imread('exposure11.nef')
pro11=raw11.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro11=cv2.cvtColor(pro11,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure11.tiff', pro11)

raw12 = rawpy.imread('exposure12.nef')
pro12=raw12.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro12=cv2.cvtColor(pro12,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure12.tiff', pro12)

raw13 = rawpy.imread('exposure13.nef')
pro13=raw13.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro13=cv2.cvtColor(pro13,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure13.tiff', pro13)

raw14 = rawpy.imread('exposure1.nef')
pro14=raw14.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro14=cv2.cvtColor(pro14,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure14.tiff', pro14)

raw15 = rawpy.imread('exposure15.nef')
pro15=raw15.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro15=cv2.cvtColor(pro15,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure15.tiff', pro15)

raw16 = rawpy.imread('exposure16.nef')
pro16=raw16.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)
pro16=cv2.cvtColor(pro16,cv2.COLOR_RGB2BGR)
cv2.imwrite('processed_exposure16.tiff', pro16)

"""
def time(k):
    return(np.power(2,k-1)/2048)

def weight(z):
    return(np.exp((-4*np.power(z-0.5,2))/0.25))

def mainfunction():


    img1 = cv2.imread('processed_exposure1.tiff')
    img1=cv2.resize(img1,None,fx=0.25,fy=0.25)
    img1=img1.astype(np.float16)
    img1 = img1/ 65535


    img2 = cv2.imread('processed_exposure2.tiff')
    img2 = cv2.resize(img2, None, fx=0.25, fy=0.25)
    img2 = img2.astype(np.float16)
    img2 = img2 / 65535

    img3 = cv2.imread('processed_exposure3.tiff')
    img3 = cv2.resize(img3, None, fx=0.25, fy=0.25)
    img3 = img3.astype(np.float16)
    img3 = img3 / 65535

    img4 = cv2.imread('processed_exposure4.tiff')
    img4 = cv2.resize(img4, None, fx=0.25, fy=0.25)
    img4 = img4.astype(np.float16)
    img4 = img4 / 65535

    img5 = cv2.imread('processed_exposure5.tiff')
    img5 = cv2.resize(img5, None, fx=0.25, fy=0.25)
    img5 = img5.astype(np.float16)
    img5 = img5 / 65535

    img6 = cv2.imread('processed_exposure6.tiff')
    img6 = cv2.resize(img6, None, fx=0.25, fy=0.25)
    img6 = img6.astype(np.float16)
    img6 = img6 / 65535

    img7 = cv2.imread('processed_exposure7.tiff')
    img7 = cv2.resize(img7, None, fx=0.25, fy=0.25)
    img7 = img7.astype(np.float16)
    img7 = img7 / 65535

    img8 = cv2.imread('processed_exposure8.tiff')
    img8 = cv2.resize(img8, None, fx=0.25, fy=0.25)
    img8 = img8.astype(np.float16)
    img8 = img8 / 65535

    img9 = cv2.imread('processed_exposure9.tiff')
    img9 = cv2.resize(img9, None, fx=0.25, fy=0.25)
    img9 = img9.astype(np.float16)
    img9 = img9 / 65535


    img10 = cv2.imread('processed_exposure10.tiff')
    img10 = cv2.resize(img10, None, fx=0.25, fy=0.25)
    img10 = img10.astype(np.float16)
    img10 = img10 / 65535

    img11 = cv2.imread('processed_exposure11.tiff')
    img11 = cv2.resize(img11, None, fx=0.25, fy=0.25)
    img11 = img11.astype(np.float16)
    img11 = img11 / 65535

    img12 = cv2.imread('processed_exposure12.tiff')
    img12 = cv2.resize(img12, None, fx=0.25, fy=0.25)
    img12 = img12.astype(np.float16)
    img12 = img12 / 65535

    img13 = cv2.imread('processed_exposure13.tiff')
    img13= cv2.resize(img13, None, fx=0.25, fy=0.25)
    img13 = img13.astype(np.float16)
    img13 = img13 / 65535

    img14 = cv2.imread('processed_exposure14.tiff')
    img14 = cv2.resize(img14, None, fx=0.25, fy=0.25)
    img14 = img14.astype(np.float16)
    img14 = img14 / 65535

    img15 = cv2.imread('processed_exposure15.tiff')
    img15 = cv2.resize(img15, None, fx=0.25, fy=0.25)
    img15 = img15.astype(np.float16)
    img15 = img15 / 65535

    img16 = cv2.imread('processed_exposure16.tiff')
    img16 = cv2.resize(img16, None, fx=0.25, fy=0.25)
    img16 = img16.astype(np.float16)
    img16 = img16 / 65535

    listofimages = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12, img13, img14, img15, img16]



    #=np.zeros([4016,6016,3])
    I=np.zeros([1004,1504,3])
    #I=np.zeros([1000,1000,3])
    for i in range(0,1004):#g1.shape[0])
        for j in range(0,1504): #img1.shape[1]
            for c in range(0,3):
                s=0
                s1=0
                for k in range(0,16):
                    x=listofimages[k]#Reteiving the image from the list
                    t=time(k+1)#Calculating thr time
                    value=x[i,j,c]#Storing the value
                    weightk=weight(x[i,j,c])#Calculating weight
                    s=s+(weightk*value)/t
                    s1=s1+weightk
                I[i,j,c]=s/s1
                print(I[i,j,c],i,j,c)

    #Performing Photographic tone mapping

    tonemap=np.divide(I,(1+I)) #Using the formula IHDR/1+IHDR
    tonemap8=(tonemap/np.max(tonemap)*255).astype(np.uint8) #Performing the conversion to uint8
    cv2.imwrite('Photographic_tonemapping.png',tonemap8) #Displaying the image

    #Performing tonemapping using the opencv functions
    I = I.astype(np.float32) #Change to float32
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.7, -0.1)#Creating object of the tonemap class
    ldrMantiuk = tonemapMantiuk.process(I) #Performing the process of the mapping
    cv2.imwrite('opencv_tonemapping2.png', ldrMantiuk)


mainfunction()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
'''
define the fuction of converting RGB image into gray image
'''
def rgb2gray(image):
    return np.round(0.2989*image[:,:,0]+0.5870*image[:,:,1]+0.1140*image[:,:,2])


'''
compute the integral image representation
'''
def integralimage(x):
    total_num=x.shape[0]
    iiimages=np.zeros((total_num,65,65))
    for j in range(total_num):
        for k in range(65):
            for l in range(65):
                if k==0 or l==0:
                    iiimages[j,k,l]=0
                else:
                    iiimages[j,k,l]=np.sum(x[j,:k,:l])
    return iiimages


'''
compute the coordinates of Harr-like feature
'''
def feature_coord(width,height,step):
    v_num=int((64-height)/step)
    h_num=int((64-width)/step)
    feature=[]
    for l in range(h_num+1):
        for j in range(v_num+1):
            if width*2+step*l-1<64:
                coord=list(map(int,[step*j,step*l,height+step*j-1,width+step*l-1,\
                                    step*j,width+step*l,height+step*j-1,width*2+step*l-1]))
                feature.append(coord)
            if height*2+step*j-1<64:
                coord=list(map(int,[step*j,step*l,height+step*j-1,width+step*l-1,\
                                    height+step*j,step*l,height*2+step*j-1,width+step*l-1]))
                feature.append(coord)
    return feature


'''
compute the fth feature value of ith images
'''
def computeFeature(iimages,featurebl,i,f):
    iimage=iimages[i]
    feature=featurebl[f]
    x1,y1,x11,y11,x2,y2,x22,y22=feature
    black=iimage[x1,y1]+iimage[x11+1,y11+1]-iimage[x1,y11+1]-iimage[x11+1,y1]
    white=iimage[x2,y2]+iimage[x22+1,y22+1]-iimage[x2,y22+1]-iimage[x22+1,y2]
    return (black-white)/(x11-x1+1)/(y11-y1+1)


'''
stump decision learner for given fth feature and weight
'''
def singleLearner(iimages,featurebl,f,w,label):
    total_num=len(iimages)
    temp=np.zeros(total_num)
    for l in range(total_num):
        temp[l]=computeFeature(iimages,featurebl,l,f)
    index=np.argsort(temp)
    new=w[index]
    newlabel=label[index]
    tem=temp[index]
    epsilon=np.zeros(total_num+1)
    p=np.zeros(total_num+1)
    tplus=np.sum(new[newlabel==1])
    tminus=np.sum(new[newlabel==-1])
    for j in range(total_num+1):
        if j==0:
            splus=0
            sminus=0
        else:
            if newlabel[j-1]==1:
                splus+=new[j-1]
            else:
                sminus+=new[j-1]
        a=splus+tminus-sminus
        b=sminus+tplus-splus
        if a<b:
            epsilon[j]=a
            p[j]=1
        else:
            epsilon[j]=b
            p[j]=-1
    
    min_ind=np.argmin(epsilon)
    if len(tem)==0:
        print(len(iimages))
    if min_ind==0:
        theta=tem[0]-0.00001
    elif min_ind==total_num:
        theta=tem[min_ind-1]+0.00001
    else:
        theta=(tem[min_ind-1]+tem[min_ind])/2
        
    return epsilon[min_ind],theta,p[min_ind]


'''
best weaker learner for given weight over all features
'''
def bestLearner(iimages,featurebl,w,label):
    feature_num=len(featurebl)
    error=np.zeros(feature_num)
    p=np.zeros(feature_num)
    theta=np.zeros(feature_num)
    for j in range(feature_num):
        error[j],theta[j],p[j]=singleLearner(iimages,featurebl,j,w,label)
    min_ind=np.argmin(error)
    return min_ind,error[min_ind],theta[min_ind],p[min_ind]


'''
compute false positive and false negative
'''
def PN(y,label):
    nsample=len(label)
    fn=0
    fp=0
    for j in range(nsample):
        if y[j]==-1 and label[j]==1:
            fn+=1
        if y[j]==1 and label[j]==-1:
            fp+=1
    n1=np.sum(label==-1)
    n2=np.sum(label==1)
    return fp/n1,fn/n2


'''
compute the labels for best weaker learner
'''
def h(iimages,featurebl,feature_ind,theta,p,sample_ind):
    num=len(sample_ind)
    temp=np.zeros(num)
    for j in range(num):
        temp[j]=computeFeature(iimages,featurebl,sample_ind[j],feature_ind)
    return np.sign(p*(temp-theta))


'''
compute the labels for adaptive booster
'''
def boosth(iimages,featurebl,alpha,feature_ind,theta,p,Theta):
    f0=0
    T=len(alpha)
    total_num=len(iimages)
    for t in range(T):
        f0+=alpha[t]*h(iimages,featurebl,feature_ind[t],theta[t],p[t],np.arange(total_num))
    h0=np.sign(f0-Theta)
    return h0


'''
train the Adaboost
'''
def Adaboost(iimages,featurebl,T_max,label):
    total_num=len(iimages)
    start=datetime.datetime.now()
    m=np.sum(label==1)
    n=np.sum(label==-1)
    w=np.array([1/2/m]*m+[1/2/n]*n)
    w=w/np.sum(w)
    D=np.zeros((T_max+1,total_num))
    D[0,:]=w
    feature_ind=np.zeros(T_max,dtype="int")
    error=np.zeros(T_max)
    theta=np.zeros(T_max)
    p=np.zeros(T_max,dtype="int")
    alpha=np.zeros(T_max)
    fp=1
    t=0
    f0=0
    while fp>=0.3 and t<T_max:
        feature_ind[t],error[t],theta[t],p[t]=bestLearner(iimages,featurebl,D[t,:],label)
        alpha[t]=0.5*np.log((1-error[t])/error[t])
        for l in range(total_num):
            predict=h(iimages,featurebl,feature_ind[t],theta[t],p[t],[l])
            D[t+1,l]=D[t,l]*np.exp(-alpha[t]*label[l]*predict)
        D[t+1,:]=D[t+1,:]/np.sum(D[t+1,:])
        f0+=alpha[t]*h(iimages,featurebl,feature_ind[t],theta[t],p[t],np.arange(total_num))
        Theta=min(f0[label==1])-0.01
        h0=np.sign(f0-Theta)
        fp,fn=PN(h0,label)
        t=t+1
    end=datetime.datetime.now()
    print(end-start)
    
    return fp,alpha[:t],feature_ind[:t],theta[:t],p[:t],Theta


'''
train the cascade classifier
'''
def cascade(iimages,featurebl,label):
    fps=[]
    alphas=[]
    feature_inds=[]
    thetas=[]
    ps=[]
    Thetas=[]
    errors=[]
    iter_num=0
    total=len(iimages)
    index=np.arange(total)
    detection_error=100
    while detection_error>20:
        fp,alpha,feature_ind,theta,p,Theta=Adaboost(iimages[index],featurebl,40,label[index])
        y=boosth(iimages[index],featurebl,alpha,feature_ind,theta,p,Theta)
        detection_error=np.sum(y!=label[index])
        index=index[y==1]
        fps.append(fp)
        alphas.append(alpha)
        feature_inds.append(feature_ind)
        thetas.append(theta)
        ps.append(p)
        Thetas.append(Theta)
        errors.append(detection_error)
        iter_num+=1
    return alphas,feature_inds,thetas,ps,Thetas,errors,index


'''
get the test images of class.jpg
'''
def getslidewindow(test_image,step):
    size=test_image.shape
    h_num=int((size[1]-64)/step)
    v_num=int((size[0]-64)/step)
    n=(h_num+1)*(v_num+1)
    images=np.zeros((n,64,64))
    count=0
    for vv in range(v_num):
        for hh in range(h_num):
            image=test_image[step*vv:(step*vv+64),step*hh:(step*hh+64)]
            images[count]=image
            count+=1
    return images


'''
detect the face of test images
'''
def detectface(iimages,alphas,feature_inds,thetas,ps,Thetas):
    detector_num=len(alphas)
    total=len(iimages)
    index=np.arange(total)
    for j in range(detector_num):
        y=boosth(iimages,featurebl,alphas[j],feature_inds[j],thetas[j],ps[j],Thetas[j])
        index=index[y==1]
        iimages=iimages[y==1]
    return index


'''
plot the rectangle on the detected faces
'''
def drawrectangle(test_image,index,size,step):
    n=len(index)
    h_num=int((size[1]-64)/step)
    fig,ax = plt.subplots()
    ax.imshow(test_image,cmap='gray')
    for j in range(n):
        a=int(index[j]/(h_num+1))
        b=index[j]-a*(h_num+1)
        if b==0:
            x=step*h_num
            y=step*(a-1)
        else:
            x=step*(b-1)
            y=step*a
        rect = patches.Rectangle((x,y),64,64,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

if __name__ == "__main__":

    # read the face images and non-face images
    face_num = 2000
    nonface_num = 2000
    total_num = face_num + nonface_num
    label = np.array([1] * face_num + [-1] * nonface_num)
    x = np.zeros((total_num, 64, 64))

    for j in range(face_num):
        filename = 'face{0:d}.jpg'.format(j)
        x[j] = rgb2gray(plt.imread(filename))
    for j in range(nonface_num):
        filename = '{0:d}.jpg'.format(j)
        x[j + face_num] = rgb2gray(plt.imread(filename))

    # compute the integral image representation of train data
    iimages = integralimage(x)

    # compute features
    width = np.arange(4, 50, 4)
    height = np.arange(4, 50, 4)
    featurebl = []
    for ww in width:
        for hh in height:
            feature = feature_coord(ww, hh, 4)
            featurebl += feature

    # train the cascade classifier
    alphas, feature_inds, thetas, ps, Thetas, errors, index = cascade(iimages, featurebl, label)

    # read the test images
    test_image = plt.imread('class.jpg')
    step = 18
    newimages = getslidewindow(test_image, step)
    test_images = integralimage(newimages)

    # plot the predicted result
    size = test_image.shape
    index = detectface(test_images, alphas, feature_inds, thetas, ps, Thetas)
    drawrectangle(test_image, index, size, step)





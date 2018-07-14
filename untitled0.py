import numpy as np
import cv2

image=cv2.imread("digits.png")

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

small=cv2.pyrDown(image)

cv2.imshow("gray",small)
cv2.waitKey(0)
cv2.destroyAllWindows()

cells= [np.hsplit(row,100) for row in np.vsplit(gray,50)]

cells=np.array(cells)

print(cells.shape)

Train = cells[:,:70].reshape(-1,400).astype(np.float32)
Test = cells[:,70:100].reshape(-1,400).astype(np.float32)

k=[0,1,2,3,4,5,6,7,8,9]

Train_labels=np.repeat(k,350)[:,np.newaxis]
Test_labels=np.repeat(k,150)[:,np.newaxis]

knn=cv2.KNearest()
knn.train(Train,Train_labels)





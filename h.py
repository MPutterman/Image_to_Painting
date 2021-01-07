from skimage import morphology, filters,color,segmentation,util,measure
import numpy as np 
class Drawer:
    def __init__(self):
        pass
    def drawit(self,img):
        edges = self.skeletonized_edges(img)
        num_colors = np.max(edges)
        edges-=1
        mi = np.min(edges)
        new_arr = self.find_mean(img,edges,num_colors)
        to_fill =np.zeros_like(img)
        for i in range(len(edges)):
            for j in range(len(edges[0])):
                to_fill[i][j]=new_arr[edges[i][j]]
        return to_fill
    def skeletonized_edges(self,img):
        edges = filters.sobel(img)
        edges=color.rgb2gray(edges)
        b_size = self.auto_detect_painting_level(img)
        edges = edges>filters.threshold_local(edges,block_size=b_size)
        edges = morphology.skeletonize(edges)
        for i in range(len(img)):
            for j in range(len(img[i])):
                if i==0 or i ==len(img) or j==0 or j==len(img[i]):
                    edges[i][j]=True
        edges = morphology.dilation(edges,morphology.disk(1))
        edges = segmentation.watershed((edges))
        return edges
    def find_mean(self,img,edges,num_colors):
        arr = [[] for i in range(num_colors)]
        for i in range(len(edges)):
            for j in range(len(edges[0])):
                arr[edges[i][j]].append(img[i][j].tolist())
        arr=np.asarray(arr)
        new_arr = [0 for i in range(num_colors)]
        for i in range(len(arr)):
            ar = np.array(arr[i])
            mean = np.array([np.mean(ar[:,0]),np.mean(ar[:,1]),np.mean(ar[:,2])])
            new_arr[i] = mean
        return new_arr
    def auto_detect_painting_level(self,img):
        orig = 10*abs((np.std(img[:,:,2])+np.std(img[:,:,1])+np.std(img[:,:,0]))/(np.max(img)-np.min(img)))
        offset =1 
        interval=2
        num = round((orig-offset)/interval)*interval+offset
        return max(num,3)




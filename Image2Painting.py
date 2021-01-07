from skimage import morphology, filters,color,segmentation,util,measure
import numpy as np 
class Drawer:
    def __init__(self):
        pass
    def drawit(self,img):
        edges = self.ideal_edges(img)
        patches = segmentation.watershed((edges))
        to_fill=self.fill_in(patches,img)
        
        new_edges = self.likely_aliasing(to_fill,img)
        new_edges = new_edges | edges
        new_patches = segmentation.watershed(new_edges)
        return self.fill_in(new_patches,img)
    def likely_aliasing(self,to_fill,img):
        diff =abs(color.rgb2gray(img-to_fill))
        diff = diff>np.mean(diff)
        diff = morphology.opening(diff,morphology.disk(3))
        diff = filters.sobel(diff)>0
        return diff
    def fill_in(self,patches,img):
        num_colors = np.max(patches)
        patches-=1
        mi = np.min(patches)
        new_arr = self.find_mean(img,patches,num_colors)
        to_fill =np.zeros_like(img)
        for i in range(len(patches)):
            for j in range(len(patches[0])):
                to_fill[i][j]=new_arr[patches[i][j]]
        patches+=1
        return to_fill
        

    def ideal_edges(self,img):
        im1=img[:,:,1]
        im2=img[:,:,2]
        im0=img[:,:,0]
        i0=(np.digitize(im0,filters.threshold_multiotsu(im0,3)))
        i1=(np.digitize(im1,filters.threshold_multiotsu(im1,3)))
        i2=(np.digitize(im2,filters.threshold_multiotsu(im2,3)))
        edges=(abs(filters.sobel(i0))+abs(filters.sobel(i1))+abs(filters.sobel(i2)))
        for i in range(len(img)):
            for j in range(len(img[i])):
                if i==0 or i ==len(img) or j==0 or j==len(img[i]):
                    edges[i][j]=1
        edges=edges>0
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



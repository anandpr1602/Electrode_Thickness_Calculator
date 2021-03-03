# -*- coding: utf-8 -*-
"""
Created Jan/Feb 2021 for UCL EIL

@author: - Anand Pallipurath
"""
import os, tkinter, multiprocessing
import tkinter.filedialog as filedialog
import tkinter.font as font
import imageio
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage import filters, morphology, segmentation, measure
from sklearn.cluster import KMeans

classes = 3 # Specify the number of clusters to categorise the pixels.
# Current collectors are grouped into one cluster, Si and Li phases are grouped together into a second cluster, and separator material (and sometimes crystalline Si) is grouped into a third cluster.

separator_dilate = 7 # This dilates the pores of the separator layer for accurate estimation of contours.
# Decrease this value to distinguish between the separator and uncycled crystalline silicon especially before lithiation. Increase this value to obtain a smoother contour of the separator layer.
# e.g., COL-SAM 279 cycled = 7; COL-SAM 279 uncycled = 4.5; COL-SAM 282 uncycled 20x = 22


# List of compatible image file types that can be opened using ImageIO
ImageFileExtensions = ['.3fr', '.arw', '.bay', '.bmp', '.bmq', '.bufr', '.bw', '.cap', '.cine', '.cr2', '.crw', '.cs1', '.cur', '.cut', '.dc2', '.dcm', '.dcr', '.dcx', '.dds', '.dicom', '.dng', '.drf', '.dsc', '.ecw', '.emf', '.eps', '.erf', '.exr', '.fff', '.fit', '.fits', '.flc', '.fli',
 '.fpx', '.ftc', '.ftu', '.g3', '.gbr', '.gdcm', '.gif', '.gipl', '.grib', '.h5', '.hdf', '.hdf5', '.hdp', '.hdr', '.ia', '.icns', '.ico', '.iff', '.iim', '.iiq', '.im', '.ipl', '.j2c', '.j2k', '.jfif', '.jng', '.jp2', '.jpc', '.jpe', '.jpeg', '.jpf', '.jpg', '.jpx', '.jxr', '.k25', '.kc2', '.kdc', '.koa', '.lbm', '.lfp', '.lfr', '.lsm', '.mdc', '.mef', '.mgh', '.mha', '.mhd', '.mic', '.mnc', '.mnc2',
 '.mos', '.mpo', '.mrw', '.msp', '.nef', '.nhdr', '.nia', '.nii', '.nrrd', '.nrw', '.orf', '.pbm', '.pcd', '.pct', '.pcx', '.pef', '.pfm', '.pgm', '.pic', '.pict', '.png', '.ppm', '.ps', '.psd', '.ptx', '.pxn', '.pxr','.qtk', '.raf', '.ras', '.raw', '.rdc', '.rgb', '.rgba', '.rw2', '.rwl', '.rwz', '.sgi', '.spe', '.sr2', '.srf', '.srw', '.sti', '.stk', '.targa', '.tga', '.tif', '.tiff', '.vtk', '.wap', '.wbm', '.wbmp', '.wdp', '.webp', '.wmf', '.xbm', '.xpm']

#############################################################################
class FileSelector: # Asks user to choose the image to process.
    def __init__(self, root):
        self.root = root
        self.root.withdraw()
        self.save_path = filedialog.askdirectory(title = "Select a Folder of images", mustexist = True) + "/"
        self.root.destroy()
    def results(self):
        return (self.save_path)
#############################################################################
class ParamsSelector: # Asks user to input pixel size, number of current collectors in the image, and whether to rotate image to make orinetation horizontal.
    def __init__(self, root):
        self.window = root
        self.window.attributes("-topmost", True)
        self.window.title("UCL EIL - Electrode Thickness Calculator. Built by Anand Pallipurath.")
        
        self.e1_label = tkinter.Label(self.window, text = "Enter pixel size (\u03BCm/pixel). Leave blank if unknown.", state = "active")
        self.e1_label['font'] = font.Font(family="Helvetica", size=10, weight=font.BOLD, slant=font.ITALIC)
        self.e1_label.grid(row = 0, column = 0, columnspan = 2)
        
        self.pixsize = tkinter.StringVar()
        self.entry1 = tkinter.Entry(self.window, textvariable = self.pixsize, state = "normal")
        self.pixsize.set("0.1989072")
        self.entry1.grid(row = 1, column = 0, columnspan = 2)
        
        self.e2_label = tkinter.Label(self.window, text = "Electrode layers must be horizontal. Rotate image?", state = "active")
        self.e2_label['font'] = font.Font(family="Helvetica", size=10, weight=font.BOLD, slant=font.ITALIC)
        self.e2_label.grid(row = 0, column = 3, columnspan = 3)
        
        self.rotpic = tkinter.IntVar()
        self.rbutton2a = tkinter.Radiobutton(self.window, text = "Yes", variable = self.rotpic, value = 1, state = "active")
        self.rbutton2a.grid(row = 1, column = 3, columnspan = 1)
        self.rbutton2b = tkinter.Radiobutton(self.window, text = "No", variable = self.rotpic, value = 0, state = "active")
        self.rbutton2b.grid(row = 1, column = 4, columnspan = 1)
        
        self.e3_label = tkinter.Label(self.window, text = "# of current collectors in the image.\n(2 - for both, 1 - for either, 0 - neither)", state = "active")
        self.e3_label['font'] = font.Font(family="Helvetica", size=10, weight=font.BOLD, slant=font.ITALIC)
        self.e3_label.grid(row = 2, column = 0, columnspan = 2)
        
        self.cc_num = tkinter.IntVar()
        self.entry3 = tkinter.Entry(self.window, textvariable = self.cc_num, state = "normal")
        self.cc_num.set(2)
        self.entry3.grid(row = 3, column = 0, columnspan = 2)
        
        self.e4_label = tkinter.Label(self.window, text = "# edges of the separator in the image.\n(2 - for both, 1 - for either, 0 - neither)", state = "active")
        self.e4_label['font'] = font.Font(family="Helvetica", size=10, weight=font.BOLD, slant=font.ITALIC)
        self.e4_label.grid(row = 2, column = 3, columnspan = 2)
        
        self.se_num = tkinter.IntVar()
        self.entry4 = tkinter.Entry(self.window, textvariable = self.se_num, state = "normal")
        self.se_num.set(2)
        self.entry4.grid(row = 3, column = 3, columnspan = 2)
        
        self.button1 = tkinter.Button(self.window, text = "Continue", command = self.selection, state = "active")
        self.button1['font'] = font.Font(family="Helvetica", size=10, weight=font.BOLD, slant=font.ITALIC)
        self.button1.grid(row = 4, column = 2, columnspan = 1)
        
        self.window.protocol("WM_DELETE_WINDOW", self.onclosingroot)
        self.window.mainloop()
    
    def onclosingroot(self):
        self.pixel_size = None
        self.rotate_image = None
        self.num_cc = None
        self.num_separator_edges = None
        self.window.destroy()
    
    def selection(self):
        if self.pixsize.get() == '':
            self.pixel_size = None
        else:
            self.pixel_size = float(self.pixsize.get())
        if int(self.rotpic.get()) == 1:
            self.rotate_image = True
        else:
            self.rotate_image = False
        self.num_cc = int(self.cc_num.get())
        self.num_separator_edges = int(self.se_num.get())
        self.window.destroy()
    
    def results(self):
        return (self.pixel_size, self.rotate_image, self.num_cc, self.num_separator_edges)

#############################################################################
class Si_Thickness_Processing: # Main class that performs the processing and estimation of thickness.
    def __init__(self, path, image, central_frame, rotate_image, num_currentcollectors, pixel_size, num_separator_edges):
        self.path = path
        self.rotate_image = rotate_image
        self.num_currentcollectors = num_currentcollectors
        self.pixel_size = pixel_size
        self.central_frame = central_frame
        if self.pixel_size == None:
            self.pixel_size = 1
            self.units = " (px)"
        else:
            self.units = " (\u03BCm)"
        self.final_contours = []
        _, self.file = os.path.split(self.path[:-1])
        self.location = self.path
        print("Processing folder - ", self.file, "\n")
        self.name = self.file + "_slice_" +  str(self.central_frame)
        self.save_path = os.path.join(self.location, "Output-"+self.name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # Open the image using ImageIO.
        self.img = image
        if self.rotate_image == True:
            self.img = np.rot90(self.img, 3)
        self.x, self.y = np.nonzero(self.img)
        self.img = self.img[self.x.min(): self.x.max()+1, self.y.min(): self.y.max()+1]
        self.plotting(self.img, "Original - slice " + str(self.central_frame) + " - " + self.file, "gray", None, self.save_path)
        self.img1 = np.zeros_like(self.img)
        
        # Identify the current collectors through Otsu thresholding and separate them.
        if self.num_currentcollectors > 0:
            self.thresh1 = filters.threshold_otsu(self.img, nbins=4096)
            self.img1[self.img >= self.thresh1] = 1
            self.mask = self.img1
            self.seed = np.copy(self.img1)
            self.seed[1:-1, 1:-1] = self.img1.min()
            self.img1 = morphology.reconstruction(self.seed, self.mask, method='dilation')
            self.img1 = morphology.dilation(self.img1, selem=np.ones((12,12)))
            self.contours1 = measure.find_contours(self.img1, 0.5, fully_connected='high', positive_orientation = 'high')
            self.contours1.sort(key=len)
            self.plotting(self.img, "Contours of current collectors - slice " + str(self.central_frame) + " - " + self.file, "gray", self.contours1[-self.num_currentcollectors:], self.save_path)
            self.hist_title = "Otsu binarization and K-Means classification, clusters = {0}".format(classes-1)
            self.CC_contours = pd.DataFrame()
            self.count=1
            for self.contour in self.contours1[-self.num_currentcollectors:]:
                self.final_contours.append(self.contour)
                self.cols = pd.DataFrame({"CC X"+str(self.count)+self.units: self.pixel_size*self.contour[:, 1], "CC Y"+str(self.count)+self.units: self.pixel_size*self.contour[:, 0]})
                self.cols = self.cols.drop_duplicates(subset=["CC X"+str(self.count)+self.units], ignore_index=True)
                self.CC_contours = pd.concat([self.CC_contours, self.cols], axis=1)
                self.count += 1
        else:
            self.thresh1 = 0
            self.hist_title = "K-Means classification, clusters = {0}".format(classes-1)
            self.CC_contours = None
        
        # Use K-Means to find other regions - separator and the electrodes.
        print("Calculating K-Means clusters from slice ", str(self.central_frame), "...\n")
        self.kmeans_finding(self.img, classes-1, self.thresh1)
        self.values1 = np.append(self.values1, self.thresh1)
        # Plot the pixel histogram and the locations of the clusters.
        self.plot_histogram(self.img.reshape(-1,1), self.values1, self.hist_title, self.save_path)
        self.elevation_map = filters.sobel(self.first_classification)
        self.markers = np.zeros_like(self.img)
        for i in range (0, len(self.values1)-1):
            self.markers[self.img >= self.values1[i]] = i+1
        self.watershed_img = segmentation.watershed(self.elevation_map, self.markers)
        self.watershed_img[self.img1 == 1] = 3
        self.watershed_img = ndi.maximum_filter(self.watershed_img, footprint=morphology.disk(separator_dilate))
        self.plotting(self.watershed_img, "Watershed - slice " + str(self.central_frame) + " - " + self.file, "nipy_spectral", None, self.save_path)
        
        self.contours2 = measure.find_contours(self.watershed_img, 1.5, fully_connected='high', positive_orientation = 'low')
        self.contours2.sort(key=len, reverse = True)
        self.plotting(self.img, "Contours of electrode layers - slice " + str(self.central_frame) + " - " + self.file, "gray", self.contours2[:num_separator_edges], self.save_path)
        
        self.Electrode_contours = pd.DataFrame()
        self.count=1
        for self.contour in self.contours2[:num_separator_edges]:
            self.final_contours.append(self.contour)
            self.cols = pd.DataFrame({"electrode X"+str(self.count)+self.units: self.pixel_size*self.contour[:, 1], "electrode Y"+str(self.count)+self.units: self.pixel_size*self.contour[:, 0]})
            self.cols = self.cols.drop_duplicates(subset=["electrode X"+str(self.count)+self.units], ignore_index=True)
            self.Electrode_contours = pd.concat([self.Electrode_contours, self.cols], axis=1)
            self.count += 1
        if self.CC_contours is not None:
            if len(self.CC_contours) >= len(self.Electrode_contours):
                self.all_contours = pd.concat([self.CC_contours, self.Electrode_contours], axis=1).reindex(self.CC_contours.index)
            else:
                self.all_contours = pd.concat([self.CC_contours, self.Electrode_contours], axis=1).reindex(self.Electrode_contours.index)
        else:
            self.all_contours = self.Electrode_contours
        # Using the locations of the contours, calculate the thickness of each layer.
        print("Calculating thickness of each layer in slice ", str(self.central_frame), "...\n")
        self.thickness_estimation(num_currentcollectors, num_separator_edges)
        
        # Save all the results into an Excel file.
        print("Saving the results from slice - " + str(self.central_frame) + " as an .xlsx file...\n")
        self.outpath = os.path.join(self.save_path, self.name+"-result.xlsx")
        with pd.ExcelWriter(self.outpath) as self.writer:
            self.all_contours.to_excel(self.writer, sheet_name = 'XY Coordinates')
            self.all_thickness.to_excel(self.writer, sheet_name = 'Thickness estimates')
        self.plotting(self.img, "Final contours - slice " + str(self.central_frame) + " - " + self.file, "gray", self.final_contours, self.save_path)
    
    #############################################################################
    def thickness_estimation(self, num_currentcollectors, num_separator_edges):       
        # Calculate separator thickness only if num_separator_edges == 2.
        if num_separator_edges == 2:
            self.cols1 = []
            self.cols2 = []
            for x1 in self.Electrode_contours["electrode X1"+self.units]:
                if x1 in self.Electrode_contours["electrode X2"+self.units].tolist():
                    self.y1 = self.Electrode_contours.loc[self.Electrode_contours["electrode X1"+self.units].tolist().index(x1), "electrode Y1"+self.units]
                    self.y2 = self.Electrode_contours.loc[self.Electrode_contours["electrode X2"+self.units].tolist().index(x1), "electrode Y2"+self.units]
                    self.cols1.append(x1)
                    self.cols2.append(abs(self.y2-self.y1))
            self.sep_thickness = pd.DataFrame({"X position"+self.units: self.cols1, "Separator thickness "+self.units: self.cols2})
        else:
            self.sep_thickness = pd.DataFrame()
        
        if num_currentcollectors == 2: 
            # This assumes that if both CCs are in the images then the whole separator is also included in the image.
            if abs(self.Electrode_contours["electrode Y1"+self.units].mean(axis=0) - self.all_contours["CC Y1"+self.units].mean(axis=0)) < abs(self.Electrode_contours["electrode Y1"+self.units].mean(axis=0) - self.all_contours["CC Y2"+self.units].mean(axis=0)):
                self.cols1 = []
                self.cols2 = []
                for x1 in self.Electrode_contours["electrode X1"+self.units]:
                    if x1 in self.all_contours["CC X1"+self.units].tolist():
                        self.y1 = self.Electrode_contours.loc[self.Electrode_contours["electrode X1"+self.units].tolist().index(x1), "electrode Y1"+self.units]
                        self.y2 = self.all_contours.loc[self.all_contours["CC X1"+self.units].tolist().index(x1), "CC Y1"+self.units]
                        self.cols1.append(x1)
                        self.cols2.append(abs(self.y2-self.y1))
                self.layer1_thickness = pd.DataFrame({"X position"+self.units: self.cols1, "Layer 1 thickness "+self.units: self.cols2})
                        
                self.cols1 = []
                self.cols2 = []        
                for x2 in self.Electrode_contours["electrode X2"+self.units]:
                    if x2 in self.all_contours["CC X2"+self.units].tolist():
                        self.y1 = self.Electrode_contours.loc[self.Electrode_contours["electrode X2"+self.units].tolist().index(x2), "electrode Y2"+self.units]
                        self.y2 = self.all_contours.loc[self.all_contours["CC X2"+self.units].tolist().index(x2), "CC Y2"+self.units]
                        self.cols1.append(x2)
                        self.cols2.append(abs(self.y2-self.y1))
                self.layer2_thickness = pd.DataFrame({"X position"+self.units: self.cols1, "Layer 2 thickness "+self.units: self.cols2})
                        
            else:
                self.cols1 = []
                self.cols2 = []
                for x1 in self.Electrode_contours["electrode X1"+self.units]:
                    if x1 in self.all_contours["CC X2"+self.units].tolist():
                        self.y1 = self.Electrode_contours.loc[self.Electrode_contours["electrode X1"+self.units].tolist().index(x1), "electrode Y1"+self.units]
                        self.y2 = self.all_contours.loc[self.all_contours["CC X2"+self.units].tolist().index(x1), "CC Y2"+self.units]
                        self.cols1.append(x1)
                        self.cols2.append(abs(self.y2-self.y1))
                self.layer1_thickness = pd.DataFrame({"X position"+self.units: self.cols1, "Layer 1 thickness "+self.units: self.cols2})
                self.cols1 = []
                self.cols2 = []        
                for x2 in self.Electrode_contours["electrode X2"+self.units]:
                    if x2 in self.all_contours["CC X1"+self.units].tolist():
                        self.y1 = self.Electrode_contours.loc[self.Electrode_contours["electrode X2"+self.units].tolist().index(x2), "electrode Y2"+self.units]
                        self.y2 = self.all_contours.loc[self.all_contours["CC X1"+self.units].tolist().index(x2), "CC Y1"+self.units]
                        self.cols1.append(x2)
                        self.cols2.append(abs(self.y2-self.y1))
                self.layer2_thickness = pd.DataFrame({"X position"+self.units: self.cols1, "Layer 2 thickness "+self.units: self.cols2})
        
        elif num_currentcollectors == 1:
            self.layer2_thickness = pd.DataFrame()
            if num_separator_edges == 2:
                if abs(self.Electrode_contours["electrode Y1"+self.units].mean(axis=0) - self.all_contours["CC Y1"+self.units].mean(axis=0)) < abs(self.Electrode_contours["electrode Y2"+self.units].mean(axis=0) - self.all_contours["CC Y1"+self.units].mean(axis=0)):
                    self.cols1 = []
                    self.cols2 = []
                    for x1 in self.Electrode_contours["electrode X1"+self.units]:
                        if x1 in self.all_contours["CC X1"+self.units].tolist():
                            self.y1 = self.Electrode_contours.loc[self.Electrode_contours["electrode X1"+self.units].tolist().index(x1), "electrode Y1"+self.units]
                            self.y2 = self.all_contours.loc[self.all_contours["CC X1"+self.units].tolist().index(x1), "CC Y1"+self.units]
                            self.cols1.append(x1)
                            self.cols2.append(abs(self.y2-self.y1))
                    self.layer1_thickness = pd.DataFrame({"X position"+self.units: self.cols1, "Layer 1 thickness "+self.units: self.cols2})
                else:
                    self.cols1 = []
                    self.cols2 = []
                    for x2 in self.Electrode_contours["electrode X2"+self.units]:
                        if x2 in self.all_contours["CC X1"+self.units].tolist():
                            self.y1 = self.Electrode_contours.loc[self.Electrode_contours["electrode X2"+self.units].tolist().index(x2), "electrode Y2"+self.units]
                            self.y2 = self.all_contours.loc[self.all_contours["CC X1"+self.units].tolist().index(x2), "CC Y1"+self.units]
                            self.cols1.append(x2)
                            self.cols2.append(abs(self.y2-self.y1))
                    self.layer1_thickness = pd.DataFrame({"X position"+self.units: self.cols1, "Layer 1 thickness "+self.units: self.cols2})
            elif num_separator_edges == 1:
                self.cols1 = []
                self.cols2 = []
                for x1 in self.Electrode_contours["electrode X1"+self.units]:
                    if x1 in self.all_contours["CC X1"+self.units].tolist():
                        self.y1 = self.Electrode_contours.loc[self.Electrode_contours["electrode X1"+self.units].tolist().index(x1), "electrode Y1"+self.units]
                        self.y2 = self.all_contours.loc[self.all_contours["CC X1"+self.units].tolist().index(x1), "CC Y1"+self.units]
                        self.cols1.append(x1)
                        self.cols2.append(abs(self.y2-self.y1))
                self.layer1_thickness = pd.DataFrame({"X position"+self.units: self.cols1, "Layer 1 thickness "+self.units: self.cols2})
            else:
                self.layer1_thickness = pd.DataFrame()
        
        elif num_currentcollectors == 0:
            self.layer1_thickness = pd.DataFrame()
            self.layer2_thickness = pd.DataFrame()
                
        if len(self.sep_thickness) > len(self.layer1_thickness) and len(self.sep_thickness) > len(self.layer2_thickness):
            self.all_thickness = pd.concat([self.sep_thickness, self.layer1_thickness, self.layer2_thickness], axis=1).reindex(self.sep_thickness.index)
        elif len(self.layer1_thickness) > len(self.sep_thickness) and len(self.layer1_thickness) > len(self.layer2_thickness):
            self.all_thickness = pd.concat([self.sep_thickness, self.layer1_thickness, self.layer2_thickness], axis=1).reindex(self.layer1_thickness.index)
        else:
            self.all_thickness = pd.concat([self.sep_thickness, self.layer1_thickness, self.layer2_thickness], axis=1).reindex(self.layer2_thickness.index)
    
    #############################################################################
    def plotting(self, image, title, cmap, contours, save_path):
        self.fig = plt.figure()
        plt.imshow(image, cmap=cmap)
        if contours is not None:
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=1.5)
        plt.title(title)
        plt.axis('off')
        plt.savefig(os.path.join(save_path,title), dpi=150, facecolor='w', edgecolor='w', bbox_inches='tight', pad_inches = 0.1)
        plt.close()
    
    #############################################################################
    def plot_histogram(self, pix, values, title, save_path):
        self.fig3 = plt.figure()
        plt.hist(pix, bins=4096, color='.5', edgecolor='.5')
        plt.xlabel("Gray scale")
        plt.ylabel("Pixel Count")
        for line in values:
            if line == values[-1]:
                if line != 0:
                    plt.axvline(line, color='r')
            else:
                plt.axvline(line, color='b')
        plt.title(title)
        plt.savefig(os.path.join(save_path,title), dpi=150, facecolor='w', edgecolor='w', bbox_inches='tight', pad_inches = 0.1)
        plt.close()
    
    #############################################################################
    def kmeans_finding(self, image, n, thresh1):
        self.image_gray = image.reshape(image.shape[0] * image.shape[1], 1)
        if thresh1 > 0:
            self.keep_ind = np.where(self.image_gray < thresh1)
        else:
            self.keep_ind = np.where(self.image_gray > 0)
        self.image_gray_crop = self.image_gray[self.keep_ind].reshape(-1,1)
        self.kmeans = KMeans(n_clusters=n, random_state=None).fit(self.image_gray_crop)
    
        self.values1 = self.kmeans.cluster_centers_.squeeze()
        self.labels = self.kmeans.labels_
        self.clustered = self.kmeans.cluster_centers_[self.labels]
        
        self.values1 = np.sort(self.values1)
        self.merged_label = np.zeros(len(self.image_gray))
        self.merged_clustered = np.zeros(len(self.image_gray))
        for i in range(0,len(self.keep_ind[0])):
            self.merged_label[self.keep_ind[0][i]] = self.labels[i]+1
            self.merged_clustered[self.keep_ind[0][i]] = self.clustered[i]
        
        self.first_classification = np.array(self.merged_label).reshape(image.shape)
        self.values2 = []        
        for a in range(0, len(self.values1)):
            self.indi = np.where(self.merged_clustered.reshape(image.shape) == self.values1[a])
            self.values2.append([np.min(image[self.indi]), np.max(image[self.indi])])
    #############################################################################    
    def results(self):
        return(self.values1, self.values2)


#############################################################################
def plotting(image, title, cmap, contours, save_path):
    fig = plt.figure()
    plt.imshow(image, cmap=cmap)
    colours_list = ['orange', 'blue', 'red', 'green']
    if contours is not None:
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=1.5)
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, dpi=150, facecolor='w', edgecolor='w', bbox_inches='tight', pad_inches = 0.1)
    plt.close()

#############################################################################        
def process_all(process_params):
    path, img, slice_num, rotate_image, num_currentcollectors, pixel_size, num_separator_edges, thresh_values, cluster_values = process_params
    print("Processing slice ", str(slice_num), "...\n")
    
    _, file = os.path.split(path[:-1])
    name = file + "_slice_" +  str(slice_num) + ".tiff"
    save_path = os.path.join(path, "Output_all_frames")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if pixel_size == None:
        pixel_size = 1
        units = " (px)"
    else:
        units = " (\u03BCm)"
    thresh1 = thresh_values[-1]
    values1 = thresh_values[:-1]
    final_contours = []
    # Open the image using ImageIO.
    if rotate_image == True:
        img = np.rot90(img, 3)
    x, y = np.nonzero(img)
    img = img[x.min(): x.max()+1, y.min(): y.max()+1]
    img1 = np.zeros_like(img)    
    # Identify the current collectors through Otsu thresholding and separate them.
    if num_currentcollectors > 0:
        img1[img >= thresh1] = 1
        mask = img1
        seed = np.copy(img1)
        seed[1:-1, 1:-1] = img1.min()
        img1 = morphology.reconstruction(seed, mask, method='dilation')
        img1 = morphology.dilation(img1, selem=np.ones((12,12)))
        contours1 = measure.find_contours(img1, 0.5, fully_connected='high', positive_orientation = 'high')
        contours1.sort(key=len) # sort the contours by length in ascending order.
        contours1 = contours1[-num_currentcollectors:] # pick only the longest "num_currentcollectors" number of contours.
        if num_currentcollectors > 1: # if more than 1 CC contours, sort them in ascending order based on Y positions.
            contours1 = sorted(contours1, key=lambda x: np.average(x[:,0]))
        CC_contours = pd.DataFrame()
        count=1
        for contour in contours1:
            final_contours.append(contour)
            cols = pd.DataFrame({"CC X"+str(count)+units: pixel_size*contour[:, 1], "CC Y"+str(count)+units: pixel_size*contour[:, 0]})
            cols = cols.drop_duplicates(subset=["CC X"+str(count)+units], ignore_index=True)
            CC_contours = pd.concat([CC_contours, cols], axis=1)
            count += 1
    else:
        CC_contours = None
    first_classification = np.zeros_like(img)
    markers = np.zeros_like(img)
    for a in range(0, len(values1)):
        first_classification[img >= cluster_values[a][0]] = a+1
        first_classification[img >= cluster_values[a][1]] = 0
        markers[img >= values1[a]] = a+1

    elevation_map = filters.sobel(first_classification)
    watershed_img = segmentation.watershed(elevation_map, markers)
    watershed_img[img1 == 1] = 3
    watershed_img = ndi.maximum_filter(watershed_img, footprint=morphology.disk(separator_dilate))

    contours2 = measure.find_contours(watershed_img, 1.5, fully_connected='high', positive_orientation = 'low')
    Electrode_contours = pd.DataFrame()
    
    contours2 = sorted(contours2, key=len, reverse=True) # sort the contours by length in descending order.
    if num_currentcollectors > 1 or num_separator_edges > 1:
        contours2 = contours2[:10] if len(contours2) >= 10 else contours2
        # Calculate if any of the separator contours are quite close to the CC contours, if so remove them from the list
        df2 = CC_contours.filter(regex='CC Y').mean(axis=0)
        sorter = df2.to_numpy()
        contours22 = []
        for contour in contours2:
            checker = (abs(sorter - pixel_size*np.average(contour[:,0])) < 7.0)
            if True not in checker:
                contours22.append(contour)
        contours2 = contours22
        contours2 = sorted(contours2, key=len, reverse=True) # sort again by length in descending order.
    contours2 = contours2[:num_separator_edges] # pick only the longest "num_separator_edges" number of contours
    if num_currentcollectors > 1 or num_separator_edges > 1: # if more than 1 separator contours, sort them in ascending order based on Y positions.
            contours2 = sorted(contours2, key=lambda x: np.average(x[:,0]))
    count=1
    for contour in contours2:
        final_contours.append(contour)
        cols = pd.DataFrame({"electrode X"+str(count)+units: pixel_size*contour[:, 1], "electrode Y"+str(count)+units: pixel_size*contour[:, 0]})
        cols = cols.drop_duplicates(subset=["electrode X"+str(count)+units], ignore_index=True)
        Electrode_contours = pd.concat([Electrode_contours, cols], axis=1)
        count += 1
    plotting(img, "Final contours - slice " + str(slice_num) + " - " + file, "gray", final_contours, os.path.join(save_path, name))
    if CC_contours is not None:
        if len(CC_contours) >= len(Electrode_contours):
            all_contours = pd.concat([CC_contours, Electrode_contours], axis=1).reindex(CC_contours.index)
        else:
            all_contours = pd.concat([CC_contours, Electrode_contours], axis=1).reindex(Electrode_contours.index)
    else:
        all_contours = Electrode_contours
    # Using the locations of the contours, calculate the thickness of each layer.
    slice_num, sp_measure, l1_measure, l2_measure = thickness_estimation(num_currentcollectors, num_separator_edges, units, Electrode_contours, all_contours, slice_num)
    return slice_num, sp_measure, l1_measure, l2_measure, final_contours

#############################################################################  
def thickness_estimation(num_currentcollectors, num_separator_edges, units, Electrode_contours, all_contours, slice_num):
    # Calculate separator thickness only if num_separator_edges == 2.
    if num_separator_edges == 2:
        cols1 = []
        cols2 = []
        for x1 in Electrode_contours["electrode X1"+units]:
            if x1 in Electrode_contours["electrode X2"+units].tolist():
                y1 = Electrode_contours.loc[Electrode_contours["electrode X1"+units].tolist().index(x1), "electrode Y1"+units]
                y2 = Electrode_contours.loc[Electrode_contours["electrode X2"+units].tolist().index(x1), "electrode Y2"+units]
                cols1.append(x1)
                cols2.append(abs(y2-y1))
        sep_thickness = pd.DataFrame({"X position"+units: cols1, "Separator thickness "+units: cols2})
    else:
        sep_thickness = pd.DataFrame()
    
    if num_currentcollectors == 2: 
        # This assumes that if both CCs are in the images then the whole separator is also included in the image.
        E1C1 = abs(Electrode_contours["electrode Y1"+units].mean(axis=0) - all_contours["CC Y1"+units].mean(axis=0))
        E1C2 = abs(Electrode_contours["electrode Y1"+units].mean(axis=0) - all_contours["CC Y2"+units].mean(axis=0))
        E2C1 = abs(Electrode_contours["electrode Y2"+units].mean(axis=0) - all_contours["CC Y1"+units].mean(axis=0))
        E2C2 = abs(Electrode_contours["electrode Y2"+units].mean(axis=0) - all_contours["CC Y2"+units].mean(axis=0))
        if E1C1 < E1C2  and E1C1 < E2C1:
            cols1 = []
            cols2 = []
            for x1 in Electrode_contours["electrode X1"+units]:
                if x1 in all_contours["CC X1"+units].tolist():
                    y1 = Electrode_contours.loc[Electrode_contours["electrode X1"+units].tolist().index(x1), "electrode Y1"+units]
                    y2 = all_contours.loc[all_contours["CC X1"+units].tolist().index(x1), "CC Y1"+units]
                    cols1.append(x1)
                    cols2.append(abs(y2-y1))
            layer1_thickness = pd.DataFrame({"X position"+units: cols1, "Layer 1 thickness "+units: cols2})
                    
            cols1 = []
            cols2 = []        
            for x2 in Electrode_contours["electrode X2"+units]:
                if x2 in all_contours["CC X2"+units].tolist():
                    y1 = Electrode_contours.loc[Electrode_contours["electrode X2"+units].tolist().index(x2), "electrode Y2"+units]
                    y2 = all_contours.loc[all_contours["CC X2"+units].tolist().index(x2), "CC Y2"+units]
                    cols1.append(x2)
                    cols2.append(abs(y2-y1))
            layer2_thickness = pd.DataFrame({"X position"+units: cols1, "Layer 2 thickness "+units: cols2})
            
        elif E2C1 < E2C2 and E2C1 < E1C1:
            cols1 = []
            cols2 = []
            for x1 in Electrode_contours["electrode X2"+units]:
                if x1 in all_contours["CC X1"+units].tolist():
                    y1 = Electrode_contours.loc[Electrode_contours["electrode X2"+units].tolist().index(x1), "electrode Y2"+units]
                    y2 = all_contours.loc[all_contours["CC X1"+units].tolist().index(x1), "CC Y1"+units]
                    cols1.append(x1)
                    cols2.append(abs(y2-y1))
            layer1_thickness = pd.DataFrame({"X position"+units: cols1, "Layer 1 thickness "+units: cols2})
                    
            cols1 = []
            cols2 = []        
            for x2 in Electrode_contours["electrode X1"+units]:
                if x2 in all_contours["CC X2"+units].tolist():
                    y1 = Electrode_contours.loc[Electrode_contours["electrode X1"+units].tolist().index(x2), "electrode Y1"+units]
                    y2 = all_contours.loc[all_contours["CC X2"+units].tolist().index(x2), "CC Y2"+units]
                    cols1.append(x2)
                    cols2.append(abs(y2-y1))
            layer2_thickness = pd.DataFrame({"X position"+units: cols1, "Layer 2 thickness "+units: cols2}) 
        elif E1C2 < E1C1 and E1C2 < E2C2:
            cols1 = []
            cols2 = []
            for x1 in Electrode_contours["electrode X1"+units]:
                if x1 in all_contours["CC X2"+units].tolist():
                    y1 = Electrode_contours.loc[Electrode_contours["electrode X1"+units].tolist().index(x1), "electrode Y1"+units]
                    y2 = all_contours.loc[all_contours["CC X2"+units].tolist().index(x1), "CC Y2"+units]
                    cols1.append(x1)
                    cols2.append(abs(y2-y1))
            layer1_thickness = pd.DataFrame({"X position"+units: cols1, "Layer 1 thickness "+units: cols2})
                    
            cols1 = []
            cols2 = []        
            for x2 in Electrode_contours["electrode X2"+units]:
                if x2 in all_contours["CC X1"+units].tolist():
                    y1 = Electrode_contours.loc[Electrode_contours["electrode X2"+units].tolist().index(x2), "electrode Y2"+units]
                    y2 = all_contours.loc[all_contours["CC X1"+units].tolist().index(x2), "CC Y1"+units]
                    cols1.append(x2)
                    cols2.append(abs(y2-y1))
            layer2_thickness = pd.DataFrame({"X position"+units: cols1, "Layer 2 thickness "+units: cols2})
        else:
            cols1 = []
            cols2 = []
            for x1 in Electrode_contours["electrode X2"+units]:
                if x1 in all_contours["CC X2"+units].tolist():
                    y1 = Electrode_contours.loc[Electrode_contours["electrode X2"+units].tolist().index(x1), "electrode Y2"+units]
                    y2 = all_contours.loc[all_contours["CC X2"+units].tolist().index(x1), "CC Y2"+units]
                    cols1.append(x1)
                    cols2.append(abs(y2-y1))
            layer1_thickness = pd.DataFrame({"X position"+units: cols1, "Layer 1 thickness "+units: cols2})
            cols1 = []
            cols2 = []        
            for x2 in Electrode_contours["electrode X1"+units]:
                if x2 in all_contours["CC X1"+units].tolist():
                    y1 = Electrode_contours.loc[Electrode_contours["electrode X1"+units].tolist().index(x2), "electrode Y1"+units]
                    y2 = all_contours.loc[all_contours["CC X1"+units].tolist().index(x2), "CC Y1"+units]
                    cols1.append(x2)
                    cols2.append(abs(y2-y1))
            layer2_thickness = pd.DataFrame({"X position"+units: cols1, "Layer 2 thickness "+units: cols2})
    
    elif num_currentcollectors == 1:
        layer2_thickness = pd.DataFrame()
        if num_separator_edges == 2:
            if abs(Electrode_contours["electrode Y1"+units].mean(axis=0) - all_contours["CC Y1"+units].mean(axis=0)) < abs(Electrode_contours["electrode Y2"+units].mean(axis=0) - all_contours["CC Y1"+units].mean(axis=0)):
                cols1 = []
                cols2 = []
                for x1 in Electrode_contours["electrode X1"+units]:
                    if x1 in all_contours["CC X1"+units].tolist():
                        y1 = Electrode_contours.loc[Electrode_contours["electrode X1"+units].tolist().index(x1), "electrode Y1"+units]
                        y2 = all_contours.loc[all_contours["CC X1"+units].tolist().index(x1), "CC Y1"+units]
                        cols1.append(x1)
                        cols2.append(abs(y2-y1))
                layer1_thickness = pd.DataFrame({"X position"+units: cols1, "Layer 1 thickness "+units: cols2})
            else:
                cols1 = []
                cols2 = []
                for x2 in Electrode_contours["electrode X2"+units]:
                    if x2 in all_contours["CC X1"+units].tolist():
                        y1 = Electrode_contours.loc[Electrode_contours["electrode X2"+units].tolist().index(x2), "electrode Y2"+units]
                        y2 = all_contours.loc[all_contours["CC X1"+units].tolist().index(x2), "CC Y1"+units]
                        cols1.append(x2)
                        cols2.append(abs(y2-y1))
                layer1_thickness = pd.DataFrame({"X position"+units: cols1, "Layer 1 thickness "+units: cols2})
        elif num_separator_edges == 1:
            cols1 = []
            cols2 = []
            for x1 in Electrode_contours["electrode X1"+units]:
                if x1 in all_contours["CC X1"+units].tolist():
                    y1 = Electrode_contours.loc[Electrode_contours["electrode X1"+units].tolist().index(x1), "electrode Y1"+units]
                    y2 = all_contours.loc[all_contours["CC X1"+units].tolist().index(x1), "CC Y1"+units]
                    cols1.append(x1)
                    cols2.append(abs(y2-y1))
            layer1_thickness = pd.DataFrame({"X position"+units: cols1, "Layer 1 thickness "+units: cols2})
        else:
            layer1_thickness = pd.DataFrame()
    
    elif num_currentcollectors == 0:
        layer1_thickness = pd.DataFrame()
        layer2_thickness = pd.DataFrame()
            
    # "all_thickness" DataFrame has the list of thicknesses from each slices
    """if len(sep_thickness) > len(layer1_thickness) and len(sep_thickness) > len(layer2_thickness):
        all_thickness = pd.concat([sep_thickness, layer1_thickness, layer2_thickness], axis=1).reindex(sep_thickness.index)
    elif len(layer1_thickness) > len(sep_thickness) and len(layer1_thickness) > len(layer2_thickness):
        all_thickness = pd.concat([sep_thickness, layer1_thickness, layer2_thickness], axis=1).reindex(layer1_thickness.index)
    else:
        all_thickness = pd.concat([sep_thickness, layer1_thickness, layer2_thickness], axis=1).reindex(layer2_thickness.index)"""

    sp_measure = []
    if len(sep_thickness) > 0:
        sp_measure.append([np.average(sep_thickness["Separator thickness "+units].dropna().to_numpy()), np.std(sep_thickness["Separator thickness "+units].dropna().to_numpy())])
    else:
        sp_measure.append([0, 0])
    l1_measure = []
    if len(layer1_thickness) > 0:
        l1_measure.append([np.average(layer1_thickness["Layer 1 thickness "+units].dropna().to_numpy()), np.std(layer1_thickness["Layer 1 thickness "+units].dropna().to_numpy())])
    else:
        l1_measure.append([0, 0])
    l2_measure = []
    if len(layer2_thickness) > 0:
        l2_measure.append([np.average(layer2_thickness["Layer 2 thickness "+units].dropna().to_numpy()), np.std(layer2_thickness["Layer 2 thickness "+units].dropna().to_numpy())])
    else:
        l2_measure.append([0, 0])
    return slice_num, sp_measure, l1_measure, l2_measure
    
#############################################################################
def main(save_path, all_images, rotate_image, num_currentcollectors, pixel_size, num_separator_edges, thresh_values, cluster_values):
    if pixel_size == None:
        pixel_size = 1
        units = " (px)"
    else:
        units = " (\u03BCm)"
    input_data = []
    for i in range(int(np.ceil(len(all_images)/3)), int(np.ceil(2*len(all_images)/3)+1)):
        input_data.append([save_path, all_images[i], i, rotate_image, num_currentcollectors, pixel_size, num_separator_edges, thresh_values, cluster_values])
    
    all_thickness_measurements = processpool.map(process_all, input_data, chunksize=1)
    slices, thickness1, thickness2, thickness3, all_final_contours  = np.squeeze(all_thickness_measurements).T
    
    slices = np.squeeze(np.asarray([item for item in slices]))
    thickness1 = np.squeeze(np.asarray([item for item in thickness1]))
    thickness2 = np.squeeze(np.asarray([item for item in thickness2]))
    thickness3 = np.squeeze(np.asarray([item for item in thickness3]))
    
    all_thickness_measurements = pd.DataFrame({"Z Slices"+units: pixel_size*slices})
    if num_separator_edges == 2:
        all_thickness_measurements = all_thickness_measurements.join(pd.DataFrame({"Average Separator Thickness"+units: thickness1[:,0], "Separator Std. Dev."+units: thickness1[:,1]}))
    if num_currentcollectors == 2:
        all_thickness_measurements = all_thickness_measurements.join(pd.DataFrame({"Average Layer 1 Thickness"+units: thickness2[:,0], " Layer 1 Std. Dev."+units: thickness2[:,1]}))
        all_thickness_measurements = all_thickness_measurements.join(pd.DataFrame({"Average Layer 2 Thickness"+units: thickness3[:,0], " Layer 2 Std. Dev."+units: thickness3[:,1]}))
    elif num_currentcollectors == 1:
        all_thickness_measurements = all_thickness_measurements.join(pd.DataFrame({"Average Layer 1 Thickness"+units: thickness2[:,0], " Layer 1 Std. Dev."+units: thickness2[:,1]}))
    outpath = os.path.join(os.path.join(save_path, "Output_all_frames"), "Results.xlsx")
    with pd.ExcelWriter(outpath) as writer:
        all_thickness_measurements.to_excel(writer, sheet_name = 'results')
    writer.close()
    
    # plot 3D projection of all slices:
    counter = num_currentcollectors + num_separator_edges
    cmap = plt.get_cmap("tab10")
    cmaplist = [cmap(i) for i in range(counter)]
    ax = plt.axes(projection='3d')
    plt.title(os.path.split(save_path[:-1])[1])
    for b in range(0, len(all_final_contours)):
        ii = 0
        all_final_contours[b] = all_final_contours[b]
        for a in range(0, len(all_final_contours[b])):
            ax.plot(pixel_size*all_final_contours[b][a][:,1], pixel_size*np.full(len(all_final_contours[b][a][:,1]), pixel_size*slices[b]), all_final_contours[b][a][:,0], color = cmaplist[ii], alpha = 0.1)
            if (ii + 1) % counter == 0:
                ii = 0
            else:
                ii += 1
    ax.azim = -50
    ax.elev = 12
    ax.locator_params(nbins=5)
    ax.set_xlabel('X '+units)
    ax.set_ylabel('Z '+units)
    ax.set_zlabel('Y '+units)
    outpath1 = os.path.join(os.path.join(save_path, "Output_all_frames"), "Orthogonal_projection.tiff")
    plt.savefig(outpath1, dpi=150, facecolor='w', edgecolor='w')#, bbox_inches='tight', pad_inches = 0.1)
    plt.close()
    
# Code starts here
#############################################################################
if __name__ == "__main__":
    # Multiple processes must be created within if __name__ == "__main__"
    processpool = multiprocessing.Pool(multiprocessing.cpu_count())
    # Call the FileSelector class to open a folder of images.
    save_path = FileSelector(tkinter.Tk()).results()
    # Call the ParamsSelector class to input required parameters.
    pixel_size, rotate_image, num_currentcollectors, num_separator_edges = ParamsSelector(tkinter.Tk()).results()

    # Send file path and parameters to the Si_Thickness_Processing class.
    if rotate_image == None:
        print("Window closed during selection. Please restart code and try again!\n")
    else:
        # Load all frames in the file into an array
        print("Loading image stack to make orthogonal slices. Please wait...\n")
        all_images = []
        for file in os.listdir(save_path):
            name = os.path.join(save_path,file)
            path, ext = os.path.splitext(name)
            if ext in ImageFileExtensions and "Orthogonal_projection" not in path:
                img = imageio.imread(name)
                img = np.array(img)
                all_images.append(img)
        all_images = np.stack(all_images, axis=0)
        all_images = np.transpose(all_images, (2, 0, 1))
        central_frame = int(np.ceil(len(all_images)/2))
        thresh_values, cluster_values = Si_Thickness_Processing(save_path, all_images[central_frame], central_frame, rotate_image, num_currentcollectors, pixel_size, num_separator_edges).results()
        main(save_path, all_images, rotate_image, num_currentcollectors, pixel_size, num_separator_edges, thresh_values, cluster_values)
        processpool.close()
        processpool.join()
        print("Processing complete...\n")
    print("Project developed at UCL EIL - www.ucl.ac.uk/electrochemical-innovation-lab/\n")

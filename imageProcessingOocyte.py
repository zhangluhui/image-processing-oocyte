import cv2 #Package for handling images in Python
import sys #Here: for reading parameter from the command line
import math #Here: used the math.ceil function: returns the smallest integer that is bigger than or equal to the number itself
import time #Here: used for calculating the program run time
import progressbar #Here: show a progress bar on the screen
import javabridge #Start and interact with java virtual machine in Python. Here: needed by bioformats package
import bioformats #Read and write life sciences image file
import numpy as np #Mamatical calculation for matrices
from xml import etree as et #Use for parsing the xml format metdata from bioformats package, to know information about picture number, size, and resolution 
from matplotlib import pyplot as plt, cm #Generate pictures

def parse_xml_metadata(xml_string):
    names, sizes, resolutions = [], [], [] #Create empty variables for storing information about picture number, size, and resolution in later steps
    size_tags = ['SizeT', 'SizeZ', 'SizeY', 'SizeX', 'SizeC']
    res_tags = ['PhysicalSizeZ', 'PhysicalSizeY', 'PhysicalSizeX']
    metadata_root = et.ElementTree.fromstring(xml_string) #Find the root node
    for child in metadata_root:
        if child.tag.endswith('Image'): #Find the child nodes that store the image infomation
            names.append(child.attrib['Name']) #Add the image name to varible "names"
            for grandchild in child: #Explore the information inside the image node
                if grandchild.tag.endswith('Pixels'): #Find the 'Pixels' node inside the image node
                    att = grandchild.attrib #Retrive the infomation stored in the 'Pixels' node
                    sizes.append(tuple([int(att[t]) for t in size_tags])) #'size_tags' includes 'SizeT', 'SizeZ', 'SizeY', 'SizeX', 'SizeC'; here, retrive the values for these size tags from the variable att; the variable type of att is a dictionary
                    resolutions.append(tuple([float(att[t]) for t in res_tags])) #'res_tags' is ['PhysicalSizeZ', 'PhysicalSizeY', 'PhysicalSizeX']; here, retrive the values for these size tags from the variable att
    return names, sizes, resolutions #Return the results

def kmeans_segmentation(image, K, iteration, eps, repeat): # "K" is how many clusters will the data points will be clustered into or how many parts will the image be divided into; "iteration" is the maxium iteration number that algorithm can be; "eps" is the termination criterion, and if the score between two iterations is smaller than "eps", the program will end and return the results; "repeat" is the number that the kmeans algorithm will run, and each time, results might be different slightly, and after that the program will return the best result.
    img = np.float32(image.reshape((-1,1))) #reshape the image dataset into one column; "-1" means unknow row number; "1" means 1 column. np.float32 means storing the data as single precision float.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iteration, eps) #Set the criteria parameters
    ret, label, center = cv2.kmeans(img, K, None, criteria, repeat, cv2.KMEANS_RANDOM_CENTERS) #Run the kmeans algorithm; "label" is the cluster number for each data point; "center" is the average density value for each cluster; #ret: the sum of squared distance from each point to their corresponding centers
    center = np.uint16(center) #Store variable center as unsigned 16-bit number (can store numbers from 0 to 2^16).
    image_gray = center[label.flatten()] #Create a image with density as center, but the data is in one dimension; flatten(): return a copy of the array collapsed into one dimension
    image_gray = image_gray.reshape((image.shape)) #Reshape the image to normal
    density = np.empty([K]) #Create a empty variable to store the density information
    for i in range(0, K):
        density[i] = np.average(img[label == i]) #Get the average density value for each cluster
    density=np.flip(np.sort(density)) #Sort the density value from the biggest to the samllest
    return image_gray, density

def extract_images(idx, reader, size):
    image5d = np.empty([size[-1], size[0], size[1], size[2], size[3]], np.uint16) #Create a empty variable to store the image data; np.uint16 means unsigned 16-bit number (can store numbers from 0 to 2^16); size[-1] is how many channels are here; size[0] is how many time points are here; size[1] is how many z stacks are here; size[2] is the range of the y axis; size[3] is the range of the x axis
    for c in range(0, size[-1]):
        for t in range(0, size[0]):
            for z in range(0, size[1]):
                image5d[c, t, z, :, :] = reader.read(c=c, z=z, t=t, series=idx, rescale=False) #Read the image from variable "idx" at channel "c", time point "t", and z stack "z"; "rescale=False" means do not change the original value
    return image5d #Return the result

def density_images(image5d, nc, nt, nz, names, idx):
    for c in range(0, nc): #"for loop" for the channels
        for t in range(0, nt): #"for loop" for the time points
            pdf = plt.figure(figsize=(30, 25)) #Set up the figure size for storing the plots
            for z in range(0, nz): #"for loop" for z stack
                image = image5d[c, t, z, :, :] #Retive the singel image data for image at channel "c", time point "t", and z stack "z"
                plt.subplot(math.ceil(2*nz/6), 6, 2*z+1) #Set up row number (math.ceil(2*nz/6)) and column number (6) in the big figure, and set the image number for the current samll image inside the whole figure; these small images will be put in the whole figure one by one, from left to right, from top to bottom.
                plt.imshow(image, cmap=cm.gray) #Draw the image in gray
                plt.title("%s\nChanel: %d; Time: %d; Z stack: %d" % (names[idx],c,t,z)) #The title of this small plot; the first part (before the "%") is the format; "%s" is a placeholder for a string; "\n" means newline; "\d" is a placeholder for integer; information in (names[idx],c,t,z) is the data to be put in the place of the placeholders.
                
                plt.subplot(math.ceil(2*nz/6), 6, 2*z+2) #Set up another small image
                image_gray, density = kmeans_segmentation(image, 3, 50, 0.05, 10) #Use the kmeans algorithm to do segmentation for the image
                plt.imshow(image_gray, cmap=cm.gray) #Draw the image in gray
                plt.title("%.2f; %.2f; %.2f;\nDiff: %.2f" % (density[0],density[1],density[2],density[0]-density[1])) #The title; "%.2f" is the placeholder for a float number with two decimals; density[0] is the average density value of the nucleus; density[1] is the average density value of the cytoplasm; density[2] is the average density value of the background.
            pdf.savefig(str.split(names[idx], '/')[0] + '_' + str.split(names[idx], '/')[1] + '_Chanel' + str(c) + '_Time' +str(t) + ".pdf") #Save the figure. "names[idx]"" is like "eto50/Position001". str.split(names[idx], '/') will split the "eto50/Position001" into two parts by "/". str.split(names[idx], '/')[0] will get the first part: eto50.
            plt.close() #End the image drawing process

def main():
    javabridge.start_vm(class_path=bioformats.JARS) #Start the java virtual machine, and the bioformats need to use
    filename = sys.argv[1] #Get the file name from the parameter on the command line; sys.argv[1] means the first parameter
    md = bioformats.get_omexml_metadata(filename) #Retrive the metdata of the image file, save as xml format
    names, sizes, resolutions = parse_xml_metadata(md) #Parse the xml file, and get information about picture number, size, and resolution 
    reader = bioformats.ImageReader(filename) #Read the images from the file into the variable "reader"
    
    nidx = sizes.__len__() #Get the number of the image groups; each image group is for one biological sample
    bar = progressbar.ProgressBar(maxval=nidx) #Set up the progress bar
    bar.start() #Start the progress bar
    for idx in range(0, nidx):
        size = sizes[idx] #Get the size of this image group
        nc = size[-1] #Get the channel number
        nt = size[0] #Get how many time points are in the image group
        nz = size[1] #Get how many z stacks are in the image group
        image5d = extract_images(idx, reader, size) #Extract the image from the variable "reader"
        density_images(image5d, nc, nt, nz, names, idx) #Use the function density_images to analyze the image
        bar.update(idx+1)
    bar.finish() # end the progress bar

if __name__ == "__main__": #If this script is run directly, the "__name__" will be "__main__", and the "if" will be "true", and the function "main()" will be run; if this script is imported into another script, the "__name__" here will be the name of this script file.
    main()
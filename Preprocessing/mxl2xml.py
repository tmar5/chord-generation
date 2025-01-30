# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:12:25 2020

@author: acer
"""

from zipfile import ZipFile
import os

path_in = 'Wikifonia/MXL/'
path_out = 'Wikifonia/XML/'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path_in):
    for file in f:
        if file.split('.')[-1] == "mxl":
            files.append(os.path.join(r, file))
            
for thisFile in files:
    #thisFile = "C:/Users/Victouf/Desktop/inf8225/projet/Wikifonia/Donald Kahn, Stanley Styne - A Beautiful Friendship.mxl"
    base = os.path.splitext(thisFile)[0]
    name = base.split('/')[-1]
    print(name)
    os.rename(thisFile, base + ".zip")
    
    
    zf = ZipFile(base + ".zip", 'r')
    try:
        zf.extract(name + '.xml', path_out)
    except:
        zf.extract('musicXML.xml', path_out)
    zf.close()
    os.rename(path_out + 'musicXML.xml', path_out + name + ".xml")
    os.rename(base + ".zip", thisFile)

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:34:14 2020

@author: Victouf
"""

import xml.etree.ElementTree as ET
import copy
import os
import zipfile
import shutil
## ATTENTION créer le dossier CSV avant de lancer le code
path = 'test/XML/'
name = 'test_papa_noel'
versionAvecVoix = True
files = [path + name + ".xml"]
# r=root, d=directories, f = files
dicAlter = {"b" : "-1", "#" : "1" , "bb" : "-2", "##" : "2"}

            
#Faire commencer la liste des harmonie de measure à 0!!!!! rajouter un élt au début sinon

#sol_regrets = ['A:min', 'G:maj', 'E:min', 'C:maj', 'A:min', 'D:min', 'A:min', 'G:maj', 'A:min', 'G:maj', 'E:min', 'C:maj', 'A:min', 'D:min', 'E:maj', 'E:maj', 'E:min', 'G:maj', 'E:maj', 'E:maj', 'E:min', 'D:min', 'E:maj', 'E:maj']
#transpose = -5
#sol_regrets_real = ['G:maj', 'D:maj', 'D:maj', 'G:maj', 'C:maj', 'C:maj', 'A:min', 'D:maj', 'G:maj', 'D:maj', 'D:maj', 'G:maj','C:maj', 'A:min', 'D:maj', 'G:maj', 'G:maj', 'D:maj', 'D:maj', 'G:maj', 'G:maj', 'D:maj', 'D:maj', 'G:maj']
sol = ['A:min', 'D:min', 'A:min', 'D:min', 'D:min', 'G:maj', 'E:maj', 'E:maj', 'A:min', 'D:min', 'A:min', 'A:min']
transpose = 0
'''
0 : time
1 : measure
2 : key_fifths
3 : key_mode
4 : chord_root
5 : chord_type
6 : note_root
7 : note_octave
8 : note_duration
'''

def modifyHarmony(elt, chord_root, chord_type) :
    elt.find('root/root-step').text = chord_root[0]
    elt.find('kind').text = chord_type
    if len(chord_root)>1 : 
        if (dicAlter.get(chord_root[1::]) is not None) :
            if elt.find('root/root-alter') is None :
                newAlter = ET.Element("root-alter")
                newAlter.text = dicAlter.get(chord_root[1::])
                elt.find('root').insert(1,newAlter)
            else :
                elt.find('root/root-alter').text = dicAlter.get(chord_root[1::])
            elt.find('kind').set("text", "")
        else :
            elt.find('kind').set("text", chord_root[1::])
    else :
        if elt.find('root/root-alter') is not None :
            elt.find('root/root-alter').text = "0"


def toMXL(path, output_name):
    path_output = path + output_name + "/"
    try :
        os.makedirs(path_output +"META-INF/")
    except : 
        pass
    containers = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<container><rootfiles><rootfile full-path=\"" + output_name + ".xml\"/></rootfiles></container>"
    with open(path_output + 'META-INF/'+ 'container.xml', 'w+', newline='') as myfile:
        myfile.write(containers)
         
    shutil.make_archive(path + output_name, 'zip', path_output)
    os.rename(path + output_name + ".zip", path + output_name + ".mxl")

        
def add2ndVoix(root):
    partlist = root.find('part-list')
    newScorePart = copy.deepcopy(partlist.find('score-part'))
    
    newScorePart.set('id','P2')
    
    if newScorePart.find('part-name') is None : 
        newScorePart.insert(0, ET.Element('part-name'))
    newScorePart.find('part-name').text = 'Guitare acoustique'
    
    if newScorePart.find('part-abbreviation') is None : 
        newScorePart.insert(1, ET.Element('part-abbreviation'))
    newScorePart.find('part-abbreviation').text = 'Guit.'
    
    if newScorePart.find('score-instrument') is None : 
        newScorePart.insert(2, ET.Element('score-instrument'))
    newScorePart.find('score-instrument').set('id', 'P2-I1')
    
    if newScorePart.find('score-instrument/instrument-name') is None : 
        newScorePart.find('score-instrument').insert(0, ET.Element('instrument-name'))
    newScorePart.find('score-instrument/instrument-name').text = "Guitare acoustique"
    
    if newScorePart.find('midi-device') is None :
        newScorePart.insert(3, ET.Element('midi-device'))
    newScorePart.find('midi-device').set('id', 'P2-I1')
    
    if newScorePart.find('midi-instrument') is None :
        newScorePart.insert(4, ET.Element('midi-instrument'))
    newScorePart.find('midi-instrument').set('id', 'P2-I1')
    
    if newScorePart.find('midi-instrument/midi-channel') is None :
        newScorePart.find('midi-instrumen').insert(0, ET.Element('midi-channel'))
    newScorePart.find('midi-instrument/midi-channel').text = str(2)
    
    if newScorePart.find('midi-instrument/midi-program') is None :
        newScorePart.find('midi-instrumen').insert(1, ET.Element('midi-program'))
    newScorePart.find('midi-instrument/midi-program').text = str(26)
    
    partlist.insert(1, newScorePart)
    
def addPart2(root, transpose):
    newPart = copy.deepcopy(root.find('part'))
    newPart.set('id', 'P2')
    extype = ""
    for measures in newPart.findall('measure'):
        # si measure commence à 1
        if len(sol) < int(measures.get('number')):
            continue
        rootHar = sol[int(measures.get('number'))-1].split(':')[0]
        typeHar = sol[int(measures.get('number'))-1].split(':')[1]
        nbNote = 0
        if measures.find("attributes") is None and typeHar != extype :
            measures.insert(0, ET.Element("attributes"))
        for child in measures.getchildren() :
            if child.tag == "note":
                measures.remove(child)
            elif child.tag =="attributes":
                if child.find("divisions") is None :
                    child.insert(0, ET.Element("divisions"))
                child.find("divisions").text = "1"
                if child.find("key") is None :
                    child.insert(1, ET.Element("key"))
                if child.find("key/mode") is None :
                    child.find("key").insert(0, ET.Element("mode"))
                child.find("key/mode").text = typeHar
        chord = getChord(rootHar, typeHar, transpose)        
        for i in range(3) :
            note = chord[i]
            child = ET.Element('note')
            if child.find('pitch') is None : 
                child.insert(0, ET.Element('pitch'))
            if child.find('type') is None:
                child.insert(2, ET.Element('type'))
            for notechild in child.getchildren():
                if notechild.tag=="pitch":
                    if notechild.find('step') is None : 
                        notechild.insert(0, ET.Element('step'))
                    if notechild.find('octave') is None:
                        notechild.insert(1, ET.Element('octave'))   
                    notechild.find('step').text = note[0]
                    if notechild.find('alter') is None and len(note)>1 :
                        notechild.insert(1, ET.Element('alter'))
                    elif notechild.find('alter') is not None and len(note)==1:
                        notechild.remove(notechild.find('alter'))
                    if len(note)>1:
                        notechild.find('alter').text = dicAlter.get(note[1::])
                    notechild.find('octave').text = "3"
                elif notechild.tag=="duration":
                    notechild.text = "4"
                elif notechild.tag == "type":
                    hasType = True
                    notechild.text = "whole"
                elif notechild.text == "voice":
                    pass
                else : 
                    child.remove(notechild)
            nbNote+=1
            measures.insert(i+1, child)
            
        extype = typeHar
    root.insert(len(root.getchildren()), newPart)
            
 
def getChord(rootHar, typeHar, transpose):
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    l = len(notes)
    i = (notes.index(rootHar) + transpose)%l 
    chord = [notes[i]]
    if typeHar == 'maj':
        j = (i+4)%l
        chord.append(notes[j])
        chord.append(notes[(j+3)%l])
    if typeHar == 'min':
        j = (i+3)%l
        chord.append(notes[j])
        chord.append(notes[(j+4)%l])
    
    return chord
        
    

for f in files :       
    print(name)
    root = ET.parse(f).getroot()
    #ATTENTION prendre en compte l'ordre des balises pour si l'harmonie change en cours de measure 
    for measures in root.findall('part/measure'):
        # si emasure comence à 1
        print(measures.get('number'))
        if len(sol) < int(measures.get('number')):
            continue
        print(measures.get('number'))
        rootHar = sol[int(measures.get('number'))-1].split(':')[0]
        typeHar = sol[int(measures.get('number'))-1].split(':')[1]
        nbHarmony = 0
        for child in measures.getchildren() :
            # ajouter une balise harmony ...
            if child.tag == "note" and nbHarmony ==0 :
                if measures.find('harmony') is not None : 
                    newHarmony = copy.deepcopy(measures.find('harmony'))
                else : 
                    newHarmony = ET.Element("harmony")
                    newHarmony.insert(0, ET.Element("root"))
                    newHarmony.insert(1, ET.Element("kind"))
                    newHarmony.find('root').insert(0, ET.Element('root-step'))
                modifyHarmony(newHarmony, rootHar, typeHar)
                nbHarmony +=1
                measures.insert(0, newHarmony)
            
            elif child.tag == 'harmony' :
                if nbHarmony == 0:
                    modifyHarmony(child, rootHar, typeHar)
                    nbHarmony +=1
                # s'il y a plusieurs harmony dans la mesure
                else :
                    measures.remove(child)
            
    tree = ET.ElementTree()
    if versionAvecVoix :
        add2ndVoix(root)
        addPart2(root, transpose)
    tree._setroot(root)
    output_name = "No Regrets"    
    try:
        os.makedirs(path + output_name)
    except:
        pass        
    tree.write(path + output_name + "/" + output_name + '.xml')
    add2ndVoix(root)
    toMXL(path, output_name)
    
        
        
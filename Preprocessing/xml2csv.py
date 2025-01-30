import xml.etree.ElementTree as ET
import csv
import os

## ATTENTION créer le dossier CSV avant de lancer le code
path = 'Wikifonia/'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path + 'XML/'):
    for file in f:
        if file.split('.')[-1] == "xml":
            files.append(os.path.join(r, file))

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
alterDic = {"-1" : "b", "1" : '#', "-2" : "bb", "2" : "##", "0" : ""}
for f in files :
    base = os.path.splitext(f)[0]
    name = base.split('/')[-1]
    print(name)
    root = ET.parse(f).getroot()
    try:
        os.makedirs(path + 'CSV/')
    except:
        pass 
    with open(path + 'CSV/'+ name + '.csv', 'w+', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        wr.writerow(['time', 'measure', 'key_fifths', 'key_mode', 'chord_root', 'chord_type', 'note_root', 'note_octave', 'note_duration'])
        noteList = ['None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None']
        # ATTENTION prendre en compte l'ordre des balises pour si l'harmonie change en cours de measure 
        for measures in root.findall('part/measure'):
            noteList[1] = measures.get('number')
            for child in measures.getchildren() :
                if child.tag == "attributes" :
                    div = child.find('divisions')
                    if div is not None :
                        division = int(div.text)
                    time = child.find('time')
                    if time is not None :
                        noteList[0] = time.find('beats').text+"/"+time.find('beat-type').text
                    key = child.find('key')
                    if key is not None :
                        noteList[2] = key.find('fifths').text
                        if key.find('mode') is not None :
                            noteList[3] = key.find('mode').text
                        # si la key n'a pas de mode (exemple dans a big hunk o'love)
                        else :
                            noteList[3] = 'None'
                elif child.tag == 'harmony' :
                # ATTENTION G7 dans exemple a guid new year
                    noteList[4] = child.find('root/root-step').text
                    # cas bémole et #
                    if child.find('root/root-alter') is not None :
                         noteList[4] += alterDic.get(child.find('root/root-alter').text)
                    kind = child.find('kind')
                    if kind is not None :
                        
                        noteList[5] = kind.text
                        # cas G7
                        number = kind.get('text')
                        try:
                            if number is not None :
                                noteList[4] += str(int(number))
                        except ValueError :
                            pass
                    # si l'harmony n'a pas de type (exemple dans falling slowly)
                    else :
                        noteList[5] = 'None'
                elif child.tag == 'note' :  
                    if child.find('rest') is not None : 
                        noteList[6] = "rest"
                        noteList[7] = 0
                    else :
                        # bémole alter (exemple dans bubbly, broadway Melody breathless (dans l'harmony aussi))
                        # # alter 1
                        # b alter -1
                        noteList[6] = child.find('pitch/step').text
                        if child.find('pitch/alter') is not None :
                            alter = child.find('pitch/alter').text
                            noteList[6] += alterDic.get(child.find('pitch/alter').text)
                        noteList[7] = child.find('pitch/octave').text  
                    if child.find('duration') is not None :
                        noteList[8] = 4*int(child.find('duration').text)/division
                    # cas où la note n'a pas de duration (exemple dans a guid new year)
                    else : 
                        noteList[8] = 0
                    wr.writerow(noteList)
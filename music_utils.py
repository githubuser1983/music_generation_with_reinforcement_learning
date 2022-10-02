import numpy as np
import random, networkx as nx
import music21 as m21


def kernPause(a1,a2):
    return  1*(a1==a2)

def kernDuration(d1,d2):
    return min(d1,d2)/max(d1,d2)

def kernVolume(v1,v2):
    return min(v1,v2)/max(v1,v2)

def getlowestfraction(x0):
    eps = 0.01

    x = np.abs(x0)
    a = int(np.floor(x))
    h1 = 1
    k1 = 0
    h = a
    k = 1

    while np.abs(x0-h/k)/np.abs(x0)> eps:
        x = 1/(x-a)
        a = int(np.floor(x))
        h2 = h1
        h1 = h
        k2 = k1
        k1 = k
        h = h2 + a*h1
        k = k2 + a*k1
    q = {"numerator": h, "denominator" :k}
    return q



def getRational(k):
    x = 2**(k*(1/12.0))
    return getlowestfraction(x)

def gcd(a,b):
    a = abs(a)
    b = abs(b)
    if (b > a):
        temp = a
        a = b
        b = temp
    while True:
        if (b == 0):
            return a
        a %= b
        if (a == 0):
            return b
        b %= a



def kernPitch(k1,k2):
    q = getRational(k2-k1)
    a,b = q["numerator"],q["denominator"]
    return gcd(a,b)**2/(a*b)


def kernNote(n1,n2):
    p1,d1,v1,r1 = n1
    p2,d2,v2,r2 = n2
    #print(n1,n2)
    k = 1.0/(1+2+4)*((1*kernPitch(p1,p2)+2*kernDuration(d1,d2)+4*kernVolume(v1,v2))*kernPause(r1,r2))
    #if n1==n2:
    #    print("n1,k = ",n1,k)
    return k 

def getCoordinatesOf(intList,kernel=None,nDim=None):
    #print("M0 ....")
    M0 = np.array([[kernel(t1,t2) for t1 in intList] for t2 in intList])
    #print("..M0")
    #print(M0)
    from sklearn.decomposition import PCA
    from sklearn.decomposition import KernelPCA
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    scaler = StandardScaler() #MinMaxScaler((0,1))
    KPCA = KernelPCA(n_components=nDim,kernel='precomputed',eigen_solver='randomized')
    
    #print("fit_transform...")
    Ch0 = KPCA.fit_transform((M0))
    #print("!")
    #print(Ch0)
    X0 = [x for x in 1.0*Ch0]    
    
    #print(X0)
        
    #X0 = scaler.fit_transform(X0)
    
    #invPitchDict = dict(zip(intList,range(len(intList))))
    return Ch0#, invPitchDict

durlist = [[sum([((2**(n-i))) for i in range(d+1)]) for n in range(-8,3+1)] for d in range(2)]
durationslist = []
for dl in durlist:
    durationslist.extend([x for x in dl])
print(durationslist)

from itertools import product

pitchlist = range(21,108+1)
vollist = range(1,128+1)
restlist = [True,False]
noteslist = list(product(pitchlist,durationslist,vollist,restlist))

maxDim = None
P = (getCoordinatesOf(intList = pitchlist, kernel = kernPitch,nDim = maxDim))
V = (getCoordinatesOf(intList = vollist, kernel = kernVolume,nDim = maxDim))
R = (getCoordinatesOf(intList = restlist, kernel = kernPause,nDim = maxDim))
D = (getCoordinatesOf(intList = durationslist, kernel = kernDuration,nDim = maxDim))


def findNearestDuration(duration,durationslist):
    return sorted([(abs(duration-nv),nv) for nv in durationslist])[0][1]


def get_knn_model(X):
    #notes = np.array([[x*1.0 for x in n] for n in notes])
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(0)
    nbrs = NearestNeighbors( algorithm='ball_tree').fit(X)
    return nbrs

def findBestMatches(nbrs,new_row,n_neighbors):
    distances,indices = nbrs.kneighbors([np.array(new_row)],n_neighbors=n_neighbors)
    dx = sorted(list(zip(distances[0],indices[0])))
    #print(dx)
    return dx

def findByRadius(nbrs,new_row,radius):
    distances,indices = nbrs.radius_neighbors([np.array(new_row)],radius=radius)
    dx = sorted(list(zip(distances[0],indices[0])))
    #print(dx)
    return dx

def getVectorForNote(note):
    pitch,duration,volume,isRest = note
    if pitch in pitchlist and volume in vollist and isRest in restlist and duration in durationslist:
        pi = pitchlist.index(pitch)
        vi = vollist.index(volume)
        di = durationslist.index(duration)
        ri = restlist.index(isRest)       
        return np.concatenate((P[pi],D[di],V[vi],R[ri]),axis=None)
    else:
        return None
    
def getMeanVectorForNotes(notes):
    return np.mean([getVectorForNote(note) for note in notes],axis=0)

def getConcatVectorForNotes(notes):
    return np.concatenate(tuple([getVectorForNote(note) for note in notes]),axis=None)

def convertFromM21Format(noteRest):
    import copy
    note = noteRest
    if note.isRest:
        start = note.offset
        duration = float(note.quarterLength)/4.0
        vol = 32 #note.volume.velocity
        pitch = 60
        return (note,((pitch,findNearestDuration(duration,durationslist),vol,True),))
    elif note.isChord:
        note = [n for n in note][0]
        duration = float(note.quarterLength)/4.0
        vol = 32 #note.volume.velocity
        pitch = 60
        return (note,((pitch,findNearestDuration(duration,durationslist),vol,True),))        
    else:
        #print(note)
        start = note.offset
        duration = float(note.quarterLength)/4.0
        pitch = note.pitch.midi
        #print(pitch,duration,note.volume)
        vol = note.volume.velocity
        if vol is None:
            vol = 64 #int(note.volume.realized * 127)
            
        return (note,((pitch,findNearestDuration(duration,durationslist),vol,False),))


def parse_file(xml):
    xml_data = m21.converter.parse(xml)
    score = []
    instruments = []
    for part in xml_data.parts:
        parts = []
        instruments.append(part.getInstrument())
        for note in part.recurse().notesAndRests:
            parts.append(convertFromM21Format(note)) 
        score.append(parts)        
    return score,instruments

def writePitches(fn,inds,tempo=82,instrument=[0,0],add21=True,start_at= [0,0],durationsInQuarterNotes=False):
    from MidiFile import MIDIFile
    import numpy as np

    track    = 0
    channel  = 0
    time     = 0   # In beats
    duration = 1   # In beats # In BPM
    volume   = 116 # 0-127, as per the MIDI standard

    ni = len(inds)
    MyMIDI = MIDIFile(ni,adjust_origin=False) # One track, defaults to format 1 (tempo track
                     # automatically created)
    MyMIDI.addTempo(track,time, tempo)


    for k in range(ni):
        MyMIDI.addProgramChange(k,k,0,instrument[k])


    times = start_at
    for k in range(len(inds)):
        channel = k
        track = k
        for i in range(len(inds[k])):
            pitch,duration,volume,isPause = inds[k][i]
            #print(pitch,duration,volume,isPause)
            track = k
            channel = k
            if not durationsInQuarterNotes:
                duration = 4*duration#*maxDurations[k] #findNearestDuration(duration*12*4)            
            #print(k,pitch,times[k],duration,100)
            if not isPause: #rest
                #print(volumes[i])
                # because of median:
                pitch = int(np.floor(pitch))
                if add21:
                    pitch += 21
                #print(pitch,times[k],duration,volume,isPause)    
                MyMIDI.addNote(track, channel, int(pitch), float(times[k]) , float(duration), int(volume))
                times[k] += duration*1.0  
            else:
                times[k] += duration*1.0
       
    with open(fn, "wb") as output_file:
        MyMIDI.writeFile(output_file)
    print("written")  
    
def writeM21Lists(fn,inds,tempo,instruments,title,author,fileType):
    from music21 import chord
    from music21 import stream
    from music21 import duration
    from music21 import clef, metadata
    import music21 as m
    import copy

    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = author
    tm = m.tempo.MetronomeMark(number=tempo)
    score.append(tm)
    lh = m.stream.Part()
    lh.append(m.instrument.Piano())
    rh = m.stream.Part()
    rh.append(m.instrument.Piano()) #Violin())
    
    def extendPart(part,ll):
        for l in ll:
            part.append(l)
        return part
    
    ourparts = []    
    for i in range(len(inds)):
        mypart = m.stream.Part()
        mypart.append(instruments[i])
        mypart = extendPart(mypart, inds[i])
        ourparts.append(mypart)
    
    for part in ourparts:
        score.append(part)
    if fileType == "musicxml":    
        score.write("musicxml",fp=fn) 
    elif fileType == "mid":    
        score.write("mid",fp=fn) 
"""
EnvironmentManager.py
"""
import io
import struct
import uuid

class ArtifactSnippet:

# struct EDF_SIGNAL {
#     char label[16 + 1];
#     char transducerType[80 + 1];
#     char physDimension[8 + 1];
#     char physMinimum[8 + 1];
#     char physMaximum[8 + 1];
#     char digiMinimum[8 + 1];
#     char digiMaximum[8 + 1];
#     char prefilter[80 + 1];
#     char numSamples[8 + 1];
#     char reserved[32 + 1];
# };
# struct EEGArtifactV4 {
#     char version[20] = "EEGArtifactV4";
#     int channel;
#     int numSamples;
#     char label[255];
#     float sampleStart;
#     float sampleEnd;
#     float artStart;
#     float artEndartEnd;
#     short recDuration;
#     EDF_SIGNAL signalMetadata;
# }

    def __init__(self):
        self.header = None
        self.version = None
        self.channel = None
        self.numRecs = None
        self.artLabel = None
        self.sampleStart = None
        self.sampleEnd = None
        self.artStart = None
        self.artEnd = None
        self.recDuration = None
        self.chanLabel = None
        self.transducerType = None
        self.physDim = None
        self.physMin = None
        self.physMax = None
        self.digiMin = None
        self.digiMax = None
        self.prefilter = None
        self.numSamples = None
        self.reserved = None
        self.data = None
        self.flags = None
        self.idx = 0

    def loadfile(self, filename=None):
        if not filename:
            return

        with open(filename, "rb") as f:


            rawHeader = f.read(568)
            self.version, self.channel, self.numRecs, self.artLabel, self.sampleStart, self.sampleEnd, \
            self.artStart, self.artEnd, self.recDuration, self.chanLabel, self.transducerType, self.physDim,\
            self.physMin, self.physMax, self.digiMin, self.digiMax, self.prefilter, self.numSamples,\
            self.reserved = struct.unpack('20sii255sffffh16sx80sx8sx8sx8sx8sx8sx80sx8sx32sx', rawHeader)

            # now there are two more things to read: an array of shorts of len numRecs an::d
            # an array of bools of len numRecs.

            rawData = f.read(self.numRecs*2)
            rawFlags = f.read(self.numRecs*1)

            self.data = struct.unpack(str(self.numRecs)+'h', rawData)
            self.flags = struct.unpack(str(self.numRecs)+'?', rawFlags)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self.data):
            raise StopIteration

        self.idx = self.idx + 1

        retVal = ((float(self.data[self.idx-1])) + \
                  (float((self.digiMax.decode()))-(float(self.digiMin.decode())) )
                   / 2) / (float(self.digiMax.decode())-float(self.digiMin.decode()))

        return (retVal, self.flags[self.idx-1])


class EnvironmentManager:

    envId = None
    snip = None
    vals = None

    def __init__(self):
        self.envId = uuid.uuid4()

    def startEpisode(self, fileName=None):
        self.snip = ArtifactSnippet()

        if fileName == None:
            self.snip.loadfile("/home/robabbott/dev/edfviewer/002_art/00000254_s005_t000_ch009-8.art")
        else:
            self.snip.loadfile(filename)

        self.vals = iter(self.snip)

    def getInitialState(self):
        nextState = self.snip.__next__()
        return nextState

    def submitAction(self, action):
        isTerminal = None
        nextState = self.snip.__next__()
        score = None

        return  nextState


if (__name__ == "__main__"):
    em = EnvironmentManager()

    em.startEpisode()

    initState = em.getInitialState()

    print(initState)

    for i in em.snip:
        print(i)

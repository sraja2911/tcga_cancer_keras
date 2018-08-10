import os, sys
from os.path import join as oj
import girder_client
### You need to change this depending on where you want to store the rawImage Data

## Make the below folder appropriate for your configuration
imageFolderOnLocalMachine = "/home/dagutman/devel/KerasSimpsons_Tensorflow/rawImageData/"
imageFolderUID = '5b0ebfab92ca9a001733549d'

testSetFolderUID = '5b197de892ca9a001735466a'
testSetID = 'Kaggle20ClassTestSet'

class LinePrinter():
    """
    Print things to stdout on one line dynamically
    """
    def __init__(self,data):
        sys.stdout.write("\r\x1b[K"+data.__str__())
        sys.stdout.flush()

def syncImageDataFromGirder( imageFolderOnLocalMachine=imageFolderOnLocalMachine ):
    if not os.path.isdir(imageFolderOnLocalMachine):
        os.makedirs(imageFolderOnLocalMachine)

    gc = girder_client.GirderClient(apiUrl='http://candygram.neurology.emory.edu:8080/api/v1')
    # ### Get a list of all the image folders on the DSA Server
    ### Go through the image folders on the DSA Server and see how many images are in each folder and if all the
    ### local images are available/uploaded
    ## This determines which images are already uploaded to Girder, and in the next block I check if it's uploaded

    ### Now download testingSetData
    print "Now downloading the validation set"
    imagesProcessed = imagesDownloaded = 0

    for cf in gc.listFolder(testSetFolderUID):  ## cf = characterFolder
        testingSetCharImages = list( gc.listItem(cf['_id']))
        print "There are a total of %d items for %s" % (len(testingSetCharImages), cf['name'])

        ## Check each image item for appropriate tags.. i.e. characterClass and largeItem
        localFolderForChar = os.path.join(imageFolderOnLocalMachine,'validation',cf['name'])
        if not os.path.isdir(localFolderForChar):
            os.makedirs(localFolderForChar)

        for itm in testingSetCharImages:
            imagesProcessed +=1
            ### Check and see if the image has already been downloaded
            imageNameWPath = os.path.join(localFolderForChar, itm['name'])
            if not os.path.isfile( imageNameWPath):
                ### I first need to get the list of files uassociatd with the item, and download the first one
                filesForItem = list(gc.listFile(itm['_id']))

                gc.downloadFile(filesForItem[0]['_id'],imageNameWPath)
                imagesDownloaded+=1
            LinePrinter("A total of %d images have been processed, and %d have been just Downloaded" % (imagesProcessed, imagesDownloaded))
        print ### Adds a linefeed between each character folder

    imagesProcessed = imagesDownloaded = 0
    ## This will download the training data
    for cf in gc.listFolder(imageFolderUID):  ## cf = characterFolder
        characterImages = list( gc.listItem(cf['_id']))
        print "There are a total of %d items for %s" % (len(characterImages), cf['name'])
        
        trainingSetCharImages = []

        for x in characterImages:
            try:
                tsName = x['meta']['trainingSetName']
                trainingSetCharImages.append(x)
            except:
                print "tag not fond.."

        if len(trainingSetCharImages) > 0:
            localFolderForChar = os.path.join(imageFolderOnLocalMachine,'training',cf['name'])
            if not os.path.isdir(localFolderForChar):
                os.makedirs(localFolderForChar)


            for itm in trainingSetCharImages:
                imagesProcessed +=1
                ### Check and see if the image has already been downloaded
                imageNameWPath = os.path.join(localFolderForChar, itm['name'])
                if not os.path.isfile( imageNameWPath):
                	### I first need to get the list of files uassociatd with the item, and download the first one
                	filesForItem = list(gc.listFile(itm['_id']))

                	gc.downloadFile(filesForItem[0]['_id'],imageNameWPath)
                	imagesDownloaded+=1
                LinePrinter("A total of %d images have been processed, and %d have been just Downloaded" % (imagesProcessed, imagesDownloaded))
        print ### Adds a linefeed between each character folder


if __name__ == "__main__":
    syncImageDataFromGirder()
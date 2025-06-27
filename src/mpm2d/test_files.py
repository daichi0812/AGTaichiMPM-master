import glob, os, ntpath, shutil
import time
from enum import Enum

class Mode(Enum):
    IDLE = 1
    PROCESSING = 2

path_to_processing = './to_process/'
path_processing = './processing/'
path_processed = './processed/'
file_wild_card = '*.xml'

count = 0

mode = Mode.IDLE
file_processing = ''

main_loop_count = 0

def fetchFileToProcess():
    files = glob.glob(path_to_processing + file_wild_card)
    if len(files) > 0:
        return files[0]
    else:
        return ''

while True:
    if mode == Mode.IDLE:
        print('idle... [' + str(count) + ']')
        fn = fetchFileToProcess()
        if fn != '':
            # cleaning up idle mode
            count = 0

            # preparing for switching to processing mode
            print('found :' + fn)
            file_processing = ntpath.basename(fn)
            shutil.move(fn, path_processing)
            mode = Mode.PROCESSING
            main_loop_count = 0
            print('processing phase will start')
            continue
        
        count += 1
        time.sleep(0.5)

    elif mode == Mode.PROCESSING:
        if main_loop_count >= 10:
            # cleaning up processing mode
            main_loop_count = 0
            shutil.move(path_processing + file_processing, path_processed)
            # save output 

            mode = Mode.IDLE
            print('process done. turning to idle mode')

        else:
            main_loop_count += 1
            time.sleep(1)
            print('processing (count until main_loop_count = 10) current: ' + str(main_loop_count))

    


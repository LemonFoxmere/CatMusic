import os

# read all files within midcsv folder
all_file_paths = os.listdir('midcsv')
# fix foledr prefix
all_file_paths = list(map(lambda x : os.path.join('midcsv',x), all_file_paths))
all_files = []
for path in all_file_paths:
    all_files.append(open(path))

# iterate through every file and generate output files for each. TODO, can cope with tqdm for visualized progress
for fin in all_files:
    tempo = 0 # extract tempo rate. Refer to devnote for converting tempo to absolute time unit
    line = fin.readline().split() # initial read line

    # read in meta data and move to read position
    while(line[1] == '0,' or 'End_track' in line[2]):
        if('Tempo' in line[2]): # extract tempo
            tempo = int(line[3])
        line = fin.readline().split()

    # audio extraction and compression starts here
    output = ''
    time = 0 # timestamp for tracking and syncing
    noteOn = False
    stm = [] # short term memory storage
    while(line[2].strip(',') != 'End_track'):
        newTime = int(line[1].strip(','))
        noteOn = line[2].strip(',') == 'Note_on_c' # update it to on or off
        # this if statement checks whether or not
        if(len(stm) == 0 and noteOn): # this means that a new line midi line has started, append a silence
            output += str(time) + ',' + str(newTime) + ',' + '{0}\n'
            time = newTime # update the last time pointer

        if(noteOn): # if the note is pressed down
            if(len(stm) != 0): # if the note is on for this one, and the last section is not silent, it means another frequency band has started
                output += str(time) + ',' + str(newTime) + ',' + str(set(stm)).replace(' ', '').replace(',','x') + '\n'
                time = newTime # update last time pointer
            # add the current Frequency to stm
            stm.append(int(line[4].strip(','))) # add whatever frequency to stm
        else: # if it is noteOff, that means that one of the portion has ended, and we add what we have
            # first add the desired note sections to the output
            output += str(time) + ',' + str(newTime) + ',' + str(set(stm)).replace(' ', '').replace(',','x') + '\n'
            # second remove ended frequency from stm
            stm.remove(int(line[4].strip(',')))
            # third update last time pointer
            time = newTime
        line = fin.readline().split()
        # end while
    # end extraction
    output_file_name = os.path.join("gen_raw_midi", fin.name.split(os.sep)[-1].split('.')[0] + '.rawMid')
    fout = open(output_file_name, 'w')
    fout.write(str(tempo)+'\n')
    fout.write(output)
    fout.close()

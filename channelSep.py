# ╭━━━┳━━━┳━━━┳━━━┳━━━┳━━━┳━━━┳━━━━┳━━━┳━━━╮╱╭━━━┳━━━╮╭━╮╱╭┳━━━┳━━━━╮╭╮╱╭┳━━━┳━━━╮
# ╰╮╭╮┃╭━━┫╭━╮┃╭━╮┃╭━━┫╭━╮┃╭━╮┃╭╮╭╮┃╭━━┻╮╭╮┃╱╰╮╭╮┃╭━╮┃┃┃╰╮┃┃╭━╮┃╭╮╭╮┃┃┃╱┃┃╭━╮┃╭━━╯
# ╱┃┃┃┃╰━━┫╰━╯┃╰━╯┃╰━━┫┃╱╰┫┃╱┃┣╯┃┃╰┫╰━━╮┃┃┃┣╮╱┃┃┃┃┃╱┃┃┃╭╮╰╯┃┃╱┃┣╯┃┃╰╯┃┃╱┃┃╰━━┫╰━━╮
# ╱┃┃┃┃╭━━┫╭━━┫╭╮╭┫╭━━┫┃╱╭┫╰━╯┃╱┃┃╱┃╭━━╯┃┃┃┣╯╱┃┃┃┃┃╱┃┃┃┃╰╮┃┃┃╱┃┃╱┃┃╱╱┃┃╱┃┣━━╮┃╭━━╯
# ╭╯╰╯┃╰━━┫┃╱╱┃┃┃╰┫╰━━┫╰━╯┃╭━╮┃╱┃┃╱┃╰━━┳╯╰╯┣╮╭╯╰╯┃╰━╯┃┃┃╱┃┃┃╰━╯┃╱┃┃╱╱┃╰━╯┃╰━╯┃╰━━╮
# ╰━━━┻━━━┻╯╱╱╰╯╰━┻━━━┻━━━┻╯╱╰╯╱╰╯╱╰━━━┻━━━┻╯╰━━━┻━━━╯╰╯╱╰━┻━━━╯╱╰╯╱╱╰━━━┻━━━┻━━━╯

# fin = open("bwv772.csv")
#
# newMidi = ""
# currentLine = fin.readline()
# currentLine
# currentLine.split()[1]
# # the third element must contain the word NOTE to start filtration process
# while(str(currentLine.split()[1]) == '0,' or "End_track" in currentLine):
#     newMidi += currentLine
#     currentLine = fin.readline()
#
# newMidi
#
# while(not "End_track" in currentLine):
#     lineArr = currentLine.split()
#     if(len(lineArr) > 3):
#         lineArr[3] = '1,'
#     yes = ""
#     for line in lineArr:
#          yes += line
#     newMidi += yes[:-1] + '\n'
#     currentLine = fin.readline()
#
# newMidi
#
# fout = open("outputFilteredMidi.csv", 'w')
# newMidi += currentLine
# newMidi += "0, 0, End_of_file\n"
# fout.write(newMidi)
# fout.close()

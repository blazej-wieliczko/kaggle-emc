from numpy import *
import re

def logloss(h, y):
    sample_size = h.shape[0]

    sum = 0.0
    count = 0
    for rowid, row in enumerate(h):
        class_id = y[rowid]
        if class_id < h.shape[1]:
            sum += log(row[class_id])
            count += 1
    #print 'logloss included', count, "entries, while there were", sample_size, "in total"
    return -sum/float(sample_size)

from collections import defaultdict

def precision(h, y):
    count = 0
    for rowid, row in enumerate(h):
        class_id = y[rowid]
        count += argmax(row) == class_id
    return count / float(h.shape[0])

def precision_with_missed(h, y):
    count = 0
    missed_rowids = []
    for rowid, row in enumerate(h):
        class_id = y[rowid]
        count += argmax(row) == class_id
        if (argmax(row) != class_id):
            missed_rowids.append(rowid)

    return missed_rowids


header = '"id","class0","class1","class2","class3","class4","class5","class6","class7","class8","class9","class10","class11","class12","class13","class14","class15","class16","class17","class18","class19","class20","class21","class22","class23","class24","class25","class26","class27","class28","class29","class30","class31","class32","class33","class34","class35","class36","class37","class38","class39","class40","class41","class42","class43","class44","class45","class46","class47","class48","class49","class50","class51","class52","class53","class54","class55","class56","class57","class58","class59","class60","class61","class62","class63","class64","class65","class66","class67","class68","class69","class70","class71","class72","class73","class74","class75","class76","class77","class78","class79","class80","class81","class82","class83","class84","class85","class86","class87","class88","class89","class90","class91","class92","class93","class94","class95","class96"\n'


def add_lineno_and_header(filename):
    infile = open(filename)
    outfile = open(filename + '-numbered', 'w')

    pattern = re.compile(r'^\d\.[^,]*,')

    outfile.write(header)
    i = 1
    for line in infile:
        replacement = pattern.sub(str(i) + ',', line)
        outfile.write(replacement)
        i += 1

    outfile.close()
    infile.close()

#addlinenumbers('../submission/submission-20.23.08')



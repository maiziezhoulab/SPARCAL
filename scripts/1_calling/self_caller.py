from pysam import FastaFile
from pysam import FastaFile
from collections import OrderedDict 
import pandas as pd
import numpy as np
import pysam
from argparse import ArgumentParser

def parse_chrName(chromo_name):
  chromo_name = chromo_name.split("chr")[1]
  # chromo_name = chromo_name
  try:
    int(chromo_name)
    return int(chromo_name) - 1
  except ValueError:
    return chromo_name


def binary_search(lst, low, high, x):
  if high >= low:
    mid = (high + low) // 2
    if lst[mid] == x:
      return mid
    elif lst[mid] > x:
      return binary_search(lst, low, mid - 1, x)
    else:
      return binary_search(lst, mid + 1, high, x)
  else:
    return -1

def compare(pos, seqread, seqref, po, ref,re,qn,q):
  seqr = seqread.capitalize()
  seqre = seqref.capitalize()
  dif = [i for i,(a1,a2)  in enumerate(zip(seqr,seqre)) if a1!=a2]
  for i in dif:
    pos.append(int(po + i)) # differences pos on ref
    ref.append(seqref[i:i + 1]) # AGCT ON REF
    re.append(seqread[i:i + 1]) # AGCT ON READ
    qn.append(q) # READ NAme
    
    
def getDuplicatesWithInfo(listOfElems):
    ''' Get duplicate element in a list along with thier indices in list
     and frequency count'''
    dictOfElems = dict()
    index = 0
    # Iterate over each element in list and keep track of index
    for elem in listOfElems:
        # If element exists in dict then keep its index in lisr & increment its frequency
        if elem in dictOfElems:
            dictOfElems[elem][0] += 1
            dictOfElems[elem][1].append(index)
        else:
            # Add a new entry in dictionary 
            dictOfElems[elem] = [1, [index]]
        index += 1    
    return dictOfElems


def extract_chromosome_regions(bed_file_path, chromo_number):
    # Initialize lists to hold the start and end positions
    region_start = []
    region_end = []
    
    # Open and read the BED file
    with open(bed_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # Split each line into its components
            chromosome = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            
            # Check if the current line's chromosome matches the specified chromosome number
            if chromosome == str(chromo_number):
                region_start.append(start)
                region_end.append(end)
    
    return region_start, region_end


def gene(standard, chromo, bamfile, bedfile, header, out):
  #-------------------------------------------------------------#
  #-------------------------------------------------------------#
  seqo = FastaFile(standard)
  tmp = seqo.fetch(chromo)
  try:
      samfile = pysam.AlignmentFile(bamfile, "rb")
  except OSError as e:
      print(f"Error opening BAM file: {e}")
      return
  position = []
  reference = []
  readss = []
  qna = []
  chromo_name = ''
  for read in samfile.fetch(until_eof = True):

    seq = read.seq
    pos = read.pos # read start pos on ref (initial ref pos of a read)
    a = read.cigar # list of tuple (N:Len)
    a = list(map(list, a)) 
    c = list(map(list, a))
    readpo = 0 # initial read pos
    chromo_name = read.reference_name
    # print(len(seq))
    # read pos & refe pos correlation
    # if (read.rname == parse_chrName(chromo)) and (seq is not None) and (read.qname == 'A00836:217:HVHCFDSXX:1:1137:19117:22232'):
    # if (read.reference_name == chromo) and (seq is not None) and (read.qname == 'A00836:217:HVHCFDSXX:4:1548:27453:30655'):
    # if (read.reference_name == chromo) and (seq is not None) and (read.qname == 'A00836:217:HVHCFDSXX:3:2453:20211:36119'):
    if (read.reference_name == chromo) and (seq is not None):
      qn = read.qname # read name

      # print("ref pos: ", pos)
      # print("query name: ", qn)
      # print("Read Cigar: ", a)
      for q in range(len(a)): # loop cigar
        # record ref cordinate changes
        # checking N = 0, 2, 3
        
        if a[q][0] in [0, 2, 3]:
          a[q][0] = pos
          pos = pos + a[q][1]
          a[q][1] = pos
          # a[q][1] += pos
        if a[q][0] in [1, 4, 5]:
          a[q][0] = 0
          a[q][1] = 0

        # record read cordinate changes
        if c[q][0] in [0, 1, 4]:
          c[q][0] = readpo
          readpo = readpo + c[q][1] 
          c[q][1] = readpo
          # c[q][1] += readpo
        if c[q][0] in [2, 3, 5]:
          c[q][0] = 0
          c[q][1] = 0

        # if both the matched reference seg and matched read seg has a non-zero length
        if(a[q][1] != 0 and c[q][1] != 0):
          readseq = seq[c[q][0]:c[q][1]]
          refseq = tmp[a[q][0]:a[q][1]]
          # print(position, readseq, refseq)
          compare(position, readseq, refseq, a[q][0], reference, readss, qna, qn)

  samfile.close()

  #-------------------------------------------------------------#
  #-------------------------------------------------------------#
  result = np.column_stack((position, reference, readss, qna))
  res = sorted(result, key=lambda x : int(x[0]))

  # print(res)
  regions_start, regions_end = extract_chromosome_regions(bedfile, chromo)
  # print(chromo, regions_start, regions_end)
  result_po = []
  result_ref = []
  result_re = []
  result_qn = []
  
  # filter snv by bed regions
  for a in range(len(res)): 
    for i in range(len(regions_start)):
      if int(res[a][0]) > int(regions_start[i]) and int(res[a][0]) < int(regions_end[i]):
        result_po.append(int(res[a][0]))
        result_ref.append(res[a][1])
        result_re.append(res[a][2])
        result_qn.append(res[a][3])

  #-------------------------------------------------------------#
  #-------------------------------------------------------------#
  res = list(OrderedDict.fromkeys(result_po))
  samfile = pysam.AlignmentFile(bamfile, "rb")
  newpo = []
  ref = []
  readsq = []
  for read in samfile.fetch(until_eof = True):
    seq = read.seq
    pos = read.pos
    a = read.cigar
    a = list(map(list, a))
    c = list(map(list, a))
    readpo = 0
    # if (read.rname == parse_chrName(chromo) and seq != None):
    # if (read.reference_name == chromo) and (seq is not None) and (read.qname == 'A00836:217:HVHCFDSXX:4:1548:27453:30655'):
    if (read.reference_name == chromo) and (seq is not None):
      qn = read.qname
      for q in range(len(a)):
        if a[q][0] in [0, 2, 3]:
          a[q][0] = pos
          pos = pos + a[q][1]
          a[q][1] = pos
        if a[q][0] in [1, 4, 5]:
          a[q][0] = 0
          a[q][1] = 0
        if c[q][0] in [0, 1, 4]:
          c[q][0] = readpo
          readpo = readpo + c[q][1] 
          c[q][1] = readpo
        if c[q][0] in [2, 3, 5]:
          c[q][0] = 0
          c[q][1] = 0
        if(a[q][1] != 0 and c[q][1] != 0):
          readseq = seq[c[q][0]:c[q][1]]
          refseq = tmp[a[q][0]:a[q][1]]
          for k in res:
            # compare each different pos on ref with current ref
            if k <= a[q][1] and k >= a[q][0]:
              newpo.append(k) # record diff pos on ref
              readsq.append(readseq[k - a[q][0]:k + 1 - a[q][0]]) # record corresponding read base 


  samfile.close()

  # count how many alt and refs
  a = getDuplicatesWithInfo(newpo)
  # print(a)
  pos_new = []
  gt_new = []
  ref_count = []
  alt_count = []
  for key, value in a.items():
    stri0 = ""
    stri1 = ""
    cur_ref_alio_count = 0
    cur_alt_alio_count = 0
    for i in value[1]:
      ind = binary_search(result_po, 0, len(result_po) - 1, int(key))
      if result_ref[ind].upper() != readsq[i].upper():
        stri1 += "1"
        cur_alt_alio_count += 1
      if result_ref[ind].upper() == readsq[i].upper():
        stri0 += "0"
        cur_ref_alio_count += 1
    if len(stri1) != 0 and len(stri0) == 0:
      gt_new.append("1/1")
    else:
      gt_new.append("0/1")
    pos_new.append(int(key))
    ref_count += [cur_ref_alio_count]
    alt_count += [cur_alt_alio_count]
  # print("result_po: ", result_po)
  # print("result_ref:", result_ref)
  # print("result_re:",result_re)
  # print("result_qn:",result_qn)
  # print("newpo:",newpo)
  # print("readsq:", readsq)
  # print("a: ", a)
  # print("pos_new:",pos_new)
  # print("ref_count: ", ref_count)
  # print("alt_count: ", alt_count)
  #-------------------------------------------------------------#
  #-----------------------Output---------------------#
  #-------------------------------------------------------------#
  # f = open(header, "r")
  # bam_name = bamfile.split('/')[-1].split('.')[0]
  with open(out, 'a', encoding='UTF8', newline='') as my_file:
    # for i in f:
    #   my_file.write(i)
    for i in range(0, len(pos_new)):
      arr = np.empty(10, dtype=object)
      gt = gt_new[i]
      # chromo_name = chromo_lst_in_bam[chromo_lst_defined.index(chromo_name)]
      # arr[0] = chromo_lst_in_bam[chromo_lst_defined.index(chromo)]
      arr[0] = chromo
      # arr[1] = pos_new[i] + 1
      arr[1] = pos_new[i] + 1
      arr[2] = "."
      arr[3] = result_ref[result_po.index(pos_new[i])]
      arr[4] = result_re[result_po.index(pos_new[i])]
      arr[5] = "20"
      arr[6] = "PASS"
      arr[7] = "SVTYPE=SNV|" + str(ref_count[i]) + '|' + str(alt_count[i])
      arr[8] = "GT:PS"
      gtstr = ""
      ps = result_qn[result_po.index(pos_new[i])].split('_')[0][2:]
      arr[9] = gt + ":" + ps
      my_file.write('\t'.join(str(item) for item in arr) + '\n')

def main():
    parser = ArgumentParser(description="Script description.")
    parser.add_argument('--reference_seq')
    parser.add_argument('--chromosome', type=str)
    parser.add_argument('--bamfile')
    parser.add_argument('--bedfile')
    parser.add_argument('--header')
    parser.add_argument('--out')
    args = parser.parse_args()
    gene(args.reference_seq, args.chromosome, args.bamfile, args.bedfile, args.header, args.out)

if __name__ == "__main__":
    main()
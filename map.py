
import csv

#---------------------------------------

# matching file produced by similarity.lua
match_file = 'match_dic.csv'

# mapping table between local image names and flickr urls
map_file = 'im2url.csv'

# output results csv file
out_file = 'results.csv'
#---------------------------------------

def dic2csv(dic, csv_path, field1, field2):

  with open(csv_path, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = [field1, field2])
    #writer.writeheader()

    for key, val in dic.iteritems():
        writer.writerow({field1 : key,
                         field2 : val
                        })

def csv2dic(csv_path, field1, field2):
  d = {}
  with open(csv_path, 'r') as csvfile:
      reader = csv.DictReader(csvfile)

      for row in reader:
          d[row[field1]] = row[field2]

  return d


match_d = csv2dic(match_file, 'mod', 'ori')
map_d = csv2dic(map_file, 'imname', 'url')
out_d = {}

for mod, ori in match_d.iteritems():
  out_d[mod] = map_d[ori]

dic2csv(out_d, out_file, 'mod', 'url')


import flickrapi
import xml.etree.ElementTree as ET
import urllib
import os
import csv

def dic2csv(dic, csv_path, field1, field2):

  with open(csv_path, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = [field1, field2])
    writer.writeheader()

    for key, val in dic.iteritems():
        writer.writerow({field1 : key,
                         field2 : val
                        })


flickr_photos_dir = 'flickr_photos'
api_key = u'c871b759c44a9c7d3ced100ba2cf4dd0'
api_secret = u'da8022494490182e'

name_to_url = {}

flickr = flickrapi.FlickrAPI(api_key, api_secret)
for photo in flickr.walk(user_id='143060054@N02', tags = 'people'):
    #print ET.tostring(photo, encoding='utf8', method='xml')
    #url = 'http://farm' + photo.get('farm') + '.staticflickr.com/' + \
    #photo.get('server') + '/' + photo.get('id') + '_' + photo.get('secret') + '_n.jpg'
    imname = photo.get('id') + '_' + photo.get('secret') + '.jpg'
    url = 'http://static.flickr.com/' + photo.get('server') + '/' + imname

    print 'Fetching ' + url
    impath = os.path.join(flickr_photos_dir, imname) 
    urllib.urlretrieve(url, impath)
    name_to_url[imname] = url 



dic2csv(name_to_url, 'im2url.csv', 'imname', 'url')

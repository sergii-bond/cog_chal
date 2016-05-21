import simplejson as json
import sys
import csv

def load_csv(csv_name):
    output_dict = {}
    with open(csv_name) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            output_dict[row[0]] = row[1]
    return output_dict

def score(ground_truth, predictions):
    num_correct = 0
    num_matched = len(ground_truth.keys())
    for filename in predictions.keys():
        try:
            if ground_truth[filename] == to_http(predictions[filename]):
                num_correct+=1
        except:
            pass
    accuracy  = float(num_correct)/float(num_matched)
    return accuracy

def to_http(url):
    if url.startswith('https'):
        url = url.split('https')[1]
        http = "http"+url
        return http
    return url

if __name__ == "__main__":
    #make sure the script is being used correctly
    if len(sys.argv) != 2:
        print("Incorrect usage! Correct usage is:\npython score_results.py <your csv file>")
        sys.exit()

    #make sure file is a .csv
    filename = str(sys.argv[1])
    if not filename.endswith(".csv"):
        print("must be csv file ending with '.csv' !")
        sys.exit()

    #load the validation set data
    val_gt = load_csv('validation_set.csv')

    #go through the csv and score the results
    val_preds = load_csv(filename)

    #get results
    results = score(val_gt, val_preds)
    print("Accuracy: {}".format(results))

    
            


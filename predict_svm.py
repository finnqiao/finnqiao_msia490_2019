import pickle
import sys
import json

if __name__ == '__main__':

    svm_model = pickle.load(open('svm_model.pkl', 'rb'))

    text = ' '.join([str(word) for word in sys.argv[1:]])

    label = svm_model.predict([text])

    print({'label': str(label)})

    with open('label_results.txt', 'w') as outfile:
        json.dump({'label': str(label)}, outfile)

import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc

import torch
import cv2
from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import pandas as pd

app = Flask(__name__)

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
   transforms.ToTensor(),
   normalize
])

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':

        # get uploaded image file if it exists
        file = request.files['image']
        if not file: return render_template('index.html', label="No file")

        # read in file as raw pixels values
        # (ignore extra alpha channel and reshape as its a single image)
        #img = misc.imread(file)

        img_pil = Image.open(file)
        #img_cv = cv2.imread(file)

        img_pil.thumbnail((224,224),Image.ANTIALIAS)
        img_pil_processed = preprocess(img_pil)
        img_pil_input = Variable(img_pil_processed.unsqueeze(0))
        result = model(img_pil_input)
        set_labels =['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
             'Fibrosis','Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
             'Pneumonia','Pneumothorax']

        h_x = (result).data.squeeze()
        probs, idx = h_x.sort(0, True)

        label = pd.DataFrame([(set_labels[idx[i]], probs[i]) for i in range(0,14)], columns = ["Disease", "Probability"])



		#img = img[:,:,:3]
		#img = img.reshape(1, -1)

		# make prediction on new image
		#prediction = model.predict(img)

		# squeeze value from 1D array and convert to string for clean return
		#label = str(np.squeeze(prediction))

		# switch for case where label=10 and number=0
		#if label=='10': label='0'

        return render_template('index.html', label=label.to_html())


if __name__ == '__main__':
	# load ml model
	#model = joblib.load('model.pkl')
    model = torch.load("Model_65_10_4-21-18.model", map_location=lambda storage, loc: storage)
    # start api
    app.run(host='0.0.0.0', port=8000, debug=True)

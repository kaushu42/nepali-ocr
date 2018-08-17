# nepali-ocr
# Handwriting recognition for nepali characters (क-ज्ञ and ०-९)

## Not an actual OCR!
A nepal character recogition program.
Created in Keras using CNN. There are only convolutional and pooling layers. No dense layers are present.

Weights are included in the weights folder.

'model.h5' is the final trained model file. Load it in keras using
  > keras.models.load_model('./weights/model.h5')
Intermediate training weights are also preset in the folder.


Trained by splitting dataset into train, test and cross validation data.

   -> 69000 were used for training out of which 46229 were used as training set and the rest for validation.
 
  -> 23000 were used as test set.

The train accuracy was 96.62%, the validation accuracy was 89.579% and the test accuracy was 88.62%.

The dataset used in this project was taken from github user Prasanna1991, https://github.com/Prasanna1991/DHCD_Dataset

## Bibtex
```
@inproceedings{acharya2015deep,
  title={Deep learning based large scale handwritten Devanagari character recognition},
  author={Acharya, Shailesh and Pant, Ashok Kumar and Gyawali, Prashnna Kumar},
  booktitle={Software, Knowledge, Information Management and Applications (SKIMA), 2015 9th International Conference on},
  pages={1--6},
  year={2015},
  organization={IEEE}
}
```

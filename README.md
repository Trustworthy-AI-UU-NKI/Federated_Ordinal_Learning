# Heterogeneous Federated Ordinal Learning

github repository of the submission paper "Understanding the Impact of Client Heterogeneity on Ordinal Classification in Federated Medical Image Analysis", submitted to [MIDL 2025](https://2025.midl.io/). 

To split the data according to the Bernoulli-Dirichlet distribution you can use the ```Splitter.py``` script in the folder ```data```. The ```.csv``` file with the pre-generated splits is in the same folder, ```classification_8_classes.csv```.

You can train the different FL methods by running the appropriate ```train_FLMODELNAME.py``` script. You can change the config options as command line arguments. 

To test, you can use ```test.py``` to test the models and generate the confusion matrices. 

### TO-DOs

* Code refactoring: create a base class for the local update, with child classes that inherit from this base class for the different FL methods.
* Add requirements.txt
* Upload sample of bash file to run the scripts.

### Acknowledgements

We thank [wnn2000](https://github.com/wnn2000/FedIIC) for the code base for federated learning.

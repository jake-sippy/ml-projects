## ML-projects
Just a lazily thought out repo to dump any of the few cute little ML-based
things I made while trying to learn sklearn, numpy, pandas and the like.
- iris:
reads in the the famous iris classification dataset from a csv
and then makes a prediction about some random parameters I made up.

- mnist:
has a trainer that makes a classifier from the MNIST handwritten 
character database and then checks its accuracy on the testing database.
It also has the window.py file that uses python's tkinter library to
create a drawing canvas and tries to make live predictions about what
digit you're drawing. It requires a saved classifier, so mnist-trainer
has to berun first.


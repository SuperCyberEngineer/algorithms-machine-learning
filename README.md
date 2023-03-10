---------------
ML experiments
---------------

## ML questions
- can we define arbitrary model like non-sequential models on existing toolings like tensorflow and pytorch?
- can we have some methods to denoise the dataset / prevent to involve lots of noise?
  - on large dataset
    - RANSAC
    - cross validation to classify out the wrong data
- can we prevent overfitting and really find a ~ unique model for a ~ clean dataset via statistic methods?
  - by introducing special loss metric to the dataset and successfully outline the manual added noise (i.e. during under-fitting)
    - like {0:0, 100:10000, 300:50}

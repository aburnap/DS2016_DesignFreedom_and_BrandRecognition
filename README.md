## Balancing Design Freedom and Brand Recognition in the Evolution of Brand Character

Code for the 2016 Design Science Journal paper, and 2015 International Design Engineering Technical Conference Paper for measuring both design freedom and brand recognition using both 2D images and 3D morphable models.  [The open-access paper may be found here](https://www.cambridge.org/core/journals/design-science/article/balancing-design-freedom-and-brand-recognition-in-the-evolution-of-automotive-brand-styling/933E383D85830C569F8F77BD2FD314A0).

This code repo contains all code necessary to replicate this experiment.  In particular, three major code pieces are separated: (1) code for calculating design attribute values as a function of design variables using a partial rank Markov chain; (2) code for calculating design freedom based on a design freedom distance metric defined in the paper; and (3) code for calculating brand recognition amongst the four brands in this survey using a multinomial logit model regularized with L1 penalty.

The following may be useful to you:  (1) 3D morphable models (note that we could not include the Cadillac model due to licensing issues), (2) 2D images normalized and with their brands removed, (3) respondant data regarding which car images received which partial rankings for design attributes, and (4) analysis code for using the design freedom distance metric.

Note that all data is anonymized from the crowds used in this study.  Moreover, the web application code is not included in this open source repository due to security conflicts.  If you would like help setting up your own experiment, please email the corresponding author at aburnap@umich.edu.


## Installation

This code will depend on many other python packages. To run this code, please install using the following:

```python
pip install -r requirements.txt
```

## To Run
First calculate the sigmas using /webapp_1st_exp/extractdata_1st_exp.py
Then extract all data from 1st and 2nd experiment for accuracy threshold above T using extractdata scripts
Then obtain brand recognition on the 24 morphed vehicles using the extractdata_2nd_exp.py script
Then combine partial rankings from the data extractions using experiment from 1_24_15_calculating_etc
Then aggregate the combined partial rankings using /rank_aggregation/compute_full_rank.py
Then run classification to calculate the omegas using 1_24_15_classification
Then calculate design freedom on the 24 morphed vehicles and plot with 1_25_15_etc

## License

The code is licensed under the Apache v2 license. Feel free to use all or portions
for your research.  If you find this code useful, please use the following citation information:

Burnap, A., Hartley, J., Pan, Y., Gonzalez, R., and Papalambros, P. Y., "Balancing Design Freedom and Brand Recognition in the Evolution of Automobile Brand Styling", Design Science Journal, 2.9 (2016).




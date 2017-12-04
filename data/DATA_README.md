#-----------------------------------------------------------------------------
#
#
#
#
#-----------------------------------------------------------------------------

2D_images/
    /baseline_cleaned_images  -- This contains all baseline images used for partial rankings of the 2D portion of both experiment 1 and experiment 2.  The MY2014 vehicles that these 2D images correspond to may be found in Table 3 of the paper.
    /morphed_car_images_cleaned  -- This contains the images taken from the face view of the morphed 3D cars from experiment 1.  These images were used in experiment 2 by during the partial rankings for attributes by mixing with the baseline images found in ./baseline_cleaned_images

3D_models/
    /audi.json
    /bmw.json
    /lexus.json
    NOTE: The Cadillac model did not have an open source license; therefore, we can not release it in this package.

raw_data/
    /db.sqlite3_AFTER_1st_EXP -- Thie is the entire database generated during the first experiment.  This experiment anonymized all crowd participants by randomly assigning UUIDs as pk values.
    /db.sqlite3_AFTER_2nd_EXP -- Thie is the entire database generated during the first experiment.  This experiment anonymized all crowd participants by randomly assigning UUIDs as pk values.
    /extractdata_1st_exp.py  -- This script extracts all data from the database, including number and UUIDs of valid users for the entire crowd and the filtered crowd.  This script also extracts morphed geometric variable values from the 3D portion of experiment 1
    /extractdata_2nd_exp.py  -- This script extracts all data from the database, including number and UUIDs of valid users for the entire crowd and the filtered crowd.

processed_data/
    /morphed_brands.csv  -- These are the brands that were randomly assigned to crowd participants during the 3D morphing from experiment 1
    /morphed_features.csv  -- These are the geometric design variable values for the 3D morphed models from experiment 1
    /omega_30percent_brand_recognition.csv  -- These are the omega values of the multinomial brand recognition logit model trained with L1 sparsity penalty
    /design_variable_sigmas.csv -- These are the standard deviations of the geometric design variables to assess significance for the design freedom distance metric
    /morphed_car_brand_recognition_baseline_acc_30.csv  -- These are the brand actual brand recognition of the morphed vehicles for all participants of experiment 2 with at least 30% brand recognition accuracy on the baseline set of vehicles

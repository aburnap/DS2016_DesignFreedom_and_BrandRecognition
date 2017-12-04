
import numpy as np
from sklearn.preprocessing import StandardScaler

#------------- Get Processed Data ---------------------------------------------
# Create the significance matrix of the sensitivities of the design variables.  See hypothesis test for details using the below data:
# sigmas = np.loadtxt("../../data/processed_data/design_variable_sigmas.csv", delimiter=",)
R= np.ones([4,10])
# Get the omegas of the brand recognition multinomial logit
omegas = np.loadtxt("../../data/processed_data/omega_30percent_brand_recognition.csv", delimiter=",")
# Convert to indicator variables using L0 norm
omegas[omegas!=0] = 1

# Get the set of all baseline and morphed design attribute values. 
# Recall that this is obtained by using the partial rank Markov chain on all vehicles (baseline and morphed)
DESIGNS = [elm+str(num) for elm in ['a','b','c','l'] for num in range(0,5)]
morphed_DESIGNS = [elm+str(num) for elm in ['A','B','C','L'] for num in range(0,8)]
DESIGNS.extend(morphed_DESIGNS)

ATTRIBUTES = ['Active', 'Aggressive','Distinctive','Expressive','Innovative','Luxurious','Powerful','Sporty','Well Proportioned','Youthful']

design_attributes = np.zeros( [len(DESIGNS) ,len(ATTRIBUTES)])
for ind, attr in enumerate(ATTRIBUTES):
    design_attributes[:, ind] = np.loadtxt("../../data/processed_data/attribute_values/"+attr+"_full_rank.csv", delimiter=',')

# Get the attribute values of the Audi A6, BMW 3 Series, Cadillac CTS, and Lexus LS as the baseline attribute values
baseline_design_attributes = design_attributes[[1,6, 11, 17], :]
# Get the set of all morphed vehicle attributes
morphed_design_attributes = design_attributes[20:52, :]

# Get the set of design variables for all morphed designs; Note that these are out of 100 and must be normalized to [0, 1]
morphed_design_variables = np.loadtxt("../../data/processed_data/morphed_design_variables.csv", delimiter=",")/100
# Get the set of brand recognition accuracy for the morphed vehicles
morphed_brand_recognition = np.loadtxt("../../data/processed_data/morphed_car_brand_recognition_baseline_acc_30.csv", delimiter=',')

# Create Brand Labels
a=[0]*8
b=[1]*8
c=[2]*8
l=[3]*8
a.extend(b)
a.extend(c)
a.extend(l)
brand_labels = np.array(a)


# Assume that baseline attributes and variables are [1,...,1] and [0,0,0,0], respectively?
def calculate_design_freedom(variables, attributes, baseline_variables, baseline_attributes, R, omega, lam, var_scaler, attr_scaler):
    variable_difference = variables - baseline_variables
    variable_quadratic = np.dot( variable_difference.T, np.dot(np.diag(np.dot(R,omega)), variable_difference))

    attribute_difference = attributes - baseline_attributes

    
    attribute_quadratic = np.dot(attribute_difference.T, np.dot(np.diag(omega), attribute_difference))
    try:
        return lam*var_scaler.transform(variable_quadratic) + attr_scaler.transform(attribute_quadratic)
    except AttributeError:
        return lam*variable_quadratic+attribute_quadratic


#def calculate_design_freedom_ONLY_quads(variables, attributes, baseline_variables, baseline_attributes, R, omega, lam):
    #variable_difference = variables - baseline_variables
    #variable_quadratic = np.dot( variable_difference.T, np.dot(np.diag(np.dot(R,omega)), variable_difference))

    #attribute_difference = attributes - baseline_attributes
    
    #attribute_quadratic = np.dot(attribute_difference.T, np.dot(np.diag(omega), attribute_difference))
    #return (variable_quadratic, attribute_quadratic)


# Create the set of unnormalized design freedom for each brand
design_freedoms = np.zeros([len(brand_labels), 1])
variable_quadratics=np.zeros([len(brand_labels), 1])
attribute_quadratics=np.zeros([len(brand_labels), 1])
lam = 1
for ind in range(len(brand_labels)):
    attributes = morphed_design_attributes[ind, :].reshape([10,1])
    variables = morphed_design_variables[ind, :].reshape([4,1])
    omega = omegas[brand_labels[ind], :]
    b_variables = np.zeros([4,1])
    b_attributes = baseline_design_attributes[brand_labels[ind],:].reshape([10,1])
    R = R
    variable_quadratics[ind], attribute_quadratics[ind]  = calculate_design_freedom_ONLY_quads(variables, attributes, b_variables, b_attributes, R, omega, lam)


#------- Calculate Standardized Design Freedoms for each Brand
audi_attr_scaler = StandardScaler()
audi_var_scaler = StandardScaler()
bmw_var_scaler = StandardScaler()
bmw_attr_scaler = StandardScaler()
cadillac_attr_scaler = StandardScaler()
cadillac_var_scaler = StandardScaler()
lexus_attr_scaler = StandardScaler()
lexus_var_scaler = StandardScaler()

audi_attr_scaler.fit(attribute_quadratics[0:8])
audi_var_scaler.fit(variable_quadratics[0:8])

bmw_attr_scaler.fit(attribute_quadratics[8:16])
bmw_var_scaler.fit(variable_quadratics[8:16])

cadillac_attr_scaler.fit(attribute_quadratics[16:24])
cadillac_var_scaler.fit(variable_quadratics[16:24])

lexus_attr_scaler.fit(attribute_quadratics[24:32])
lexus_var_scaler.fit(variable_quadratics[24:32])

lam = 1

for ind in range(len(brand_labels)):
    attributes = morphed_design_attributes[ind, :].reshape([10,1])
    variables = morphed_design_variables[ind, :].reshape([4,1])
    omega = omegas[brand_labels[ind], :]
    b_variables = np.zeros([4,1])
    b_attributes = baseline_design_attributes[brand_labels[ind],:].reshape([10,1])
    R = R
    var_scaler = [audi_var_scaler, bmw_var_scaler, cadillac_var_scaler, lexus_var_scaler][brand_labels[ind]]
    attr_scaler = [audi_attr_scaler, bmw_attr_scaler, cadillac_attr_scaler, lexus_attr_scaler][brand_labels[ind]]
    design_freedoms[ind]  = calculate_design_freedom(variables, attributes, b_variables, b_attributes, R, omega, lam, var_scaler, attr_scaler)

design_freedoms

morphed_brand_recognition

plt.scatter(design_freedoms[0:8], morphed_brand_recognition[0:8])

plt.scatter(design_freedoms[8:16], morphed_brand_recognition[8:16])

plt.scatter(design_freedoms[16:24], morphed_brand_recognition[16:24])

plt.scatter(design_freedoms[24:32], morphed_brand_recognition[24:32])




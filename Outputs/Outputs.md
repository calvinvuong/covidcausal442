# IP Weighting Difference 1000 samples 
Method:  ip

Measure:  difference

Bootstrap samples:  1000

(3.0902015481850227, (1.8934580136460075, 4.286945082724038))
## Parameters
Outcome = 'Sick'

Treatment = 'DistancingGrade'

Covariates = ['SVISocioeconomic', 'StatePctTested', 
'PctGE65', 'DaytimePopDensity']


# IP Weighting Ratio 1000 samples 
Method:  ip

Measure:  ratio

Bootstrap samples:  1000

(2.3111946753050487, (1.8134638227352033, 2.8089255278748944))
## Parameters
Outcome = 'Sick'

Treatment = 'DistancingGrade'

Covariates = ['SVISocioeconomic', 'StatePctTested', 
'PctGE65', 'DaytimePopDensity']





# Standardization Difference 1000 samples 

Method:  standardization

Measure:  difference

Bootstrap samples:  1000

(1.278338529218136, (1.18217327056061, 1.3745037878756619))
## Parameters

Outcome = 'Sick'

Treatment = 'DistancingGrade'

Covariates = ['SVISocioeconomic', 'StatePctTested', 
'PctGE65', 'DaytimePopDensity']


# Standardization Ratio 1000 samples 

Method:  standardization

Measure:  ratio

Bootstrap samples:  1000

(1.5112526988720665, (1.466593200325613, 1.55591219741852))
## Parameters

Outcome = 'Sick'

Treatment = 'DistancingGrade'

Covariates = ['SVISocioeconomic', 'StatePctTested', 
'PctGE65', 'DaytimePopDensity']



# **Using `DaytimePopDensity` as treatment instead of `DistancingGrade`**


# Ip Weighting Difference 100 samples 

Method:  ip

Measure:  difference

Bootstrap samples:  100

(-2.696517644114828, (-3.8243600564531164, -1.56867523177654))
## Parameters

OUTCOME_VAR = 'Sick'

TREATMENT_VAR = 'DaytimePopDensity'

COVARIATES = ['SVISocioeconomic', 'StatePctTested', 'PctGE65', 'DistancingGrade']

TREATMENT_CUTOFF = 300.0


# Standardization Difference 100 samples 


Method:  standardization

Measure:  difference

Bootstrap samples:  100

(-0.551217625725461, (-0.6674777842024279, -0.43495746724849405))
## Parameters

OUTCOME_VAR = 'Sick'

TREATMENT_VAR = 'DaytimePopDensity'

COVARIATES = ['SVISocioeconomic', 'StatePctTested', 'PctGE65', 'DistancingGrade']

TREATMENT_CUTOFF = 300.0










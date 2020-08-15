library(dplyr)   
library(PRROC)
library(nnet)
library(varhandle)


#load dataset
wids = read.csv("/Users/miriamnwaru/Documents/Data_Science_Graduate_Program/DS_740_Data_Mining_and_Machine_Learning/Final/training_v2.csv", na.strings=c("","NA"))


#convert variables to factors
wids$hospital_death = as.factor(wids$hospital_death)
wids$elective_surgery = as.factor(wids$elective_surgery)
wids$readmission_status = as.factor(wids$readmission_status)
wids$apache_2_diagnosis = as.factor(wids$apache_2_diagnosis)
wids$apache_3j_diagnosis = as.factor(wids$apache_3j_diagnosis)
wids$apache_post_operative = as.factor(wids$apache_post_operative)
wids$arf_apache = as.factor(wids$arf_apache)
wids$gcs_unable_apache = as.factor(wids$gcs_unable_apache)
wids$intubated_apache = as.factor(wids$intubated_apache)
wids$ventilated_apache = as.factor(wids$ventilated_apache)
wids$aids = as.factor(wids$aids)
wids$cirrhosis = as.factor(wids$cirrhosis)
wids$diabetes_mellitus = as.factor(wids$diabetes_mellitus)
wids$hepatic_failure = as.factor(wids$hepatic_failure)
wids$immunosuppression = as.factor(wids$immunosuppression)
wids$leukemia = as.factor(wids$leukemia)
wids$lymphoma = as.factor(wids$lymphoma)
wids$solid_tumor_with_metastasis = as.factor(wids$solid_tumor_with_metastasis)

#convert na's in factors to unknown
wids = wids %>% 
  mutate_if(is.factor, fct_explicit_na, na_level = 'Unknown')

is.fact <- sapply(wids, is.factor)
summary(wids[, is.fact])

#keep consistency in ethnicity variable
wids$ethnicity[which(wids$ethnicity == "Unknown")] = "Other/Unknown" 

#frequency table of na's in dataset
head(freq.na(wids))

wids30 = wids[, which(colMeans(!is.na(wids)) > 0.3)] #remove columns with more than 30% missing
wids30 = wids30[which(rowMeans(!is.na(wids30)) > 0.3), ] #remove rows with more than 30% missing


#compare to make sure percentage did not change
summary(wids$hospital_death)
summary(wids30$hospital_death)

#knn imputation method 
wids_imp = knnImputation(wids30, 3)

#convert gender to binary
gender01 = rep(0, length(wids_imp$gender))
gender01[which(wids_imp$gender == "F")] = 1
wids_imp$gender = gender01

#find multicolinearity between variables
fit = glm(hospital_death ~ ., data = wids_imp, family = "binomial")
vif(fit)
summary(fit)


#create a subset of african american women
wids_AA = filter(wids_imp, ethnicity == "African American" & gender == 1)

#remove unknowns so variables can be changed to binary
wids_filter = filter(wids_AA, arf_apache != "Unknown" & gcs_unable_apache != "Unknown" & intubated_apache != "Unknown"
                     &  ventilated_apache != "Unknown" & aids != "Unknown" & cirrhosis != "Unknown"
                     & diabetes_mellitus != "Unknown" & hepatic_failure != "Unknown" & immunosuppression != "Unknown" 
                     & leukemia != "Unknown" & lymphoma != "Unknown" & solid_tumor_with_metastasis != "Unknown")

#convert categorical var to binary
wids_df <- data.frame( 
  to.dummy(wids_filter$icu_admit_source, "icu_admit_source"),
  to.dummy(wids_filter$icu_stay_type, "icu_stay_type"), 
  to.dummy(wids_filter$icu_type, "icu_type"), wids_filter)

#list variables i no longer need
drops_tem = c("ethnicity", "gender","readmission_status","apache_3j_diagnosis", "apache_2_bodysystem", "apache_2_diagnosis", "weight", "hight", 
              "hospital_admit_source", "d1_bilirubin_max", "d1_bilirubin_min", "d1_bun_max", "d1_bun_min", 
              "d1_creatinine_max", "d1_creatinine_min", "d1_diasbp_invasive_max", "d1_diasbp_invasive_min", 
              "d1_diasbp_noninvasive_max", "d1_diasbp_noninvasive_min", "d1_mbp_noninvasive_max", 
              "d1_mbp_noninvasive_min", "d1_mbp_invasive_max", "d1_mbp_invasive_min", "d1_sysbp_noninvasive_max", 
              "d1_sysbp_noninvasive_min", "d1_sysbp_invasive_max", "d1_sysbp_invasive_min", 
              "h1_diasbp_noninvasive_max", "h1_diasbp_invasive_max", "h1_diasbp_noninvasive_min", 
              "h1_diasbp_invasive_max", "h1_mbp_noninvasive_max", "h1_mbp_noninvasive_min", "h1_mbp_invasive_max", 
              "h1_mbp_invasive_min", "h1_sysbp_noninvasive_max", "h1_sysbp_noninvasive_min", "h1_sysbp_invasive_max", 
              "h1_sysbp_invasive_min", "d1_albumin_max", "d1_albumin_min", "d1_hematocrit_max", "d1_hematocrit_min", 
              "h1_creatinine_max", "h1_creatinine_min", "h1_hemaglobin_max", "h1_hemaglobin_min", "h1_platelets_max", 
              "h1_platelets_min", "d1_wbc_max", "d1_wbc_min", "h1_wbc_max", "h1_wbc_min", "h1_glucose_max", 
              "h1_glucose_min", "d1_glucose_max", "d1_glucose_min", "h1_inr_max", "h1_inr_min", "icu_admit_source", 
              "icu_stay_type", "icu_type", "apache_3j_bodysystem")

#remove variables no longer needed
wids_binary = wids_df[ , !(names(wids_df) %in% drops_tem)]


##### model assessment OUTER 10-fold CV (with model selection INNER 10-fold CV as part of model-fitting) #####

xy.out = wids_binary
n.out = dim(xy.out)[1]

set.seed(8, sample.kind = "Rounding")


##### model assessment OUTER shell #####



#create validation for outershell
train = sample(1:n.out, 3110)
trainxy.out = xy.out[train,]
testxy.out = xy.out[-train,]

##############################################
###   model selection on trainxy.out       ###
##############################################
##entire model-fitting process##
xy.in = trainxy.out  # fixed to be fit ONLY to the training data from the outer split
n.in = dim(xy.in)[1]
ncv = 5
sizes = 1:8
if ((n.in%%ncv) == 0) {
  groups.in= rep(1:ncv,floor(n.in/ncv))} else {
    groups.in=c(rep(1:ncv,floor(n.in/ncv)),(1:(n.in%%ncv)))
  }

cvgroups.in = sample(groups.in,n.in)
# set up storage
allpredictedcvlog = matrix(,ncol=1,nrow=n.in)
allpredictedcvnnet = matrix(,ncol = 8,nrow = n.in)
# with model selection 
for (i in 1:ncv) { #iterate over folds in inner CV
  newdata.in = xy.in[cvgroups.in==i,]
  
  logfit = glm(hospital_death ~., data=xy.in, subset=(cvgroups.in!=i), family=binomial)
  logprob = predict(logfit,newdata.in,type="response")
  logfact = rep(1,dim(newdata.in)[1]); logfact[logprob > 0.5] = 2
  allpredictedcvlog[cvgroups.in==i,1] = logfact
  print(paste("log",i))
  
  for(current_size in 1: length(sizes)){
    fit_nnet = nnet(hospital_death ~ ., data=xy.in, subset=(cvgroups.in!=i), size = sizes[current_size], maxit = 1000)
    print(paste("nnet","size", current_size))
    nnetprob = predict(fit_nnet, newdata.in)
    nnetfact = rep(1,dim(newdata.in)[1]); nnetfact[nnetprob > 0.5] = 2
    allpredictedcvnnet[cvgroups.in == i, current_size] = nnetfact
    print(paste("nnet",i))
    
  }#end iteration over size
  
  
  
}# end inner for loop
#relabel as original values, not factor levels
allpredictedcvlog = allpredictedcvlog-1  # now a table of predicted 0-1 values for hospital_death
allpredictedcvnnet = allpredictedcvnnet-1
#compute the CV values
allcv10 = rep(0,2)
allcv10[1] = sum(xy.in$hospital_death!=allpredictedcvlog[,1])/n.in
y.array = array(xy.in$hospital_death, dim = c(n.in, 8)) #make an array of responses, to match dimensions of allpredictedcvnnet
diffs = (allpredictedcvnnet != y.array)
nnetErrorMatrix = (apply(diffs, c(2), sum))/n.in
min_loc = as.vector(which(nnetErrorMatrix==min(nnetErrorMatrix), arr.ind=T))
allcv10[2] = min(nnetErrorMatrix)
bestmodels = (1:2)[allcv10 == min(allcv10)]
##############################################
###   resulting in bestmodels              ###
##############################################

bestmodel = ifelse(length(bestmodels)==1,bestmodels,sample(bestmodels,1))
print(allcv10)
print(paste("Best model at outer loop",bestmodel))

if (bestmodel == 1)  {
  logfit.train = glm(hospital_death ~., data=trainxy.out, family=binomial)
  logprob.test = predict(logfit.train,testxy.out,type="response")
  predictvalid = rep(1,dim(testxy.out)[1]); predictvalid[logprob.test > 0.5] = 2
}

if (bestmodel == 2)  {
  fit_nnet.train = nnet(hospital_death ~ ., data=trainxy.out, size = min_loc[1], trace = F, maxit = 1000) 
  nnetprob.test = predict(fit_nnet.train, testxy.out)
  predictvalid = rep(1,dim(testxy.out)[1]); predictvalid[nnetprob.test > 0.5] = 2
}

#relabel as original values, not factor levels
predictvalid = predictvalid-1  # now a vector of predicted 0-1 values for hospital_death in validation set

#proportion of the validated classifications match the actual observations
table(testxy.out$hospital_death, predictvalid)
cvout = sum(testxy.out$hospital_death != predictvalid)/n.out
p.out = 1- cvout; p.out

#find area under precision curve
fg = logprob.test[testxy.out$hospital_death == 1]
bg = logprob.test[testxy.out$hospital_death == 0]
pr = pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)


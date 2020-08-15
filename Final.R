library(dplyr)   
library(forcats)
library(questionr)
library(DMwR)
library(car)
library(ggfortify)
library(pROC)
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

write.csv(dataset, "C:/folder/filename.csv")
write.csv(wids_imp,"/Users/miriamnwaru/Documents/Data_Science_Graduate_Program/DS_740_Data_Mining_and_Machine_Learning/Final/wids_imp.csv")

wids_imp = read.csv("/Users/miriamnwaru/Documents/Data_Science_Graduate_Program/DS_740_Data_Mining_and_Machine_Learning/Final/wids_imp.csv")

#convert variables to factors
wids_imp$hospital_death = as.factor(wids_imp$hospital_death)
wids_imp$elective_surgery = as.factor(wids_imp$elective_surgery)
wids_imp$readmission_status = as.factor(wids_imp$readmission_status)
wids_imp$apache_2_diagnosis = as.factor(wids_imp$apache_2_diagnosis)
wids_imp$apache_3j_diagnosis = as.factor(wids_imp$apache_3j_diagnosis)
wids_imp$apache_post_operative = as.factor(wids_imp$apache_post_operative)
wids_imp$arf_apache = as.factor(wids_imp$arf_apache)
wids_imp$gcs_unable_apache = as.factor(wids_imp$gcs_unable_apache)
wids_imp$intubated_apache = as.factor(wids_imp$intubated_apache)
wids_imp$ventilated_apache = as.factor(wids_imp$ventilated_apache)
wids_imp$aids = as.factor(wids_imp$aids)
wids_imp$cirrhosis = as.factor(wids_imp$cirrhosis)
wids_imp$diabetes_mellitus = as.factor(wids_imp$diabetes_mellitus)
wids_imp$hepatic_failure = as.factor(wids_imp$hepatic_failure)
wids_imp$immunosuppression = as.factor(wids_imp$immunosuppression)
wids_imp$leukemia = as.factor(wids_imp$leukemia)
wids_imp$lymphoma = as.factor(wids_imp$lymphoma)
wids_imp$solid_tumor_with_metastasis = as.factor(wids_imp$solid_tumor_with_metastasis)

wids_imp = wids_imp[,c(-1)]
wids_imp = wids_imp[,c(-17)]


fit = glm(hospital_death ~ ., data = wids_imp, family = "binomial")


vif(fit)
summary(fit)

summary(wids_imp$apache_3j_diagnosis)

ld.vars <- attributes(alias(fit)$Complete)$dimnames[[1]]
ld.vars


new.wids <- wids_imp[, !sapply(wids_imp, is.factor)] # remove the y-variable and only keep quantitative variables


unlabled = read.csv("/Users/miriamnwaru/Documents/Data_Science_Graduate_Program/DS_740_Data_Mining_and_Machine_Learning/Final/unlabeled.csv")

wids.test = data.frame(wids_imp$hospital_death, wids.rem)

fit.new = glm(hospital_death~., data = wids_imp, family = "binomial")
vif(fit.new)
summary(fit.new)

drops_perm = c("apache_2_bodysystem", "apache_2_diagnosis", "weight", "hight", 
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
          "h1_glucose_min", "d1_glucose_max", "d1_glucose_min", "h1_inr_max", "h1_inr_min")

drops_tem = c("apache_3j_diagnosis", "intubated_apache", "ventilated_apache", "aids", 
          "cirrhosis", "diabetes_mellitus", "hepatic_failure", "immunosuppression", "leukemia", "lymphoma", 
          "solid_tumor_with_metastasis", "apache_2_bodysystem", "apache_2_diagnosis", "weight", "hight", 
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
          "h1_glucose_min", "d1_glucose_max", "d1_glucose_min", "h1_inr_max", "h1_inr_min")

wids_rem = wids_imp[ , !(names(wids_imp) %in% drops)]

fit.new = glm(hospital_death~., data = wids_rem, family = "binomial")
summary(fit.new)
vif(fit.new)

wids_rem = wids_imp[ , !(names(wids_imp) %in% drops_perm)]
fit.new = glm(hospital_death~., data = wids_rem, family = "binomial")

probs = predict(fit.new, wids_rem, type = "response")

newroc = roc(response = wids_rem$hospital_death, predictor = probs)
newroc


fg <- probs[wids_rem$hospital_death == 1]
bg <- probs[wids_rem$hospital_death == 0]

pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)


n = dim(wids.test)[1]
k = 10 #using 10-fold cross-validation
groups = c(rep(1:k,floor(n/k)),1:(n%%k))
sizes = 1:8
misclassError = matrix( , nr = k, nc = length(sizes) )
conv = matrix(, nr = k, nc = length(sizes) ) 
set.seed(4, sample.kind = "Rounding")
cvgroups = sample(groups,n) 
for(i in 1:k){
  groupi = (cvgroups == i)
  mywids_test.train = scale(wids.test[!groupi, 2:104])
  mywids_test.valid = scale(wids.test[groupi, 2:104], center = attr(mywids_test.train, "scaled:center"), 
                          scale = attr(mywids_test.train, "scaled:scale"))
  mywids_test.train = data.frame(wids_imp.hospital_death=wids.test[!groupi, 1], mywids_test.train)
  mywids_test.valid = data.frame(wids_imp.hospital_death=wids.test[groupi, 1], mywids_test.valid)
  for(j in 1:length(sizes)){
    fit = nnet(wids_imp.hospital_death ~ ., data=mywids_test.train, size = sizes[j], trace = F, maxit = 1000) 
    predictions = predict(fit, mywids_test.valid, type = "class")
    misclassError[i, j] = length(which(predictions != mywids_test.valid[ , 1])) / length(predictions)
    conv[i, j] = fit$convergence
  } # end iteration over j
} # end iteration over i

colSums(conv)




wids_scale <- scale(wids_nofactor)
pc.info.wids = prcomp(wids_scale)
summary(pc.info.wids)

par(mfrow=c(1,1))
plot(pc.info.wids)

vjs.wids = pc.info.wids$sdev^2 
pve.wids = vjs.wids/sum(vjs.wids) 
plot(cumsum(pve.wids ), type = "o", ylab="Cumulative PVE", xlab="Principal Component") 

biplot(pc.info.wids,scale=0) 
pc.info.wids$rotation[,1] 
pc.info.wids$rotation[,2]  
pc1scores.wids = pc.info.wids$x[,1]  
pc2scores.wids = pc.info.wids$x[,2]   
#pc1scores.wids[237] # pete rose: 7.75
#pc2scores.wids[237]  # pete rose: -7.45

coords = pc.info.wids$x
plot(coords[,1], coords[,2],xlab="PC1",ylab="PC2", type="n")
text(coords[,1], coords[,2], rownames(coords))

load = pc.info.wids$rotation 
plot(load[,1],load[,2])
text(load[,1],load[,2], rownames(load))

autoplot(pc.info.wids, data = wids_imp, colour = 'hospital_death', scale = 0)
autoplot(pc.info.wids, data = wids_imp, colour = 'hospital_death', scale = 0, x=2, y=3)
autoplot(pc.info.wids, data = wids_imp, colour = 'hospital_death', scale = 0, x=1, y=11)

pc.info.wids$x[,1]

wids.pc = data.frame(wids_imp[, sapply(wids_imp, is.factor)], pc.info.wids$x[,1:2])

wids.pc = data.frame(wids_imp$hospital_death, pc.info.wids$x[,1:50])


fit.pc = glm(wids_imp.hospital_death~.,data = wids.pc, family = "binomial")
predictvalsnew = predict(fit.pc, wids.pc, type = "response")

#newroc = pr.curve(response = wids.pc$wids_imp.hospital_death, predictor = predictvalsnew)
#newroc


fg <- predictvalsnew[wids.pc$wids_imp.hospital_death == 1]
bg <- predictvalsnew[wids.pc$wids_imp.hospital_death == 0]

pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)


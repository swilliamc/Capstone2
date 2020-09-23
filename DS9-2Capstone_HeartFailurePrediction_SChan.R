##########################################################
# HarvardX: PH125.9x
# Data Science: Capstone
# Heart Failure Prediction by Suhaimi William Chan (Sep 2020)
# Create edx set, validation set (final hold-out test set)
##########################################################

# 1. Introduction/Overview 

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# 1a. Importing data

# Heart Failure Prediction dataset:
# https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv
# https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/download
# https://www.kaggle.com/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv

# URL for heart failure prediction data set in csv file format
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"

# Read the data directly from the URL and store it into dl variable
dl <- read_csv(url)

# We can see a glimpse of our dl dataset
glimpse(dl)
#Rows: 299
#Columns: 13
#$ age                      <dbl> 75, 55, 65, 50, 65, 90, 75, 60, 65, 80, 75, 62, 45, 50, 49, 82, 87, 45, 70, 48, 65, 65, 68, 53...
#$ anaemia                  <dbl> 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0...
#$ creatinine_phosphokinase <dbl> 582, 7861, 146, 111, 160, 47, 246, 315, 157, 123, 81, 231, 981, 168, 80, 379, 149, 582, 125, 5...
#$ diabetes                 <dbl> 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0...
#$ ejection_fraction        <dbl> 20, 38, 20, 20, 20, 40, 15, 60, 65, 35, 38, 25, 30, 38, 30, 50, 38, 14, 25, 55, 25, 30, 35, 60...
#$ high_blood_pressure      <dbl> 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0...
#$ platelets                <dbl> 265000, 263358, 162000, 210000, 327000, 204000, 127000, 454000, 263358, 388000, 368000, 253000...
#$ serum_creatinine         <dbl> 1.90, 1.10, 1.30, 1.90, 2.70, 2.10, 1.20, 1.10, 1.50, 9.40, 4.00, 0.90, 1.10, 1.10, 1.00, 1.30...
#$ serum_sodium             <dbl> 130, 136, 129, 137, 116, 132, 137, 131, 138, 133, 131, 140, 137, 137, 138, 136, 140, 127, 140,...
#$ sex                      <dbl> 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1...
#$ smoking                  <dbl> 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0...
#$ time                     <dbl> 4, 6, 7, 7, 8, 8, 10, 10, 10, 10, 10, 10, 11, 11, 12, 13, 14, 14, 15, 15, 16, 20, 20, 22, 23, ...
#$ DEATH_EVENT              <dbl> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1...
# It shows that we have 299 rows and 13 columns

# checking and counting missing values (NA or NULL)
dl %>% summarize(NA_age = sum(is.na(age)),
            NA_anaemia = sum(is.na(anaemia)),
            NA_cp = sum(is.na(creatinine_phosphokinase)),
            NA_diabetes = sum(is.na(diabetes)),
            NA_ef = sum(is.na(ejection_fraction)),
            NA_hbp = sum(is.na(high_blood_pressure)),
            NA_platelets = sum(is.na(platelets)),
            NA_sc = sum(is.na(serum_creatinine)),
            NA_ss = sum(is.na(serum_sodium)),
            NA_sex = sum(is.na(sex)),
            NA_smoking = sum(is.na(smoking)),
            NA_time = sum(is.na(time)),
            NA_DEATH_EVENT = sum(is.na(DEATH_EVENT)))

# A tibble: 1 x 13
#NA_age NA_anaemia NA_cp NA_diabetes NA_ef NA_hbp NA_platelets NA_sc NA_ss NA_sex NA_smoking NA_time NA_DEATH_EVENT
#<int>      <int> <int>       <int> <int>  <int>        <int> <int> <int>  <int>      <int>   <int>          <int>
#  1      0          0     0           0     0      0            0     0     0      0          0       0              0
# It shows that data are clean, so we have no missing values

# 1b. Create data partitions

# Validation set will be 20% of dl dataset, edx set will be 80% of dl dataset
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = dl$DEATH_EVENT, times = 1, p = 0.2, list = FALSE)
edx <- dl[-test_index,]
validation <- dl[test_index,]

# Validation dataset will be broken out to 50% of test_set and 50% of validation_set
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
vtest_index <- createDataPartition(y = validation$DEATH_EVENT, times = 1, p = 0.5, list = FALSE)
vtest_set <- validation[-vtest_index,]
validation_test <- validation[vtest_index,]

# Here is the dimension of our edx set
dim(edx)

# Here is the dimension of our validation set
dim(validation)


# 2. Exploratory Data Analysis
# Now we are doing our data exploration using edx data set, to see a more complete data set, instead of using train set

# We can see some examples of our edx data set with available columns
head(edx)

# We can see classes of our edx data set
str(edx)

#Attribute Information [6]:
#Thirteen (13) clinical features:
#- age: age of the patient (years)
#- anaemia: decrease of red blood cells or hemoglobin (boolean)
#- high blood pressure: if the patient has hypertension (boolean)
#- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
#- diabetes: if the patient has diabetes (boolean)
#- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
#- platelets: platelets in the blood (kiloplatelets/mL)
#- sex: woman or man (binary)
#- serum creatinine: level of serum creatinine in the blood (mg/dL)
#- serum sodium: level of serum sodium in the blood (mEq/L)
#- smoking: if the patient smokes or not (boolean)
#- time: follow-up period (days)
#- [target] death event: if the patient deceased during the follow-up period (boolean)

#Based on the clinical features above, here is a list of our boolean data type with the meaning of its values:
#.	Age - Age of patient
#.	Sex - Gender of patient 0 = Female, 1 = Male  
#.	Diabetes - 0 = No, 1 = Yes
#.	Anaemia - 0 = No, 1 = Yes
#.	High_blood_pressure - 0 = No, 1 = Yes
#.	Smoking - 0 = No, 1 = Yes
#.	DEATH_EVENT - 0 = No, 1 = Yes

# We are going to convert our data into data frame and all the boolean data type into factor class for better plotting of our data analysis
train_set <- as.data.frame(edx) %>%
  select(age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, 
         serum_creatinine, serum_sodium, smoking, time, DEATH_EVENT) %>%
  mutate(sex = factor(sex),
         diabetes = factor(diabetes),
         anaemia = factor(anaemia),
         high_blood_pressure = factor(high_blood_pressure),
         smoking = factor(smoking),
         DEATH_EVENT = factor(DEATH_EVENT))

test_set <- as.data.frame(vtest_set) %>%
  select(age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, 
         serum_creatinine, serum_sodium, smoking, time, DEATH_EVENT) %>%
  mutate(sex = factor(sex),
         diabetes = factor(diabetes),
         anaemia = factor(anaemia),
         high_blood_pressure = factor(high_blood_pressure),
         smoking = factor(smoking),
         DEATH_EVENT = factor(DEATH_EVENT))

validation_set <- as.data.frame(validation_test) %>%
  select(age, anaemia, high_blood_pressure, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, sex, 
         serum_creatinine, serum_sodium, smoking, time, DEATH_EVENT) %>%
  mutate(sex = factor(sex),
         diabetes = factor(diabetes),
         anaemia = factor(anaemia),
         high_blood_pressure = factor(high_blood_pressure),
         smoking = factor(smoking),
         DEATH_EVENT = factor(DEATH_EVENT))

# Let's check the class of our new data frame
str(train_set)
str(test_set)
str(validation_set)

# Install library for themes and scales
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
library(ggthemes)
library(scales)

options(pillar.sigfig = 4)

# [chart 1] We can visually see the Age Distribution of our edx dataset
qplot(train_set$age) +
  ggtitle("Distribution of Age", 
          subtitle = "The distribution is almost normal distribution with longer tail") +
  xlab("Age") +
  ylab("Count") 
# The distribution is almost normal distribution with a little longer tail at the end. Majority of patient ages are between 40 and 80.

# [chart 2] We can also visually see the distribution of age by sex in a box plot
train_set %>%
  group_by(sex, age) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = sex, y = age, group = sex)) +
  geom_point() +
  geom_boxplot() +
  #  scale_y_continuous(labels = comma) + 
  ggtitle("Age Distribution by sex", subtitle = "Most males are a bit higher in age than females") + 
  xlab("Sex") +
  ylab("Age") +
  theme_economist()
# According to Age Distribution by sex, most males are a bit higher in age than females.

# A question that is worth to investigage: Is age or sex an indicator for death event?
# Now, we are going to exploit the relationship of age or sex to Death_Event.
# [chart 3] We can also visually see the distribution of age by Death Event
train_set %>%
  group_by(DEATH_EVENT, age) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = DEATH_EVENT, y = age, group = DEATH_EVENT)) +
  geom_point() +
  geom_boxplot() +
  #  scale_y_continuous(labels = comma) + 
  ggtitle("Age Distribution by DEATH_EVENT", subtitle = "Higher Age has more possibility in DEATH") + 
  xlab("DEATH_EVENT") +
  ylab("Age") +
  theme_economist()
# According to Age Distribution by DEATH_EVENT, higher age has more possibility in death event. 

# [chart 4] We can also visually see the distribution of sex by Death Event
train_set %>%
  ggplot(aes(DEATH_EVENT, fill = sex )) +
  geom_bar(position = position_dodge()) +
  ggtitle("Distribution of sex by DEATH_EVENT", subtitle = "The proportion is about the same, no significant differences") + 
  xlab("DEATH_EVENT") 
# According to the Distribution of sex by DEATH_EVENT, the proportion is about the same, no significant differences.

# [chart 5] Now, let's explore and analyze further on age by DEATH_EVENT based on our findings on chart 3.
train_set %>%
  ggplot(aes(age, fill = DEATH_EVENT)) +
  geom_density(alpha = 0.2) +
  ggtitle("Overlap Density Distribution of age by DEATH_EVENT", subtitle = "Death event has much higher chance for patients with age 68 or above") 
# From the chart above, we can see that death event has much higher possibility/proportion for the patients with age of 68 or above.

# [chart 6] Let's explore further details on age by DEATH_EVENT using a box plot with jitter based on our findings on chart 5.
train_set %>%
  group_by(age) %>%
  ggplot(aes(age, DEATH_EVENT)) +
  geom_boxplot() +
  geom_jitter(width = 0.1, alpha = 0.2) +
  ggtitle("Distribution of age by DEATH_EVENT in a box plot with jitter", subtitle = "Possibility of death event would likely double at age 65 or above") 
# The chart above shows us that the possibility of death event would likely double at age 65 or above. 
# It also shows us that the chance of survival would likely almost double at age 58 or less.

# [chart 7] Let's explore further details on age by DEATH_EVENT using Empirical Cumulative Density Function (ECDF).
ggplot(train_set, aes(age, color = DEATH_EVENT)) + stat_ecdf(geom = "step") +
  labs(title="Empirical Cumulative Density Function (ECDF) for age by DEATH_EVENT",
       subtitle = "The largest delta between survived and death is at age 58",
       y = "F(age)", x="age")
# According to ECDF chart, there are big variance between survival and death at age 55, 58, 65, and 70.

# [chart 8] Let's look at the distribution of creatinine_phosphokinase by Death Event
train_set %>%
  ggplot(aes(creatinine_phosphokinase, fill = DEATH_EVENT)) +
  geom_density(alpha = 0.2) +
  scale_x_log10() +
  xlab("creatinine_phosphokinase in log10") +
  ggtitle("Overlap Distribution of creatinine_phosphokinase by DEATH_EVENT", 
          subtitle = "Death event proportion is about the same, no significant differences") 
# From the chart above, we can see that death event proportion is about the same, no significant differences.

# [chart 9] Let's look at the distribution of ejection_fraction by Death Event
train_set %>%
  ggplot(aes(ejection_fraction, fill = DEATH_EVENT)) +
  geom_density(alpha = 0.2) +
  ggtitle("Overlap Density Distribution of ejection_fraction by DEATH_EVENT", 
          subtitle = "Death is significantly higher with ejection_fraction value at 30 or below") 
# From the chart above, Death is significantly higher with ejection_fraction value at 30 or below.
# It also shows that survival is higher with ejection_fraction between 35 and low 40s. 

# [chart 10] Let's explore further details on ejection_fraction by DEATH_EVENT using Empirical Cumulative Density Function (ECDF).
ggplot(train_set, aes(ejection_fraction, color = DEATH_EVENT)) + stat_ecdf(geom = "step") +
  labs(title="Empirical Cumulative Density Function (ECDF) for ejection_fraction by DEATH_EVENT",
       subtitle = "Death is significantly higher with ejection_fraction value at 30 (28% delta)",
       y = "F(ejection_fraction)", x="ejection_fraction")
# From the ECDF chart above, it shows cumulative death rate is more than double at 28% delta with 52% Death and 24% survival for ejection_fraction value at 30.

# [chart 11] Let's look at the distribution of platelets by Death Event
train_set %>%
  ggplot(aes(platelets, fill = DEATH_EVENT)) +
  geom_density(alpha = 0.2) +
  scale_x_log10() +
  xlab("platelets in log10") +
  ggtitle("Overlap Density Distribution of platelets by DEATH_EVENT", 
          subtitle = "there might be higher survival in the bell shape of platelets values") 
# From the chart above, there might be higher survival in the bell shape of platelets values.

# [chart 12] Let's explore further details on platelets by DEATH_EVENT using Empirical Cumulative Density Function (ECDF).
ggplot(train_set, aes(platelets, color = DEATH_EVENT)) + stat_ecdf(geom = "step") +
  scale_x_log10() +
  xlab("platelets in log10") +
  labs(title="Empirical Cumulative Density Function (ECDF) for platelets by DEATH_EVENT",
       subtitle = "platelets proportion by death event shows no significant differences",
       y = "F(platelets)", x="platelets log10")
# From the ECDF chart above, we can see that platelets proportion by death event shows no significant differences.

# [chart 13] Let's look at the distribution of serum_creatinine by Death Event
train_set %>%
  ggplot(aes(serum_creatinine, fill = DEATH_EVENT)) +
  geom_density(alpha = 0.2) +
  ggtitle("Overlap Density Distribution of serum_creatinine by DEATH_EVENT", 
          subtitle = "survival is significantly higher with serum_creatinine value below 1.25") 
# From the chart above, survival is significantly higher with serum_creatinine value below 1.25.

# [chart 14] Let's explore further details on serum_creatinine by DEATH_EVENT using Empirical Cumulative Density Function (ECDF).
ggplot(train_set, aes(serum_creatinine, color = DEATH_EVENT)) + stat_ecdf(geom = "step") +
  labs(title="Empirical Cumulative Density Function (ECDF) for serum_creatinine by DEATH_EVENT",
       subtitle = "survival is significantly higher with serum_creatinine value at 1.25 (34% delta)",
       y = "F(serum_creatinine)", x="serum_creatinine")
# From the ECDF chart above, the cumulative survival rate is almost double at 34% delta with 44% Death and 78% survival for serum_creatinine value at 1.25.

# [chart 15] Let's look at the distribution of serum_sodium by Death Event
train_set %>%
  ggplot(aes(serum_sodium, fill = DEATH_EVENT)) +
  geom_density(alpha = 0.2) +
  ggtitle("Overlap Density Distribution of serum_sodium by DEATH_EVENT", 
          subtitle = "Death is slightly higher with serum_sodium value at 135 or below") 
# From the chart above, survival is slightly higher with serum_sodium value at 135 or below.

# [chart 16] Let's explore further details on serum_sodium by DEATH_EVENT using Empirical Cumulative Density Function (ECDF).
ggplot(train_set, aes(serum_sodium, color = DEATH_EVENT)) + stat_ecdf(geom = "step") +
  labs(title="Empirical Cumulative Density Function (ECDF) for serum_sodium by DEATH_EVENT",
       subtitle = "Death rate is double with serum_sodium value at 135",
       y = "F(serum_sodium)", x="serum_sodium")
# From the ECDF chart above, the cumulative death rate is double at 25% delta with 51% Death and 26% survival for serum_sodium value at 135.

# [chart 17] Let's look at the distribution of time by Death Event
train_set %>%
  ggplot(aes(time, fill = DEATH_EVENT)) +
  geom_density(alpha = 0.2) +
  ggtitle("Overlap Density Distribution of time by DEATH_EVENT", 
          subtitle = "Death is significantly higher with time value at 75 or below") 
# From the chart above, Death is significantly higher with time value at 75 or below.

# [chart 18] Let's explore further details on time by DEATH_EVENT using Empirical Cumulative Density Function (ECDF).
ggplot(train_set, aes(time, color = DEATH_EVENT)) + stat_ecdf(geom = "step") +
  labs(title="Empirical Cumulative Density Function (ECDF) for time by DEATH_EVENT",
       subtitle = "Cumulative Death rate is significantly higher with time value at 75 (53% delta)",
       y = "F(time)", x="time")
# From the ECDF chart above, the cumulative death rate is over 6.8 times at 53% delta with 61% Death and 8% survival for time value at 75.
# It also means that with time value above 75, survival rate is 2.5 times with 53% delta for 39% death and 92% survival.


# 3. Method/Analysis

# We are going to train our train_set using the following methods/algorithms:
# 1. Generalized Linear Model (glm)
# 2. Linear Discriminant Analysis (lda)
# 3. Quadratic Discriminant Analysis (qda)
# 4. Random Forest (rf)
# 5. Decision Tree (rpart)
# 6. Support Vector Machine (svmLinear2)
# 7. Ensemble

#First, we are going to train all our classifier models using our train_set dataset. 
#Then, We are going to test all our trained models using our test_set.
#And finally, we are going to cross-validate our trained models using our validation_set as our final results.

#names(fits) <- c("Generalized Linear Model", "Linear Discriminant Analysis", "Quadratic Discriminant Analysis", 
#                 "Random Forest", "Decision Tree", "Support Vector Machine")

# Let's start with adding the following library for our models
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")

library(randomForest)
library(e1071)

# Let's start with our first model
# Logistic Regression / Generalized Linear Model (glm)

train_glm <- train(DEATH_EVENT ~ ., method = "glm", data = train_set)
y_hat_glm <- predict(train_glm, test_set, type = "raw")
confusionMatrix(y_hat_glm, test_set$DEATH_EVENT)$overall[["Accuracy"]]
#> [1] 0.833333

# Let's check our confusion matrix detail for our logistic regression model
confusionMatrix(y_hat_glm, test_set$DEATH_EVENT)


# To create our models more efficiently, we are going to use caret package with lapply and sapply functions as follow:

models <- c("glm", "lda", "qda", "rf", "rpart", "svmLinear2")

# set.seed(1) # if using R 3.5 or earlier
set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later

fits <- lapply(models, function(model){ 
#  print(model)
  train(DEATH_EVENT ~., method = model, data = train_set)
}) 

names(fits) <- models
#fits
#class(fits)

pred <- sapply(fits, function(object) 
  predict(object, newdata = test_set))

#dim(pred)
acc <- colMeans(pred == test_set$DEATH_EVENT)
acc
#glm        lda    qda    gamLoess   rf   rpart     svmLinear2 
#0.833333  0.8     0.7    0.8        0.9  0.866666  0.833333 

# Based on the results of testing our trained models on the test_set, 
#it looks quite promising as we have test accuracy ranges from 70% to 90%, 
#with most of our classifier models have over 80% of accuracy.


mean(acc)
#[1] 0.819

votes <- rowMeans(pred == "1")
y_hat <- ifelse(votes > 0.5, "1", "0")
mean(y_hat == test_set$DEATH_EVENT)
#[1] 0.8

# Since the accuracy of our classifier models on the test_set are quite promising and acceptable, we will go ahead to cross-validate our trained classifier models with our validation set.

# Let's cross-validate our trained models using our validation_set

validation_pred <- sapply(fits, function(object) 
  predict(object, newdata = validation_set))

#dim(validation_pred)
validation_acc <- colMeans(validation_pred == validation_set$DEATH_EVENT)
validation_acc
#glm        lda        qda            rf      rpart svmLinear2 
#0.9666667  0.9666667  0.9000000  0.9666667  0.9333333  0.9666667 

mean(validation_acc)
#[1] 0.943

validation_votes <- rowMeans(validation_pred == "1")
validation_y_hat <- ifelse(validation_votes > 0.5, "1", "0")
mean(validation_y_hat == validation_set$DEATH_EVENT)
#[1] 0.9666667

# We have a very promising accuracy of our models on the validation_set that have accuracy ranges from 90% to 96.67%

# We are going to store all the methods and accuracy values in results variable as follow:
results <- tibble(method = "Decision Tree (rpart)", accuracy = validation_acc[5])
results <- bind_rows(results, tibble(method = "Ensemble", accuracy = mean(validation_y_hat == validation_set$DEATH_EVENT)))
results <- bind_rows(results, tibble(method = "Generalized Linear Model (glm)", accuracy = validation_acc[1]))
results <- bind_rows(results, tibble(method = "Linear Discriminant Analysis (lda)", accuracy = validation_acc[2]))
results <- bind_rows(results, tibble(method = "Quadratic Discriminant Analysis (qda)", accuracy = validation_acc[3]))
results <- bind_rows(results, tibble(method = "Random Forest (rf)", accuracy = validation_acc[4]))
results <- bind_rows(results, tibble(method = "Support Vector Machine (svmLinear2)", accuracy = validation_acc[6]))
results


# Accuracy of different Classified Models in a bar chart
chart_results <- as.data.frame(results) %>% 
  mutate(accuracy = round(accuracy * 100, digits = 2)) %>%
  select(method, accuracy)
#str(chart_results)

ggplot(data=chart_results, aes(x=method, y=accuracy, fill=method)) + 
  geom_bar(position = 'dodge', stat='identity') +
  geom_text(aes(label=accuracy), position=position_dodge(width=0.9), vjust=-0.25) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none") +
  ggtitle("Accuracy of different Classifier Models") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ylim(0,100) +
  xlab("Classifier Models") +
  ylab("% of Accuracy") 


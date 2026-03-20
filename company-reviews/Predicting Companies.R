library("readxl")
library(keras)
library(dplyr)
library(ggplot2)
library(purrr)
library(magrittr)
library(tm)
library(SnowballC)
library(caTools)
library(randomForest)
library(e1071)
library(caret)
library(zoo)
library(rpart)
library(rpart.plot)
library(forcats)
library(tidytext)
library(rsample)
library(glmnet)
library(doMC)
library(broom)
library(yardstick)
library(ggplot2)

setwd("/Users/lowieholemans/Desktop/HIR Master 2/Erasmus CPH/COURSES/Big Data/exam project/scraped data")

df <- read_excel('glassdoor_and_indeed_merged.xlsx')
df
attach(df)


#VECTOR REPRESENTATION OF TEXT DATA 

#create the corpus 
corpus = Corpus(VectorSource(df$Text))
corpus[[1]][1]
df$Rating[1]

#conversion to lowercase 
corpus = tm_map(corpus, PlainTextDocument)
corpus = tm_map(corpus, tolower)
corpus[[1]][1]

#remove punct 
corpus = tm_map(corpus, removePunctuation)
corpus[[1]][1]

#remove stopwords 
corpus = tm_map(corpus, removeWords, c(stopwords("english")))
corpus[[1]][1]

#remove our own stopwords 
corpus = tm_map(corpus, removeWords, c("work","office","job","indeed","glassdoor","salary","position"))
corpus[[1]][1]

#stemming
corpus = tm_map(corpus, stemDocument)
corpus[[1]][1]

#create document term matrix 
frequencies = DocumentTermMatrix(corpus)
#remove sparse words (to lower dimensions)
sparse = removeSparseTerms(frequencies, 0.995)


#convert to dataframe 
tSparse = as.data.frame(as.matrix(sparse))
colnames(tSparse) = make.names(colnames(tSparse))
tSparse$Rating = df$Rating
tSparse$Company = df$Company

prop.table(table(tSparse$Rating)) #clearly class imbalance but many classes so less of an issue 
prop.table(table(tSparse$Company)) #also class imbalance but many classes so less of an issue 

######## -------- PREDICT THE RATING -------- ###########
#split into 
set.seed(100)
split = sample.split(tSparse$Rating, SplitRatio = 0.7) #Company instead of Rating for company classif. 
trainSparse = subset(tSparse, split==TRUE)
testSparse = subset(tSparse, split==FALSE)

trainSparse$Rating = as.factor(trainSparse$Rating)
testSparse$Rating = as.factor(testSparse$Rating )

#Naive bayes 
set.seed(120)  # Setting Seed
NB_model <- naiveBayes(Rating ~ ., data = trainSparse)
predictNB = predict(NB_model, newdata=testSparse)
NB_table <- table(testSparse$Rating, predictNB)
confusionMatrix(NB_table) #accuracy 35.85%

#random forest 
RF_model = randomForest(trainSparse$Rating ~ ., data=trainSparse)
predictRF = predict(RF_model, newdata=testSparse)
RF_table <- table(testSparse$Rating, predictRF)
confusionMatrix(RF_table) #accuracy 44.11%

#classification tree 
CT_model <- rpart(Rating ~ ., data = trainSparse, method = "class")
rpart.plot(CT_model)
#summary based on complexity parameter 
summary(CT_model, "cp")
plotcp(CT_model, upper = "splits")
# Prune the Tree To have ? Splits
CT_model_pruned <- prune(CT_model, cp = 0.026)
rpart.plot(CT_model_pruned)
#predictions
predict(CT_mode_pruned, testSparse, type="class")

#also see KNIME workflow for rating prediction 



######## -------- PREDICT THE COMPANY -------- ###########

trainSparse$Company = as.factor(trainSparse$Company)
testSparse$Company = as.factor(testSparse$Company )


#RF
RF_model = randomForest(trainSparse$Company ~ ., data=trainSparse)
predictRF = predict(RF_model, newdata=testSparse)
RF_table <- table(testSparse$Company, predictRF)
confusionMatrix(RF_table) #accuracy:73.86%

#NB
set.seed(120)  # Setting Seed
NB_model <- naiveBayes(trainSparse$Company ~ ., data = trainSparse)
NB_model
predictNB = predict(NB_model, newdata=testSparse)
NB_table <- table(testSparse$Company, predictNB)
confusionMatrix(NB_table) #accuracy: 2.85% ... only predicts Carlsberg. 

#logistic regression 
library(tidytext)
tidy_companies <- df %>%
  unnest_tokens(word, Text) %>%
  group_by(word) %>%
  filter(n() > 10) %>%
  ungroup()

tidy_companies %>%
  count(Company, word, sort = TRUE) %>%
  anti_join(get_stopwords()) %>%
  group_by(Company) %>%
  top_n(20) %>%
  ungroup() %>%
  ggplot(aes(reorder_within(word, n, Company), n,
             fill = Company
  )) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  scale_x_reordered() +
  coord_flip() +
  facet_wrap(~Company, scales = "free") +
  scale_y_continuous(expand = c(0, 0)) +
  labs(
    x = NULL, y = "Word count",
    title = "Most frequent words after removing stop words",
    subtitle = "Words like 'company' and 'work' occupy similar ranks but other words are quite different"
  )

library(rsample)
companies_split <- df %>%
  select(Document) %>%
  initial_split()
train_data <- training(companies_split)
test_data <- testing(companies_split)

#transform to a sparse matrix to use in ML algo 
sparse_words <- tidy_companies %>%
  count(Document, word) %>%
  inner_join(train_data) %>%
  cast_sparse(Document, word, n)

class(sparse_words)
dim(sparse_words) #7539 train observations and 2717 features so a very high 
#dimensional text feature space 

word_rownames <- as.integer(rownames(sparse_words))
companies_joined<- data_frame(Document = word_rownames) %>%
  left_join(df %>%
              select(Document, Company))

library(glmnet)
library(doMC)
registerDoMC(cores = 8)

y <- companies_joined$Company
model <- cv.glmnet(sparse_words, y,
                   family = "multinomial",
                   parallel = TRUE, keep = TRUE)


y_predict_lr <- predict(model, newx = sparse_words,type ='class',s=0.01)

#via python: ConfusionMatrix (see lab report)
#accuracy (via python): 76.177%
#or via R: 
y_true <- read_excel('y_true.xlsx')
y_pred <- read_excel('y_pred.xlsx')

LR_table <- table(y_true$Actual, y_pred$Predicted)
confusionMatrix(LR_table) #accuracy:76.18%

#try ridge regression with all possible parameter values 
grid = 10^seq(10, -2, length = 100)
ridge_mod = cv.glmnet(sparse_words, y, family = "multinomial", alpha = 0, lambda = grid)
y_pred_ridge <- predict(ridge_mod, newx = sparse_words,type ='class',s=0.01)


write.csv(y_pred_ridge,"/Users/lowieholemans/Desktop/HIR Master 2/Erasmus CPH/COURSES/Big Data/exam project/ridgepred.csv", row.names = FALSE)
ridge_pred <- read_excel('ridgepred.xlsx')
LR_table <- table(y_true$Actual, ridge_pred$Predicted)
confusionMatrix(LR_table) #accuracy:76.18%

#try lasso regression with all possible parameter values 
grid = 10^seq(10, -2, length = 100)
lasso_mod = cv.glmnet(sparse_words, y, family = "multinomial", alpha = 1, lambda = grid)
y_pred_lasso <- predict(lasso_mod, newx = sparse_words,type ='class',s=0.01)


write.csv(y_pred_lasso,"/Users/lowieholemans/Desktop/HIR Master 2/Erasmus CPH/COURSES/Big Data/exam project/scraped data/lassopred.csv", row.names = FALSE)
lasso_pred <- read_excel('lassopred.xlsx')
LR_table <- table(y_true$Actual, lasso_pred$lassopred)
confusionMatrix(LR_table) #accuracy:76.18%



##OVERSAMPLING 

#company class highly imbalanced --> oversampling
#plot classes without oversampling: 
barplot(prop.table(table(trainSparse$Company)))

attach(trainSparse)
df1 <- filter(trainSparse,Company == "Carlsberg") #147 rows 
df2 <- filter(trainSparse,Company == "Danske Bank") #258 rows 
df3 <- filter(trainSparse,Company == "Lego") #426 rows 
df4 <- filter(trainSparse,Company == "Maersk") #3044 rows 
df5 <- filter(trainSparse,Company == "Novo Nordisk") #1233 rows 
df6 <- filter(trainSparse,Company == "Pandora") #1906 rows 

library(zoo)
df1 <- coredata(df1)[rep(seq(nrow(df1)),40),]
df2 <- coredata(df2)[rep(seq(nrow(df2)),23),]
df3 <- coredata(df3)[rep(seq(nrow(df3)),14),]
df4 <- coredata(df4)[rep(seq(nrow(df4)),2),]
df5 <- coredata(df5)[rep(seq(nrow(df5)),5),]
df6 <- coredata(df6)[rep(seq(nrow(df6)),3),]


df_tot <- rbind(df1,df2,df3,df4,df5,df6) #now we have around 36k observations
#plot class distribution without oversampling 
barplot(prop.table(table(df_tot$Company)))

df_tot$Company = as.factor(df_tot$Company)
testSparse$Company = as.factor(testSparse$Company )
attach(df_tot)

#RF oversampled 
RF_model = randomForest(df_tot$Company ~ ., data=df_tot)
predictRF = predict(RF_model, newdata=testSparse)
RF_table <- table(testSparse$Company, predictRF)
confusionMatrix(RF_table) #accuracy:73.86%

#LR oversampled: to be added 
y_predict_lr <- predict(model, newx = df_tot,type ='class',s=0.01)

#GRID SEARCH 

metric <- "Accuracy"
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(123)
tunegrid <- expand.grid(.mtry=c(1:15))
rf_gridsearch <- train(tSparse$Company~., data=tSparse, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)


####### ------ PREDICT THE RATING -------- ########
#see KNIME workflow - 33% accuracy 



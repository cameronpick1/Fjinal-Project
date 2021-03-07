
x <- read.csv("data4.csv")
final <- x[!(is.na(x$Attrition1)),]

final$Attrition1 <- factor(final$Attrition1)

set.seed(1029)
library(caTools)
split <- sample.split(final$Attrition1, SplitRatio = 0.75)

dresstrain <- subset(final, split == TRUE)
dresstest <- subset(final, split == FALSE)
as.data.frame(table(dresstrain$Attrition1))
library(DMwR)
balanced.data <- SMOTE(Attrition1 ~., dresstrain, perc.over = 4800, k = 5, perc.under = 1000)
as.data.frame(table(balanced.data$Attrition1))
library(caret)  

model <- glm (Attrition1~., data=balanced.data, family = binomial)
summary(model)
predict <- predict(model, dresstest, type = 'response')
table(dresstest$Attrition1, predict > 0.5)

res <- predict(model, dresstest, type = "response")
res <- predict(model, dresstrain, type = "response")
confmatrix <- table(Actual_Value = dresstrain$Attrition1, Predicted_Value = res > 0.5)
confmatrix
(confmatrix[[1,1]] + confmatrix[[2,2]])/ sum(confmatrix)

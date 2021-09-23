
# Load TwitterSpam.csv
twitter.spam <- read.csv("TwitterSpam.csv")
View(twitter.spam)
library(ggplot2)  # for ggplot
library(dplyr)    # for pipe operators
library(scales)   # to remove scientific notation in graphs 


head(twiter.spam, 20)


dim(twitter.spam) # 2,000 observations and 14 variables 
# subset insted of dim?

# Check some of the first and last data
head(twitter.spam) # label = non-spammer
tail(twitter.spam) # label = spammer

# Define spammer and non-spammer variables 
twitter.spam.spammer <- with(twitter.spam, account_age[label == "spammer"]) 
twitter.spam.nonspammer <- with(twitter.spam, account_age[label == "non-spammer"])
head(twitter.spam.nonspammer)
mean(twitter.spam$no_follower)

# 
count(with(twitter.spam, no_follower))
count(with(twitter.spam, no_follower[label == "spammer"]))
count(with(twitter.spam, no_follower[label == "non-spammer"]))

# Variance and standard deviation for spammer
var(twitter.spam.spammer)     # 206990.5
sd(twitter.spam.spammer)      # 454.962

# Variance and standard deviation for non-spammer
var(twitter.spam.nonspammer)  # 305372.4
sd(twitter.spam.nonspammer)   # 552.6051


# 4. Compute the numeric summary of the 'no_tweets' column
twitter.spam.notweets <- with(twitter.spam, no_tweets)
summary(twitter.spam.notweets)
quantile(twitter.spam.notweets)

# 5. Plot the histogram to show the distribution of tweets number (column: no_tweets).
#    In the generated plot, the plot name is "histogram of Posted Tweets", Y-axis is frequency,
#    and X-axis is Tweets number.

hist(twitter.spam.notweets, 
     main = "Histogram of Posted Tweets", 
     xlab = "Tweets number", 
     ylab = "Frequency") + options(scipen = 999)


ggplot(twitter.spam, aes(x=no_tweets)) + 
  geom_histogram(bins = 12) +
  ggtitle("Histogram of Posted Tweets") +
  labs(x="Tweets number", y="Frequency")


# Without a range limitation
twitter.spam %>%
  ggplot(aes(x=no_follower, fill = label)) +
  geom_density(alpha=0.4) +
  facet_grid(label ~ .)

# With a range limitation
twitter.spam %>%
  filter(no_follower<6000) %>%
  ggplot(aes(x=no_follower, fill = label)) +
  geom_density(alpha=0.4) +
  facet_grid(. ~ label)  # Shows in horizontal direction
#facet_grid(label ~ .) # Shows in vertical direction


twitter.spam %>%
  filter(no_following<6000) %>%
  ggplot(aes(x=no_following, fill = label)) +
  geom_density(alpha = 0.4) +
  facet_grid(. ~ label)


############### no_follower ##################

# total followers
length(twitter.spam$no_follower) # 2000

# spammers in no_follower
with(twitter.spam, no_follower[label == "spammer"]) %>% length() #1000

# all followers the sum of spammer and non-spammer followers 
all.follower <- twitter.spam %>% 
  group_by(no_follower) %>%
  summarise( no_follower) %>%
  count()

View(all.follower)

# labelled followers separated into spammer and non-spammer caterogries
lab.follower <- twitter.spam %>% 
  group_by(no_follower, label) %>%
  summarise( no_follower, label) %>%
  count()

View(lab.follower)
summary(lab.follower)


############### no_following ##################

# total followewing
length(twitter.spam$no_following) # 2000

# spammers in no_followewing
with(twitter.spam, no_following[label == "spammer"]) %>% length() #1000

# all followewing the sum of spammer and non-spammer followers 
all.following <- twitter.spam %>% 
  group_by(no_following) %>%
  summarise( no_following) %>%
  count()

View(all.following)

# labelled followewing seperated into spammer and non-spammer caterogries
lab.following <- twitter.spam %>% 
  group_by(no_following, label) %>%
  summarise( no_following, label) %>%
  count()

View(lab.following)
summary(lab.following)

# spammer followewing only
spam.following <-twitter.spam %>%
  group_by(no_following, label) %>%
  summarise( no_following, label) %>%
  filter(label == "spammer") %>%
  count()

##############################################

# no_follower < 10 --> 545
twitter.spam %>%
  group_by(no_follower < 10, label) %>%
  summarise( no_follower, label) %>%
  filter(label == "spammer") %>% 
  count() %>%
  View()

# number of people who have followers fewer than 10 --> 643
twitter.spam %>%
  group_by(no_follower < 10) %>%
  summarise( no_follower) %>%
  count() %>%
  View()

# no_following < 10 --> 273
twitter.spam %>%
  group_by(no_following < 10, label) %>%
  summarise( no_following, label) %>%
  filter(label == "spammer") %>% 
  count() %>%
  View()

# number of people who have following fewer than 10 --> 353
twitter.spam %>%
  group_by(no_following < 10) %>%
  summarise( no_following) %>%
  count() %>%
  View()

# no_follower < 1,000 --> 885
twitter.spam %>%
  group_by(no_follower < 1000, label) %>%
  summarise( no_follower, label) %>%
  filter(label == "spammer") %>% 
  count() %>%
  View()

# number of people who have followers fewer than 1,000 --> 1,638
twitter.spam %>%
  group_by(no_follower < 1000) %>%
  summarise( no_follower) %>%
  count() %>%
  View()


# no_following < 1,000 --> 876
twitter.spam %>%
  group_by(no_following < 1000, label) %>%
  summarise( no_following, label) %>%
  filter(label == "spammer") %>% 
  count() %>%
  View()

# number of people who have following fewer than 1,000 --> 1,656
twitter.spam %>%
  group_by(no_following < 1000) %>%
  summarise( no_following) %>%
  count() %>%
  View()

twitter.spam %>%
  ggplot(aes(x=no_tweets, y = no_follower)) +
  geom_point() +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma)

# test the correlation between no_tweets and no_follower
cor.test(twitter.spam$no_tweets, twitter.spam$no_follower)

twitter.spam %>%
  filter(no_tweets<10000, no_follower<10000) %>%
  ggplot(aes(x=no_tweets, y = no_follower)) +
  geom_point() +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma)

twitter.spam %>%
  ggplot(aes(x=no_tweets, y = no_follower)) +
  geom_point() +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  geom_smooth(method=lm)

twitter.spam %>%
  filter(no_tweets<10000, no_follower<10000) %>%
  ggplot(aes(x=no_tweets, y = no_follower)) +
  geom_point() +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  geom_smooth(method=lm)

# total number of people having userfavorites fewer than 10
twitter.spam %>%
  group_by(no_userfavorites < 10) %>%
  summarise(no_userfavorites) %>%
  count() %>%
  View()

# when user favourites are fewer than 10, 708 out of 1,000 are a spammer
twitter.spam %>%
  group_by(no_userfavorites < 10, label) %>%
  summarise(no_userfavorites) %>%
  filter(label == "spammer") %>%
  count() %>%
  View()

# put no_follower, no_following, no_userfavorites, and label together
twitter.spam %>%
  filter(label == "spammer") %>%
  group_by(no_userfavorites, no_follower, no_following, label) %>%
  summarise(label) %>%
  arrange(no_following) %>%
  count() %>%
  View()


# Machine Learning

library(caret)
library(ranger)
library(tidyverse)


# # add a logical variable for "old" (age > 10)
# twitter.spam <- twitter.spam %>%
#   # create a column named old at the end of the last column
#   mutate(spammer = ifelse(label=="spammer", 1, 0)) %>%
#   # remove the "age" variable
#   select(-label)


set.seed(1400) # set a seed 
# split train and test datasets with the ratio of 8:2 
train.index <- sample(1:nrow(twitter.spam), 0.8 * nrow(twitter.spam))
tw.train <- twitter.spam[train.index, ] # 1600 observations
tw.test <- twitter.spam[-train.index, ] # 400 observations
# remove the original dataset
rm(twitter.spam)

summary(tw.train)
dim(tw.train)


# ML model with the Random Forest algorithm.

# fit a random forest model (using ranger)
# Ranger is a fast implementation of random forests 
rf.fit <- train(as.factor(label) ~ .,
                data = tw.train,
                method = "ranger")
rf.fit

# predict the outcome on a test set
tw.rf.pred <- predict(rf.fit, tw.test)
# compare predicted outcome and true outcome
confusionMatrix(tw.rf.pred, as.factor(tw.test$label))


# ML model with the K Nearest neighbours algorithm.

# knn.fit <- train(as.factor(spammer) ~ .,
knn.fit <- train(as.factor(label) ~ .,
                 data = tw.train,
                 method = "knn")
knn.fit

# predict the outcome on a test set
tw.knn.pred <- predict(knn.fit, tw.test)
# compare predicted outcome and true outcome
confusionMatrix(tw.knn.pred, as.factor(tw.test$label))













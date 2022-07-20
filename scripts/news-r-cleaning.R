#ORDER:
#importing libraries
#importing dataset
#investigating data types
#checking for missing values in dataset & online news respectively + plot of missing values
#checking for duplicate counts + number of duplicate entries in dataset & online news respectively
#EDA:
#bar chart of observations by source
#bar chart of observations by language
#bar chart of articles by language
#bar chart of articles by domain
#line chart of articles over time
#histogram of article lengths
#histogram of sentiments
#bar chart of articles by country

#importing libraries
library(readxl)
library(tidyverse)
library(lubridate)

#importing dataset
data <- read_xlsx("./nlp_proj/Copy of SG sanctions on Russia.xlsx")

#investigating data types
str(data)

#checking for missing values in entire dataset
for(col in colnames(data)){
  temp <- sum(is.na(data[col]))
  pcnt <- temp/nrow(data)
  print(paste(col,":", round(pcnt, 4)))
}

#plot of missing values by source: all missing values come from forums & reddit
missing <- data[which(is.na(data["content"])),]
ggplot(missing) +
  geom_bar(aes(source))


# checking for exact duplicates (approximate duplicates tbc)
dup2 <- data %>%
  group_by(source, title, content, date) %>%
  summarise(cnt = n())
  
dup1 <- dup2 %>%
  filter(cnt > 1)

print(paste("There are a total of", sum(dup1["cnt"]), "duplicated entries"))
print(paste("There are a total of", sum((dup1 %>% filter(source == "Online News"))["cnt"]), " EXACTLY duplicated entries for Online News"))
#print(paste("There should be", nrow(dup2 %>% filter(source == "Online News")), "distinct entries for Online News"))
#12240


#EXPLORATORY DATA ANALYSIS

#Bar chart of frequency of observations by source
freq_sources <- data.frame(table(data$source))
ggplot(data) +
  geom_bar(aes(source))


#Frequency table & Bar chart of frequency of observations by language
table(data$language)
ggplot(data) +
  geom_bar(aes(language)) +
  ggtitle("Bar chart of frequency of observations by language")


#Bar chart of frequency of observations by language (excluding English, which, based on previous chart, takes up HUGE majority)
ggplot(data %>% filter(language != "en")) +
  geom_bar(aes(language)) + 
  ggtitle("Bar chart of frequency of observations by language (excluding English)")


#Bar chart of frequency of observations in online news by language
online_news <- data %>%
  filter(source == "Online News")
ggplot(online_news) +
  geom_bar(aes(language)) +
  ggtitle("Bar chart of frequency of observations in online news by language")

ggplot(online_news %>% filter(language != "en")) +
  geom_bar(aes(language)) +
  ggtitle("Bar chart of frequency of observations in online news by language (excluding English)")

#Analysis of online news by website domain
print(paste("Number of distinct website domains:", n_distinct(online_news$domain)))

freq_sources <- data.frame(table(online_news$domain)) %>%
  arrange(desc(Freq)) %>%
  rename(domain_name = "Var1") %>%
  head(10)
freq_sources
#Bar plot of frequency of online news observations based on website domain (for top 10 website domains)
ggplot(freq_sources) +
  geom_col(aes(x = domain_name, y = Freq)) +
  ggtitle("Bar plot of frequency of online news observations based on website domain (for top 10 website domains)")

# Frequency of articles over time
temp <- online_news %>%
  mutate(date_col = as.character(date(date))) %>%
  group_by(date_col) %>%
  summarize(cnt = n()) %>%
  mutate(date_col = as.Date(date_col))

ggplot(data=temp, aes(x=date_col, y=cnt, group=1)) +
  geom_line() +
  scale_x_date(date_breaks = "2 weeks") +
  ggtitle("Frequency of articles over time")


#Frequent topics 
print(paste("Ratio of articles without topics:", round(sum(is.na(online_news["topics"]))/nrow(online_news), 5)))
freq_topics <- data.frame(table(online_news$topics)) %>%
  arrange(desc(Freq)) %>%
  head(30)
print("Frequent topics:")
freq_topics

#Histogram of Distribution of Article Lengths (by Number of Words)
article_lengths <- lengths(strsplit(online_news$content, '\\S+'))
ggplot(mapping = aes(article_lengths)) + geom_histogram()

# Histogram of Distribution of Sentiments
ggplot(data = online_news,aes(x = sentiment)) + geom_histogram()

# Bar Chart of Number of Observations of Online news by Country (for top 10 countries)
freq_countries <- data.frame(table(online_news$country)) %>%
  arrange(desc(Freq)) %>%
  rename(country = Var1) %>%
  head(10)
freq_countries

ggplot(freq_countries) +
  geom_col(aes(x = country, y = Freq)) +
  ggtitle("Bar Chart of Number of Observations of Online news by Country (for top 10 countries)")


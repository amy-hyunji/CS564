library(readr)
df <- read_csv("R.csv")
dff <- read_csv("RR.csv")
dfg <- read_csv("R_gender.csv")

tmp <- subset(df, df$author_num <= 15)
hist(tmp$author_num)

tmp_u <- subset(tmp, tmp$con == 1)
hist(tmp_u$author_num)
tmp_c <- subset(tmp, tmp$con == 0)
hist(tmp_c$author_num)
boxplot(tmp_u$author_num, tmp_c$author_num, horizontal = T)

tmp <- subset(dff, dff$author_num <= 15)
boxplot(tmp$author_num ~ tmp$category, horizontal = T)

boxplot(df$subjectivity ~ df$country)
boxplot(df$polarity ~ df$country)

boxplot(we ~ category, horizontal = T, data=dff)

summary(lm(formula = df$author_num ~ df$year + df$con))
summary(lm(formula = i ~ year + con + author_num, data = df))

tmp <- subset(df, df$i > 0)
summary(lm(formula = i ~ year + con + author_num, data = tmp))
summary(lm(formula = we ~ year + con + author_num, data = df))

tmp <- subset(df, df$we > 0)
summary(lm(formula = we ~ year + con + author_num, data = tmp))

summary(glm(formula = gen ~ year + con + author_num, data = dfg))

tmp <- subset(df, df$gender > 0)
summary(lm(formula = gender ~ year + con + author_num, data = df))
summary(lm(formula = gender ~ year + con + author_num, data = tmp))
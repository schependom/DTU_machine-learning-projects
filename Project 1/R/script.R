abalone <- read.csv("~/Documents/Code/DTU_machine-learning-projects/Project 1/R/abalone.data", header=FALSE)

View(abalone)

names(abalone) <- c("Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings")



attach(abalone)

X = abalone[-c(1,9)]

pca = prcomp(scale(X))
summary(pca)
attributes(pca)
pca$rotation
plot(pca)

detach(abalone)




glass <- read.csv("~/Documents/Code/DTU_machine-learning-projects/Project 1/R/glass.data", header=FALSE)


summary(glass)

pca2 = prcomp(scale(glass))
plot(pca2)
pca2$rotation


tomodata <- read.csv("~/Software/NN_QTomo/fids_dim2.csv")

library(reshape2)
library(ggplot2)

melted <- melt(tomodata)

ggplot(data = melted, aes(value, fill = variable)) +
  geom_density(alpha = 0.2) +
  ggtitle("Distribution of fidelities for Neural Network vs. LBMLE incomplete tomography")

library(ggplot2)
library(reshape2)

d_zy <- read.csv("~/Desktop/angle_investigation_dim2_zy_lbmle.csv")
d_xy <- read.csv("~/Desktop/angle_investigation_dim2_xy_lbmle.csv")

melted_zy <- melt(d_zy, id = "Angle")
melted_xy <- melt(d_xy, id = "Angle")

ggplot(melted_zy, aes(Angle, value, colour = variable)) +
  geom_point() +
  ggtitle("Fidelity vs. Bloch polar angle for single qubit 
          reconstruction, measurements in Z and Y MUBs") +
  ylab("Fidelity") +
  xlab("Angle (/ pi)") 

ggplot(melted_xy, aes(Angle, value, colour = variable)) +
  geom_point() +
  ggtitle("Fidelity vs. Bloch polar angle for single qubit 
          reconstruction, measurements in X and Y MUBs") +
  ylab("Fidelity") +
  xlab("Angle (/ pi)")

ggplot(d_zy, aes(Angle)) +
  geom_histogram()
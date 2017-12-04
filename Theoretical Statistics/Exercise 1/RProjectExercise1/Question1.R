# Fitting a simple linear regression model to the data.
# The "y" axis is the stopping distance and the "x" axis is the speed of the car.
# This is because the cause of the distance is the speed of the car.

cars <- read.table ('cars.txt', header=TRUE, sep=' ')
cars[1:10,]
